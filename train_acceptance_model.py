"""Collect plain-MCMF offers and evaluate binary driver acceptance learning."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from src.acceptance_model import (
    BinaryAcceptanceModel, FEATURE_NAMES, collect_offers, probability_metrics,
)
from src.charging_metrics import charging_session_metrics
from src import synthetic_scenario as scenario


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["synthetic", "nyc"], default="synthetic")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(100, 120)))
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=list(range(200, 206)))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(range(300, 310)))
    parser.add_argument("--num-vehicles", type=int, default=60)
    parser.add_argument("--num-ev", type=int, default=30)
    parser.add_argument("--mcmf-backend", choices=["primal_dual", "ortools", "gurobi_network"], default="primal_dual")
    parser.add_argument("--grid-size", type=int, default=scenario.DEFAULT_GRID_SIZE)
    parser.add_argument("--simulation-period", type=int, default=scenario.DEFAULT_SIMULATION_PERIOD)
    parser.add_argument("--episode-days", type=int, default=scenario.DEFAULT_EPISODE_DAYS)
    parser.add_argument("--synthetic-demand-scale", type=float, default=scenario.DEFAULT_SYNTHETIC_DEMAND_SCALE)
    parser.add_argument("--parquet-path", type=Path, default=ROOT / "nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet")
    parser.add_argument("--station-csv", type=Path, default=ROOT / "nyedata/nyc_all_charging_stations.csv")
    parser.add_argument("--date", default="2025-12-18")
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--stop-hour", type=float, default=10.0)
    parser.add_argument("--epoch-length", type=float, default=30.0)
    parser.add_argument('--nn-epochs', type=int, default=120)
    parser.add_argument('--nn-patience', type=int, default=20)
    parser.add_argument('--nn-seed', type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    groups = [args.train_seeds, args.validation_seeds, args.test_seeds]
    flat = [seed for group in groups for seed in group]
    if len(flat) != len(set(flat)):
        parser.error("Training, validation and test seeds must be unique and disjoint")
    if not 0 < args.num_ev <= args.num_vehicles:
        parser.error("Require 0 < num-ev <= num-vehicles")
    if min(args.grid_size, args.simulation_period, args.episode_days, args.epoch_length) <= 0:
        parser.error("Grid size and simulation durations must be positive")
    if not 0 <= args.start_hour < args.stop_hour <= 24:
        parser.error("Require 0 <= start-hour < stop-hour <= 24")
    if min(args.nn_epochs, args.nn_patience) <= 0:
        parser.error('Neural epochs and patience must be positive')
    return args


def make_environment(args, seed):
    shared = dict(num_vehicles=args.num_vehicles, ev_num_vehicles=args.num_ev,
                  random_seed=seed, assignmentgurobi=True, usemcmf=True,
                  mcmf_solver="exact", mcmf_backend=args.mcmf_backend,
                  mcmf_strict=True, knownreject=False, daily_drop_off=False,
                  ifreject=True)
    if args.environment == "synthetic":
        from src.Environment import ChargingIntegratedEnvironment
        env = ChargingIntegratedEnvironment(
            **shared, grid_size=args.grid_size,
            simulation_period=args.simulation_period, episode_days=args.episode_days,
            synthetic_demand_profile=scenario.DEFAULT_SYNTHETIC_DEMAND_PROFILE,
            synthetic_demand_scale=args.synthetic_demand_scale,
            num_stations=scenario.DEFAULT_NUM_STATIONS,
            station_capacity=scenario.DEFAULT_STATION_CAPACITY,
            station_queue_capacity=scenario.DEFAULT_STATION_QUEUE_CAPACITY,
            charge_duration=scenario.DEFAULT_CHARGE_DURATION,
            aev_initial_battery_scale=scenario.DEFAULT_AEV_INITIAL_BATTERY_SCALE,
            critical_charging_battery=scenario.DEFAULT_CRITICAL_CHARGING_BATTERY,
            charging_wait_penalty_per_step=scenario.DEFAULT_WAIT_PENALTY_PER_STEP,
            battery_first=False,
        )
    else:
        from src.NYCEnvironment import NYCEnvironment
        env = NYCEnvironment(
            **shared, parquet_path=str(args.parquet_path), station_csv=str(args.station_csv),
            start_date=args.date, end_date=args.date,
            start_hour=args.start_hour, stop_hour=args.stop_hour,
            epoch_length_sec=args.epoch_length,
            episode_length=int(np.ceil((args.stop_hour - args.start_hour) * 3600 / args.epoch_length)),
        )
    # Ordinary MCMF, no ADP score, no known-probability MCMF-K correction.
    # Evaluation here disables RL updates only; our passive offer collector
    # still collects labels for supervised fitting after all episodes finish.
    env.adp_value = 0.0
    env.evaluatemode = True
    env.reset()
    return env


def collect_split(args, split, seeds, output):
    all_rows, episodes = [], []
    for seed in seeds:
        started = time.perf_counter()
        episode_id = f"{args.environment}:{split}:seed-{seed}"
        solver_calls, backends = 0, set()
        with (output / f"{split}-{seed}.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                env = make_environment(args, seed)
                with collect_offers(env, episode_id=episode_id, seed=seed) as rows:
                    for _ in range(env.episode_length):
                        previous_result = getattr(env, "mcmf_last_result", None)
                        actions, stored, stored_ev = env.simulate_motion(
                            agents=[], current_requests=list(env.active_requests.values()), rebalance=True
                        )
                        result = getattr(env, "mcmf_last_result", None)
                        if result is not None and result is not previous_result:
                            if not result["optimal"] or result["solver_fallback_used"]:
                                raise RuntimeError(f"Plain exact MCMF requirement violated: {result}")
                            solver_calls += 1
                            backends.add(result["backend"])
                        _, _, _, done, _ = env.step(actions, stored, stored_ev)
                        if done:
                            break
                days = (env.current_time * env.EPOCH_LENGTH / 86400
                        if args.environment == "nyc"
                        else env.current_time / env.simulation_period)
                charge_metrics = charging_session_metrics(env.vehicles, days)
        if not solver_calls:
            raise RuntimeError("No exact MCMF solves were observed in this episode")
        all_rows.extend(rows)
        episode = {"episode_id": episode_id, "seed": seed, "offers": len(rows),
                   "accepted": sum(row["accepted"] for row in rows),
                   "mcmf_calls": solver_calls, "mcmf_backends": sorted(backends),
                   "elapsed_seconds": time.perf_counter() - started,
                   "charging": charge_metrics}
        episodes.append(episode)
        print(f"{episode_id}: offers={len(rows)}, accepted={episode['accepted']}, "
              f"MCMF={solver_calls}, seconds={episode['elapsed_seconds']:.1f}", flush=True)
    with (output / f"{split}_offers.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    if not all_rows:
        raise RuntimeError(f"No human-driver offers in {split}; increase simulation coverage")
    return all_rows, episodes


def clustered_improvement_intervals(rows, probabilities, baseline):
    """Paired 95% bootstrap CIs; resample whole seeds, never individual offers."""
    groups = sorted({row["episode_id"] for row in rows})
    if len(groups) < 2:
        return None
    summaries = []
    for group in groups:
        indexes = [i for i, row in enumerate(rows) if row["episode_id"] == group]
        subset = [rows[i] for i in indexes]
        model = probability_metrics(subset, probabilities[indexes])
        constant = probability_metrics(subset, np.full(len(subset), baseline))
        summaries.append([len(subset),
                          len(subset) * (constant["log_loss"] - model["log_loss"]),
                          len(subset) * (constant["brier_score"] - model["brier_score"])])
    summaries = np.asarray(summaries)
    indexes = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    boot = summaries[indexes].sum(axis=1)
    gains = boot[:, 1:] / boot[:, :1]
    return {key: np.quantile(gains[:, column], [0.025, 0.975]).tolist()
            for column, key in enumerate(("log_loss_gain", "brier_score_gain"))}


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir or ROOT / "results/acceptance_model" / (
        args.environment + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    data, episode_stats = {}, {}
    for split, seeds in (("train", args.train_seeds), ("validation", args.validation_seeds)):
        data[split], episode_stats[split] = collect_split(args, split, seeds, output)
    candidates = []
    for l2 in (0.0, 1e-5, 1e-4, 1e-3):
        model = BinaryAcceptanceModel(l2=l2, max_epochs=args.nn_epochs, patience=args.nn_patience,
                                      seed=args.nn_seed).fit(data["train"], validation_rows=data['validation'])
        validation_loss = model.loss_history[model.selected_epoch]['validation_binary_cross_entropy']
        candidates.append((validation_loss, model))
    _, model = min(candidates, key=lambda item: item[0])
    model.save(output / "model.json")
    restored = BinaryAcceptanceModel.load(output / "model.json")
    (output / 'loss_history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in model.loss_history))
    (output / 'candidate_loss_histories.json').write_text(json.dumps(
        {str(candidate.l2): candidate.loss_history for _, candidate in candidates}, indent=2) + '\n')
    data['test'], episode_stats['test'] = collect_split(args, 'test', args.test_seeds, output)
    metrics = {}
    for split, rows in data.items():
        predictions = model.predict_proba(rows)
        if not np.array_equal(predictions, restored.predict_proba(rows)):
            raise AssertionError("Saved-model inference differs from training-time inference")
        metrics[split] = {
            "model": probability_metrics(rows, predictions),
            "constant_train_rate": probability_metrics(rows, np.full(len(rows), model.training_acceptance_rate)),
            "simulator_oracle": probability_metrics(rows, [row["oracle_acceptance_probability"] for row in rows]),
        }
    test_p = model.predict_proba(data["test"])
    confidence = clustered_improvement_intervals(data["test"], test_p, model.training_acceptance_rate)
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, probability in zip(data["test"], test_p):
            stream.write(json.dumps({**row, "predicted_acceptance_probability": float(probability)}) + "\n")
    summary = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "collection": "integrated / plain exact MCMF / ADP=0 / knownreject=False / recourse=legacy",
        "label": "Actual Bernoulli response: accepted=1, rejected=0; no class balancing",
        "oracle_usage": "Evaluation only; excluded from features, fitting and model selection",
        "split": "Disjoint simulation seeds; NYC splits reuse the same fixed demand day, not held-out real-driver data",
        "model": model.to_dict(), "metrics": metrics, "episodes": episode_stats,
        "validation_selection": [{"l2": candidate.l2, "log_loss": loss} for loss, candidate in candidates],
        "test_seed_clustered_95pct_ci": confidence,
        "checkpoint_prediction_roundtrip_exact": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("train_acceptance_model.py", "src/acceptance_model.py", "src/acceptance_inputs.py", "src/Environment.py", "src/NYCEnvironment.py")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test = metrics["test"]["model"]
    baseline = metrics["test"]["constant_train_rate"]
    report = ["# MCMF 人类司机接单概率验证", "",
              f"环境：{args.environment}；采集策略：普通 exact MCMF，未使用真实接单概率。", "",
              "模型：30→64→32→1 神经网络（ReLU / sigmoid）；标签 1=接单、0=拒单；保持实际类别比例。",
              f"特征：{', '.join(FEATURE_NAMES)}；单位模式：{model.feature_schema}。",
              "损失 BCEWithLogitsLoss + 权重 L2；Adam 优化。训练/验证/测试按种子分离；仅验证 BCE 选正则强度并早停。",
              f"训练 {model.epochs_run} epochs，恢复最佳 epoch {model.selected_epoch}；无逻辑回归回退。", "",
              f"样本量：训练 {len(data['train'])}，验证 {len(data['validation'])}，测试 {len(data['test'])}。", "",
              "| 测试指标 | Binary 模型 | 训练集接单率常数基线 |", "|---|---:|---:|"]
    for key in ("log_loss", "brier_score", "roc_auc", "accuracy_at_0_5", "predicted_acceptance_rate", "ece_10_bins", "oracle_probability_mae"):
        report.append(f"| {key} | {test[key]} | {baseline[key]} |")
    report.extend(["", f"测试实际接单率：{test['acceptance_rate']:.4%}。",
                   f"按种子重采样的改进量 95% CI（正数表示优于常数基线）：{confidence}。", "",
                   "已验证保存/重新加载模型后的概率与保存前逐项一致。原有充电次数统计保留在 summary.json。", "",
                   "## 结论边界", "",
                   "这是对仿真司机响应规律的预测验证，不是对真实人类司机的外部验证。随机响应不能被逐单确定性预测。",
                   "NYC 使用真实订单，但司机标签仍是仿真生成；不同种子复用同一天需求，不能据此声称跨日期泛化。",
                   "模型尚未接入 MCMF 打分，采集时不会改变原始分配或使用模型自生成标签。", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"test": test, "baseline": baseline, "ci": confidence}, indent=2), flush=True)
    print(f"Model and report saved: {output}", flush=True)


if __name__ == "__main__":
    main()
