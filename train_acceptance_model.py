"""Train calibrated EV rejection probabilities from mixed feasible real offers."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from src.acceptance_model import (
    BinaryAcceptanceModel, FEATURE_NAMES, collect_offers, probability_metrics,
)
from src.charging_metrics import charging_session_metrics
from src.rejection_collection import mixed_feasible_offers, parse_mixture
from src.acceptance_inputs import FEATURE_VARIANTS
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
    parser.add_argument("--nyc-demand-scale", type=float, default=1.0)
    parser.add_argument("--station-capacity-scale", type=float, default=1.0)
    parser.add_argument("--battery-consumption-ratio", type=float, default=1.0)
    parser.add_argument("--initial-battery-mean", type=float, default=0.875)
    parser.add_argument("--charge-duration-scale", type=float, default=1.0)
    parser.add_argument("--parquet-path", type=Path, default=ROOT / "nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet")
    parser.add_argument("--station-csv", type=Path, default=ROOT / "nyedata/nyc_all_charging_stations.csv")
    parser.add_argument("--date", default="2025-12-18")
    for split in ('train', 'validation', 'test'):
        parser.add_argument(f'--{split}-dates', nargs='+', default=None)
    parser.add_argument('--ev-response-target', choices=['rejection'], default='rejection')
    parser.add_argument('--ev-response-feature-variant', choices=list(FEATURE_VARIANTS), default='driver_offer_core')
    parser.add_argument('--ev-response-calibration', choices=['temperature', 'platt', 'none'], default='temperature')
    parser.add_argument('--ev-response-behavior-policy-mixture', default='0.8,0.1,0.1',
                        help='MCMF,stratified,random probabilities; test always uses pure MCMF')
    parser.add_argument('--max-pickup-distance-km', type=float, default=2.)
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--stop-hour", type=float, default=10.0)
    parser.add_argument("--epoch-length", type=float, default=30.0)
    parser.add_argument('--nn-epochs', type=int, default=120)
    parser.add_argument('--nn-patience', type=int, default=20)
    parser.add_argument('--nn-seed', type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        parse_mixture(args.ev_response_behavior_policy_mixture)
    except ValueError as exc:
        parser.error(str(exc))
    date_groups = [getattr(args, split + '_dates') for split in ('train', 'validation', 'test')]
    if any(date_groups):
        if not all(date_groups):
            parser.error('Provide all three date splits or none (explicit same-day response holdout)')
        dates = [day for group in date_groups for day in group]
        if len(dates) != len(set(dates)):
            parser.error('Whole train/validation/test dates must be disjoint')
    groups = [args.train_seeds, args.validation_seeds, args.test_seeds]
    flat = [seed for group in groups for seed in group]
    if len(flat) != len(set(flat)):
        parser.error("Training, validation and test seeds must be unique and disjoint")
    if not 0 < args.num_ev <= args.num_vehicles:
        parser.error("Require 0 < num-ev <= num-vehicles")
    if min(args.grid_size, args.simulation_period, args.episode_days, args.epoch_length) <= 0:
        parser.error("Grid size and simulation durations must be positive")
    if min(args.nyc_demand_scale, args.station_capacity_scale,
           args.battery_consumption_ratio, args.initial_battery_mean,
           args.charge_duration_scale) <= 0:
        parser.error('NYC demand/energy sensitivity parameters must be positive')
    if args.initial_battery_mean > 1.0:
        parser.error('initial-battery-mean must not exceed 1')
    if not 0 <= args.start_hour < args.stop_hour <= 24:
        parser.error("Require 0 <= start-hour < stop-hour <= 24")
    if min(args.nn_epochs, args.nn_patience) <= 0:
        parser.error('Neural epochs and patience must be positive')
    if not np.isfinite(args.max_pickup_distance_km) or args.max_pickup_distance_km <= 0:
        parser.error('Pickup radius must be finite and positive')
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
            demand_scale=args.nyc_demand_scale,
            station_capacity_scale=args.station_capacity_scale,
            battery_consumption_ratio=args.battery_consumption_ratio,
            initial_battery_mean=args.initial_battery_mean,
            charge_duration_scale=args.charge_duration_scale,
        )
    # Ordinary MCMF, no ADP score, no known-probability MCMF-K correction.
    # Evaluation here disables RL updates only; our passive offer collector
    # still collects labels for supervised fitting after all episodes finish.
    env.adp_value = 0.0
    env.reject_uniform = True
    if args.environment == 'nyc':
        env.assignmentrange = args.max_pickup_distance_km
    env.evaluatemode = True
    env.reset()
    return env


def collect_split(args, split, seeds, output):
    all_rows, episodes = [], []
    dates = getattr(args, split + '_dates', None) or [args.date if args.environment == 'nyc' else 'synthetic']
    feasible_rows = []
    for day, seed in ((day, seed) for day in dates for seed in seeds):
        started = time.perf_counter()
        episode_id = f"{args.environment}:{split}:{day}:seed-{seed}"
        solver_calls, backends = 0, set()
        with (output / f"{split}-{day}-{seed}.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                episode_args = copy(args)
                episode_args.date = day
                env = make_environment(episode_args, seed)
                mixture = (1., 0., 0.) if split == 'test' else parse_mixture(args.ev_response_behavior_policy_mixture)
                with mixed_feasible_offers(env, seed=seed, mixture=mixture,
                        feature_variant=args.ev_response_feature_variant) as support, collect_offers(
                        env, episode_id=episode_id, seed=seed, day_id=day,
                        feature_variant=args.ev_response_feature_variant) as rows:
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
        feasible_rows.extend(support['feasible_rows'])
        episode = {"episode_id": episode_id, "seed": seed, "offers": len(rows),
                   "accepted": sum(row["accepted"] for row in rows),
                   "mcmf_calls": solver_calls, "mcmf_backends": sorted(backends),
                   "elapsed_seconds": time.perf_counter() - started,
                   "charging": charge_metrics, "day_id": day,
                   "behavior_policy_counts": support['policy_counts']}
        episodes.append(episode)
        print(f"{episode_id}: offers={len(rows)}, accepted={episode['accepted']}, "
              f"MCMF={solver_calls}, seconds={episode['elapsed_seconds']:.1f}", flush=True)
    with (output / f"{split}_offers.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    if not all_rows:
        raise RuntimeError(f"No human-driver offers in {split}; increase simulation coverage")
    with (output / f'{split}_feasible_features.jsonl').open('w', encoding='utf-8') as stream:
        for row in feasible_rows:
            stream.write(json.dumps(row) + '\n')
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
    source_paths = sorted([ROOT / 'train_acceptance_model.py', *ROOT.glob('src/**/*.py')])
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    with tarfile.open(output / 'source_archive.tar.gz', 'w:gz') as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    data, episode_stats = {}, {}
    for split, seeds in (("train", args.train_seeds), ("validation", args.validation_seeds)):
        data[split], episode_stats[split] = collect_split(args, split, seeds, output)
    candidates = []
    for l2 in (0.0, 1e-5, 1e-4, 1e-3):
        model = BinaryAcceptanceModel(l2=l2, max_epochs=args.nn_epochs, patience=args.nn_patience,
                                      feature_variant=args.ev_response_feature_variant,
                                      calibration=args.ev_response_calibration,
                                      seed=args.nn_seed).fit(data["train"], validation_rows=data['validation'])
        validation_loss = model.calibration['validation_nll_after']
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
            "constant_train_rate": probability_metrics(rows, np.full(len(rows), model.training_rejection_rate)),
            "simulator_oracle": probability_metrics(rows, [row["oracle_rejection_probability"] for row in rows]),
            "selected_support": model.support_diagnostics(rows),
            "feasible_support": model.support_diagnostics([json.loads(line) for line in
                (output / f'{split}_feasible_features.jsonl').read_text().splitlines()]),
        }
    test_p = model.predict_proba(data["test"])
    confidence = clustered_improvement_intervals(data["test"], test_p, model.training_rejection_rate)
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, probability in zip(data["test"], test_p):
            stream.write(json.dumps({**row, "predicted_rejection_probability": float(probability),
                "rejection_logit": float(model.predict_logits([row])[0])}) + "\n")
    summary = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "collection": "integrated / mixed feasible proposals + exact MCMF / test pure MCMF / ADP=0 / knownreject=False / recourse=legacy",
        "label": "Actual Bernoulli response: rejected=1, accepted=0; no class balancing",
        "oracle_usage": "Evaluation only; excluded from features, fitting and model selection",
        "split": "disjoint whole dates" if args.train_dates else "same-demand-day stochastic-response holdout; not cross-date generalization",
        "model": model.to_dict(), "metrics": metrics, "episodes": episode_stats,
        "validation_selection": [{"l2": candidate.l2, "log_loss": loss} for loss, candidate in candidates],
        "test_seed_clustered_95pct_ci": confidence,
        "checkpoint_prediction_roundtrip_exact": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes,
    }
    if source_hashes != {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}:
        raise RuntimeError('Source changed during rejection training; do not attribute the run to a single version')
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test = metrics["test"]["model"]
    baseline = metrics["test"]["constant_train_rate"]
    report = ["# EV 拒单概率 v3 验证", "",
              f"环境：{args.environment}；训练/验证混合可行提案，测试纯 MCMF；未使用 oracle 打分。", "",
              f"模型：{len(model.feature_names)}→{model.hidden_dims[0]}→{model.hidden_dims[1]}→1；标签 1=拒单；保持自然比例。",
              f"特征：{', '.join(model.feature_names)}；单位模式：{model.feature_schema}。",
              f"验证集校准：{model.calibration}；数据划分：{summary['split']}。",
              "损失 BCEWithLogitsLoss + 权重 L2；Adam 优化。训练/验证/测试按种子分离；仅验证 BCE 选正则强度并早停。",
              f"训练 {model.epochs_run} epochs，恢复最佳 epoch {model.selected_epoch}；无逻辑回归回退。", "",
              f"样本量：训练 {len(data['train'])}，验证 {len(data['validation'])}，测试 {len(data['test'])}。", "",
              "| 测试指标 | 拒单网络 | 训练集拒单率常数基线 |", "|---|---:|---:|"]
    for key in ("log_loss", "brier_score", "roc_auc", "accuracy_at_0_5", "predicted_rejection_rate", "ece_10_bins", "oracle_probability_mae"):
        report.append(f"| {key} | {test[key]} | {baseline[key]} |")
    report.extend(["", f"测试实际拒单率：{test['rejection_rate']:.4%}。",
                   f"按种子重采样的改进量 95% CI（正数表示优于常数基线）：{confidence}。", "",
                   "已验证保存/重新加载模型后的概率与保存前逐项一致。原有充电次数统计保留在 summary.json。", "",
                   "## 结论边界", "",
                   "这是对仿真司机响应规律的预测验证，不是对真实人类司机的外部验证。随机响应不能被逐单确定性预测。",
                   f"NYC 使用真实订单、仿真司机标签；本次划分：{summary['split']}。",
                   "训练/验证的混合行为策略会改变所选报价；测试使用纯 MCMF。所有标签来自实际执行的报价，不对未选边生成回答。",
                   "selection_probability 仅在可计算时记录条件提案概率；它不是整条轨迹或最终边的边际 propensity，未用于加权 BCE。",
                   "独立监督训练不调用 Q/residual 更新；部署是否改善平台性能需要另外的单阶段学习实验。", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"test": test, "baseline": baseline, "ci": confidence}, indent=2), flush=True)
    print(f"Model and report saved: {output}", flush=True)


if __name__ == "__main__":
    main()
