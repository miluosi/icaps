"""Validate paired NYC evaluations and summarize the observed policy changes.

Reads JSON/JSONL from benchmark_nyc_frozen_policy.py. No simulator execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np

ARMS = ("episode1", "episode2", "myopic")
LABELS = ("Frozen episode 1", "Frozen episode 2", "Myopic MCMF")
COLORS = ("#193557", "#f43b52", "#28a397")


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def keyed_offers(rows):
    """Match the attempt index as well as epoch/vehicle/request CRN identity."""
    attempts = Counter()
    keyed = {}
    for row in rows:
        key = (row["date"], row["epoch"], row["vehicle_id"], row["request_id"])
        attempts[key] += 1
        keyed[(*key, attempts[key])] = row
    return keyed


def load_experiment(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    seeds = manifest["seeds"]
    records, trajectories, checks, distributions = [], {}, [], []
    for seed in seeds:
        identities, offers, demand = {}, {}, {}
        for arm in ARMS:
            folder = root / f"{arm}_seed{seed}"
            result = read(folder / "result.json")
            assert result["frozen_weights_verified"], folder
            assert result["optimizer_budget"]["optimizer_steps_total"] == 0, folder
            assert len(result["daily_metrics"]) == 1, folder
            identity = result["identities"][0]
            identities[arm] = {key: identity[key] for key in (
                "date", "initial_vehicle_hash", "crn_episode_index", "request_seed")}
            stats = result["episode_statistics"][0]
            metrics = next(iter(result["daily_metrics"].values()))
            assert metrics["ev_offers"] == stats["ev_offer_count"], folder
            assert np.isclose(metrics["ev_rejection_rate"], stats["ev_conditional_rejection_rate"]), folder
            steps = jsonl(folder / "steps.jsonl")
            trajectories[arm, seed] = steps
            demand[arm] = [(s["step"], s["generated"]) for s in steps]
            offers[arm] = keyed_offers(jsonl(folder / "offers.jsonl"))
            assert stats["request_lifecycle_gap"] == 0, folder
            assert len(steps) == round((manifest["stop_hour"] - manifest["start_hour"]) * 120), folder
            assert np.isclose(metrics["reward"], result["episode_rewards"][0]), folder
            row = {"arm": arm, "seed": seed, **metrics,
                   "completed": stats["completed_orders"],
                   "completed_ev": stats["completed_ev_orders"],
                   "completed_aev": stats["completed_orders"] - stats["completed_ev_orders"],
                   "generated": stats["whole_req_num"],
                   "lost": stats["expired_request_count"],
                   "service_ratio": stats["completed_orders"] / max(1, stats["whole_req_num"]),
                   "charge_finished": stats["charge_finished"],
                   "elapsed_seconds": result["elapsed_seconds"]}
            for fleet in ("ev", "aev"):
                for status in ("onboard", "pickup", "idle_moving", "fully_idle", "charging", "to_charge"):
                    row[f"{fleet}_{status}_mean_vehicles"] = metrics.get(f"{fleet}_{status}_vehicle_steps", 0) / len(steps)
            row["ev_mean_offer_utility"] = (1.810 - .017 * row["offer_idle_minutes_mean"]
                - .050 * row["offer_pickup_minutes_mean"] + .101 * row["offer_surge_bonus_mean"])
            records.append(row)
        assert all(identities[a] == identities[ARMS[0]] for a in ARMS), identities
        assert all(demand[a] == demand[ARMS[0]] for a in ARMS), f"Demand mismatch seed {seed}"
        for left, right in itertools.combinations(ARMS, 2):
            common = offers[left].keys() & offers[right].keys()
            mismatch = sum(offers[left][key]["uniform"] != offers[right][key]["uniform"] for key in common)
            assert mismatch == 0, (seed, left, right, mismatch)
            checks.append({"seed": seed, "left": left, "right": right,
                           "matched_offer_draws": len(common), "mismatches": mismatch,
                           "same_initial_state_and_demand": True, "frozen_weights": True})
        baseline = {s["step"]: s for s in trajectories["episode1", seed] if "ev_zones" in s}
        for arm in ("episode2", "myopic"):
            for step in trajectories[arm, seed]:
                if step["step"] not in baseline:
                    continue
                reference = baseline[step["step"]]
                row = {"arm": arm, "seed": seed, "hour": step["hour"], "step": step["step"]}
                for key in ("ev_zones", "aev_zones", "ev_idle_zones"):
                    a, b = reference[key], step[key]
                    na, nb = sum(a.values()), sum(b.values())
                    # Normalized location distributions. Empty idle fleet has
                    # no conditional distribution, so leave that sample null.
                    row[key + "_tv"] = (0.5 * sum(abs(a.get(z, 0) / na - b.get(z, 0) / nb)
                        for z in a.keys() | b.keys())) if na and nb else None
                distributions.append(row)
    return manifest, records, trajectories, checks, distributions


def summarize(records):
    numeric = sorted(set.intersection(*(set(r) for r in records)) - {"arm", "seed"})
    result = {}
    for arm in ARMS:
        group = [r for r in records if r["arm"] == arm]
        result[arm] = {key: {"mean": float(np.mean([r[key] for r in group])),
                            "std": float(np.std([r[key] for r in group], ddof=1)) if len(group) > 1 else 0.}
                       for key in numeric}
    return result


def checkpoint_diagnostics(root):
    import torch
    result = {}
    for fleet in ("ev", "aev"):
        states = [torch.load(Path(root) / "checkpoints" / fleet / f"full_state_episode_{i}.pth",
                             map_location="cpu", weights_only=False) for i in (1, 2)]
        info = []
        for state in states:
            extra = state["extra_value_function_state"]
            policy_step = max(state["training_step"], extra["joint_training_step"])
            info.append({"episode": state["episode"], "training_step": state["training_step"],
                         "joint_training_step": extra["joint_training_step"],
                         "beta_with_current_defaults": min(.3, .3 * policy_step / 500),
                         "training_reward": state["combined_reward"],
                         "pair_id": state["checkpoint_pair_id"]})
        changes = {}
        left = {"network_state_dict": states[0]["network_state_dict"], **states[0]["extra_value_function_state"]}
        right = {"network_state_dict": states[1]["network_state_dict"], **states[1]["extra_value_function_state"]}
        for name, values in left.items():
            if not name.endswith("state_dict") or not isinstance(values, dict):
                continue
            deltas = [(right[name][key].double() - tensor.double()).square().sum().item()
                      for key, tensor in values.items() if torch.is_tensor(tensor)]
            changes[name] = float(np.sqrt(sum(deltas)))
        result[fleet] = {"checkpoints": info, "parameter_change_l2": changes}
    return result


def figures(root, records, trajectories, distributions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    root = Path(root)
    manifest = read(root / "manifest.json")
    seeds = sorted({r["seed"] for r in records})
    keys = [("reward", "System reward ($)"), ("completed", "Completed requests"),
            ("ev_rejection_rate", "EV rejection rate"), ("reward_ev", "EV reward ($)"),
            ("offer_idle_minutes_mean", "EV idle time before offer (min)"),
            ("offer_pickup_minutes_mean", "EV offer pickup time (min)"),
            ("ev_empty_km", "EV empty distance (km)"),
            ("offer_surge_bonus_mean", "EV offer surge bonus ($)"),
            ("ev_onboard_mean_vehicles", "Mean occupied EVs")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (key, title) in zip(axes.flat, keys):
        values = np.array([[next(r[key] for r in records if r["arm"] == arm and r["seed"] == seed)
                            for arm in ARMS] for seed in seeds])
        for row in values:
            ax.plot(range(3), row, color="#acb1b9", alpha=.65, linewidth=1)
        for i in range(3):
            ax.scatter([i] * len(seeds), values[:, i], color=COLORS[i], s=25, alpha=.7)
            ax.scatter(i, values[:, i].mean(), marker="D", color=COLORS[i], s=65, edgecolors="white", zorder=5)
        ax.set_xticks(range(3), ["Episode 1", "Episode 2", "Myopic"])
        ax.set_title(title)
        if key == "ev_rejection_rate":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{manifest['vehicles']} vehicles · full Yellow demand · "
                 f"{manifest['start_hour']:g}–{manifest['stop_hour']:g} h\nPaired seeds; diamonds = means", fontsize=16)
    fig.tight_layout()
    fig.savefig(root / "policy_comparison.png", dpi=180)
    fig.savefig(root / "policy_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        time = [s["hour"] for s in trajectories[arm, seeds[0]]]
        metrics = [lambda rows: np.cumsum([s["reward"] for s in rows]),
                   lambda rows: [s["completed"] for s in rows],
                   lambda rows: [s["rejections"] for s in rows],
                   lambda rows: [s["ev"].get("onboard", 0) for s in rows],
                   lambda rows: [s["ev"].get("idle_moving", 0) + s["ev"].get("fully_idle", 0) for s in rows]]
        for ax, fn in zip(axes.flat, metrics):
            array = np.array([fn(trajectories[arm, seed]) for seed in seeds])
            avg, sd = array.mean(axis=0), array.std(axis=0, ddof=1 if len(seeds) > 1 else 0)
            ax.plot(time, avg, color=color, label=label)
            ax.fill_between(time, avg-sd, avg+sd, color=color, alpha=.12)
    for arm, color in zip(ARMS[1:], COLORS[1:]):
        rows = [r for r in distributions if r["arm"] == arm]
        times = sorted({r["hour"] for r in rows})
        means = [np.mean([r["ev_zones_tv"] for r in rows if r["hour"] == t]) for t in times]
        axes[1, 2].plot(times, means, color=color, label=arm)
    titles = ["Cumulative system reward ($)", "Completed requests", "EV rejection events",
              "Occupied EVs", "Idle / cruising EVs", "EV location TV distance from episode 1"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Time of day (hour)")
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Frozen policies on 16 December · paired seeds · shading = ±1 seed SD", fontsize=15)
    fig.tight_layout()
    fig.savefig(root / "policy_trajectories.png", dpi=180)
    fig.savefig(root / "policy_trajectories.pdf")
    plt.close(fig)


def analyze(root):
    root = Path(root)
    manifest, rows, trajectories, checks, distributions = load_experiment(root)
    summary = summarize(rows)
    paired = []
    for seed in manifest["seeds"]:
        first = next(r for r in rows if r["arm"] == "episode1" and r["seed"] == seed)
        second = next(r for r in rows if r["arm"] == "episode2" and r["seed"] == seed)
        delta = {key: second[key] - first[key] for key in summary["episode1"]}
        paired.append({"seed": seed, **delta,
                       "reward_change_percent": 100 * delta["reward"] / abs(first["reward"]) if first["reward"] else None,
                       "acceptance_utility_idle_contribution": -.017 * delta["offer_idle_minutes_mean"],
                       "acceptance_utility_pickup_contribution": -.050 * delta["offer_pickup_minutes_mean"],
                       "acceptance_utility_surge_contribution": .101 * delta["offer_surge_bonus_mean"]})
    write(root / "evaluation_records.json", rows)
    write(root / "summary.json", summary)
    write(root / "paired_episode2_minus_episode1.json", paired)
    write(root / "pairing_validation.json", checks)
    write(root / "distribution_changes.json", distributions)
    checkpoints = checkpoint_diagnostics(root)
    write(root / "checkpoint_diagnostics.json", checkpoints)
    figures(root, rows, trajectories, distributions)
    train = read(root / "training/result.json")
    vehicles = manifest["vehicles"]
    report = [f"# {vehicles} 车冻结策略诊断", "",
        f"完整 Yellow Manhattan 需求，{vehicles // 2} EV + {vehicles - vehicles // 2} AEV；训练 2025-12-15、16 日 "
        f"{manifest['start_hour']:g}–{manifest['stop_hour']:g} 时，统一测试 16 日同一窗口。",
        f"训练种子 32；测试种子 {manifest['seeds']}。使用生产 NYCTrainer、macro、P3 residual learner、OR-Tools MCMF+SSG、单线程。",
        "", f"两天在线训练 reward：{train['episode_rewards']}。两天需求不同，不能直接据此认定学习改善或退化。", "",
        "|Method|Reward ($)|EV reward ($)|AEV reward ($)|Completed|EV reject rate|EV empty km|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, label in zip(ARMS, LABELS):
        s = summary[arm]
        report.append(f"|{label}|{s['reward']['mean']:.2f}|{s['reward_ev']['mean']:.2f}|{s['reward_aev']['mean']:.2f}|"
                      f"{s['completed']['mean']:.2f}|{s['ev_rejection_rate']['mean']:.2%}|{s['ev_empty_km']['mean']:.2f}|")
    first, second = summary["episode1"], summary["episode2"]
    change = lambda key: second[key]["mean"] - first[key]["mean"]
    relative = lambda key: 100 * change(key) / first[key]["mean"]
    ev_tv = float(np.mean([r["ev_zones_tv"] for r in distributions if r["arm"] == "episode2"]))
    report += ["", f"第二轮相对第一轮：系统 reward {relative('reward'):+.3f}%，完成订单 {change('completed'):+.2f} 单。"
        f"其中 EV 完成订单 {change('completed_ev'):+.2f} 单、EV reward {change('reward_ev'):+.2f}；"
        f"AEV 完成订单 {change('completed_aev'):+.2f} 单、AEV reward {change('reward_aev'):+.2f}。", "",
        f"EV 平均空驶里程 {relative('ev_empty_km'):+.3f}%，offer 前平均空闲时间 "
        f"{first['offer_idle_minutes_mean']['mean']:.3f} → {second['offer_idle_minutes_mean']['mean']:.3f} 分钟，"
        f"接驾时间 {first['offer_pickup_minutes_mean']['mean']:.3f} → {second['offer_pickup_minutes_mean']['mean']:.3f} 分钟。"
        f"EV 分区位置分布相对第一轮的平均 TV 距离为 {ev_tv:.3f}，说明调度轨迹发生变化。", "",
        f"实际 EV 条件拒单率变化 {100*change('ev_rejection_rate'):+.3f} 个百分点，"
        f"但实际 offer 的模型期望接单率变化为 {100*change('offer_acceptance_probability_mean'):+.3f} 个百分点。"
        "实际拒单率同时受所提供的订单组合及其抽样结果影响，不能只看拒单事件总数来判断接单概率恶化。"]
    if change("completed_ev") >= 0 and change("ev_empty_km") <= 0:
        report += ["", "本次结果没有支持“后续学习使 EV 更空闲/更远接驾，再由拒单反馈减少 EV 服务量”的解释："
                   "EV 完成量和收益增加，平均空驶没有增加。车辆分布改变与收益恶化不是同一结论。"
                   "这不能排除原先 3000 车、全天训练、较大 residual beta 下出现不同结果。"]
    report += ["", "上述为同一对训练模型、多个测试种子的均值，不是多个独立训练结果。", "",
        "|Test seed|Episode 2 − 1 reward ($)|Change (%)|Completed change|EV reject-rate change (pp)|",
        "|---|---:|---:|---:|---:|"]
    for p in paired:
        report.append(f"|{p['seed']}|{p['reward']:+.2f}|{p['reward_change_percent']:+.2f}|{p['completed']:+.0f}|{100*p['ev_rejection_rate']:+.2f}|")
    report += ["", "验证：同种子的初始车辆状态、每步生成需求、CRN 编号一致；相同 offer 的随机 draw 相同；测试期间网络权重不变；订单生命周期计数无缺口。", "",
        "EV 接单模型的平均 utility 变化可以精确拆为 −0.017×Δidle −0.050×Δpickup +0.101×Δsurge。见 paired_episode2_minus_episode1.json。"
        "这描述不同策略实际提供的订单组合，并不单独识别拒单反馈的因果效应；sigmoid 的均值也不等于均值 utility 的 sigmoid。", "",
        "限制：测试日参与第二轮训练，属于机制诊断；500 车承受完整需求，不代表 3000 车供需比例。五小时截断不追踪 17:00 后的订单完成，窗口末未完成订单不等于最终丢失。"
        "训练每个车队累计约 120 次联合 critic 更新，保留各检查点自己的学习步数及 residual beta；不能直接重现全天训练的优化预算和低电量累积。",
        "位置 TV 差异反映车辆分布发生变化，不自动代表变化有害。车辆状态是在每步结束采样，空驶距离来自实际移动增量。"]
    report += ["", "检查点实际学习步数与权重变化见 checkpoint_diagnostics.json；beta 使用现有默认上限 0.3、warmup 500 次计算。"
               "本实验保留每轮自己的 beta，因此比较的是完整部署策略（权重和 beta），不是固定 beta 的纯权重消融。"]
    report += ["旧的 per-edge training_step 保持 0 不表示未训练；实际学习使用 joint_training_step，"
               f"两轮分别为 {checkpoints['ev']['checkpoints'][0]['joint_training_step']}、"
               f"{checkpoints['ev']['checkpoints'][1]['joint_training_step']}，两个车队的 critic/图编码器/mixer 参数均发生变化。"]
    report += ["", "|训练日|12–17 时生成订单|完成订单|EV 条件拒单率|",
               "|---|---:|---:|---:|"]
    for day, stats in zip(manifest["training_dates"], train["episode_statistics"]):
        report.append(f"|{day}|{stats['whole_req_num']}|{stats['completed_orders']}|{stats['ev_conditional_rejection_rate']:.2%}|")
    (root / "report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(analyze(parser.parse_args().output), indent=2))
