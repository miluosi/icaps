"""Paired frozen macro checkpoints versus myopic MCMF on one NYC demand window.

Runs the production NYCTrainer with observation-only instrumentation. Separate
worker processes release replay RAM between training and evaluation arms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def model_hash(value):
    if value is None:
        return None
    digest = hashlib.sha256()
    for name in ("network", "critic2", "graph_encoder", "mixer", "actor",
                 "queue_predictor", "post_demand_predictor"):
        module = getattr(value, name, None)
        if module is None:
            continue
        for key, tensor in sorted(module.state_dict().items()):
            digest.update(f"{name}:{key}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Observer:
    """Read existing outcomes; do not draw RNG or change actions/costs."""
    def __init__(self, env, args, folder):
        self.env, self.args, self.folder = env, args, folder
        self.offers = (folder / "offers.jsonl").open("w")
        self.steps = (folder / "steps.jsonl").open("w")
        self.days = defaultdict(lambda: defaultdict(float))
        self.identities = []
        self.started = time.perf_counter()
        reset, step = env.reset, env.step
        reject = env._should_reject_request
        move = env._move_vehicle_one_step

        def observed_reset():
            result = reset()
            # Production trainer creates a fresh UUID per call. Offer-keyed
            # CRN must instead share a run identity across the frozen arms.
            env.recourse_run_id = f"frozen-policy-seed-{args.seed}"
            if args.stage != "train":
                # Loading episode N advances the trainer's resume counter.
                # That counter participates in simulator CRN keys, so reset
                # it for evaluation without changing the model's beta/steps.
                env.cumulative_episode_index = 0
                env.set_request_generation_seed(args.seed)
            vehicles = [[vid, v["type"], v["location"], v["battery"], v.get("coordinates")]
                        for vid, v in sorted(env.vehicles.items())]
            self.identities.append({
                "date": self.day,
                "crn_episode_index": int(env.cumulative_episode_index),
                "request_seed": int(env.request_generation_seed),
                "initial_vehicle_hash": hashlib.sha256(json.dumps(vehicles, default=str).encode()).hexdigest(),
                "model_hash_aev": model_hash(getattr(env, "value_function", None)),
                "model_hash_ev": model_hash(getattr(env, "value_function_ev", None)),
            })
            return result

        def observed_reject(vid, request):
            v = env.vehicles[vid]
            if v["type"] != 1:
                return reject(vid, request)
            idle = float(v.get("idle_timer", 0)) * env.EPOCH_LENGTH / 60
            pickup = env.get_travel_time_minutes(v["location"], request.pickup)
            surge = env._request_surge_bonus(request)
            result = reject(vid, request)
            realization = env._last_offer_realizations[(env._epoch_id(), int(vid), int(request.request_id))]
            row = dict(date=self.day, epoch=env.current_time, vehicle_id=vid,
                       request_id=request.request_id, location=v["location"], pickup_zone=request.pickup,
                       idle_minutes=idle, pickup_minutes=pickup, surge_bonus=surge,
                       final_value=float(request.final_value), **realization)
            self.offers.write(json.dumps(row) + "\n")
            stats = self.days[self.day]
            stats["ev_offers"] += 1
            stats["ev_rejections"] += int(result)
            for key, number in (("offer_idle_minutes_sum", idle), ("offer_pickup_minutes_sum", pickup),
                                ("offer_surge_bonus_sum", surge), ("offer_value_sum", request.final_value),
                                ("offer_acceptance_probability_sum", realization["acceptance_probability"])):
                stats[key] += float(number)
            return result

        def observed_move(vid, target):
            v = env.vehicles[vid]
            fleet = "ev" if v["type"] == 1 else "aev"
            occupied = v.get("passenger_onboard") is not None
            pickup = v.get("assigned_request") is not None
            charging = v.get("charging_target") is not None
            distance = move(vid, target)
            if not occupied:
                self.days[self.day][f"{fleet}_empty_km"] += distance
                kind = "pickup" if pickup else "to_charge" if charging else "other_empty"
                self.days[self.day][f"{fleet}_{kind}_km"] += distance
            return distance

        def observed_step(actions, storeactions, storeactions_ev=None):
            result = step(actions, storeactions, storeactions_ev)
            rewards, info = result[1], result[4]
            stats = self.days[self.day]
            stats["steps"] += 1
            stats["reward"] += sum(rewards.values())
            stats["reward_ev"] += sum(r for vid, r in rewards.items() if env.vehicles[vid]["type"] == 1)
            stats["reward_aev"] += sum(r for vid, r in rewards.items() if env.vehicles[vid]["type"] == 2)
            stats["rejection_reward"] += info.get("step_rejection_reward_total", 0.)
            for kind, reward in info.get("reward_breakdown", {}).items():
                stats[f"reward_action_{kind}"] += reward
            row = {"date": self.day, "step": int(stats["steps"]),
                   "hour": float(env.get_hour_of_day()), "reward": sum(rewards.values()),
                   "generated": env.whole_req_num, "completed": len(env.completed_requests),
                   "completed_ev": len(env.completed_requests_ev),
                   "lost": len(env.expired_request_ids), "offers": stats["ev_offers"],
                   "rejections": stats["ev_rejections"]}
            for fleet, vtype in (("ev", 1), ("aev", 2)):
                counts = Counter()
                zones = Counter()
                idle_zones = Counter()
                for v in env.vehicles.values():
                    if v["type"] != vtype:
                        continue
                    zones[int(v["location"])] += 1
                    if v["charging_station"] is not None:
                        status = "charging"
                    elif v["passenger_onboard"] is not None:
                        status = "onboard"
                    elif v["assigned_request"] is not None:
                        status = "pickup"
                    elif v.get("charging_target") is not None:
                        status = "to_charge"
                    elif v.get("idle_target") is not None or v.get("target_location") is not None:
                        status = "idle_moving"
                    else:
                        status = "fully_idle"
                    counts[status] += 1
                    if status in {"idle_moving", "fully_idle"}:
                        idle_zones[int(v["location"])] += 1
                row[fleet] = dict(counts)
                for status, count in counts.items():
                    stats[f"{fleet}_{status}_vehicle_steps"] += count
                if int(stats["steps"]) % 10 == 0:
                    row[f"{fleet}_zones"] = dict(zones)
                    row[f"{fleet}_idle_zones"] = dict(idle_zones)
            self.steps.write(json.dumps(row) + "\n")
            if int(stats["steps"]) % 25 == 0:
                self.offers.flush()
                self.steps.flush()
                dump(self.folder / "progress.json", {"day": self.day, **stats,
                     "elapsed_seconds": time.perf_counter() - self.started})
            return result

        env.reset, env.step = observed_reset, observed_step
        env._should_reject_request = observed_reject
        env._move_vehicle_one_step = observed_move

    @property
    def day(self):
        return str(self.env._current_date_label().date())

    def finish(self):
        self.offers.close()
        self.steps.close()
        for date, stats in self.days.items():
            offers = max(1, stats["ev_offers"])
            stats["ev_rejection_rate"] = stats["ev_rejections"] / offers
            for name in ("idle_minutes", "pickup_minutes", "surge_bonus", "value", "acceptance_probability"):
                stats[f"offer_{name}_mean"] = stats.get(f"offer_{name}_sum", 0) / offers
        frozen = True
        if self.args.stage != "train":
            for fleet in ("aev", "ev"):
                vf = getattr(self.env, "value_function" if fleet == "aev" else "value_function_ev", None)
                frozen &= model_hash(vf) == self.identities[-1][f"model_hash_{fleet}"]
            if not frozen:
                raise AssertionError("Evaluation modified model weights")
        return {"daily_metrics": dict(self.days), "identities": self.identities,
                "frozen_weights_verified": frozen if self.args.stage != "train" else None,
                "elapsed_seconds": time.perf_counter() - self.started}


def worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch
    import run_nyctrainer as entry
    from src.ADPtrainer import ADPTrainer
    from src.NYCtrainer import NYCTrainer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = args.output / ("training" if args.stage == "train" else f"{args.arm}_seed{args.seed}")
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint_root = args.output / "checkpoints"
    observations = []

    class Helper(ADPTrainer):
        def __init__(self):
            super().__init__(log_file=str(folder / "helper.log"))

        @staticmethod
        def find_checkpoint_pair(ev_dir, aev_dir, **unused):
            episode = int(args.arm[-1])
            paths = tuple(str(Path(d) / f"full_state_episode_{episode}.pth") for d in (ev_dir, aev_dir))
            assert all(Path(p).is_file() for p in paths)
            identities = [ADPTrainer._checkpoint_identity(p) for p in paths]
            assert identities[0]["pair_id"] == identities[1]["pair_id"]
            return paths

        @staticmethod
        def _save_episode_stats_to_excel(env, rows, *unused, **kwargs):
            path = folder / "episode_statistics.json"
            dump(path, rows)
            return str(path), None

    class Trainer(NYCTrainer):
        @staticmethod
        def _checkpoint_dirs(**unused):
            return str(checkpoint_root / "ev"), str(checkpoint_root / "aev")

        @staticmethod
        def _loss_output_dir():
            path = folder / "losses"
            path.mkdir(exist_ok=True)
            return path

    def create_env(**kwargs):
        env = entry._create_nyc_environment(**kwargs)
        observations.append(Observer(env, args, folder))
        return env

    training = args.stage == "train"
    kwargs = dict(
        adpvalue=0. if args.arm == "myopic" and not training else 1.,
        num_episodes=2 if training else 1, use_intense_requests=True,
        assignmentgurobi=True, batch_size=256, num_vehicles=args.vehicles,
        num_ev=args.vehicles // 2, heuristic_battery_threshold=.5,
        transportation_mode="evfirst", start_training_episode=0, usemcmf=True,
        knownreject=False, mcmf_use_gpu=False, useauction=False, auction_use_gpu=False,
        auction_epsilon=.001, auction_max_rounds=None, auction_top_k=None,
        ifloadcheckpoint=not training and args.arm != "myopic", trainnetwork=training,
        random_seed=args.seed, parquet_path=str(ROOT / "nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet"),
        start_year_month="2025-12", end_year_month="2025-12",
        start_date="2025-12-15" if training else "2025-12-16", end_date="2025-12-16",
        coord_csv=None, station_csv=str(ROOT / "nyedata/nyc_all_charging_stations.csv"),
        station_capacity_scale=1., start_hour=args.start_hour, stop_hour=args.stop_hour,
        epoch_length=30., only_manhattan_zones=True, prestep=0,
        training_frequency=10, checkpoint_selection="latest",
        aev_charging_center_count=3, conservative_charging=False,
        recourse_variant="recourse_macro", common_random_numbers=True,
        state_variant="joint_state_separate_critics", learner_variant="optimization_anchored_residual",
        zone_distribution_mode="optimization_anchored_residual", predictor_variant="p3",
        mcmf_solver="exact", mcmf_backend="ortools", mcmf_strict=True,
        mcmf_graph_reduction=True, mcmf_verify=True, checkpoint_replay="none",
    )
    dump(folder / "settings.json", kwargs)
    with (folder / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        trainer = Trainer(create_environment=create_env,
                          resolve_parquet_paths=entry.resolve_nyc_parquet_paths,
                          get_value_function_class=entry._get_value_function_class,
                          set_random_seeds=entry._set_random_seeds, trainer_helper=Helper())
        results, env = trainer.run_nyc_training(**kwargs)
        observation = observations[0].finish()
        observation["episode_rewards"] = results["episode_rewards"]
        observation["episode_statistics"] = results["episode_detailed_stats"]
        observation["optimizer_budget"] = results["optimizer_budget"]
        dump(folder / "result.json", observation)
    print(f"DONE {folder.name}: rewards={results['episode_rewards']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vehicles", type=int, default=500)
    parser.add_argument("--start-hour", type=float, default=12.)
    parser.add_argument("--stop-hour", type=float, default=17.)
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 257, 258])
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--arm", choices=["episode1", "episode2", "myopic"], default="episode1")
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage != "all":
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "manifest.json", {**vars(args), "demand": "full Yellow Manhattan; scale=1",
         "training_dates": ["2025-12-15", "2025-12-16"], "evaluation_date": "2025-12-16",
         "purpose": "in-sample mechanism diagnosis, not held-out generalization", "threads": 1})
    common = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(args.output),
              "--vehicles", str(args.vehicles), "--start-hour", str(args.start_hour),
              "--stop-hour", str(args.stop_hour)]
    subprocess.run(common + ["--stage", "train", "--seed", "32"], check=True, cwd=ROOT)
    for seed in args.seeds:
        for arm in ("episode1", "episode2", "myopic"):
            subprocess.run(common + ["--stage", "evaluate", "--arm", arm, "--seed", str(seed)],
                           check=True, cwd=ROOT)
    dump(args.output / "completion.json", {"completed": True, "seeds": args.seeds})


if __name__ == "__main__":
    main()
