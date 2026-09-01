"""Machine-readable experiment manifests for reproducible recourse runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from .metrics import summarize_metric_with_uncertainty
from .types import REPLAY_SCHEMA_VERSION
from .config import OBJECTIVE_POLICY, method_metadata


METRIC_DEFINITIONS = {
    "lost_requests": "requests expired or otherwise terminal without completion",
    "unresolved_requests": "generated requests not completed by the reporting time",
    "recourse_requests": "EV-rejected requests assigned to an AEV in the same integer epoch",
    "recovery_rate_assignment": "same-epoch AEV assignments divided by unique rejected residual requests",
    "recovery_rate_pickup": "AEV pickups after EV rejection divided by unique rejected residual requests",
    "recovery_rate_completion": "completions after EV rejection divided by unique rejected residual requests",
    "conditional_recovery_rate_assignment": "same-epoch AEV assignments divided by eligible rejected residual requests",
    "conditional_recovery_rate_pickup": "pickups by the assigned AEV divided by eligible rejected residual requests",
    "conditional_recovery_rate_completion": "completions by the assigned AEV divided by eligible rejected residual requests",
}


def write_experiment_manifest(
    output_path: str | Path,
    *,
    arguments: Mapping[str, Any],
    results: Mapping[str, Any],
    data_paths: Iterable[str | Path] = (),
    checkpoint_paths: Iterable[str | Path] = (),
    value_functions: Iterable[Any] = (),
    test_status: Mapping[str, Any] | None = None,
) -> Path:
    """Write resolved arguments, provenance, hashes, and uncertainty summaries."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _result_rows(arguments, results)
    summaries: dict[str, dict[str, float | int]] = {}
    for metric in (
        "reward",
        "completed_orders",
        "service_ratio",
        "recovery_rate_assignment",
        "recovery_rate_pickup",
        "recovery_rate_completion",
        "conditional_recovery_rate_assignment",
        "conditional_recovery_rate_pickup",
        "conditional_recovery_rate_completion",
    ):
        if any(row.get(metric) is not None for row in rows):
            summaries[metric] = summarize_metric_with_uncertainty(rows, metric)
    manifest = {
        "recourse_configuration": method_metadata(
            arguments.get('transportation_mode', 'integrated'),
            arguments.get('recourse_variant', 'legacy')),
        "manifest_version": 3,
        "git_commit": _git_commit(output_path.parent),
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "resolved_config": _json_safe(dict(arguments)),
        "data_hashes": {
            str(path): _sha256_file(path)
            for raw_path in data_paths
            if (path := Path(raw_path)).is_file()
        },
        "checkpoint_namespace": arguments.get("checkpoint_suffix")
        or arguments.get("checkpoint_scenario_suffix"),
        "checkpoint_hashes": {
            str(path): _sha256_file(path)
            for raw_path in checkpoint_paths
            if (path := Path(raw_path)).is_file()
        },
        "effective_model_hyperparameters": _effective_model_hyperparameters(
            value_functions
        ),
        "effective_replay_hyperparameters": _effective_replay_hyperparameters(
            value_functions
        ),
        "target_builder_version": "solver_consistent_v3",
        "objective_policy": OBJECTIVE_POLICY,
        "solver_config": _json_safe({
            "rollout_solver": arguments.get("mcmf_solver", "exact"),
            "backend": arguments.get("mcmf_backend", "primal_dual"),
            "graph_reduction": arguments.get("mcmf_graph_reduction", True),
            "verify": arguments.get("mcmf_verify", True),
            "cost_scale": arguments.get("mcmf_cost_scale", 10_000),
            "strict": arguments.get("mcmf_strict", True),
            "target_policy": arguments.get(
                "target_solver_policy", "same_as_rollout_exact"
            ),
        }),
        "optimizer_budget": _json_safe(
            dict(results.get("optimizer_budget", {}) or {})
        ),
        "run_identity": _json_safe({
            "training_run_id": results.get("training_run_id"),
            "resume_episode_offset": results.get(
                "resume_episode_offset", 0
            ),
            "episodes": results.get("episode_identity_rows", ()),
        }),
        "test_status": _json_safe(
            dict(test_status or results.get("test_status", {}) or {})
        ),
        "environment": _environment_metadata(),
        "metric_definitions": METRIC_DEFINITIONS,
        "seed_day_rows": rows,
        "uncertainty": summaries,
    }
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _result_rows(
    arguments: Mapping[str, Any], results: Mapping[str, Any]
) -> list[dict[str, Any]]:
    detailed = list(results.get("episode_detailed_stats", ()) or ())
    rewards = list(results.get("episode_rewards", ()) or ())
    count = max(len(detailed), len(rewards))
    rows = []
    for index in range(count):
        detail = dict(detailed[index]) if index < len(detailed) else {}
        row = {
            "seed": int(arguments.get("random_seed", 0) or 0),
            "day_id": str(
                detail.get("current_real_date")
                or detail.get("day_id")
                or index
            ),
            "reward": float(rewards[index]) if index < len(rewards) else None,
            "recourse_variant": str(arguments.get("recourse_variant", "legacy")),
        }
        for metric in (
            "completed_orders",
            "service_ratio",
            "recovery_rate_assignment",
            "recovery_rate_pickup",
            "recovery_rate_completion",
            "conditional_recovery_rate_assignment",
            "conditional_recovery_rate_pickup",
            "conditional_recovery_rate_completion",
        ):
            if detail.get(metric) is not None:
                row[metric] = float(detail[metric])
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_model_hyperparameters(
    value_functions: Iterable[Any],
) -> list[dict[str, Any]]:
    fields = (
        "gamma",
        "within_epoch_gamma",
        "tau",
        "learning_rate",
        "huber_kappa",
        "gradient_clip_norm",
        "beta_max",
        "beta_warmup_steps",
        "eta_pi",
        "residual_clip_rho",
        "lambda_actor",
        "lambda_alpha",
        "lambda_orth",
        "lambda_cql",
        "hidden_dim",
        "graph_node_dim",
        "edge_local_dim",
        "edge_dim",
        "queue_loss_weight",
        "queue_edge_loss_weight",
        "checkpoint_replay",
        "checkpoint_replay_recent",
        "state_variant",
        "learner_variant",
        "recourse_variant",
    )
    rows = []
    seen = set()
    for value_function in value_functions:
        if value_function is None or id(value_function) in seen:
            continue
        seen.add(id(value_function))
        rows.append(
            {
                "class": (
                    f"{type(value_function).__module__}."
                    f"{type(value_function).__name__}"
                ),
                **{
                    field: _json_safe(getattr(value_function, field))
                    for field in fields
                    if hasattr(value_function, field)
                },
            }
        )
    return rows


def _effective_replay_hyperparameters(
    value_functions: Iterable[Any],
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for value_function in value_functions:
        replay = getattr(value_function, "joint_replay_buffer", None)
        if replay is None or id(replay) in seen:
            continue
        seen.add(id(replay))
        rows.append(
            {
                field: _json_safe(getattr(replay, field))
                for field in (
                    "capacity",
                    "alpha",
                    "beta_start",
                    "beta_end",
                    "beta_anneal_steps",
                    "beta",
                    "beta_step",
                    "epsilon",
                    "rejection_bonus",
                    "recourse_bonus",
                    "seed",
                )
            }
        )
    return rows


def _environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
    }
    try:
        import numpy

        metadata["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import pandas

        metadata["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        import scipy

        metadata["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        pass
    try:
        import gurobipy

        metadata["gurobi"] = '.'.join(map(str, gurobipy.gurobi.version()))
    except (ImportError, AttributeError):
        pass
    try:
        import ortools

        metadata["ortools"] = ortools.__version__
    except ImportError:
        pass
    return metadata


def environment_metadata() -> dict[str, Any]:
    """Public environment snapshot used by standalone formal runners."""
    return _environment_metadata()


def git_commit(start: str | Path) -> str | None:
    """Return the checked-out revision for an experiment source tree."""
    return _git_commit(Path(start))


def _git_commit(start: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
