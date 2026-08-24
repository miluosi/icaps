"""Machine-readable experiment manifests for reproducible recourse runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from .metrics import summarize_metric_with_uncertainty
from .types import REPLAY_SCHEMA_VERSION


METRIC_DEFINITIONS = {
    "lost_requests": "requests expired or otherwise terminal without completion",
    "unresolved_requests": "generated requests not completed by the reporting time",
    "recourse_requests": "EV-rejected requests assigned to an AEV in the same integer epoch",
    "recovery_rate_assignment": "same-epoch AEV assignments divided by unique rejected residual requests",
    "recovery_rate_pickup": "AEV pickups after EV rejection divided by unique rejected residual requests",
    "recovery_rate_completion": "completions after EV rejection divided by unique rejected residual requests",
}


def write_experiment_manifest(
    output_path: str | Path,
    *,
    arguments: Mapping[str, Any],
    results: Mapping[str, Any],
    data_paths: Iterable[str | Path] = (),
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
    ):
        if any(row.get(metric) is not None for row in rows):
            summaries[metric] = summarize_metric_with_uncertainty(rows, metric)
    manifest = {
        "manifest_version": 1,
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
        }
        for metric in (
            "completed_orders",
            "service_ratio",
            "recovery_rate_assignment",
            "recovery_rate_pickup",
            "recovery_rate_completion",
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
