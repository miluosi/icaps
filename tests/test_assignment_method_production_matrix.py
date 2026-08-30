"""Short real checkpoint smoke for every canonical assignment architecture."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.recourse.config import PAPER_METHODS


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_methods_train_checkpoint_reload_and_evaluate(tmp_path):
    output = tmp_path / "assignment-matrix"
    command = [
        sys.executable,
        str(ROOT / "run_recourse_audit.py"),
        "--environment", "synthetic",
        "--methods", *PAPER_METHODS,
        "--seeds", "71",
        "--num-vehicles", "8",
        "--num-ev", "4",
        "--max-steps", "8",
        "--batch-size", "1",
        "--train-every", "2",
        "--event-contract-mode", "record",
        "--output-dir", str(output),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert {row["recourse_variant"] for row in summary["runs"]} == set(PAPER_METHODS)
    for row in summary["runs"]:
        assert row["checkpoint_inference_verified"] is True
        assert row["training"]["steps"] == 8
        assert row["evaluation"]["steps"] == 8
        assert sum(row["training"]["optimizer_steps_joint"]) > 0
        checkpoint = output / f"20251218-{row['recourse_variant']}-71" / "checkpoint.pt"
        assert checkpoint.is_file()
