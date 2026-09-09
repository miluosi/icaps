from __future__ import annotations

import csv
import hashlib
import json

import pytest

from benchmark_cplex_mcmf_ssg import (
    METHODS,
    build_adp_case,
    build_case,
    parse_args,
    parse_scale,
    run_benchmark,
    resolve_case_specs,
    save_case_input,
    load_case_input,
)


def test_parse_scale_supports_short_and_explicit_forms():
    short = parse_scale("200:1000")
    explicit = parse_scale("200:1000:40:20")

    assert short == explicit
    assert short.label == "200V/1000R"


def test_all_four_exact_methods_match_and_outputs_are_saved(tmp_path):
    output_dir = tmp_path / "benchmark"
    args = parse_args([
        "--scales", "8:24:4:3",
        "--case-distribution", "sparse", "--mcmf-backend", "legacy",
        "--repeats", "1",
        "--request-candidates", "3",
        "--charge-candidates", "2",
        "--relocation-candidates", "2",
        "--output-dir", str(output_dir),
        "--no-plot",
    ])

    saved_dir, rows = run_benchmark(args)

    assert saved_dir == output_dir.resolve()
    assert [row["method"] for row in rows] == list(METHODS)
    assert len({row["objective_int"] for row in rows}) == 1
    assert all(row["objective_match"] for row in rows)
    assert all(row["uses_primal_dual"] is False for row in rows)
    assert all(row["status"] == "OPTIMAL" for row in rows)
    assert all(row["flow"] == 8 for row in rows)
    assert all(
        row["graph_edges"] == row["original_graph_edges"]
        for row in rows if not row["method"].startswith("ssg_")
    )
    assert all(
        row["graph_edges"] < row["original_graph_edges"]
        for row in rows if row["method"].startswith("ssg_")
    )

    with (output_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(raw_rows) == 4
    assert metadata["all_objectives_match"] is True
    assert metadata["primal_dual_used"] is False


def test_case_generation_is_seed_deterministic():
    scale = parse_scale("10:30:5:3")
    kwargs = dict(
        ev_ratio=0.5,
        request_candidates=4,
        charge_candidates=2,
        relocation_candidates=2,
        station_capacity=4,
    )
    left = build_case(scale, 99, **kwargs)
    right = build_case(scale, 99, **kwargs)

    assert left.feasible_edges == right.feasible_edges
    assert (left.feasibility == right.feasibility).all()
    assert (left.q_values == right.q_values).all()
    assert (left.capacities == right.capacities).all()


def test_ortools_backend_preserves_ssg_objective_and_records_engine(tmp_path):
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    args = parse_args([
        "--scales", "12:24:4:3", "--repeats", "2",
        "--case-distribution", "sparse",
        "--mcmf-backend", "ortools", "--warmup", "--no-plot",
        "--output-dir", str(tmp_path / "ortools"),
    ])
    output_dir, rows = run_benchmark(args)
    assert len(rows) == 8
    assert all(row["objective_match"] and row["warmed_up"] for row in rows)
    assert all(row["flow"] == 12 for row in rows)
    assert all(
        row["backend"] == "ortools_cost_scaling"
        for row in rows if row["method"] in {"mcmf", "ssg_mcmf"}
    )
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["args"]["mcmf_backend"] == "ortools"


def test_defaults_use_ortools_and_adp_workload():
    args = parse_args([])
    assert args.mcmf_backend == "ortools"
    assert args.case_distribution == "adp"
    assert args.repeats == 10
    assert args.seeds == list(range(2101, 2111))
    assert args.scenarios == ["adp_control", "reloc_rich", "aev_joint"]
    assert args.vehicle_counts == [100, 500, 1000, 2000, 3000, 6000]
    assert args.save_inputs and args.no_plot
    assert args.warmup and args.cplex_threads == 1


@pytest.mark.parametrize("scale,expected", [
    ("8:24:4:3", (
        "c35cf7ef14d5d99d7cdc18684c6eeaa0d1e1d6fae20de900cc9f6afab9adb5b3",
        "b35a9f24ef48169fd4762714afdf1e71e2ecdb178dadfb33f0ac7d074c0985d8",
        "656516f00248abe9f3c0a27c4632d24e5526174454d22b5b8b01f9b4e6c16b95",
    )),
    ("100:500:50:10", (
        "439c62a013c8c2ab63af6d1846739ddfa4554f07a9ad54358a8cd239e485cf58",
        "389622910257bb7ad945d45a3434457c7b9bbc24ffca6fc70b5211ad1f2dd70f",
        "ca7e9718afae1f7575bfc7b475bc7aaf9a19ae91ed678337c918dfe3539348ad",
    )),
])
def test_adp_inputs_match_reference_project_fingerprints(scale, expected):
    # Recorded from adp_trainer/test_alg_time.py build_matrix, source SHA256:
    # 8697266f23a392647c1f785d8c1cdee89d90ab6eb6dd3a42550588b5578a86dd
    case = build_adp_case(parse_scale(scale), 2101)
    arrays = (case.feasibility, case.q_values.astype("<f8"), case.capacities.astype("<i8"))
    assert tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in arrays) == expected
    assert case.fallback_values is None  # A real explicit wait column, as in ADP.


def test_adp_mode_uses_same_seeds_at_every_scale(tmp_path):
    pytest.importorskip("ortools.graph.python.min_cost_flow")
    args = parse_args([
        "--scales", "8:24:4:3", "10:30:5:3", "--repeats", "2",
        "--methods", "mcmf", "ssg_mcmf", "--no-plot",
        "--output-dir", str(tmp_path / "adp"),
    ])
    _, rows = run_benchmark(args)
    assert {r["seed"] for r in rows if r["vehicles"] == 8} == {2101, 2102}
    assert {r["seed"] for r in rows if r["vehicles"] == 10} == {2101, 2102}
    assert all(r["objective_match"] and r["case_distribution"] == "adp" for r in rows)
    assert all(r["time_ms"] == 1000.0 * r["end_to_end_seconds"] for r in rows)


def test_favorable_scenarios_preserve_reward_generator_and_grow_from_small_to_large():
    args = parse_args(["--vehicle-counts", "3000", "100", "1000"])
    specs = resolve_case_specs(args)
    for scenario in args.scenarios:
        selected = [s for s in specs if s["scenario"] == scenario]
        assert [s["scale"]["vehicles"] for s in selected] == [100, 1000, 3000]
        assert all(s["generator"]["fixed_charge_capacity"] == 4 for s in selected)
    large = {s["scenario"]: s for s in specs if s["scale"]["vehicles"] == 3000}
    assert large["reloc_rich"]["scale"]["relocation"] == 6000
    assert large["aev_joint"]["scale"]["charging"] == 3000
    assert large["aev_joint"]["generator"]["ev_ratio"] == pytest.approx(0.1)
    assert large["adp_control"]["generator"]["ev_ratio"] == 0.5


@pytest.mark.parametrize("distribution", ["adp", "sparse"])
def test_input_archive_losslessly_restores_solver_inputs(tmp_path, distribution):
    import numpy as np
    scale = parse_scale("8:24:4:3")
    if distribution == "adp":
        case = build_adp_case(scale, 2101)
    else:
        case = build_case(scale, 2101, ev_ratio=0.5, request_candidates=3,
                          charge_candidates=2, relocation_candidates=2, station_capacity=4)
    path = tmp_path / "input.npz"
    digest = save_case_input(case, path, cost_scale=10000, case_distribution=distribution)
    restored = load_case_input(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("feasibility", "q_values", "capacities"):
        a, b = getattr(case, name), getattr(restored, name)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    if case.fallback_values is None:
        assert restored.fallback_values is None
    else:
        np.testing.assert_array_equal(case.fallback_values, restored.fallback_values)


def test_scenarios_do_not_mix_summaries_and_resume_preserves_completed_data(tmp_path, monkeypatch):
    import benchmark_cplex_mcmf_ssg as benchmark
    argv = ["--scenarios", "adp_control", "aev_joint", "--vehicle-counts", "8", "12",
            "--seeds", "91", "92", "--methods", "mcmf", "ssg_mcmf",
            "--output-dir", str(tmp_path / "suite")]
    output, rows = run_benchmark(parse_args(argv))
    assert len(rows) == 16
    assert len(list((output / "inputs").rglob("*.npz"))) == 8
    summary = list(csv.DictReader((output / "summary.csv").open()))
    assert len(summary) == 8 and all(int(r["runs"]) == 2 for r in summary)
    assert {r["scenario"] for r in summary} == {"adp_control", "aev_joint"}
    first = [r for r in rows if r["scenario"] == "adp_control" and r["vehicles"] == 8]
    assert [r["execution_index"] for r in first if r["method"] == "mcmf"] == [0, 1]
    monkeypatch.setattr(benchmark, "solve_method", lambda *a, **kw: pytest.fail("completed case was solved again"))
    _, resumed = run_benchmark(parse_args(argv + ["--resume"]))
    assert resumed == rows
    with pytest.raises(ValueError, match="settings"):
        run_benchmark(parse_args(argv + ["--resume", "--aev-ratio", "0.8"]))


def test_failure_keeps_completed_cases_and_can_resume(tmp_path, monkeypatch):
    import benchmark_cplex_mcmf_ssg as benchmark
    original = benchmark.solve_method
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected second-case failure")
        return original(*args, **kwargs)
    argv = ["--scales", "8:24:4:3", "--seeds", "91", "92", "--methods", "mcmf", "ssg_mcmf",
            "--no-warmup", "--output-dir", str(tmp_path / "partial")]
    monkeypatch.setattr(benchmark, "solve_method", interrupted)
    with pytest.raises(RuntimeError, match="injected"):
        run_benchmark(parse_args(argv))
    metadata = json.loads((tmp_path / "partial/metadata.json").read_text())
    assert metadata["run_status"] == "failed" and metadata["completed_cases"] == 1
    saved = json.loads((tmp_path / "partial/raw_results.json").read_text())
    assert len(saved) == 3 and sum(r["case_complete"] for r in saved) == 2
    monkeypatch.setattr(benchmark, "solve_method", original)
    _, rows = run_benchmark(parse_args(argv + ["--resume"]))
    assert len(rows) == 4 and all(r["objective_match"] for r in rows)
    assert rows[:2] == saved[:2]


@pytest.mark.parametrize("argv", [
    ["--seeds", "1", "1"], ["--vehicle-counts", "100", "100"],
    ["--scales", "8:24", "--scenarios", "aev_joint"], ["--cplex-threads", "2"],
    ["--resume"], ["--methods", "mcmf", "mcmf"],
])
def test_invalid_experiment_settings_fail_before_running(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
