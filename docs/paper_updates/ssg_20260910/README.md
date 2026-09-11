# SSG manuscript update — 2026-09-10

Applied in the browser to the Overleaf project:
https://www.overleaf.com/project/6a834a5015344234d20053d9

## Applied source changes

- `AnonymousSubmission/icaps_main.tex`: replace subsections 5.1 and 5.2 with `main_scaling.tex`; update the waiting-rule author comment to refer to forced rows; replace the old ILP/SGG experiment-method wording with CPLEX and OR-Tools, each with and without SSG; reflow the action-exclusivity and SoC/wait equations without changing their constraints.
- `AnonymousSubmission/icaps_supplement.tex`: replace assignment/flow and scaling proofs through the start of the existing integer-rounding proposition with `supplement_scaling.tex`; replace the certified-reduction subsection with `supplement_algorithm.tex`.
- `AnonymousSubmission/aaai2026.bib`: append the verified entry in `reduction_reference.bib`.

The existing experimental table, its data, millisecond units and two-column layout were preserved. The unfinished experiment and overall-conclusion sections remain reserved for actual final results.

## Mathematical and implementation findings

Initial private fallback actions are optional and distinct from the ordinary resource set. A null marker has baseline minus infinity and is not an executable action. After reduction, rows with a certified witness may leave the shared core unmatched; all other rows must choose exactly one real shared action. The value offset sums finite certified baselines only. The proof preserves feasibility in both directions, including infeasibility, and preserves the optimum of each feasible fixed-score assignment instance.

Deterministic simultaneous batches retain the old witness on equality and use the smallest action index among new improving ties. The fast path attempts four vectorized batches; if more folding is possible, the incremental implementation restarts from the original initial state and computes the complete fixed point. This is not a four-round truncation.

The complexity includes dense validation/quantization and retained dense matrices. The measured OR-Tools `SimpleMinCostFlow` uses cost-scaling push–relabel; the successive-shortest-path bound is explicitly a separate solver bound. End-to-end timing includes preprocessing, graph/model construction, solving, decoding and per-solve verification. The theorem guarantees neither strict graph shrinkage nor faster runtime on every instance.

## Outside-action question

The ADP-layout benchmark supplies an admissible wait action to every vehicle (`benchmark_cplex_mcmf_ssg.py`, `build_adp_case`). The sparse benchmark can explicitly supply private zero-valued fallbacks (`build_assignment_case`). The default NYC adapter supplies no separate fallback, and the low-SoC gate can prohibit wait when an individually feasible charging edge exists.

Individual charging feasibility does not reserve capacity against other newly assigned vehicles. Two must-charge vehicles that can each reach both of two capacity-one stations have a feasible joint assignment, but neither station initially meets the capacity-slack folding certificate. Both vehicles must remain in the constrained shared core. With only one such station, that restricted instance is infeasible.

Current temporal conflict repair outside the fixed assignment solver is not covered by the theorem. It can delete charging edges greedily. The paper's independent reservation-compatible quota assumption must not be presented as a proof that this outer repair procedure globally optimizes interval scheduling.

## Validation

- Independent code audit: 153 tests passed in `tests/test_ssg_preprocessing.py` and `tests/test_exact_mcmf.py`; a separate targeted audit reported seven passing checks for the NYC/outside-action paths.
- All three modified Overleaf source buffers were copied back through the editor and checked against their expected text after saving.
- Main PDF: 7 pages including references; supplementary PDF: 6 pages. Main page count did not increase.
- Visually checked the revised mathematical model, proofs, complexity, algorithm and the two reflowed physical constraints. No new overlapping or overfull mathematics was found.
- Main and supplement compiles: zero errors and zero warnings; no overfull boxes after reflow. Underfull spacing notices remain.

## Verified primary references

- Großmann et al., *Engineering Hypergraph b-Matching Algorithms*, JGAA 30(1), 1–24 (2026), https://doi.org/10.7155/jgaa.v30i1.3166. Used as motivation, not as a substitute for the outside-action proof.
- OR-Tools 9.14 implementation and documented solver bound: https://github.com/google/or-tools/blob/v9.14/ortools/graph/min_cost_flow.h.
