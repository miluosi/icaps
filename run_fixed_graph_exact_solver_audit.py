"""Compare exact solvers on identical immutable checkpoint replay graphs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from src.recourse.target_builder import RecourseTargetBuilder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backends', nargs='+',
                        choices=['primal_dual', 'ortools', 'gurobi_network'],
                        default=['primal_dual', 'ortools', 'gurobi_network'])
    parser.add_argument('--reductions', nargs='+', choices=['on', 'off'],
                        default=['on', 'off'])
    parser.add_argument('--max-graphs', type=int, default=100)
    parser.add_argument('--allow-unavailable-backends',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_graphs <= 0:
        parser.error('max-graphs must be positive')
    return args


def replay_graphs(payload):
    seen = set()
    for learner in payload.get('learners', ()):
        replay = learner.get('extra', {}).get('joint_replay_state_dict', {})
        for transition in replay.get('items', ()):
            for graph in (
                getattr(transition, 'ev_stage_graph', None),
                getattr(transition, 'aev_stage_graph', None),
            ):
                if graph is None or not graph.edges or graph.graph_id in seen:
                    continue
                seen.add(graph.graph_id)
                yield graph


def gurobi_api():
    try:
        import gurobipy as gp
        return gp, gp.GRB
    except (ImportError, RuntimeError):
        return None, None


def main(argv=None):
    args = parse_args(argv)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    graphs = list(replay_graphs(payload))[:args.max_graphs]
    if not graphs:
        raise RuntimeError(
            'checkpoint contains no replay graphs; train with '
            '--checkpoint-replay recent/full'
        )
    gp, grb = gurobi_api()
    rows, unavailable = [], []
    for graph in graphs:
        for backend in args.backends:
            for reduction in args.reductions:
                candidate = replace(
                    graph, solver_backend=backend, solver_family='exact',
                    graph_reduction=reduction == 'on', solver_verify=True,
                    solver_strict=True,
                    target_solver_policy='same_as_rollout_exact',
                )
                builder = RecourseTargetBuilder(gp=gp, grb=grb)
                try:
                    selected = builder.project(candidate, {
                        edge.edge_id: edge.collection_score
                        for edge in candidate.edges
                    })
                except (ImportError, RuntimeError) as error:
                    if not args.allow_unavailable_backends:
                        raise
                    unavailable.append({
                        'graph_id': graph.graph_id, 'backend': backend,
                        'graph_reduction': reduction == 'on',
                        'error': f'{type(error).__name__}: {error}',
                    })
                    continue
                builder.verify_feasible(candidate, selected)
                selected_set = set(selected)
                rows.append({
                    'graph_id': graph.graph_id,
                    'backend': backend,
                    'graph_reduction': reduction == 'on',
                    'selected_edge_ids': list(selected),
                    'selected_edge_trace_hash': builder.last_solver_diagnostics[
                        'selected_edge_trace_hash'
                    ],
                    'objective': sum(
                        edge.collection_score for edge in candidate.edges
                        if edge.edge_id in selected_set
                    ),
                    'quantized_objective': builder.last_solver_diagnostics['objective_q'],
                    'feasibility_violations': 0,
                    'runtime_seconds': builder.last_solver_runtime_seconds,
                    'diagnostics': builder.last_solver_diagnostics,
                })
    comparisons = []
    for graph in graphs:
        graph_rows = [row for row in rows if row['graph_id'] == graph.graph_id]
        if len(graph_rows) < 2:
            raise RuntimeError(f'not enough exact configurations for {graph.graph_id}')
        objectives = [row['quantized_objective'] for row in graph_rows]
        objective_gap = max(objectives) - min(objectives)
        if abs(objective_gap) > 1e-7:
            raise AssertionError(
                f'exact solver objective mismatch on {graph.graph_id}: {objectives}'
            )
        comparisons.append({
            'graph_id': graph.graph_id,
            'configuration_count': len(graph_rows),
            'objective_gap': objective_gap,
            'selected_edge_agreement': len({
                row['selected_edge_trace_hash'] for row in graph_rows
            }) == 1,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'checkpoint_schema_version': payload.get('checkpoint_schema_version'),
        'graph_count': len(graphs), 'rows': rows,
        'comparisons': comparisons, 'unavailable': unavailable,
    }, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
