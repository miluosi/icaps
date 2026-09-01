"""Audit state-ablation observations on one fixed serialized execution trace."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from src.recourse.types import STATE_VARIANTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--state-variants', nargs='+', choices=STATE_VARIANTS,
                        default=list(STATE_VARIANTS))
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def _stable_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _transitions(payload):
    learners = payload.get('learners', ())
    if not learners:
        raise ValueError('checkpoint has no paired learners')
    for learner in learners:
        replay = learner.get('extra', {}).get('joint_replay_state_dict', {})
        items = replay.get('items', ())
        if items:
            return list(items)
    raise ValueError('checkpoint does not retain joint replay; train with checkpoint_replay=recent/full')


def audit_checkpoint(checkpoint: Path, variants):
    payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
    transitions = _transitions(payload)
    trace_rows = []
    for transition in transitions:
        selected = []
        for graph in (transition.ev_stage_graph, transition.aev_stage_graph):
            if graph is not None:
                selected.extend(graph.selected_edge_ids)
        trace_rows.append((transition.transition_id, tuple(selected)))
    trajectory_hash = _stable_hash(trace_rows)
    observations = {}
    for variant in variants:
        rows = []
        for transition in transitions:
            sources = [('pre_state', transition.pre_state)]
            residual = getattr(transition, 'residual_state', None)
            if residual is not None:
                sources.append(('residual_state', residual))
            for graph_name in ('ev_stage_graph', 'aev_stage_graph'):
                graph = getattr(transition, graph_name, None)
                if graph is not None:
                    sources.append((f'{graph_name}_state', graph.state))
            for source_name, source_state in sources:
                for fleet_type in (1, 2):
                    state = source_state.masked(variant, vehicle_type=fleet_type)
                    visible = [vehicle for vehicle in state.vehicles if vehicle.online]
                    other_count = sum(
                        vehicle.vehicle_type != fleet_type for vehicle in state.vehicles
                    )
                    if variant.startswith('strict_fleet_local') and other_count:
                        raise AssertionError(
                            f'{source_name} leaks the other fleet under {variant}'
                        )
                    rows.append(dict(
                        transition_id=transition.transition_id,
                        state_source=source_name,
                        fleet_type=fleet_type,
                        node_count=len(state.vehicles),
                        online_vehicle_count=len(visible),
                        other_fleet_node_count=other_count,
                        request_label_count=len(getattr(state, 'request_labels', ())),
                        state=asdict(state),
                    ))
        source_summaries = {}
        for source_name in sorted({row['state_source'] for row in rows}):
            selected = [row for row in rows if row['state_source'] == source_name]
            source_summaries[source_name] = dict(
                row_count=len(selected),
                observation_hash=_stable_hash(selected),
                mean_node_count=(sum(row['node_count'] for row in selected) / len(selected)),
                max_other_fleet_node_count=max(
                    row['other_fleet_node_count'] for row in selected
                ),
            )
        observations[variant] = dict(
            observation_hash=_stable_hash(rows),
            trajectory_hash=trajectory_hash,
            row_count=len(rows),
            strict_local=variant.startswith('strict_fleet_local'),
            state_sources=source_summaries,
        )
    if len({row['trajectory_hash'] for row in observations.values()}) != 1:
        raise AssertionError('state ablation recollected or changed the execution trace')
    return dict(
        checkpoint=str(checkpoint.resolve()),
        transition_count=len(transitions), trajectory_hash=trajectory_hash,
        fixed_execution=True, observations=observations,
    )


def main(argv=None):
    args = parse_args(argv)
    result = audit_checkpoint(args.checkpoint, args.state_variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
