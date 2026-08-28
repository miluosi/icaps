"""Supervised-only exploration of feasible offers; never fabricate responses.

An epoch chooses ordinary MCMF, random feasible EV proposals, or stratified
feasible EV proposals. Exploration restricts only EV service choices and uses
the existing exact solver for request uniqueness and all remaining capacities.
The local RNG is independent of simulator response and learning RNG streams.
"""
from contextlib import contextmanager
import numpy as np

from src.acceptance_inputs import offer_features, feature_names


def parse_mixture(text):
    weights = np.asarray([float(x) for x in text.split(',')], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Behavior mixture must be MCMF,stratified,random nonnegative weights summing to 1')
    return weights


def stratum(row):
    return (int(np.searchsorted([5, 15, 30], row['idle_time'])),
            int(np.searchsorted([1, 3, 5], row['pickup_time'])),
            int(np.searchsorted([1, 3, 5], row['surge_bonus'])))


@contextmanager
def mixed_feasible_offers(env, *, seed, mixture=(.8, .1, .1), feature_variant='driver_offer_core'):
    """Yield unlabeled feasible support rows and policy counts, restoring hooks."""
    weights = parse_mixture(','.join(map(str, mixture)))
    rng = np.random.default_rng(int(seed) + 817_213)
    if not hasattr(env, 'gurobi_optimizer'):
        from src.GurobiOptimizer import GurobiOptimizer
        env.gurobi_optimizer = GurobiOptimizer(env)
    optimizer = env.gurobi_optimizer
    names = ('_np_vehicle_rebalancing_network', '_np_vehicle_rebalancing_network_ev')
    originals = {name: getattr(optimizer, name) for name in names}
    overrides = {name: optimizer.__dict__.get(name) for name in names}
    stats = dict(feasible_rows=[], policy_counts={key: 0 for key in ('mcmf', 'stratified', 'random')})
    old_metadata = getattr(env, '_response_collection_metadata', None)

    def wrap(original):
        def solve(vehicle_ids, requests, feasible, scores, *args, **kwargs):
            layout = optimizer._get_matrix_action_layout(requests, feasible.shape[1])
            matrix_requests = layout['requests']
            nr = layout['num_requests']
            policy_index = int(rng.choice(3, p=weights))
            policy = ('mcmf', 'stratified', 'random')[policy_index]
            stats['policy_counts'][policy] += 1
            mask, objective = np.array(feasible, copy=True), np.array(scores, dtype=float, copy=True)
            metadata, candidates = {}, {}
            for i, vid in enumerate(vehicle_ids):
                if env.vehicles[int(vid)].get('type', 1) != 1:
                    continue
                choices = {}
                for j in np.flatnonzero(np.asarray(feasible[i, :nr]) > 0):
                    request = matrix_requests[int(j)]
                    row = offer_features(env, vid, request, feature_variant=feature_variant)
                    choices[int(j)] = row
                    stats['feasible_rows'].append(row)
                candidates[i] = choices
                for j, row in choices.items():
                    metadata[(int(vid), int(matrix_requests[j].request_id))] = dict(
                        behavior_policy_id=policy, behavior_policy_probability=float(weights[policy_index]),
                        candidate_count=len(choices), selection_stratum=list(stratum(row)),
                        selection_probability=None, selection_probability_kind='not_available_for_mcmf')
            if policy != 'mcmf':
                used = set()
                # Dominates the remaining objective while preserving feasibility;
                # the proposals are unique and never use unavailable edges.
                finite = np.abs(objective[np.asarray(feasible) > 0])
                priority = (float(finite.max()) + 1.) * (2 * len(vehicle_ids) + 1) if len(finite) else 1.
                for i in rng.permutation(list(candidates)):
                    mask[i, :nr] = 0
                    available = {j: row for j, row in candidates[i].items() if j not in used}
                    if not available:
                        continue
                    if policy == 'stratified':
                        groups = {}
                        for j, row in available.items():
                            groups.setdefault(stratum(row), []).append(j)
                        group = list(groups)[int(rng.integers(len(groups)))]
                        choices = groups[group]
                        probability = 1. / len(groups) / len(choices)
                    else:
                        choices = list(available)
                        probability = 1. / len(choices)
                    j = int(rng.choice(choices))
                    used.add(j)
                    mask[i, j], objective[i, j] = 1, priority
                    meta = metadata[(int(vehicle_ids[i]), int(matrix_requests[j].request_id))]
                    meta.update(selection_probability=probability,
                                selection_probability_kind='conditional_proposal_given_policy_order_and_previous_proposals',
                                remaining_candidate_count=len(available))
            env._response_collection_metadata = metadata
            return original(vehicle_ids, requests, mask, objective, *args, **kwargs)
        return solve

    for name in names:
        setattr(optimizer, name, wrap(originals[name]))
    try:
        yield stats
    finally:
        for name in names:
            if overrides[name] is None:
                delattr(optimizer, name)
            else:
                setattr(optimizer, name, overrides[name])
        if old_metadata is None:
            env.__dict__.pop('_response_collection_metadata', None)
        else:
            env._response_collection_metadata = old_metadata
