"""Classify returned execution reward under the paper's explicit Option A.

The optimized objective is cumulative vehicle execution reward. Lost,
unresolved, and recovered orders remain separately reported service metrics;
they are not silently converted into a non-vehicle cash-flow penalty.
"""
from .config import OBJECTIVE_POLICY
from .types import ActionType, RewardLedger


REWARD_OBJECTIVE_POLICY = OBJECTIVE_POLICY


def build_reward_ledger(env, pending, rewards, lifecycle):
    chosen = {}
    for graph in (pending.ev_stage_graph, pending.aev_stage_graph):
        if graph is not None:
            chosen.update({e.vehicle_id: e for e in graph.edges if e.edge_id in graph.selected_edge_ids})
    rejected = set(lifecycle.rejection_outcome(transition_id=pending.transition_id).rejected_request_ids)
    labels = dict(pending.residual_state.request_labels) if pending.residual_state else {}
    components = {name: 0.0 for name in RewardLedger.__dataclass_fields__}
    for vid, reward in rewards.items():
        reward = float(reward)
        edge = chosen.get(vid)
        if int(env.vehicles[vid].get('type', 1)) == 1:
            rejection_penalty = float(getattr(env, '_epoch_rejection_reward_components', {}).get(vid, 0.0))
            components['ev_rejection_penalty'] += rejection_penalty
            key = 'ev_accepted_service' if edge and edge.action_type == ActionType.SERVICE and edge.request_id not in rejected else 'ev_other'
            components[key] += reward - rejection_penalty
        else:
            key = 'aev_other'
            if edge:
                if edge.action_type == ActionType.SERVICE:
                    category = labels.get(edge.request_id)
                    key = ('aev_rejected_repair_service' if category == 'rejected' else
                           'aev_unoffered_service' if category == 'unoffered' else 'aev_other_service')
                else:
                    key = {ActionType.CHARGE: 'aev_charging', ActionType.RELOCATE: 'aev_relocation',
                           ActionType.WAIT: 'aev_waiting'}.get(edge.action_type, key)
            components[key] += reward
    # Neither simulator currently debits request expiry as a separate returned
    # system reward. Do not invent the unused `unserved_penalty` as cash flow.
    # A future non-vehicle reward must be included in both the environment's
    # reported return and this ledger before it can enter a macro target.
    ledger = RewardLedger(**components)
    if abs(ledger.system - sum(map(float, rewards.values()))) > 1e-6 * max(1., abs(ledger.system)):
        raise AssertionError('execution reward ledger does not reconcile')
    return ledger
