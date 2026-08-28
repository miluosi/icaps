"""Pure response-aware structured scores, independent of neural TD rewards."""
import math


def rejection_penalty(*, final_value, pickup_distance, ratio=None, base=0., per_km=0.):
    """The explicit rejection penalty, shared with NYC execution (not motion)."""
    if ratio is not None and final_value > 0:
        return -float(ratio) * float(final_value)
    return -float(base + per_km * pickup_distance)


def rejection_score(env, *, request_value, pickup_distance):
    # Synthetic execution has no explicit rejection penalty. Do not invent one
    # or reuse NYC's defaults; realized movement still belongs to the TD reward.
    return rejection_penalty(final_value=float(request_value), pickup_distance=float(pickup_distance),
        ratio=getattr(env, 'rejection_penalty_final_value_ratio', None),
        base=float(getattr(env, 'rejection_penalty_base', 0.)),
        per_km=float(getattr(env, 'rejection_penalty_per_km', 0.)))


def expected_structured_score(success_score, rejected_score, q_reject, human_response_mask):
    """Only unanswered human service offers have a response mixture.

    success_score is the existing service-option surrogate, not cash received
    this epoch. Neither q nor this anchor replaces realized reward in replay.
    """
    if not all(math.isfinite(float(x)) for x in (success_score, rejected_score, q_reject)):
        raise ValueError('Response anchor inputs must be finite')
    if not 0. <= float(q_reject) <= 1.:
        raise ValueError('q_reject must be a probability')
    if human_response_mask not in (False, True, 0, 1):
        raise ValueError('Human response mask must be binary')
    if not human_response_mask:
        if q_reject != 0:
            raise ValueError('Non-human/continuing edges require q=0, mask=0')
        return float(success_score)
    return (1. - float(q_reject)) * float(success_score) + float(q_reject) * float(rejected_score)


def response_graph_diagnostics(graph):
    """Unanswered feasible/selected EV risk; no oracle and no new inference."""
    selected = set(graph.selected_edge_ids)
    result = dict(graph_id=graph.graph_id, epoch_id=graph.epoch_id)
    for name, edges in (
        ('feasible', [e for e in graph.edges if e.human_response_mask]),
        ('selected', [e for e in graph.edges if e.human_response_mask and e.edge_id in selected]),
    ):
        count = len(edges)
        result[name] = dict(count=count)
        for key, values in (
            ('q_reject', [e.rejection_probability for e in edges]),
            ('success_score', [e.success_structured_score for e in edges]),
            ('risk_deduction', [e.success_structured_score - e.structured_score for e in edges]),
            ('expected_anchor', [e.structured_score for e in edges]),
            ('deployed_adjustment', [e.collection_score - e.structured_score for e in edges]),
            ('total_score', [e.collection_score for e in edges]),
        ):
            result[name][key + '_mean'] = sum(values) / count if count else None
    return result
