import pytest

from run_acceptance_ablation import parse_args, summarize


def evaluation_rows():
    rows = []
    metrics = ['rejected_offers', 'unique_rejected_requests', 'completed_orders',
               'completed_ev_orders', 'completed_aev_orders', 'ev_offers',
               'ev_rejection_rate', 'completion_rate', 'reward']
    for train_seed, delta in [(41, -2), (42, 4)]:
        for test_seed in [9001, 9002, 9003]:
            for arm in ['off', 'predicted']:
                row = dict(learner='integrated_directq', training=False,
                           train_seed=train_seed, seed=test_seed, arm=arm,
                           demand_hash=f'demand-{test_seed}')
                row.update({metric: 10 + (delta if arm == 'predicted' else 0)
                            for metric in metrics})
                rows.append(row)
    return rows


def test_ablation_summary_clusters_by_trained_model():
    result = summarize(evaluation_rows())[0]
    assert result['paired_evaluations'] == 6
    metric = result['metrics']['completed_orders']
    assert metric['delta_predicted_minus_off'] == 1
    assert metric['per_train_seed_delta'] == {'41': -2, '42': 4}
    assert metric['cluster_bootstrap_95'] == [-2, 4]


def test_ablation_summary_rejects_unpaired_demand():
    rows = evaluation_rows()
    rows[-1]['demand_hash'] = 'different-orders'
    with pytest.raises(AssertionError, match='exogenous demand'):
        summarize(rows)


def test_ablation_summary_rejects_missing_arm():
    with pytest.raises(AssertionError, match='Incomplete'):
        summarize(evaluation_rows()[:-1])


@pytest.mark.parametrize('arguments', [
    ['--train-seeds', '41', '41'],
    ['--train-seeds', '41', '--test-seeds', '54100'],
    ['--train-every', '0'],
])
def test_ablation_configuration_rejects_invalid_pairing(arguments):
    with pytest.raises(SystemExit):
        parse_args([*arguments, '--acceptance-model', '/tmp/neural-test-checkpoint.json'])


def test_ablation_no_longer_defaults_to_a_regression_checkpoint():
    with pytest.raises(SystemExit):
        parse_args([])
