import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_all_nyc_models as runner


def source_run(tmp_path, monkeypatch, methods=('no_repair',)):
    source = tmp_path / 'source'
    source.mkdir()
    settings = runner.engine.parse_args(['--methods', *methods, '--output-dir', str(source),
        '--num-vehicles', '8', '--num-ev', '4', '--smoke-steps', '4', '--train-every', '2'])
    data = tmp_path / 'data.parquet'
    data.write_bytes(b'fixture')
    settings.parquet_path = data
    runner.engine.save_json(source / 'manifest.json', dict(arguments=vars(settings),
        data_sha256=runner.engine.digest_file(data), source_sha256={}))
    payloads = {}
    for method in methods:
        folder = source / method
        folder.mkdir()
        (folder / 'checkpoint.pt').write_bytes(b'checkpoint fixture')
        stats = dict(method=method, steps=4, demand_hash='train')
        runner.engine.save_json(folder / 'training.json', stats)
        payloads[method] = dict(metadata=dict(method=method, train_date=settings.train_date,
            test_date=settings.test_date, seed=settings.seed, test_seed=settings.test_seed,
            initial_weight_hash='initial', trained_weight_hash='trained'),
            learners=[dict(network={}, target={}, extra={}) for _ in range(2)])
    monkeypatch.setattr(runner.engine.torch, 'load', lambda path, **kw: payloads[path.parent.name])
    return source, payloads


def test_defaults_cover_existing_seven_not_all_legacy_solver_products():
    args = runner.parse_args([])
    assert args.methods == runner.engine.MAIN_METHODS
    assert args.num_vehicles == 200 and args.num_ev == 100
    assert args.train_date != args.test_date
    assert args.seed != args.test_seed
    assert args.smoke_steps is None
    assert runner.METHODS['no_repair'].operating_mode == 'integrated'
    assert runner.METHODS['evfirst_no_repair'].variant == 'r1'


@pytest.mark.parametrize('arguments', [
    ['--models', 'all', 'samitha'], ['--models', 'samitha', 'samitha'],
    ['--test-date', '2025-12-18'], ['--test-seed', '71'],
    ['--epoch-length', '7'], ['--epoch-length', 'nan'],
    ['--smoke-steps', '3000'], ['--workers', '0'],
])
def test_invalid_experiments_fail_before_simulation(arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


def test_dry_run_does_not_execute_or_create_results(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'never-created'
    monkeypatch.setattr(runner, 'launch', lambda *a: pytest.fail('dry-run launched training'))
    runner.main(['--dry-run', '--output-dir', str(output), '--models', 'samitha'])
    settings = json.loads(capsys.readouterr().out)
    assert settings['arguments']['methods'] == ['samitha']
    assert settings['test_updates'] is False
    assert settings['rejection_predictor'] == 'off'
    assert not output.exists()


def test_missing_checkpoint_never_falls_back_to_training(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch, ('no_repair', 'samitha'))
    (source / 'samitha' / 'checkpoint.pt').unlink()
    output = tmp_path / 'output'
    args = runner.parse_args(['test-only', '--source-dir', str(source), '--output-dir', str(output)])
    with pytest.raises(FileNotFoundError, match='will not train'):
        runner.prepare_test_only(args)
    assert not output.exists()  # Validate ALL selected models before copying any.


@pytest.mark.parametrize('problem', ['metadata', 'schema', 'training', 'data'])
def test_incompatible_checkpoints_fail_closed(tmp_path, monkeypatch, problem):
    source, payloads = source_run(tmp_path, monkeypatch)
    payload = payloads['no_repair']
    if problem == 'metadata':
        payload['metadata']['method'] = 'samitha'
    elif problem == 'schema':
        payload['learners'].pop()
    elif problem == 'training':
        runner.engine.save_json(source / 'no_repair' / 'training.json', dict(method='no_repair', steps=2))
    else:
        (tmp_path / 'data.parquet').write_bytes(b'changed')
    args = runner.parse_args(['test-only', '--source-dir', str(source), '--output-dir', str(tmp_path / 'out')])
    with pytest.raises(ValueError):
        runner.prepare_test_only(args)
    assert not (tmp_path / 'out').exists()


def test_test_only_copies_training_boundary_but_not_old_test_results(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch, ('no_repair', 'samitha'))
    (source / 'samitha' / 'results.json').write_text('{}')
    output = tmp_path / 'output'
    args = runner.parse_args(['test-only', '--source-dir', str(source), '--output-dir', str(output),
                             '--models', 'samitha'])
    settings = runner.prepare_test_only(args)
    assert settings.resume and settings.methods == ['samitha']
    assert settings.train_date == '2025-12-18' and settings.test_date == '2025-12-19'
    assert settings.max_steps == 4
    assert (output / 'samitha' / 'training.json').is_file()
    assert (output / 'samitha' / 'checkpoint.pt').read_bytes() == (source / 'samitha' / 'checkpoint.pt').read_bytes()
    assert not (output / 'samitha' / 'results.json').exists()
    assert not (output / 'no_repair').exists()
    assert json.loads((output / 'execution.json').read_text())['phase'] == 'test-only'
    saved = json.loads((output / 'manifest.json').read_text())['arguments']
    roundtrip = json.loads(json.dumps(vars(runner.engine.parse_args(runner.engine_arguments(settings))),
                                    default=runner.engine.json_default))
    assert saved == roundtrip  # Shared engine resume will accept exactly these settings.


def test_test_only_dry_run_does_not_copy(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    output = tmp_path / 'output'
    args = runner.parse_args(['test-only', '--source-dir', str(source), '--output-dir', str(output), '--dry-run'])
    runner.prepare_test_only(args, copy_files=False)
    assert not output.exists()


def test_test_only_refuses_overwriting_source(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    args = runner.parse_args(['test-only', '--source-dir', str(source), '--output-dir', str(source)])
    with pytest.raises(FileExistsError):
        runner.prepare_test_only(args)


def test_progress_is_never_reported_as_completed_result(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.engine.save_json(source / 'no_repair' / 'progress.json', dict(
        phase='training', step=60, total_steps=2880, reward=999999))
    summary = runner.write_report(source)
    assert summary['status'] == 'incomplete'
    assert summary['completed_methods'] == 0 and summary['runs'] == []
    assert json.loads((source / 'metrics.json').read_text())['rows'] == []
    report = (source / 'REPORT.md').read_text()
    assert '不是完整一天' in report and '不是最终结果' in report
    assert '999999' not in report


def test_reports_keep_failures(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.engine.save_json(source / 'summary.json', {'failures': [dict(method='no_repair', exit_code=1)]})
    for _ in range(2):
        summary = runner.write_report(source)
        assert summary['status'] == 'failed' and summary['failures']
        assert '有方法失败' in (source / 'REPORT.md').read_text()


def test_completed_report_preserves_charging_and_recourse_semantics(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    stats = dict(method='no_repair', steps=4, demand_hash='demand', optimizer_steps_joint=[1, 1],
                 optimizer_steps_edge=[0, 0], **{key: 3 for key in runner.METRICS.values()})
    stats.update(same_epoch_aev_assignment_count=0, completed_orders=7, reward=-2.5,
                 completion_after_rejection_count=2)
    runner.engine.save_json(source / 'no_repair' / 'results.json', dict(method='no_repair',
        training=stats, testing=stats, initial_weight_hash='initial', checkpoint_loaded=True,
        test_weights_unchanged=True))
    summary = runner.write_report(source)
    assert summary['status'] == 'completed'
    flat = json.loads((source / 'metrics.json').read_text())['rows']
    assert len(flat) == 2 and flat[1]['phase'] == 'testing'
    assert flat[1]['recourse_number'] == 0 and flat[1]['completed_number'] == 7
    assert flat[1]['rejected_number'] == 3 and flat[1]['accomplished_number'] == 7
    assert flat[1]['reward'] == -2.5 and flat[1]['ev_charging_sessions'] == 3
    assert flat[1]['aev_completions_after_rejection'] == 2


def test_new_entry_preserves_production_nyc_motion_mapping():
    from src.NYCtrainer import NYCTrainer
    environment = SimpleNamespace(simulate_motion=object(), simulate_motion_evfirst=object(),
                                  simulate_motion_integrated_repair=object())
    for name in runner.engine.MAIN_METHODS:
        spec = runner.METHODS[name]
        expected = {'integrated': environment.simulate_motion, 'evfirst': environment.simulate_motion_evfirst,
                    'integrated_repair': environment.simulate_motion_integrated_repair}[spec.operating_mode]
        assert NYCTrainer._select_motion_fn(environment, spec.operating_mode) is expected


def test_short_results_cannot_be_reported_as_full_requested_run(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    stats = dict(steps=2, demand_hash='demand', **{key: 0 for key in runner.METRICS.values()})
    runner.engine.save_json(source / 'no_repair' / 'results.json', dict(method='no_repair',
        training=stats, testing=stats, initial_weight_hash='initial', checkpoint_loaded=True,
        test_weights_unchanged=True))
    with pytest.raises(ValueError, match='Incomplete rollout'):
        runner.write_report(source)
    assert not (source / 'metrics.json').exists()


def test_wrong_method_directory_is_not_silently_aggregated(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.engine.save_json(source / 'no_repair' / 'results.json', dict(method='samitha'))
    with pytest.raises(ValueError, match='wrong method directory'):
        runner.write_report(source)


@pytest.mark.parametrize('selector', [['train'], ['train-only'], ['--action', 'train'], ['--mode=train']])
def test_train_action_and_method_list_aliases(selector):
    args = runner.parse_args([*selector, '--train-models', 'no_repair', 'samitha', '--dry-run'])
    assert args.command == 'train-only'
    assert args.methods == ['no_repair', 'samitha']
    assert runner.TRAIN_MODELS == tuple(runner.engine.MAIN_METHODS)


@pytest.mark.parametrize('arguments, expected', [
    (['train', '--r', 'R1', 'r2', 'R3', 'r4', '--dry-run'],
     ['evfirst_no_repair', 'repair_only', 'repair_learning', 'recourse_nested_q2']),
    (['train-test', '--train-models', 'integrated', 'macro', 'samitha', '--dry-run'],
     ['no_repair', 'recourse_macro', 'samitha']),
    (['test', '--source-dir', '/tmp/source', '--r', 'recourse-aware', '--dry-run'],
     ['recourse_macro']),
])
def test_r_aliases_are_valid_for_training_and_testing(arguments, expected):
    args = runner.parse_args(arguments)
    assert (args.methods if args.command != 'test-only' else args.models) == expected


@pytest.mark.parametrize('selector', [['test'], ['test-only'], ['--action', 'test'], ['--mode=test']])
def test_test_action_and_method_list_aliases(selector):
    args = runner.parse_args([*selector, '--test-models', 'repair_only', 'recourse_macro',
                             '--source-dir', '/tmp/source', '--dry-run'])
    assert args.command == 'test-only'
    assert args.models == ['repair_only', 'recourse_macro']
    assert runner.TEST_MODELS == runner.TRAIN_MODELS


@pytest.mark.parametrize('arguments', [
    ['train', '--action', 'test'], ['--list-models', '--action', 'train'],
    ['train', '--test-models', 'no_repair'],
    ['test', '--source-dir', '/tmp/source', '--train-models', 'no_repair'],
    ['test', '--source-dir', '/tmp/source', '--test-models', 'all', 'no_repair'],
    ['test', '--source-dir', '/tmp/source', '--test-models', 'no_repair', 'no_repair'],
    ['train', '--r', 'R1', 'evfirst_no_repair'],
    ['train', '--train-models', 'samitha', '--worker-method', 'no_repair'],
])
def test_action_or_phase_specific_model_conflicts_are_errors(arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


def test_list_shows_training_and_testing_choices(capsys):
    runner.main(['--list-models'])
    output = capsys.readouterr().out
    assert '--train-models' in output and '--test-models' in output
    for method in runner.TRAIN_MODELS:
        assert method in output
    assert 'R1' in output and 'R2' in output and 'R3' in output and 'R4' in output
    for function in runner.TRAINING_FUNCTIONS.values():
        assert function.__name__ in output


def test_training_only_dry_run_cannot_dispatch_a_simulation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, 'launch', lambda *a: pytest.fail('launched test engine'))
    monkeypatch.setattr(runner, 'launch_training_only', lambda *a: pytest.fail('launched training'))
    output = tmp_path / 'dry-run'
    runner.main(['--action', 'train', '--train-models', 'samitha', '--dry-run', '--output-dir', str(output)])
    values = json.loads(capsys.readouterr().out)
    assert values['training_enabled'] is True and values['testing_enabled'] is False
    assert not output.exists()


def test_train_dispatch_only_uses_training_controller(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'launch', lambda *a: pytest.fail('training-only invoked testing'))
    monkeypatch.setattr(runner, 'launch_training_only', lambda settings: calls.append(settings.methods))
    runner.main(['train', '--train-models', 'samitha', '--output-dir', str(tmp_path / 'unused')])
    assert calls == [['samitha']]


def test_interactive_numbers_and_names_are_supported():
    answers = iter(['1', '1, recourse_macro,9', ''])
    args = runner.parse_args(['--interactive', '--dry-run'], input_fn=lambda prompt: next(answers))
    assert args.command == 'train-only'
    assert args.methods == ['no_repair', 'recourse_macro', 'samitha']


def test_interactive_r_names_are_canonicalized():
    answers = iter(['3', 'R1, r2 r3 R4', ''])
    args = runner.parse_args(['--interactive', '--dry-run'], input_fn=lambda prompt: next(answers))
    assert args.methods == ['evfirst_no_repair', 'repair_only', 'repair_learning',
                            'recourse_nested_q2']


def test_interactive_all_inherits_only_the_source_models(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    answers = iter(['2', '', str(source), ''])
    args = runner.parse_args(['--interactive', '--dry-run'], input_fn=lambda prompt: next(answers))
    assert args.command == 'test-only' and args.models == ['all']
    settings = runner.prepare_test_only(args, copy_files=False)
    assert settings.methods == ['no_repair']


def test_interactive_existing_arguments_are_not_overridden(tmp_path):
    args = runner.parse_args(['--interactive', '--action', 'train', '--models', 'samitha',
        '--output-dir', str(tmp_path / 'new'), '--dry-run'],
        input_fn=lambda prompt: pytest.fail('Explicit choices should not be prompted again'))
    assert args.methods == ['samitha']


def test_interactive_cancel_never_copies_or_executes(tmp_path, monkeypatch, capsys):
    answers = iter([''])  # All choices supplied; the final confirmation defaults to cancel.
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    monkeypatch.setattr(runner, 'prepare_test_only', lambda *a, **k: pytest.fail('copied before confirmation'))
    runner.main(['--interactive', '--action', 'test', '--test-models', 'no_repair',
        '--source-dir', str(tmp_path / 'source'), '--output-dir', str(tmp_path / 'output')])
    assert '已取消' in capsys.readouterr().out
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('answers', [['1', '99'], ['bad-action']])
def test_invalid_interactive_choices_do_not_default_to_training(answers):
    choices = iter(answers)
    with pytest.raises(SystemExit):
        runner.parse_args(['--interactive'], input_fn=lambda prompt: next(choices))


def training_stats(method):
    return dict(method=method, steps=4, demand_hash='training-demand', optimizer_steps_joint=[1, 1],
                optimizer_steps_edge=[0, 0], **{key: 2 for key in runner.METRICS.values()})


@pytest.mark.parametrize('method', runner.TRAIN_MODELS)
def test_training_worker_only_builds_and_rolls_out_training(tmp_path, monkeypatch, method):
    settings = runner.engine.parse_args(['--methods', method, '--output-dir', str(tmp_path),
        '--smoke-steps', '4', '--train-every', '2'])
    settings.worker_method = method
    calls = []
    monkeypatch.setattr(runner.engine.torch, 'set_num_threads', lambda *a: None)
    monkeypatch.setattr(runner.engine, 'build_env', lambda *a, **kw: calls.append(('env', kw['training'])))
    monkeypatch.setattr(runner.engine, 'seed_everything', lambda *a: None)
    monkeypatch.setattr(runner.engine, 'build_pair', lambda *a, **k: ['aev', 'ev'])
    hashes = iter(['before', 'after'])
    monkeypatch.setattr(runner.engine, 'weight_hash', lambda pair: next(hashes))

    def fake_rollout(*a, **kw):
        calls.append(('rollout', kw['training']))
        return training_stats(method)

    def fake_save(pair, path, metadata):
        assert metadata['method'] == method
        path.write_bytes(b'trained checkpoint')

    monkeypatch.setattr(runner.engine, 'rollout', fake_rollout)
    monkeypatch.setattr(runner.engine, 'save_pair', fake_save)
    runner.run_training_worker(settings)
    assert calls == [('env', True), ('rollout', True)]
    folder = tmp_path / method
    assert (folder / 'checkpoint.pt').is_file()
    assert (folder / 'training.json').is_file()
    result = json.loads((folder / 'training_result.json').read_text())
    assert result['checkpoint_saved'] and 'testing' not in result
    assert not (folder / 'results.json').exists()


def test_all_current_models_have_explicit_training_and_motion_functions():
    assert set(runner.TRAINING_FUNCTIONS) == set(runner.TRAIN_MODELS)
    assert set(runner.MOTION_FUNCTIONS) == set(runner.TRAIN_MODELS)
    assert runner.TRAINING_FUNCTIONS['no_repair'] is runner.train_integrated
    assert runner.TRAINING_FUNCTIONS['evfirst_no_repair'] is runner.train_r1
    assert runner.TRAINING_FUNCTIONS['repair_only'] is runner.train_r2
    assert runner.TRAINING_FUNCTIONS['repair_learning'] is runner.train_r3
    assert runner.TRAINING_FUNCTIONS['recourse_nested_q2'] is runner.train_r4
    assert runner.TRAINING_FUNCTIONS['samitha'] is runner.train_samitha
    assert runner.MOTION_FUNCTIONS['samitha'].endswith('simulate_motion_integrated_repair')
    assert runner.MOTION_FUNCTIONS['no_repair'].endswith('simulate_motion_integrated_control')


@pytest.mark.parametrize('method', runner.TRAIN_MODELS)
def test_named_training_function_rejects_wrong_method(monkeypatch, method):
    settings = SimpleNamespace(worker_method='wrong')
    monkeypatch.setattr(runner, 'run_training_worker', lambda *a: pytest.fail('wrong method was trained'))
    with pytest.raises(ValueError, match='received wrong'):
        runner.TRAINING_FUNCTIONS[method](settings)


def test_training_report_never_claims_evaluation(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.save_json(source / 'execution.json', {'phase': 'train-only'})
    runner.save_json(source / 'no_repair' / 'training_result.json', dict(method='no_repair',
        training=training_stats('no_repair'), initial_weight_hash='initial', checkpoint_saved=True))
    summary = runner.write_report(source)
    assert summary['status'] == 'completed'
    assert all('testing' not in row for row in summary['runs'])
    metrics = json.loads((source / 'metrics.json').read_text())
    assert [row['phase'] for row in metrics['rows']] == ['training']
    report = (source / 'REPORT.md').read_text()
    assert '仅训练' in report and '没有执行或生成测试阶段结果' in report
    assert '## 测试日' not in report


def test_saved_metric_tables_have_exact_requested_columns(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    stats = training_stats('no_repair')
    stats.update(same_epoch_aev_assignment_count=4, ev_rejected_offer_count=6,
                 completed_orders=8, reward=12.5)
    runner.engine.save_json(source / 'no_repair' / 'results.json', dict(method='no_repair',
        training=stats, testing=stats, initial_weight_hash='initial', checkpoint_loaded=True,
        test_weights_unchanged=True))
    summary = runner.write_report(source)
    assert summary['definitions']['recourse_number'] == 'same_epoch_aev_assignment_count'
    assert summary['definitions']['rejected_number'] == 'ev_rejected_offer_count'
    assert summary['definitions']['accomplished_number'] == 'completed_orders'
    machine = json.loads((source / 'metrics.json').read_text())
    for row in machine['rows']:
        assert (row['recourse_number'], row['rejected_number'], row['accomplished_number']) == (4, 6, 8)
    with (source / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {'recourse_number', 'rejected_number', 'accomplished_number'} <= set(rows[0])
    assert (rows[1]['recourse_number'], rows[1]['rejected_number'], rows[1]['accomplished_number']) == ('4', '6', '8')
    report = (source / 'REPORT.md').read_text()
    assert '| Recourse number | Rejected number | Accomplished number |' in report


def test_stopped_status_is_preserved_by_report_refresh(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.save_json(source / 'summary.json', dict(status='stopped_by_user', stop_reason='explicit stop'))
    summary = runner.write_report(source)
    assert summary['status'] == 'stopped_by_user' and summary['stop_reason'] == 'explicit stop'
    assert '已按用户要求停止' in (source / 'REPORT.md').read_text()


def test_training_controller_cleans_up_its_children_on_interrupt(tmp_path, monkeypatch):
    source, _ = source_run(tmp_path, monkeypatch)
    runner.save_json(source / 'execution.json', dict(phase='train-only'))
    _, settings = runner.read_manifest(source)
    monkeypatch.setattr(runner, 'prepare_training_directory', lambda *a: None)
    actions = []
    process = SimpleNamespace(pid=12345, poll=lambda: None,
        terminate=lambda: actions.append('terminate'), wait=lambda **kw: actions.append('wait'))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: process)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.time, 'sleep', interrupt)
    with pytest.raises(SystemExit) as error:
        runner.launch_training_only(settings)
    assert error.value.code == 130
    assert actions == ['terminate', 'wait']
    assert json.loads((source / 'summary.json').read_text())['status'] == 'stopped_by_user'
