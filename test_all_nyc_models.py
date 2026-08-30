"""Legacy filename shim for the NYC recourse-architecture experiment runner.

Examples (run from the repository root):
  python test_all_nyc_models.py --action train --train-models no_repair samitha
  python test_all_nyc_models.py --action test --source-dir results/nyc_all_models/day1
  python test_all_nyc_models.py --interactive
  python test_all_nyc_models.py train-test --output-dir results/nyc_all_models/day1
  python test_all_nyc_models.py test-only --source-dir results/nyc_all_models/day1
  python test_all_nyc_models.py report --output-dir results/nyc_all_models/day1

This is an executable experiment runner, not a pytest test module. It reuses
run_recourse_day's NYC/ADP simulation, joint-critic learning and checkpoint path.
The methods are recourse presets, not a Cartesian product of every legacy
solver and value-function architecture in test_nyc_model.py.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import run_recourse_day as engine
from src.recourse.config import METHODS
from summarize_recourse_day import LABELS, build_report

ROOT = Path(__file__).resolve().parent
TRAIN_MODELS = tuple(engine.MAIN_METHODS)
TEST_MODELS = tuple(engine.MAIN_METHODS)
METHOD_ALIASES = {
    'integrated': 'no_repair',
    'r0': 'evfirst_no_rejection',
    'r1': 'evfirst_no_repair',
    'structured_r1': 'evfirst_no_repair_structured',
    'r1_structured': 'evfirst_no_repair_structured',
    'r2': 'repair_only',
    'r3': 'repair_learning',
    'macro': 'recourse_macro',
    'recourse_aware': 'recourse_macro',
    'r4': 'recourse_nested_q2',
}
METHOD_SHORT_NAMES = {
    'no_repair': 'Integrated', 'evfirst_no_rejection': 'R0',
    'evfirst_no_repair': 'R1',
    'evfirst_no_repair_structured': 'C0',
    'repair_only': 'R2', 'repair_learning': 'R3',
    'recourse_macro': 'Macro', 'recourse_nested_q2': 'R4', 'samitha': 'Samitha',
}
MOTION_FUNCTIONS = {
    'no_repair': 'NYCEnvironment.simulate_motion_integrated_control',
    'evfirst_no_rejection': 'NYCEnvironment.simulate_motion_evfirst',
    'evfirst_no_repair': 'NYCEnvironment.simulate_motion_evfirst',
    'evfirst_no_repair_structured': 'NYCEnvironment.simulate_motion_evfirst',
    'repair_only': 'NYCEnvironment.simulate_motion_evfirst',
    'repair_learning': 'NYCEnvironment.simulate_motion_evfirst',
    'recourse_macro': 'NYCEnvironment.simulate_motion_evfirst',
    'recourse_nested_q2': 'NYCEnvironment.simulate_motion_evfirst',
    'samitha': 'NYCEnvironment.simulate_motion_integrated_repair',
}
ACTION_COMMANDS = {
    'train': 'train-only', 'train-only': 'train-only',
    'test': 'test-only', 'test-only': 'test-only',
    'train-test': 'train-test', 'report': 'report', 'list': 'list',
}
ENGINE_OPTIONS = (
    'train_date', 'test_date', 'parquet_path', 'num_vehicles', 'num_ev', 'seed',
    'test_seed', 'epoch_length', 'batch_size', 'train_every',
    'joint_replay_capacity', 'workers', 'output_dir', 'smoke_steps',
    'event_contract_mode',
)
METRICS = {
    'recourse_number': 'same_epoch_aev_assignment_count',
    'rejected_number': 'ev_rejected_offer_count',
    'accomplished_number': 'completed_orders',
    'reward': 'reward',
    # Backward-compatible names retained for existing consumers.
    'completed_number': 'completed_orders',
    'ev_rejected_offers': 'ev_rejected_offer_count',
    'aev_completions_after_rejection': 'completion_after_rejection_count',
    'ev_charging_sessions': 'human_ev_charging_sessions',
    'aev_charging_sessions': 'aev_charging_sessions',
}
CSV_COLUMNS = ('method', 'phase', 'steps', 'recourse_number', 'rejected_number',
               'accomplished_number', 'reward', 'aev_completions_after_rejection',
               'ev_charging_sessions', 'aev_charging_sessions')


def save_json(path, value):
    # A live worker coordinator and a read-only evaluator may refresh reports
    # concurrently; don't share the engine's fixed temporary filename.
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, default=engine.json_default, ensure_ascii=False, indent=2))
    temporary.replace(path)


def normalize_method_name(value):
    name = str(value).strip().lower().replace('-', '_')
    return METHOD_ALIASES.get(name, name)


def method_selection(values, available=None):
    available = [normalize_method_name(value) for value in (available or engine.MAIN_METHODS)]
    if values is None or values == ['all']:
        return available
    values = [normalize_method_name(value) for value in values]
    if 'all' in values or len(values) != len(set(values)):
        raise ValueError('Use all alone, or a list of distinct method names')
    if any(value not in available for value in values):
        raise ValueError(f'Models must belong to {available}')
    return list(values)


def engine_arguments(args):
    command = ['--methods', *args.methods]
    for name in ENGINE_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    if getattr(args, 'resume', False):
        command += ['--resume']
    return command


def interactive_arguments(argv, input_fn):
    """Only called on explicit --interactive; never implicitly prompt a worker."""
    argv = list(argv)
    if not argv or argv[0] not in ACTION_COMMANDS:
        print('选择执行操作：1=train（仅训练），2=test（仅测试），3=train-test（训练后测试），q=退出')
        answer = input_fn('请输入操作编号或名称：').strip().lower()
        if answer in {'q', 'quit', 'exit'}:
            raise SystemExit(0)
        answer = {'1': 'train', '2': 'test', '3': 'train-test'}.get(answer, answer)
        if answer not in ACTION_COMMANDS:
            raise ValueError('请选择 train、test、train-test、report 或 list')
        argv.insert(0, ACTION_COMMANDS[answer])
    command = ACTION_COMMANDS[argv[0]]
    argv[0] = command

    def has_option(*names):
        return any(token.split('=', 1)[0] in names for token in argv)

    if command in {'train-only', 'train-test', 'test-only'} and not has_option(
            '--models', '--methods', '--train-models', '--test-models', '--r'):
        models = TEST_MODELS if command == 'test-only' else TRAIN_MODELS
        for number, model in enumerate(models, 1):
            print(f'{number}. {METHOD_SHORT_NAMES[model]} = {model}: {LABELS[model]}')
        answer = input_fn('选择方法（编号/名称，用空格或逗号分隔；回车=all）：').strip()
        selected = []
        for token in (answer or 'all').replace('，', ' ').replace(',', ' ').split():
            if token.isdigit():
                index = int(token) - 1
                if not 0 <= index < len(models):
                    raise ValueError(f'方法编号必须在 1..{len(models)} 之间')
                token = models[index]
            selected.append(token)
        if selected != ['all']:
            selected = method_selection(selected, models)
        argv += ['--models', *selected]
    if command == 'test-only' and not has_option('--source-dir'):
        source = input_fn('输入已训练实验目录（source-dir）：').strip()
        if not source:
            raise ValueError('仅测试必须输入已有 checkpoint 的实验目录')
        argv += ['--source-dir', source]
    if command != 'list' and not has_option('--output-dir'):
        output = input_fn('输入结果目录（回车=自动生成；report 必填）：').strip()
        if output:
            argv += ['--output-dir', output]
        elif command == 'report':
            raise ValueError('report 必须指定既有结果目录')
    return argv


def parse_args(argv=None, *, input_fn=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    controls = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    controls.add_argument('--action', '--mode', choices=tuple(ACTION_COMMANDS))
    controls.add_argument('--interactive', action='store_true')
    controls.add_argument('--list-models', action='store_true')
    control, argv = controls.parse_known_args(argv)
    requested = 'list' if control.list_models else control.action
    if control.list_models and control.action not in {None, 'list'}:
        controls.error('--list-models cannot be combined with a training/testing action')
    if requested:
        command = ACTION_COMMANDS[requested]
        if argv and argv[0] in ACTION_COMMANDS:
            if ACTION_COMMANDS[argv[0]] != command:
                controls.error('The subcommand conflicts with --action/--mode')
            argv[0] = command
        else:
            argv.insert(0, command)
    elif argv and argv[0] in ACTION_COMMANDS:
        argv[0] = ACTION_COMMANDS[argv[0]]
    if control.interactive and not any(flag in argv for flag in ('-h', '--help')):
        try:
            argv = interactive_arguments(argv, input_fn or input)
        except (EOFError, ValueError) as error:
            controls.error(str(error) or 'No interactive input; use explicit command-line arguments')
    if not argv or (argv[0].startswith('--') and argv[0] != '--help'):
        argv.insert(0, 'train-test')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--action', '--mode', choices=tuple(ACTION_COMMANDS),
                        help='Execution phase; train/test are aliases for train-only/test-only')
    parser.add_argument('--interactive', action='store_true', help='Select action/models by keyboard input')
    parser.add_argument('--list-models', action='store_true', help='List available training and testing methods')
    commands = parser.add_subparsers(dest='command', required=True)
    defaults = engine.parse_args([])
    for action, help_text in (('train-only', 'Train one day, save checkpoints, and STOP (no evaluation)'),
                              ('train-test', 'Train one day and test a separate full day')):
        train = commands.add_parser(action, help=help_text)
        train.add_argument('--models', '--methods', '--train-models', '--r', nargs='+', default=['all'],
                           type=normalize_method_name, choices=['all', *TRAIN_MODELS],
                           help='Training method list; train-test evaluates the same selected methods')
        for name, value_type in (
            ('train_date', str), ('test_date', str), ('parquet_path', Path),
            ('num_vehicles', int), ('num_ev', int), ('seed', int), ('test_seed', int),
            ('epoch_length', float), ('batch_size', int), ('train_every', int),
            ('joint_replay_capacity', int), ('workers', int), ('smoke_steps', int),
        ):
            train.add_argument('--' + name.replace('_', '-'), type=value_type,
                               default=getattr(defaults, name))
        train.add_argument('--output-dir', type=Path)
        train.add_argument('--resume', action='store_true', help='Only resume completed phase boundaries')
        train.add_argument(
            '--event-contract-mode', choices=['required', 'record', 'off'],
            default=defaults.event_contract_mode,
            help='required for formal runs; record is suitable for short interface smoke tests',
        )
        train.add_argument('--dry-run', action='store_true', help='Print configuration; do not train or write results')
        if action == 'train-only':
            train.add_argument('--worker-method', type=normalize_method_name,
                               choices=TRAIN_MODELS, help=argparse.SUPPRESS)

    test = commands.add_parser('test-only', help='Use existing checkpoints; NEVER fall back to training')
    test.add_argument('--source-dir', required=True, type=Path,
                      help='A run_recourse_day/test_all_nyc_models experiment directory')
    test.add_argument('--output-dir', type=Path, help='New directory; source results are never overwritten')
    test.add_argument('--models', '--methods', '--test-models', '--r', nargs='+',
                      type=normalize_method_name, choices=['all', *TEST_MODELS],
                      help='Testing method list; default all models present in the source experiment')
    test.add_argument('--workers', type=int, default=1)
    test.add_argument('--dry-run', action='store_true')

    report = commands.add_parser('report', help='Read existing results/progress; no model execution')
    report.add_argument('--output-dir', required=True, type=Path)
    report.add_argument('--wait-seconds', type=float, default=0,
                        help='Optionally refresh every 30s for at most this long, or until all methods finish')
    commands.add_parser('list', help='List the current training/testing method choices')
    args = parser.parse_args(argv)
    args.interactive = control.interactive
    if args.command in {'train-only', 'train-test'}:
        try:
            args.methods = method_selection(args.models)
        except ValueError as error:
            parser.error(str(error))
        args.output_dir = (args.output_dir or new_output(args.command)).resolve()
        # Central validation: no same-day/seed leakage or invalid replay/budgets.
        engine.parse_args(engine_arguments(args))
        if not math.isfinite(args.epoch_length) or 86400 / args.epoch_length != round(86400 / args.epoch_length):
            parser.error('epoch-length must divide 86400 seconds exactly')
        if args.smoke_steps is not None and args.smoke_steps > round(86400 / args.epoch_length):
            parser.error('smoke-steps cannot exceed the full-day step count')
        if getattr(args, 'worker_method', None) and args.worker_method not in args.methods:
            parser.error('Worker method must be selected in --train-models')
    if args.command == 'test-only':
        try:
            if args.models is not None:
                method_selection(args.models, TEST_MODELS)
        except ValueError as error:
            parser.error(str(error))
    if args.command == 'test-only' and args.workers < 1:
        parser.error('workers must be positive')
    if args.command == 'report' and (not math.isfinite(args.wait_seconds) or args.wait_seconds < 0):
        parser.error('wait-seconds must be finite and nonnegative')
    return args


def new_output(label):
    return ROOT / 'results/nyc_all_models' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + label)


def read_manifest(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    args = SimpleNamespace(**manifest['arguments'])
    args.output_dir = directory.resolve()
    args.parquet_path = Path(args.parquet_path)
    args.methods = method_selection(args.methods)
    return manifest, args


def prepare_test_only(args, *, copy_files=True):
    """Fail closed before creating output if ANY selected model is unavailable.

    Copying both training.json and checkpoint.pt into a fresh resume directory
    makes every worker take the evaluation-only branch in the shared engine.
    No production checkpoint directory is searched and no retraining fallback
    is allowed. Only locally trusted checkpoints should be supplied.
    """
    source = args.source_dir.resolve()
    manifest, settings = read_manifest(source)
    selected = method_selection(args.models, settings.methods)
    if engine.digest_file(settings.parquet_path) != manifest['data_sha256']:
        raise ValueError('NYC data changed since training')
    for name, expected in manifest['source_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or engine.digest_file(path) != expected:
            raise ValueError(f'Training source version differs: {name}')
    expected_steps = settings.smoke_steps or round(86400 / settings.epoch_length)
    for method in selected:
        checkpoint, training = source / method / 'checkpoint.pt', source / method / 'training.json'
        if not checkpoint.is_file() or not training.is_file():
            raise FileNotFoundError(f'{method}: require checkpoint.pt AND training.json; will not train')
        stats = json.loads(training.read_text())
        if stats['method'] != method or stats['steps'] != expected_steps:
            raise ValueError(f'{method}: incomplete or mismatched training statistics')
        payload = engine.torch.load(checkpoint, weights_only=False, map_location='cpu')
        metadata = payload['metadata']
        expected = dict(method=method, train_date=settings.train_date, test_date=settings.test_date,
                        seed=settings.seed, test_seed=settings.test_seed)
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{method}: checkpoint metadata differs from manifest')
        if len(payload['learners']) != 2 or any(
                not {'network', 'target', 'extra'} <= saved.keys() for saved in payload['learners']):
            raise ValueError(f'{method}: incompatible checkpoint schema; require paired joint critics')
        if not metadata.get('trained_weight_hash') or not metadata.get('initial_weight_hash'):
            raise ValueError(f'{method}: checkpoint lacks weight verification metadata')
        del payload
    output = (args.output_dir or new_output('test-only')).resolve()
    if output.exists():
        raise FileExistsError(f'Use a NEW test output directory: {output}')
    settings.methods, settings.output_dir = selected, output
    settings.workers, settings.resume = args.workers, True
    settings = engine.parse_args(engine_arguments(settings))
    if copy_files:
        output.mkdir(parents=True, exist_ok=False)
        for method in selected:
            folder = output / method
            folder.mkdir()
            for name in ('checkpoint.pt', 'training.json'):
                shutil.copy2(source / method / name, folder / name)
        manifest['arguments'] = vars(settings)
        engine.save_json(output / 'manifest.json', manifest)
        engine.save_json(output / 'execution.json', dict(
            phase='test-only', training_reused_from=str(source),
            note='Training statistics/checkpoints copied; only held-out evaluation is executed.',
            entrypoint_sha256=engine.digest_file(Path(__file__))))
    return settings


def run_training_worker(settings):
    """Same training path/checkpoint schema as the day engine; no test env."""
    method = settings.worker_method
    folder = settings.output_dir / method
    folder.mkdir(exist_ok=True)
    engine.torch.set_num_threads(1)
    with (folder / 'run.log').open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        env = engine.build_env(settings, settings.seed, method, training=True)
        engine.seed_everything(settings.seed + 100000)
        pair = engine.build_pair(env, replay_buffer_size=5 * settings.joint_replay_capacity)
        initial_hash = engine.weight_hash(pair)
        trained = engine.rollout(settings, env, pair, method, training=True,
                                 progress_path=folder / 'progress.json')
        trained_hash = engine.weight_hash(pair)
        if initial_hash == trained_hash:
            raise AssertionError('Training did not change any model weights')
        if trained['steps'] != (settings.max_steps or round(86400 / settings.epoch_length)):
            raise AssertionError('Training ended before the requested step count')
        from src.recourse.config import method_metadata
        spec = METHODS[method]
        metadata = dict(method=method, initial_weight_hash=initial_hash, trained_weight_hash=trained_hash,
                        train_date=settings.train_date, test_date=settings.test_date,
                        seed=settings.seed, test_seed=settings.test_seed,
                        **method_metadata(spec.operating_mode, spec.variant),
                        state_variant=getattr(env, 'state_variant', 'joint_state_separate_critics'),
                        learner_variant=getattr(env, 'learner_variant', 'optimization_anchored_residual'),
                        solver_config=dict(
                            rollout_solver=getattr(env, 'mcmf_solver', 'exact'),
                            backend=getattr(env, 'mcmf_backend', 'primal_dual'),
                            graph_reduction=getattr(env, 'mcmf_graph_reduction', True),
                            verify=getattr(env, 'mcmf_verify', True),
                            cost_scale=getattr(env, 'mcmf_cost_scale', 10_000),
                            target_policy=getattr(env, 'target_solver_policy', 'same_as_rollout_exact'),
                        ))
        checkpoint = folder / 'checkpoint.pt'
        temporary = folder / 'checkpoint.tmp.pt'
        engine.save_pair(pair, temporary, metadata)
        temporary.replace(checkpoint)
        save_json(folder / 'training.json', trained)
        # Distinct from results.json: no held-out evaluation is claimed.
        save_json(folder / 'training_result.json', dict(training=trained, checkpoint_saved=True, **metadata))


def _train_named_method(settings, method):
    if settings.worker_method != method:
        raise ValueError(f'Training function for {method} received {settings.worker_method}')
    return run_training_worker(settings)


def train_integrated(settings):
    return _train_named_method(settings, 'no_repair')


def train_r0(settings):
    return _train_named_method(settings, 'evfirst_no_rejection')


def train_r1(settings):
    return _train_named_method(settings, 'evfirst_no_repair')


def train_structured_r1(settings):
    return _train_named_method(settings, 'evfirst_no_repair_structured')


def train_r2(settings):
    return _train_named_method(settings, 'repair_only')


def train_r3(settings):
    return _train_named_method(settings, 'repair_learning')


def train_macro(settings):
    return _train_named_method(settings, 'recourse_macro')


def train_r4(settings):
    return _train_named_method(settings, 'recourse_nested_q2')


def train_samitha(settings):
    return _train_named_method(settings, 'samitha')


TRAINING_FUNCTIONS = {
    'no_repair': train_integrated,
    'evfirst_no_rejection': train_r0,
    'evfirst_no_repair': train_r1,
    'evfirst_no_repair_structured': train_structured_r1,
    'repair_only': train_r2,
    'repair_learning': train_r3,
    'recourse_macro': train_macro,
    'recourse_nested_q2': train_r4,
    'samitha': train_samitha,
}


def prepare_training_directory(settings):
    source_paths = [Path(engine.__file__).resolve(), ROOT / 'run_recourse_audit.py',
                    ROOT / 'run_acceptance_ablation.py', ROOT / 'train_acceptance_model.py',
                    *sorted((ROOT / 'src').rglob('*.py'))]
    manifest = dict(arguments=vars(settings), data_audit=engine.audit_dates(settings),
        data_sha256=engine.digest_file(settings.parquet_path),
        source_sha256={str(p.relative_to(ROOT)): engine.digest_file(p) for p in source_paths})
    path = settings.output_dir / 'manifest.json'
    if settings.resume:
        previous, _ = read_manifest(settings.output_dir)
        execution = json.loads((settings.output_dir / 'execution.json').read_text())
        if execution.get('phase') != 'train-only':
            raise ValueError('train-only --resume requires a train-only source directory')
        ignored = {'resume', 'workers', 'output_dir', 'worker_method'}
        current = json.loads(json.dumps(vars(settings), default=engine.json_default))
        if (previous['source_sha256'] != manifest['source_sha256']
                or previous['data_sha256'] != manifest['data_sha256']
                or {k: v for k, v in previous['arguments'].items() if k not in ignored}
                != {k: v for k, v in current.items() if k not in ignored}
                or execution.get('entrypoint_sha256') != engine.digest_file(Path(__file__))):
            raise ValueError('Cannot change source/data/experiment settings on train-only resume')
    else:
        settings.output_dir.mkdir(parents=True, exist_ok=False)
        save_json(path, manifest)
        save_json(settings.output_dir / 'execution.json', dict(phase='train-only',
            note='Only training is executed; checkpoint saved for a later test-only invocation.',
            entrypoint_sha256=engine.digest_file(Path(__file__))))


def launch_training_only(settings):
    prepare_training_directory(settings)
    pending = [method for method in settings.methods if not (settings.resume and all(
        (settings.output_dir / method / name).is_file()
        for name in ('training_result.json', 'training.json', 'checkpoint.pt')))]
    active, failures = {}, []
    cancelled = False

    def stop_requested(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, stop_requested)
    print('Train only; no evaluation. Output:', settings.output_dir, flush=True)
    try:
        while pending or active:
            while pending and len(active) < settings.workers:
                method = pending.pop(0)
                folder = settings.output_dir / method
                folder.mkdir(exist_ok=True)
                log = (folder / 'worker.log').open('a', buffering=1)
                command = [sys.executable, str(Path(__file__).resolve()), 'train-only',
                           *engine_arguments(settings), '--worker-method', method]
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                except BaseException:
                    log.close()
                    raise
                active[method] = (process, log)
                print(f'Started training {method}: pid={process.pid}', flush=True)
            for method, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[method]
                if code:
                    failures.append(dict(method=method, exit_code=code))
                summary = write_report(settings.output_dir)
                summary['failures'] = failures
                save_json(settings.output_dir / 'summary.json', summary)
                print(f'Finished training {method}: exit={code}', flush=True)
            if active:
                time.sleep(.5)
    except KeyboardInterrupt:
        cancelled = True
    finally:
        # Only touch child processes owned by this invocation, not other runs.
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                log.close()
        signal.signal(signal.SIGTERM, previous_handler)
    summary = write_report(settings.output_dir)
    summary['failures'] = failures
    if cancelled:
        summary.update(status='stopped_by_user', stop_reason='Training-only invocation interrupted')
    else:
        # Clear failure/stop markers from a previous attempt after a successful resume.
        summary['status'] = 'failed' if failures else 'completed' if not pending and not active else 'incomplete'
        summary.pop('stop_reason', None)
    save_json(settings.output_dir / 'summary.json', summary)
    summary = write_report(settings.output_dir)
    print(f"Training completed: {summary['completed_methods']}/{summary['required_methods']}", flush=True)
    if cancelled:
        raise SystemExit(130)
    if failures or summary['status'] != 'completed':
        raise RuntimeError('Training-only run incomplete; inspect worker.log')


def write_report(directory):
    directory = directory.resolve()
    manifest, settings = read_manifest(directory)
    previous = directory / 'summary.json'
    previous_summary = json.loads(previous.read_text()) if previous.exists() else {}
    failures = previous_summary.get('failures', [])
    execution_path = directory / 'execution.json'
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    train_only = execution.get('phase') == 'train-only'
    phases = ('training',) if train_only else ('training', 'testing')
    runs = []
    for method in settings.methods:
        path = directory / method / ('training_result.json' if train_only else 'results.json')
        if path.exists():
            row = json.loads(path.read_text())
            if row['method'] != method:
                raise ValueError(f'Result is stored in the wrong method directory: {method}')
            if train_only and not all((directory / method / name).is_file()
                                      for name in ('checkpoint.pt', 'training.json')):
                raise ValueError(f'{method}: training result is missing its checkpoint/statistics')
            runs.append(row)
    if runs:
        for name, hashes in (
            ('initial weights', {row['initial_weight_hash'] for row in runs}),
            ('training demand', {row['training']['demand_hash'] for row in runs}),
            *[(f'{phase} demand', {row[phase]['demand_hash'] for row in runs})
              for phase in phases if phase != 'training'],
        ):
            if len(hashes) != 1:
                raise ValueError(f'Methods do not share {name}')
    summary = dict(scope='preflight' if settings.smoke_steps else '24h_train_only' if train_only else '24h_train_24h_heldout_test',
        train_date=settings.train_date, test_date=settings.test_date,
        num_vehicles=settings.num_vehicles, num_ev=settings.num_ev,
        completed_methods=len(runs), required_methods=len(settings.methods), runs=runs,
        methods=settings.methods, failures=failures,
        definitions=dict(recourse_number='same_epoch_aev_assignment_count',
            rejected_number='ev_rejected_offer_count', accomplished_number='completed_orders',
            recourse_completed_number='completion_after_rejection_count',
            reward='sum of actual env.step vehicle rewards',
            completed_number='legacy alias of accomplished_number',
            ev_rejected_offers='legacy alias of rejected_number'))
    if execution:
        summary['execution'] = execution
    completed = {row['method'] for row in summary['runs']}
    progress = {}
    for method in settings.methods:
        path = directory / method / 'progress.json'
        if method not in completed and path.exists():
            progress[method] = json.loads(path.read_text())
    summary['incomplete_progress'] = progress
    summary['status'] = ('failed' if failures else 'completed' if len(completed) == len(settings.methods)
                         else 'incomplete')
    if previous_summary.get('status') == 'stopped_by_user':
        summary.update(status='stopped_by_user', stop_reason=previous_summary.get('stop_reason', 'User requested stop'))
    summary['generated_at'] = datetime.now().astimezone().isoformat()
    metrics = []
    for row in summary['runs']:
        verified = row.get('checkpoint_saved') if train_only else row['checkpoint_loaded'] and row['test_weights_unchanged']
        if row['method'] not in settings.methods or not verified:
            raise ValueError('Result failed checkpoint/evaluation validation')
        for phase in phases:
            stats = row[phase]
            if stats['steps'] != (settings.smoke_steps or round(86400 / settings.epoch_length)):
                raise ValueError(f'Incomplete rollout cannot be a final result: {row["method"]}/{phase}')
            values = {name: stats[key] for name, key in METRICS.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f'Non-finite result for {row["method"]}/{phase}')
            metrics.append(dict(method=row['method'], phase=phase, steps=stats['steps'], **values))
    save_json(directory / 'summary.json', summary)
    save_json(directory / 'metrics.json', dict(status=summary['status'], scope=summary['scope'],
        execution=summary.get('execution', {}), definitions=summary['definitions'], rows=metrics))
    csv_path = directory / 'metrics.csv'
    csv_temporary = csv_path.with_name(f'.metrics.{os.getpid()}.csv.tmp')
    with csv_temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics)
    csv_temporary.replace(csv_path)
    report_path = directory / 'REPORT.md'
    temporary = report_path.with_name(f'.REPORT.{os.getpid()}.md.tmp')
    temporary.write_text(build_report(summary, manifest))
    temporary.replace(report_path)
    return summary


def launch(settings):
    command = [sys.executable, str(ROOT / 'run_recourse_day.py'), *engine_arguments(settings)]
    print('Output:', settings.output_dir, flush=True)
    # Inherit stdout so method start/finish messages remain visible. Workers
    # write their full simulator logs in their own result folders.
    status = subprocess.call(command, cwd=ROOT)
    if (settings.output_dir / 'manifest.json').exists():
        summary = write_report(settings.output_dir)
        print(f"Completed: {summary['completed_methods']}/{summary['required_methods']}; "
              f"report: {settings.output_dir / 'REPORT.md'}", flush=True)
        if not status and summary['status'] != 'completed':
            raise RuntimeError('Engine exited without completing all requested methods')
    if status:
        raise SystemExit(status)


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'list':
        print('Actions: train / test / train-test  (also --action or --mode)')
        print('Training choices (--train-models/--r) / testing choices (--test-models/--r): all or')
        for name in engine.MAIN_METHODS:
            spec = METHODS[name]
            print(f'{METHOD_SHORT_NAMES[name]:10s} -> {name:22s} {LABELS[name]:24s} mode={spec.operating_mode}, '
                  f'variant={spec.variant}, credit={spec.leader_credit}; '
                  f'train={TRAINING_FUNCTIONS[name].__name__}; motion={MOTION_FUNCTIONS[name]}; test=yes')
        return
    if args.command == 'report':
        deadline = time.monotonic() + args.wait_seconds
        while True:
            summary = write_report(args.output_dir)
            print(f"{summary['generated_at']}: {summary['status']} "
                  f"{summary['completed_methods']}/{summary['required_methods']} — "
                  f"{args.output_dir.resolve() / 'REPORT.md'}", flush=True)
            if summary['status'] != 'incomplete' or time.monotonic() >= deadline:
                return
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    if args.interactive and not args.dry_run:
        print(f'准备执行 {args.command}；方法：{args.models or ["all from source"]}；'
              f'输出：{args.output_dir or "自动生成新目录"}')
        if input('确认开始？输入 yes 才执行（默认取消）：').strip().lower() not in {'yes', 'y', '是'}:
            print('已取消，未启动训练或测试。')
            return
    settings = (prepare_test_only(args, copy_files=not args.dry_run) if args.command == 'test-only'
                else engine.parse_args(engine_arguments(args)))
    if args.dry_run:
        print(json.dumps(dict(phase=args.command, arguments=vars(settings),
            learner='optimization_anchored_residual', state='joint_state_separate_critics',
            rejection_predictor='off', test_updates=False,
            training_enabled=args.command != 'test-only', testing_enabled=args.command != 'train-only'),
            default=engine.json_default, indent=2))
        return
    if args.command == 'train-only':
        if args.worker_method:
            settings.worker_method = args.worker_method
            TRAINING_FUNCTIONS[args.worker_method](settings)
        else:
            launch_training_only(settings)
        return
    launch(settings)


if __name__ == '__main__':
    main()
