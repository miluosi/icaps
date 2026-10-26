"""Additional CPU-only telemetry replay; existing charging results stay intact."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import contextlib
import hashlib
import json
import platform
import sys
from pathlib import Path

from benchmark_nyc_charging_rollout import ROOT, parse_args, run_case
from src.charging_deferral_diagnostics import DeferralObserver


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--output-dir' not in argv:
        argv += ['--output-dir', str(ROOT/'results/nyc_charging_deferral_20261022')]
    if '--policies' not in argv:
        argv += ['--policies', 'current', 'real_conservative']
    args = parse_args(argv)
    if set(args.policies) != {'current', 'real_conservative'}:
        raise ValueError('This diagnostic compares current and real_conservative')
    protected = [ROOT/'results/nyc_charging_rollout_1000', ROOT/'results/nyc_real_conservative_1000_20260919']
    if any(args.output_dir.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Old result directories are protected; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    paths = ['benchmark_charging_deferral.py', 'benchmark_nyc_charging_rollout.py',
             'src/charging_deferral_diagnostics.py', 'src/NYCEnvironment.py',
             'src/conservative_charging.py', 'src/real_conservative_charging.py',
             'benchmark_conservative_charging.py', 'src/GurobiOptimizer.py', 'src/exact_mcmf.py']
    manifest = dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'resume'},
                    python=platform.python_version(),
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                    modes={'current': 'Reactive', 'real_conservative': 'Reservation-based'},
                    protocol='Unchanged burst/staggered workload, seeds 0-9, one charge/vehicle, no learning/service/relocation. Pre-mask C0 excludes reservation checks; strict wait-only refers to this isolated charge/wait graph. Supplemental full-relocation and interzone service energy probes never enter dispatch.',
                    warnings='Post-selection feasible intervals are individual insertion opportunities, not simultaneous extra throughput. Reactive/RB include the existing execution-semantics difference. New request cohort excludes initial charging/inbound vehicles; unfinished waits are horizon-censored.')
    target = args.output_dir/'manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError('Resume requires identical configuration and source hashes')
    else:
        target.write_text(json.dumps(manifest, indent=2))
    summaries, deferrals = [], []
    for scenario in args.scenarios:
        for seed in args.seeds:
            hashes = []
            for policy in args.policies:
                out = args.output_dir/f'{scenario}_seed{seed}_{policy}'
                out.mkdir(exist_ok=args.resume)
                if args.resume and (out/'deferral_summary.json').exists() and (out/'summary.json').exists():
                    summary = json.loads((out/'summary.json').read_text())
                else:
                    with (out/'run.log').open('w') as f, contextlib.redirect_stdout(f):
                        summary = run_case(args, seed, scenario, policy, out, DeferralObserver(out))
                d = json.loads((out/'deferral_summary.json').read_text())
                summaries.extend(summary)
                deferrals.append(d)
                hashes.append(summary[0]['initial_state_sha256'])
                (args.output_dir/'summary.json').write_text(json.dumps(summaries, indent=2))
                (args.output_dir/'deferral_summary.json').write_text(json.dumps(deferrals, indent=2))
                print(scenario, seed, policy, {k:d[k] for k in ['affected_requests', 'blocked_vehicle_minutes',
                      'post_selection_slot_feasible_requests', 'low_energy_probe_requests', 'not_started_by_H_count']}, flush=True)
            assert len(set(hashes)) == 1, 'Initial states must match across modes'
    print('Saved:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
