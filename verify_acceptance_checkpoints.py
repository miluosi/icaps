"""Replay saved ablation checkpoints with current code and verify frozen results."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_acceptance_ablation import (
    LEARNERS, attach_pair, build_env, json_default, load_pair, rollout,
    seed_everything, weight_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed_directory', type=Path)
    parser.add_argument('--learners', nargs='+', choices=LEARNERS, default=LEARNERS)
    parser.add_argument('--test-seed', type=int, default=9001)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    manifest = json.loads((options.seed_directory / 'manifest.json').read_text())
    args = SimpleNamespace(**manifest['arguments'])
    assert len(args.train_seeds) == 1, 'Select a directory for one training seed'
    train_seed = args.train_seeds[0]
    references = [json.loads(line) for line in
                  (options.seed_directory / 'episodes.jsonl').read_text().splitlines()]
    torch.set_num_threads(args.torch_threads)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    verified = []
    for learner in options.learners:
        for arm in ['off', 'predicted']:
            expected = next(row for row in references if row['learner'] == learner
                            and row['arm'] == arm and not row['training']
                            and row['seed'] == options.test_seed)
            checkpoint = options.seed_directory / learner / f'seed-{train_seed}' / arm / 'checkpoint.pt'
            directory = options.output_dir / learner / arm
            directory.mkdir(parents=True)
            with (directory / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                env = build_env(args, options.test_seed, arm, learner, False)
                pair = load_pair(env, learner, checkpoint)
                # Match the production evaluation lifecycle: fresh episode
                # environment after model restoration, then reset rollout RNG.
                env = build_env(args, options.test_seed, arm, learner, False)
                attach_pair(env, pair)
                seed_everything(options.test_seed)
                before = weight_hash(pair)
                actual = rollout(args, env, pair, training=False,
                                 seed=options.test_seed, directory=directory)
                assert weight_hash(pair) == before, 'Frozen inference changed weights'
            for name in ['demand_hash', 'steps', 'generated_requests', 'ev_offers',
                         'ev_accepted_offers', 'rejected_offers', 'unique_rejected_requests',
                         'completed_orders', 'completed_ev_orders', 'completed_aev_orders',
                         'expired_requests', 'active_requests', 'charging']:
                assert actual[name] == expected[name], (learner, arm, name, actual[name], expected[name])
            assert np.isclose(actual['reward'], expected['reward'], rtol=0, atol=1e-8)
            actual.update(learner=learner, arm=arm, checkpoint=str(checkpoint),
                          matches_saved_evaluation=True, trained_weights_hash=before)
            verified.append(actual)
            print(f'{learner} {arm}: exact replay verified; '
                  f'completed={actual["completed_orders"]}, rejected={actual["rejected_offers"]}', flush=True)
    (options.output_dir / 'verification.json').write_text(
        json.dumps(verified, indent=2, default=json_default) + '\n')


if __name__ == '__main__':
    main()
