"""Calibrated EV rejection probabilities learned from pre-offer observations.

This supervised model is independent of the ADP critics and the MCMF policy.
Labels are 1=rejected, 0=accepted. Simulator probabilities, random draws,
request/driver IDs and post-response state are never model inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn


from src.acceptance_inputs import FEATURE_NAMES, FEATURE_VERSION, SCHEMAS, offer_features, feature_names


@contextmanager
def collect_offers(env, *, episode_id: str, seed: int, feature_variant="driver_offer_core", day_id=None):
    """Passively observe actual driver responses without changing dispatch/RNG.

    Wrap the instance only for this context. Capture covariates before the
    original response function, then record its one actual Bernoulli label.
    The oracle probability is an evaluation-only column, not a training label.
    """
    original = env._should_reject_request
    had_override = "_should_reject_request" in env.__dict__
    previous_override = env.__dict__.get("_should_reject_request")
    rows = []

    def observe(vehicle_id, request):
        human = int(env.vehicles[vehicle_id].get("type", 1)) == 1
        features = offer_features(env, vehicle_id, request, feature_variant=feature_variant) if human else None
        metadata = dict(getattr(env, '_response_collection_metadata', {}).get(
            (int(vehicle_id), int(request.request_id)), {}))
        now = float(env.current_time)
        rejected = original(vehicle_id, request)
        if human:
            key = (env._epoch_id(), int(vehicle_id), int(request.request_id))
            realization = env._last_offer_realizations.get(key)
            if realization is None:
                raise RuntimeError("The actual offer outcome was not recorded")
            rows.append({
                **features,
                "episode_id": str(episode_id),
                "seed": int(seed),
                "offer_index": len(rows),
                "current_time": now,
                "vehicle_id": int(vehicle_id),
                "request_id": int(request.request_id),
                "accepted": int(not rejected),
                "rejected": int(rejected), "target_semantics": "rejected=1",
                "day_id": str(day_id if day_id is not None else getattr(env, 'start_date', 'synthetic')),
                "oracle_rejection_probability": 1.0 - float(realization["acceptance_probability"]),
                **metadata,
            })
        return rejected

    env._should_reject_request = observe
    try:
        yield rows
    finally:
        if had_override:
            env._should_reject_request = previous_override
        else:
            del env._should_reject_request


class EVRejectionProbabilityModel:
    """Two-hidden-layer PyTorch MLP, trained on actual unbalanced binary labels.

    BCEWithLogitsLoss + L2, Adam, validation-only early stopping. The output
    sigmoid is applied at inference. No linear/logistic-regression fallback.
    """

    VERSION = 3
    MODEL_TYPE = 'mlp_ev_rejection'
    HIDDEN_DIMS = (16, 8)

    def __init__(self, *, l2=1e-4, learning_rate=1e-3, max_epochs=120,
                 batch_size=512, patience=20, seed=42,
                 feature_variant="driver_offer_core", calibration="temperature"):
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if any(int(v) != v or v <= 0 for v in (max_epochs, batch_size, patience)):
            raise ValueError('Training epochs, batch size and patience must be positive integers')
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_epochs, self.batch_size, self.patience = int(max_epochs), int(batch_size), int(patience)
        self.seed = int(seed)
        self.feature_variant = feature_variant
        self.feature_names = feature_names(feature_variant)
        self.hidden_dims = self.HIDDEN_DIMS if feature_variant == 'driver_offer_core' else (64, 32)
        if calibration not in {'none', 'temperature', 'platt'}:
            raise ValueError('Unknown rejection calibration method')
        self.calibration_method = calibration
        self.calibration = dict(method=calibration, a=1.0, b=0.0, temperature=1.0,
                                fitted=False, validation_nll_before=None, validation_nll_after=None)
        self.fitted = False

    def _new_network(self):
        # Loading or initializing a predictor must not perturb dispatch/RL RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            h1, h2 = self.hidden_dims
            return nn.Sequential(nn.Linear(len(self.feature_names), h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def _features(self, rows) -> np.ndarray:
        if any(row["feature_schema"] != self.feature_schema for row in rows):
            raise ValueError("Acceptance feature schema/units do not match the model")
        if any(row.get('feature_version') != FEATURE_VERSION for row in rows):
            raise ValueError('Rejection feature schema v3 required; recollect legacy acceptance data')
        if any(row.get('feature_variant') != self.feature_variant for row in rows):
            raise ValueError('EV rejection feature variant mismatch; recollect the requested inputs')
        try:
            x = np.asarray([[row[name] for name in self.feature_names] for row in rows], dtype=float)
        except KeyError as exc:
            raise ValueError(f'Missing required neural acceptance feature: {exc}') from exc
        x = x.reshape(-1, len(self.feature_names))
        if not np.all(np.isfinite(x)):
            raise ValueError("Acceptance features must be finite")
        return x

    def _tensor(self, rows):
        return torch.as_tensor((self._features(rows) - self.mean) / self.scale, dtype=torch.float32)

    def _penalty(self):
        return 0.5 * self.l2 * sum(layer.weight.square().sum() for layer in self.network if isinstance(layer, nn.Linear))

    def fit(self, rows, *, validation_rows=None):
        if not rows:
            raise ValueError("No observed offers to train on")
        self.feature_schema = rows[0]["feature_schema"]
        if self.feature_schema not in SCHEMAS:
            raise ValueError("Unknown acceptance feature schema")
        x = self._features(rows)
        y = self._labels(rows)
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("Training needs both accepted and rejected offers")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        self.network = self._new_network()
        rate = float(y.mean())
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.fill_(np.log(rate / (1.0 - rate)))
        tx, ty = self._tensor(rows), torch.as_tensor(y, dtype=torch.float32)
        vx = vy = None
        if validation_rows is not None:
            if not validation_rows:
                raise ValueError('Validation rows must be nonempty')
            vx = self._tensor(validation_rows)
            vy = torch.tensor(self._labels(validation_rows), dtype=torch.float32)
            if not torch.all((vy == 0) | (vy == 1)):
                raise ValueError('Validation labels must be binary')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history = []

        def record(epoch):
            self.network.eval()
            with torch.no_grad():
                bce = float(criterion(self.network(tx).squeeze(-1), ty))
                penalty = float(self._penalty())
                validation = float(criterion(self.network(vx).squeeze(-1), vy)) if vx is not None else None
            self.loss_history.append(dict(epoch=epoch, iteration=epoch, objective=bce + penalty,
                binary_cross_entropy=bce, l2_penalty=penalty, validation_binary_cross_entropy=validation))
            return validation if validation is not None else bce + penalty

        best_score = record(0)
        best_state, best_epoch, stale = copy.deepcopy(self.network.state_dict()), 0, 0
        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            order = torch.randperm(len(rows), generator=generator)
            for indexes in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(tx[indexes]).squeeze(-1), ty[indexes]) + self._penalty()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite neural acceptance training loss')
                loss.backward()
                optimizer.step()
            score = record(epoch)
            if score < best_score - 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
            if validation_rows is not None and stale >= self.patience:
                break
        self.network.load_state_dict(best_state)
        self.network.eval().requires_grad_(False)
        self.training_rejection_rate = rate
        self.training_samples = len(rows)
        self.selected_epoch = best_epoch
        self.epochs_run = epoch
        self.fit_loss = self.loss_history[best_epoch]['objective']
        self.fitted = True
        self.train_day_ids = sorted({str(row.get('day_id', 'unspecified')) for row in rows})
        self.validation_day_ids = sorted({str(row.get('day_id', 'unspecified')) for row in (validation_rows or [])})
        self.training_support = {name: dict(min=float(x[:, i].min()), max=float(x[:, i].max()),
            quantiles=np.quantile(x[:, i], [.01, .1, .5, .9, .99]).tolist())
            for i, name in enumerate(self.feature_names)}
        if validation_rows is not None:
            self.fit_calibration(validation_rows)
        return self

    @staticmethod
    def _labels(rows):
        if any('rejected' not in row for row in rows):
            raise ValueError('Explicit rejected=1 labels required; legacy acceptance labels cannot be silently inverted')
        y = np.asarray([row['rejected'] for row in rows], dtype=float)
        if not np.isfinite(y).all() or not set(np.unique(y)).issubset({0., 1.}):
            raise ValueError('Rejection labels must be binary')
        if any('accepted' in row and int(row['accepted']) != 1 - int(row['rejected']) for row in rows):
            raise ValueError('Contradictory acceptance/rejection labels')
        return y

    def predict_logits(self, rows) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Acceptance model has not been trained")
        self.network.eval()
        with torch.no_grad():
            p = self.network(self._tensor(rows)).squeeze(-1).numpy().astype(float)
        if not np.isfinite(p).all():
            raise RuntimeError('Nonfinite neural acceptance probabilities')
        return p

    def predict_proba(self, rows, *, calibrated=True) -> np.ndarray:
        """Always P(rejected=1), never a hard decision or acceptance probability."""
        logits = self.predict_logits(rows)
        if calibrated:
            logits = self.calibration['a'] * logits + self.calibration['b']
        return expit(logits)

    def fit_calibration(self, validation_rows):
        """Fit only held-out validation responses; retain identity if NLL worsens."""
        z, y = self.predict_logits(validation_rows), self._labels(validation_rows)
        def coefficients(theta):
            return (float(np.exp(theta[0])), float(theta[1]) if self.calibration_method == 'platt' else 0.)
        def objective(theta):
            a, b = coefficients(theta)
            logits = a * z + b
            return float(np.mean(np.logaddexp(0., logits) - y * logits))
        initial = [0., 0.] if self.calibration_method == 'platt' else [0.]
        before = objective(initial)
        theta = initial
        if self.calibration_method != 'none':
            result = minimize(objective, initial, method='L-BFGS-B',
                              bounds=[(-5., 5.)] + ([(-10., 10.)] if len(initial) == 2 else []))
            if result.success and np.isfinite(result.fun) and result.fun <= before:
                theta = result.x
        a, b = coefficients(theta)
        self.calibration = dict(method=self.calibration_method, a=a, b=b, temperature=1. / a,
            fitted=self.calibration_method != 'none', validation_nll_before=before,
            validation_nll_after=objective(theta))
        return self

    @property
    def predictor_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def support_diagnostics(self, rows):
        x = self._features(rows)
        outside = np.zeros(x.shape, dtype=bool)
        for i, name in enumerate(self.feature_names):
            support = self.training_support[name]
            outside[:, i] = (x[:, i] < support['min']) | (x[:, i] > support['max'])
        return dict(count=len(rows), outside_any_count=int(outside.any(axis=1).sum()),
                    outside_any_rate=float(outside.any(axis=1).mean()) if len(rows) else None,
                    per_feature={name: int(outside[:, i].sum()) for i, name in enumerate(self.feature_names)})

    def predict_acceptance_probability(self, env, vehicle_id: int, request) -> float:
        if int(env.vehicles[vehicle_id].get("type", 1)) == 2:
            return 1.0
        return 1.0 - self.predict_rejection_probability(env, vehicle_id, request)

    def predict_rejection_probability(self, env, vehicle_id: int, request) -> float:
        if int(env.vehicles[vehicle_id].get("type", 1)) == 2:
            return 0.0
        return float(self.predict_proba([offer_features(env, vehicle_id, request,
                                                       feature_variant=self.feature_variant)])[0])

    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Acceptance model has not been trained")
        return {
            "version": self.VERSION,
            "model_type": self.MODEL_TYPE,
            "feature_version": FEATURE_VERSION,
            "hidden_dims": list(self.hidden_dims),
            "activation": "relu",
            "feature_names": list(self.feature_names),
            "feature_variant": self.feature_variant,
            "information_set": "restricted_offer" if self.feature_variant == 'driver_offer_core' else 'platform_context',
            "feature_schema": self.feature_schema,
            "label": "1=rejected, 0=accepted", "target_semantics": "rejected=1",
            "calibration": self.calibration,
            "train_day_ids": self.train_day_ids, "validation_day_ids": self.validation_day_ids,
            "training_support": self.training_support,
            "l2": self.l2,
            "learning_rate": self.learning_rate, "max_epochs": self.max_epochs,
            "batch_size": self.batch_size, "patience": self.patience, "seed": self.seed,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "network_state": {key: value.tolist() for key, value in self.network.state_dict().items()},
            "training_samples": self.training_samples,
            "training_rejection_rate": self.training_rejection_rate,
            "fit_loss": self.fit_loss,
            "selected_epoch": self.selected_epoch, "epochs_run": self.epochs_run,
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(state)

    @classmethod
    def from_dict(cls, state):
        """Restore a predictor embedded in a Q/residual checkpoint."""
        if state.get('version') in (1, 2):
            raise ValueError('Legacy acceptance checkpoint v1/v2 is incompatible; recollect/retrain rejected=1 schema v3')
        variant = state.get('feature_variant')
        names = feature_names(variant)
        hidden = cls.HIDDEN_DIMS if variant == 'driver_offer_core' else (64, 32)
        if (state.get('version') != cls.VERSION or state.get('model_type') != cls.MODEL_TYPE
                or state.get('feature_version') != FEATURE_VERSION
                or state.get('feature_names') != list(names) or state.get('target_semantics') != 'rejected=1'
                or state.get('label') != '1=rejected, 0=accepted'
                or state.get('hidden_dims') != list(hidden) or state.get('activation') != 'relu'):
            raise ValueError('Unsupported neural acceptance model schema/architecture')
        model = cls(feature_variant=variant, calibration=state['calibration']['method'],
                    **{name: state[name] for name in
                       ('l2', 'learning_rate', 'max_epochs', 'batch_size', 'patience', 'seed')})
        model.feature_schema = state["feature_schema"]
        if model.feature_schema not in SCHEMAS:
            raise ValueError("Unknown acceptance feature schema")
        for name, size in (("mean", len(names)), ("scale", len(names))):
            value = np.asarray(state[name], dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid acceptance model {name}")
            setattr(model, name, value)
        if np.any(model.scale <= 0):
            raise ValueError("Acceptance feature scales must be positive")
        model.network = model._new_network()
        expected = model.network.state_dict()
        if set(state['network_state']) != set(expected):
            raise ValueError('Invalid neural acceptance parameter names')
        tensors = {}
        for name, reference in expected.items():
            value = torch.tensor(state['network_state'][name], dtype=reference.dtype)
            if value.shape != reference.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid neural acceptance parameter {name}')
            tensors[name] = value
        model.network.load_state_dict(tensors, strict=True)
        model.network.eval().requires_grad_(False)
        for name in ("training_samples", "training_rejection_rate", "fit_loss", "selected_epoch", "epochs_run",
                     "train_day_ids", "validation_day_ids", "training_support", "calibration"):
            setattr(model, name, state[name])
        cal = model.calibration
        if (not all(np.isfinite(cal[k]) for k in ('a', 'b', 'temperature')) or cal['a'] <= 0
                or not np.isclose(cal['temperature'] * cal['a'], 1.)
                or (cal['method'] != 'platt' and cal['b'] != 0)):
            raise ValueError('Invalid rejection calibration parameters')
        model.fitted = True
        return model


# Import compatibility only: the alias also uses rejected=1 and rejects v1/v2.
BinaryAcceptanceModel = EVRejectionProbabilityModel


def probability_metrics(rows, probabilities) -> dict:
    """Probability quality, not just majority-class classification accuracy."""
    y = EVRejectionProbabilityModel._labels(rows)
    p = np.asarray(probabilities, dtype=float)
    if not len(y) or p.shape != y.shape or not np.all(np.isfinite(p)):
        raise ValueError("Nonempty aligned finite labels/probabilities are required")
    if np.any((p < 0) | (p > 1)) or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Invalid binary labels or probabilities")
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        ranks = rankdata(p)
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives))
    oracle = np.asarray([row["oracle_rejection_probability"] for row in rows])
    bins = []
    ece = 0.0
    indexes = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        mask = indexes == index
        if not mask.any():
            continue
        error = abs(float(p[mask].mean() - y[mask].mean()))
        ece += mask.mean() * error
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "count": int(mask.sum()), "predicted": float(p[mask].mean()),
                     "observed": float(y[mask].mean()), "oracle": float(oracle[mask].mean())})
    return {
        "count": len(y), "rejection_rate": float(y.mean()),
        "predicted_rejection_rate": float(p.mean()),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "roc_auc": auc,
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == y)),
        "ece_10_bins": float(ece),
        "oracle_probability_mae": float(np.mean(np.abs(p - oracle))),
        "oracle_probability_rmse": float(np.sqrt(np.mean((p - oracle) ** 2))),
        "oracle_probability_range": [float(oracle.min()), float(oracle.max())],
        "calibration_bins": bins,
    }
