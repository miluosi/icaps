"""Frozen, pre-decision EV acceptance input shared by every learner.

This is an input feature, never a reward adjustment or an oracle probability.
Replay must use immutable pre-offer snapshots; it must not consult live drivers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn

from src.acceptance_model import BinaryAcceptanceModel
from src.acceptance_inputs import offer_context, offer_features, vehicle_value


def add_acceptance_arguments(parser):
    parser.add_argument("--ev-acceptance-feature", choices=["off", "predicted"], default="off")
    parser.add_argument("--ev-acceptance-model", default=None,
                        help="Frozen neural acceptance model v2 JSON; legacy regression checkpoints require retraining")


def acceptance_checkpoint_suffix(mode, model_path):
    if mode == 'off':
        return ''
    if mode != 'predicted' or not model_path:
        raise ValueError('predicted EV feature requires a trained acceptance model')
    state = BinaryAcceptanceModel.load(model_path).to_dict()
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:12]
    return '_evaccept-' + digest


def configure_acceptance_feature(env, mode="off", model_path=None, *, model_state=None):
    if mode not in {"off", "predicted"}:
        raise ValueError("ev_acceptance_feature must be off or predicted")
    model = None
    if mode == "predicted":
        if model_state is None and not model_path:
            raise ValueError("predicted EV feature requires a trained acceptance model")
        model = (BinaryAcceptanceModel.from_dict(model_state) if model_state is not None
                 else BinaryAcceptanceModel.load(model_path))
        schema = "nyc_minutes" if callable(getattr(env, "get_travel_time_minutes", None)) else "synthetic_steps"
        if model.feature_schema != schema:
            raise ValueError("EV acceptance model units do not match the environment")
    env.ev_acceptance_feature = mode
    env.ev_acceptance_model = model


def predicted_acceptance(env, vehicle_id, request, *, vehicle=None, snapshot=None, context=None):
    """Share the supervised feature builder; replay uses only its saved state."""
    if getattr(env, "ev_acceptance_feature", "off") == "off":
        return 0.0
    if vehicle is None:
        vehicle = (next(v for v in snapshot.vehicles if v.vehicle_id == int(vehicle_id))
                   if snapshot is not None else env.vehicles[int(vehicle_id)])
    if int(vehicle_value(vehicle, 'type', 1)) != 1:
        return 0.0  # AEV and non-service edges have no human-response feature.
    if request is None:
        raise ValueError("Missing pre-offer request for EV acceptance feature")
    row = offer_features(env, vehicle_id, request, vehicle=vehicle, snapshot=snapshot, context=context)
    return float(env.ev_acceptance_model.predict_proba([row])[0])


def insert_zero_input(layer, index):
    """Add an input without changing outputs, existing weights, or RNG state."""
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(layer.in_features + 1, layer.out_features,
                             bias=layer.bias is not None).to(layer.weight)
    with torch.no_grad():
        expanded.weight[:, :index].copy_(layer.weight[:, :index])
        expanded.weight[:, index].zero_()
        expanded.weight[:, index + 1:].copy_(layer.weight[:, index:])
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias)
    return expanded


class AcceptanceFeatureMixin:
    supports_ev_acceptance_feature = True

    def _init_acceptance_feature(self):
        self.ev_acceptance_feature = getattr(self.env, "ev_acceptance_feature", "off")
        self.acceptance_input_enabled = self.ev_acceptance_feature == "predicted"
        self._acceptance_model_state = (copy.deepcopy(self.env.ev_acceptance_model.to_dict())
                                        if self.acceptance_input_enabled else None)

    def acceptance_checkpoint_state(self):
        return dict(mode=self.ev_acceptance_feature, model=self._acceptance_model_state, version=1)

    def load_acceptance_checkpoint_state(self, state):
        saved = state.get("ev_acceptance", {"mode": "off", "model": None})
        if saved["mode"] != self.ev_acceptance_feature:
            raise ValueError("Checkpoint EV acceptance feature mismatch; use the same feature flag")
        if saved.get("model") != self._acceptance_model_state:
            raise ValueError("Checkpoint EV acceptance predictor differs from the configured model")

    def acceptance_for_live_edges(self, vehicle_ids, action_type_ids, request_ids=None,
                                  acceptance_probabilities=None):
        if not self.acceptance_input_enabled:
            return np.zeros(len(vehicle_ids), dtype=np.float32)
        if acceptance_probabilities is not None:
            values = np.asarray(acceptance_probabilities, dtype=np.float32)
            if values.shape != (len(vehicle_ids),) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError("Acceptance inputs must be finite probabilities with one value per edge")
            return values
        values = np.zeros(len(vehicle_ids), dtype=np.float32)
        rows, indexes = [], []
        context = offer_context(self.env)
        for i, vid in enumerate(vehicle_ids):
            if int(action_type_ids[i]) != 2 or int(self.env.vehicles[int(vid)].get("type", 1)) != 1:
                continue
            if request_ids is None:
                raise ValueError("EV service inference requires request_ids or acceptance_probabilities")
            request = self.env.active_requests.get(int(request_ids[i]))
            if request is None:
                raise ValueError('Missing pre-offer request for EV acceptance feature')
            rows.append(offer_features(self.env, vid, request, context=context))
            indexes.append(i)
        if rows:
            values[indexes] = self.env.ev_acceptance_model.predict_proba(rows)
        return values

    def acceptance_from_experience(self, exp, candidate=None, *, next_state=False):
        if not self.acceptance_input_enabled:
            return 0.0
        source = exp if candidate is None else candidate
        prefix = "next_" if next_state and candidate is None else ""
        if next_state and (exp.get('is_system_done', False) or exp.get('is_vehicle_done', False)):
            return 0.0
        action = str(source.get(prefix + "action_type", "idle")).lower()
        action_id = source.get(prefix + "action_type_id")
        service = int(action_id) == 2 if action_id is not None else action.startswith(("assign", "service"))
        if not service or int(exp.get("vehicle_type", 1)) != 1:
            return 0.0
        saved = source.get(prefix + "acceptance_probability")
        if saved is not None:
            if not np.isfinite(saved) or not 0 <= float(saved) <= 1:
                raise ValueError("Invalid acceptance probability in replay")
            return float(saved)
        snapshot = exp.get("next_state_snapshot" if next_state else "state_snapshot")
        if snapshot is None:
            raise ValueError("EV service replay is missing its pre-offer acceptance snapshot")
        request_id = source.get(prefix + "request_id")
        if request_id is None and "_" in action:
            request_id = int(action.split("_", 1)[1])
        request = next((r for r in snapshot.requests if r.request_id == request_id), None)
        vehicle = next(v for v in snapshot.vehicles if v.vehicle_id == int(exp["vehicle_id"]))
        return predicted_acceptance(self.env, exp["vehicle_id"], request, vehicle=vehicle, snapshot=snapshot)

    def acceptance_tensor(self, rows, *, next_state=False):
        return torch.tensor([self.acceptance_from_experience(row, next_state=next_state) for row in rows],
                            dtype=torch.float32, device=self.device).unsqueeze(1)
