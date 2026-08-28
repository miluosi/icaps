"""Frozen calibrated rejected=1 inputs shared by all Q/residual learners.

Legacy acceptance CLI names are aliases only; v1/v2 data/checkpoints fail closed.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from src.acceptance_model import EVRejectionProbabilityModel
from src.acceptance_inputs import offer_features, vehicle_value
from src.rejection_anchor import expected_structured_score, rejection_score

RESPONSE_SCHEMA_VERSION = 3


def add_acceptance_arguments(parser):
    parser.add_argument("--ev-response-feature", "--ev-acceptance-feature", dest="ev_acceptance_feature",
                        choices=["off", "predicted"], default="off")
    parser.add_argument("--ev-response-model", "--ev-acceptance-model", dest="ev_acceptance_model", default=None,
                        help="Frozen calibrated rejected=1 neural model v3 JSON; retrain legacy models")
    parser.add_argument("--ev-response-anchor", choices=["auto", "off", "expected_immediate"], default="auto",
                        help="auto: expected anchor only for optimization_anchored_residual")
    parser.add_argument("--ev-response-critic-input", choices=["q_mask", "none"], default="q_mask")


def acceptance_checkpoint_suffix(mode, model_path, *, anchor="auto", critic_input="q_mask"):
    if mode == 'off':
        return ''
    if mode != 'predicted' or not model_path:
        raise ValueError('predicted EV feature requires a trained rejection model')
    digest = EVRejectionProbabilityModel.load(model_path).predictor_hash[:12]
    return f'_evreject-v3-{digest}-{anchor}-{critic_input}'


def configure_acceptance_feature(env, mode="off", model_path=None, *, model_state=None,
                                 anchor="auto", critic_input="q_mask"):
    if mode not in {"off", "predicted"} or anchor not in {"auto", "off", "expected_immediate"} or critic_input not in {"q_mask", "none"}:
        raise ValueError("Invalid EV response configuration")
    model = None
    if mode == "predicted":
        if model_state is None and not model_path:
            raise ValueError("predicted EV feature requires a trained rejection model")
        model = (EVRejectionProbabilityModel.from_dict(model_state) if model_state is not None
                 else EVRejectionProbabilityModel.load(model_path))
        schema = "nyc_minutes" if callable(getattr(env, "get_travel_time_minutes", None)) else "synthetic_steps"
        if model.feature_schema != schema:
            raise ValueError("EV rejection model units do not match the environment")
        if bool(getattr(env, 'knownreject', False)):
            raise ValueError("Learned rejection input cannot be combined with oracle knownreject scoring")
    env.ev_acceptance_feature = mode  # Legacy entry-point attribute.
    env.ev_acceptance_model = model
    env.ev_response_anchor = anchor
    env.ev_response_critic_input = critic_input
    env.ev_response_model_hash = model.predictor_hash if model is not None else None


def human_response_mask(vehicle, action_type_id=2):
    return bool(int(action_type_id) == 2 and int(vehicle_value(vehicle, 'type', 1)) == 1
                and vehicle_value(vehicle, 'assigned_request') is None
                and vehicle_value(vehicle, 'passenger_onboard') is None)


def predicted_rejection(env, vehicle_id, request, *, vehicle=None, snapshot=None, context=None):
    """Read only saved pre-offer state when called for replay."""
    if getattr(env, "ev_acceptance_feature", "off") == "off":
        return 0.0
    if vehicle is None:
        vehicle = (next(v for v in snapshot.vehicles if v.vehicle_id == int(vehicle_id))
                   if snapshot is not None else env.vehicles[int(vehicle_id)])
    if not human_response_mask(vehicle):
        return 0.0
    if request is None:
        raise ValueError("Missing pre-offer request for EV rejection feature")
    model = env.ev_acceptance_model
    row = offer_features(env, vehicle_id, request, vehicle=vehicle, snapshot=snapshot,
                         context=context, feature_variant=model.feature_variant)
    return float(model.predict_proba([row])[0])


def insert_zero_input(layer, index, count=1):
    """Add inputs without changing outputs, existing weights, or RNG state."""
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(layer.in_features + count, layer.out_features,
                             bias=layer.bias is not None).to(layer.weight)
    with torch.no_grad():
        expanded.weight[:, :index].copy_(layer.weight[:, :index])
        expanded.weight[:, index:index + count].zero_()
        expanded.weight[:, index + count:].copy_(layer.weight[:, index:])
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias)
    return expanded


class AcceptanceFeatureMixin:
    supports_ev_acceptance_feature = True

    def _init_acceptance_feature(self):
        self.ev_acceptance_feature = getattr(self.env, "ev_acceptance_feature", "off")
        self.response_enabled = self.ev_acceptance_feature == "predicted"
        self.acceptance_input_enabled = self.response_enabled and getattr(self.env, 'ev_response_critic_input', 'q_mask') == 'q_mask'
        anchor = getattr(self.env, 'ev_response_anchor', 'auto')
        if self.response_enabled and anchor == 'expected_immediate' and not getattr(self, 'uses_response_aware_anchor', False):
            raise ValueError('Expected response anchor is supported only by optimization_anchored_residual')
        self.response_anchor_enabled = self.response_enabled and (
            anchor == 'expected_immediate' or (anchor == 'auto' and bool(getattr(self, 'uses_response_aware_anchor', False))))
        if self.response_enabled and not self.acceptance_input_enabled and not self.response_anchor_enabled:
            raise ValueError('Enabled response model has neither critic inputs nor expected anchor')
        self._acceptance_model_state = copy.deepcopy(self.env.ev_acceptance_model.to_dict()) if self.response_enabled else None
        # The learner owns a frozen copy, so an external env mutation cannot
        # quietly change the predictor used by old replay.
        self.response_model = EVRejectionProbabilityModel.from_dict(self._acceptance_model_state) if self.response_enabled else None
        self.response_model_hash = self.response_model.predictor_hash if self.response_enabled else None

    def acceptance_checkpoint_state(self):
        return dict(mode=self.ev_acceptance_feature, model=self._acceptance_model_state,
                    version=RESPONSE_SCHEMA_VERSION, target_semantics='rejected=1',
                    predictor_hash=self.response_model_hash, critic_input=self.acceptance_input_enabled,
                    expected_anchor=self.response_anchor_enabled)

    def load_acceptance_checkpoint_state(self, state):
        saved = state.get("ev_response")
        if saved is None:
            if self.response_enabled or state.get('ev_acceptance', {}).get('mode', 'off') != 'off':
                raise ValueError("Legacy/missing EV response schema; v3 rejection checkpoint required")
            return
        if saved != self.acceptance_checkpoint_state():
            raise ValueError("Checkpoint EV response mismatch: predictor differs, schema or anchor/input configuration differs")

    def response_masks_for_live_edges(self, vehicle_ids, action_type_ids):
        return np.asarray([self.response_enabled and human_response_mask(self.env.vehicles[int(vid)], kind)
                           for vid, kind in zip(vehicle_ids, action_type_ids)], dtype=np.float32)

    def rejection_for_live_edges(self, vehicle_ids, action_type_ids, request_ids=None, rejection_probabilities=None):
        mask = self.response_masks_for_live_edges(vehicle_ids, action_type_ids)
        values = np.zeros(len(vehicle_ids), dtype=np.float32)
        if not self.response_enabled:
            return values
        if getattr(self.env, 'ev_response_model_hash', None) != self.response_model_hash:
            raise ValueError('Live rejection predictor differs from learner/replay identity')
        if rejection_probabilities is not None:
            values = np.asarray(rejection_probabilities, dtype=np.float32)
            if values.shape != mask.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) or np.any(values[mask == 0] != 0):
                raise ValueError("Rejection inputs must be finite probabilities with q=0 on masked edges")
            return values
        rows, indexes = [], []
        for i in np.flatnonzero(mask):
            if request_ids is None:
                raise ValueError("EV service inference requires request_ids or rejection_probabilities")
            request = self.env.active_requests.get(int(request_ids[i]))
            if request is None:
                raise ValueError('Missing pre-offer request for EV rejection feature')
            rows.append(offer_features(self.env, vehicle_ids[i], request, feature_variant=self.response_model.feature_variant))
            indexes.append(i)
        if rows:
            values[indexes] = self.response_model.predict_proba(rows)
        return values

    def response_from_experience(self, exp, candidate=None, *, next_state=False):
        if not self.response_enabled:
            return 0., 0.
        source = exp if candidate is None else candidate
        prefix = "next_" if next_state and candidate is None else ""
        if next_state and (exp.get('is_system_done', False) or exp.get('is_vehicle_done', False)):
            return 0., 0.
        if source.get(prefix + 'acceptance_probability') is not None:
            raise ValueError('Legacy acceptance replay is incompatible with rejected=1 v3')
        action = str(source.get(prefix + "action_type", "idle")).lower()
        action_id = source.get(prefix + "action_type_id")
        service = int(action_id) == 2 if action_id is not None else action.startswith(("assign", "service"))
        if not service or int(exp.get("vehicle_type", 1)) != 1:
            return 0., 0.
        saved = source.get(prefix + "rejection_probability")
        if saved is not None:
            if source.get('response_model_hash', exp.get('response_model_hash')) != self.response_model_hash:
                raise ValueError('Replay rejection predictor hash mismatch')
            mask = source.get(prefix + 'human_response_mask')
            expected_structured_score(0., 0., saved, mask)
            return float(saved), float(mask)
        snapshot = exp.get("next_state_snapshot" if next_state else "state_snapshot")
        if snapshot is None:
            raise ValueError("EV service replay is missing its pre-offer rejection snapshot")
        request_id = source.get(prefix + "request_id")
        if request_id is None and "_" in action:
            request_id = int(action.split("_", 1)[1])
        vehicle = next(v for v in snapshot.vehicles if v.vehicle_id == int(exp["vehicle_id"]))
        if not human_response_mask(vehicle):
            return 0., 0.
        request = next((r for r in snapshot.requests if r.request_id == request_id), None)
        if request is None:
            raise ValueError('Missing pre-offer request in replay snapshot')
        row = offer_features(self.env, exp['vehicle_id'], request, vehicle=vehicle, snapshot=snapshot,
                             feature_variant=self.response_model.feature_variant)
        return float(self.response_model.predict_proba([row])[0]), 1.

    def rejection_from_experience(self, exp, candidate=None, *, next_state=False):
        return self.response_from_experience(exp, candidate, next_state=next_state)[0]

    def response_mask_from_experience(self, exp, candidate=None, *, next_state=False):
        return self.response_from_experience(exp, candidate, next_state=next_state)[1]

    def rejection_tensor(self, rows, *, next_state=False):
        return torch.tensor([self.rejection_from_experience(row, next_state=next_state) for row in rows],
                            dtype=torch.float32, device=self.device).unsqueeze(1)

    def response_mask_tensor(self, rows, *, next_state=False):
        return torch.tensor([self.response_mask_from_experience(row, next_state=next_state) for row in rows],
                            dtype=torch.float32, device=self.device).unsqueeze(1)

    def response_anchor(self, success, request_value, pickup_distance, q, mask):
        reject = rejection_score(self.env, request_value=request_value, pickup_distance=pickup_distance)
        expected = expected_structured_score(success, reject, q, mask) if self.response_anchor_enabled else float(success)
        return expected, reject
