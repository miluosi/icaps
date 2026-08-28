"""
PyTorch-based Value Function for ADP with Gym Integration

This module replaces the original Keras/TensorFlow implementation with PyTorch,
while maintaining the core ADP algorithm concepts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import gym
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from collections import deque
import random
import pickle
from pathlib import Path as PathlibPath
import logging

from src.recourse.replay import PrioritizedJointReplayBuffer
from src.recourse.types import RecourseTransition


def _resolve_aux_zone_dim(env, default: int = 4) -> int:
    if env is None:
        return default
    zone_dim = getattr(env, 'aux_zone_dim', None)
    if zone_dim is None:
        zone_dim = getattr(env, 'num_zones', default)
    zone_dim = int(zone_dim)
    return zone_dim if zone_dim > 0 else default


def _env_distribution_zone_index(env, location) -> int | None:
    """0-based index into zone-distribution vectors."""
    if env is None:
        return None
    if hasattr(env, 'get_distribution_zone_index'):
        zone_idx = env.get_distribution_zone_index(location)
    elif int(getattr(env, 'aux_zone_dim', 0) or 0) > 0 and hasattr(env, 'get_aux_zone_id'):
        zone_idx = env.get_aux_zone_id(location)
    elif hasattr(env, 'get_zone_id'):
        zone_idx = env.get_zone_id(location)
    else:
        return None
    if zone_idx is None:
        return None
    zone_idx = int(zone_idx)
    return zone_idx if zone_idx >= 0 else None


def _env_zone_index(env, location) -> int:
    """1-based id for the neural zone embedding; 0 is reserved for padding/missing."""
    if env is None:
        return 0
    if hasattr(env, 'get_zone_embedding_id'):
        return int(env.get_zone_embedding_id(location))
    zone_idx = _env_distribution_zone_index(env, location)
    return int(zone_idx) + 1 if zone_idx is not None else 0

# Use fallback logger instead of tensorboard to avoid protobuf issues
TENSORBOARD_AVAILABLE = False
print("Using fallback logging instead of TensorBoard to avoid protobuf compatibility issues")

class SummaryWriter:
    """Fallback logger when TensorBoard is not available"""
    def __init__(self, *args, **kwargs):
        self.log_dir = args[0] if args else "logs"
        print(f"Fallback logger initialized for {self.log_dir}")
    
    def add_scalar(self, tag, value, step):
        print(f"[{step}] {tag}: {value:.4f}")
    
    def flush(self):
        pass
    
    def close(self):
        pass

# Import existing modules (modified for compatibility)
try:
    from LearningAgent import LearningAgent
    from Action import Action  
    from Environment import Environment
    from Experience import Experience
    from CentralAgent import CentralAgent
    from Request import Request
except ImportError:
    # Define placeholder classes if imports fail
    class LearningAgent: pass
    class Action: pass
    class Environment: pass
    class Experience: pass
    class CentralAgent: pass
    class Request: pass
    class Path: pass
    class Experience: pass
    class CentralAgent: pass
    class Request: pass


class PyTorchReplayBuffer:
    """PyTorch-based replay buffer for experience storage"""
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return [], [], []
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** 0.6  # Alpha parameter
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class PyTorchValueFunction(ABC, nn.Module):
    """Abstract base class for PyTorch-based value functions"""
    
    def __init__(self, log_dir: str, device: str = 'cpu'):
        super(PyTorchValueFunction, self).__init__()
        self.device = torch.device(device)
        
        # Setup logging
        log_path = PathlibPath(log_dir) / type(self).__name__
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_path))
        
        # Training statistics
        self.training_step = 0
        
    def add_to_logs(self, tag: str, value: float, step: int):
        """Add scalar to tensorboard logs"""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()
    
    @abstractmethod
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Get value estimates for experiences"""
        raise NotImplementedError
    
    @abstractmethod
    def update(self, central_agent: CentralAgent):
        """Update value function parameters"""
        raise NotImplementedError
    
    @abstractmethod
    def remember(self, experience: Experience):
        """Store experience for learning"""
        raise NotImplementedError


class PyTorchRewardPlusDelay(PyTorchValueFunction):
    """Simple reward + delay value function (no learning required)"""
    
    def __init__(self, delay_coefficient: float, log_dir: str = "logs/reward_plus_delay", device: str = 'cpu'):
        super().__init__(log_dir=log_dir, device=device)
        self.delay_coefficient = delay_coefficient
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PyTorchRewardPlusDelay with delay_coefficient={delay_coefficient}")
    
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Compute value as immediate reward plus delay bonus"""
        scored_actions_all_agents = []
        
        for experience in experiences:
            for feasible_actions in experience.feasible_actions_all_agents:
                scored_actions = []
                for action in feasible_actions:
                    if hasattr(action, 'new_path') and action.new_path:
                        immediate_reward = sum([getattr(request, 'value', 0) 
                                              for request in getattr(action, 'requests', [])])
                        delay_bonus = self.delay_coefficient * getattr(action.new_path, 'total_delay', 0)
                        score = immediate_reward + delay_bonus
                    else:
                        score = 0.0
                    
                    scored_actions.append((action, score))
                scored_actions_all_agents.append(scored_actions)
        
        return scored_actions_all_agents
    
    def get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                   target_location: int, current_time: float = 0.0) -> float:
        """
        Simple Q-value calculation for ChargingIntegratedEnvironment
        
        Args:
            vehicle_id: ID of the vehicle
            action_type: Type of action ('assign', 'charge', 'move', 'idle')
            vehicle_location: Current location of vehicle
            target_location: Target location (request pickup or charging station)
            current_time: Current simulation time
            
        Returns:
            Q-value for the state-action pair
        """
        # Base reward calculation
        base_reward = 0.0
        
        # Distance penalty
        if hasattr(self, '_calculate_distance'):
            distance = self._calculate_distance(vehicle_location, target_location)
        else:
            # Simple Manhattan distance approximation
            distance = abs(vehicle_location - target_location)
        
        distance_penalty = distance * 0.1
        
        # Action-specific rewards
        if action_type.startswith('assign'):
            # Passenger service reward
            base_reward = 10.0  # Base reward for serving passenger
            # Add urgency bonus based on delay
            urgency_bonus = self.delay_coefficient * max(0, 300 - current_time)  # 300 is max delay
            base_reward += urgency_bonus
            
        elif action_type.startswith('charge'):
            # Charging reward - depends on battery level (if available)
            base_reward = 5.0  # Base charging benefit
            # Negative delay penalty for charging (opportunity cost)
            charging_delay_penalty = self.delay_coefficient * 30  # Assume 30 time units charging
            base_reward -= charging_delay_penalty
            
        elif action_type == 'move':
            # Movement has small cost
            base_reward = -1.0
            
        elif action_type == 'idle' or action_type == 'reloc' or str(action_type).startswith('reloc'):
            # Idle has minimal cost
            base_reward = -1.0 if str(action_type).startswith('reloc') else -0.5
        
        # Final Q-value calculation
        q_value = base_reward - distance_penalty
        
        return float(q_value)
    
    def _calculate_distance(self, loc1: int, loc2: int, grid_size: int = 10) -> float:
        """Calculate Manhattan distance between two locations"""
        x1, y1 = loc1 // grid_size, loc1 % grid_size
        x2, y2 = loc2 // grid_size, loc2 % grid_size
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int, 
                              current_time: float = 0.0) -> float:
        """Get Q-value for vehicle assignment to request"""
        return self.get_q_value(vehicle_id, f"assign_{target_id}", 
                               vehicle_location, target_location, current_time)
    
    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int,
                           current_time: float = 0.0) -> float:
        """Get Q-value for vehicle charging decision"""
        return self.get_q_value(vehicle_id, f"charge_{station_id}",
                               vehicle_location, station_location, current_time)

    def update(self, *args, **kwargs):
        """No learning required for this value function"""
        pass
    
    def remember(self, *args, **kwargs):
        """No experience storage required"""
        pass


from src.acceptance_features import AcceptanceFeatureMixin, insert_zero_input


class PyTorchChargingValueFunction(AcceptanceFeatureMixin, PyTorchValueFunction):
    """Neural network-based value function for ChargingIntegratedEnvironment using PyTorchPathBasedNetwork"""
    
    def __init__(self, grid_size: int = 10, num_vehicles: int = 8, 
                 log_dir: str = "logs/charging_nn", device: str = 'cpu',
                 episode_length: int = 300, max_requests: int = 1000, env=None,
                 encoder: bool = True,
                 zone_distribution_mode: str = None,
                 replay_buffer_size: int = 500000,
                 iftransformer: bool = False):
        super().__init__(log_dir=log_dir, device=device)
        
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.episode_length = episode_length  # 实际episode长度
        self.max_requests = max_requests      # 最大预期请求数
        self.num_locations = grid_size * grid_size
        self.env = env  # 存储环境引用
        self._init_acceptance_feature()
        self.iftransformer = bool(iftransformer)
        self.zone_distribution_mode = zone_distribution_mode or ("bayes" if encoder else "time-only")
        if self.zone_distribution_mode not in {"bayes", "time-only", "none"}:
            raise ValueError(f"Unsupported zone_distribution_mode: {self.zone_distribution_mode}")
        self.encoder = self.zone_distribution_mode == "bayes"  # 是否启用sequential encoder
        self.debug_name = "QNet"
        
        # num_zones for cross-attention auxiliary head
        self._num_zones = _resolve_aux_zone_dim(env)
        
        # Initialize the neural network with increased capacity for complex environment
        self.network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            num_vehicles=num_vehicles,  # 添加车辆数量参数
            max_capacity=6,  # Increased capacity for longer paths
            embedding_dim=128,  # Larger embedding for complex environment
            lstm_hidden=256,   # Larger LSTM for complex patterns
            dense_hidden=512,   # Larger dense layer
            pretrained_embeddings=None,  # Explicitly set to None to ensure gradients
            num_zones=self._num_zones,
            encoder=self.encoder,
            iftransformer=self.iftransformer,
        ).to(self.device)
        
        # Target network for stable DQN training
        self.target_network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            num_vehicles=num_vehicles,  # 添加车辆数量参数
            max_capacity=6,
            embedding_dim=128,
            lstm_hidden=256,
            dense_hidden=512,
            pretrained_embeddings=None,
            num_zones=self._num_zones,
            encoder=self.encoder,
            iftransformer=self.iftransformer,
        ).to(self.device)
        
        # Copy weights from main network to target network
        self.target_network.load_state_dict(self.network.state_dict())
        for module in (self.network, self.target_network):
            module.acceptance_input_enabled = self.acceptance_input_enabled
            if self.acceptance_input_enabled:
                layer = module.state_embedding[0]
                module.state_embedding[0] = insert_zero_input(layer, layer.in_features, count=2)
        self.target_update_frequency = 500  # Update target network every 100 steps
        
        # Ensure all parameters require gradients
        total_params = 0
        grad_params = 0
        for name, param in self.network.named_parameters():
            param.requires_grad = True
            total_params += 1
            if param.requires_grad:
                grad_params += 1
        
        print(f"   Parameters requiring gradients: {grad_params}/{total_params}")
        
        # Optimizer for training - reduced learning rate for stable learning
        self.optimizer = optim.Adam(self.network.parameters(), lr=2e-3, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()


        self.zone_dist_losses = []
        self.zone_dist_predictor_leader = None
        self.zone_dist_predictor_follower = None
        self.zone_dist_optimizer_leader = None
        self.zone_dist_optimizer_follower = None
        self.time_zone_dist_predictor_leader = None
        self.time_zone_dist_predictor_follower = None
        self.time_zone_dist_optimizer_leader = None
        self.time_zone_dist_optimizer_follower = None
        self.time_zone_dist_losses = []
        self._tz_num_time_bins = 12

        # Time-and-leader-conditioned likelihood predictor for strict Bayes fusion.
        if self.encoder:
            self.zone_dist_predictor_leader = LikelihoodZonePredictor(
                prior_dim=5,
                hidden_dim=64,
                num_zones=self._num_zones,
                num_time_bins=self._tz_num_time_bins,
            ).to(self.device)
            self.zone_dist_predictor_follower = LikelihoodZonePredictor(
                prior_dim=5,
                hidden_dim=64,
                num_zones=self._num_zones,
                num_time_bins=self._tz_num_time_bins,
            ).to(self.device)
            self.zone_dist_optimizer_leader = optim.Adam(self.zone_dist_predictor_leader.parameters(), lr=1e-3)
            self.zone_dist_optimizer_follower = optim.Adam(self.zone_dist_predictor_follower.parameters(), lr=1e-3)
        
        # Time-dependent global zone distribution predictor for bayes/time-only modes.
        if self.zone_distribution_mode != "none":
            self.time_zone_dist_predictor_leader = TimeZoneDistributionPredictor(
                num_zones=self._num_zones, hidden_dim=64, num_time_bins=self._tz_num_time_bins
            ).to(self.device)
            self.time_zone_dist_predictor_follower = TimeZoneDistributionPredictor(
                num_zones=self._num_zones, hidden_dim=64, num_time_bins=self._tz_num_time_bins
            ).to(self.device)
            self.time_zone_dist_optimizer_leader = optim.Adam(self.time_zone_dist_predictor_leader.parameters(), lr=1e-3)
            self.time_zone_dist_optimizer_follower = optim.Adam(self.time_zone_dist_predictor_follower.parameters(), lr=1e-3)
        self.zone_dist_predictor = self.zone_dist_predictor_leader
        self.zone_dist_optimizer = self.zone_dist_optimizer_leader
        self.time_zone_dist_predictor = self.time_zone_dist_predictor_leader
        self.time_zone_dist_optimizer = self.time_zone_dist_optimizer_leader
        # EMA buffer per time-bin to smooth noisy per-step zone distribution targets
        self._tz_ema_alpha = 0.1  # smoothing factor: smaller = smoother
        self._tz_ema = {'leader': {}, 'follower': {}}  # {role: {time_bin_int: np.array[num_zones]}}
        
        # 修复学习率调度器：更保守的设置，避免学习率过快下降
        # 原设置：factor=0.7, patience=50, min_lr=1e-4 太激进
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=200, 
            min_lr=1e-3
        )
        
        # Training data buffer. len(buffer) is the current sliding window, not total generated samples.
        self.replay_buffer_size = int(replay_buffer_size)
        self.total_experiences_seen = 0
        self.experience_buffer = deque(maxlen=self.replay_buffer_size)
        self.joint_replay_buffer = PrioritizedJointReplayBuffer(
            capacity=max(1, self.replay_buffer_size // 5)
        )
        self._replay_collection_context = None
        
        # Training metrics tracking
        self.training_losses = []
        self.normalized_td_losses = []
        self.td_error_history = []
        self.q_values_history = []
        self.training_step = 0
        
        # Debug mode for detailed logging
        self.debug_mode = False  # 设置为False以避免过多输出，可以通过外部设置为True
        
        # 初始化拒绝概率学习网络
        self._init_rejection_predictor()
        
        print(f"✓ PyTorchChargingValueFunction initialized with neural network (encoder={self.encoder}, distribution_mode={self.zone_distribution_mode})")
        print(f"   - Path transformer self-attention: {'enabled' if self.iftransformer else 'disabled'}")
        print(f"   - Grid size: {grid_size}x{grid_size}")
        print(f"   - Network parameters: {sum(p.numel() for p in self.network.parameters())}")
        if self.encoder:
            print(f"   - Likelihood zone predictor leader parameters: {sum(p.numel() for p in self.zone_dist_predictor_leader.parameters())}")
            print(f"   - Likelihood zone predictor follower parameters: {sum(p.numel() for p in self.zone_dist_predictor_follower.parameters())}")
        if self.time_zone_dist_predictor_leader is not None:
            print(f"   - Time zone dist predictor leader parameters: {sum(p.numel() for p in self.time_zone_dist_predictor_leader.parameters())}")
            print(f"   - Time zone dist predictor follower parameters: {sum(p.numel() for p in self.time_zone_dist_predictor_follower.parameters())}")
        else:
            print("   - Time zone dist predictor parameters: disabled (no distribution input)")
        print(f"   - Rejection predictor parameters: {sum(p.numel() for p in self.rejection_predictor.parameters())}")
    

    






    def _vehicle_type_id(self, vehicle_id: int) -> int:
        vehicles = getattr(self.env, "vehicles", {}) if self.env is not None else {}
        if int(vehicle_id) not in vehicles:
            # Compatibility callers without an environment receive the
            # declared default type; critically, no id-parity inference is
            # performed.
            return 1
        return int(vehicles[int(vehicle_id)].get("type", 1))

    def set_replay_collection_context(self, action) -> None:
        self._replay_collection_context = action

    def store_recourse_transition(self, transition: RecourseTransition) -> None:
        self.joint_replay_buffer.add(transition)

    def _init_rejection_predictor(self):
        """初始化拒绝概率预测神经网络"""
        class RejectionPredictor(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()  # 输出0-1之间的拒绝概率
                )
            
            def forward(self, x):
                return self.network(x)
        
        self.rejection_predictor = RejectionPredictor().to(self.device)
        self.rejection_optimizer = optim.Adam(self.rejection_predictor.parameters(), lr=1e-3)
        self.rejection_criterion = nn.BCELoss()
        
        # 拒绝数据缓冲区
        self.rejection_buffer = deque(maxlen=5000)
        self.rejection_training_losses = []
        self.rejection_min_train_samples = 64
        self.rejection_predictor_trained = False

    def _build_rejection_sample(
        self,
        vehicle_id: int,
        vehicle_location: int,
        pickup_location: int,
        current_time: float,
        distance: float = None,
        request=None,
        was_rejected: bool = False,
    ):
        env = getattr(self, 'env', None)
        vehicle = env.vehicles.get(vehicle_id, {}) if env is not None and hasattr(env, 'vehicles') else {}
        pickup_distance_km = float(distance if distance is not None else 0.0)
        pickup_time_minutes = 0.0
        trip_distance_km = 0.0
        trip_duration_epochs = 0.0
        request_value = 0.0
        base_value = 0.0
        dropoff_location = pickup_location
        if env is not None:
            if hasattr(env, 'get_distance_km'):
                pickup_distance_km = float(env.get_distance_km(vehicle_location, pickup_location))
            if hasattr(env, 'get_travel_time_minutes'):
                pickup_time_minutes = float(env.get_travel_time_minutes(vehicle_location, pickup_location))
        if request is not None:
            dropoff_location = int(getattr(request, 'dropoff', pickup_location))
            request_value = float(getattr(request, 'final_value', getattr(request, 'value', 0.0)))
            base_value = float(getattr(request, 'value', request_value))
            trip_duration_epochs = float(getattr(request, 'travel_time', 0.0))
            if env is not None and hasattr(env, 'get_distance_km'):
                trip_distance_km = float(env.get_distance_km(pickup_location, dropoff_location))
        num_requests = len(getattr(env, 'active_requests', {})) if env is not None else 0
        return {
            'pickup_distance_km': pickup_distance_km,
            'pickup_time_minutes': pickup_time_minutes,
            'vehicle_idle_time': float(vehicle.get('idle_timer', 0.0)),
            'battery_level': float(vehicle.get('battery', 1.0)),
            'current_time': float(current_time),
            'num_requests': float(num_requests),
            'request_value': request_value,
            'surge_value': max(0.0, request_value - base_value),
            'trip_distance_km': trip_distance_km,
            'trip_duration_epochs': trip_duration_epochs,
            'vehicle_type': int(vehicle.get('type', 1)),
            'pickup_location': int(pickup_location),
            'dropoff_location': int(dropoff_location),
            'was_rejected': bool(was_rejected),
        }

    def _rejection_feature_vector(self, sample: dict):
        env = getattr(self, 'env', None)
        epoch_length = float(getattr(env, 'EPOCH_LENGTH', 30.0)) if env is not None else 30.0
        idle_minutes = float(sample.get('vehicle_idle_time', 0.0)) * epoch_length / 60.0
        aligned_time, aligned_length = self._get_aligned_time_scalar(float(sample.get('current_time', 0.0)))
        return [
            float(np.clip(sample.get('pickup_distance_km', 0.0) / 20.0, 0.0, 5.0)),
            float(np.clip(sample.get('pickup_time_minutes', 0.0) / 60.0, 0.0, 5.0)),
            float(np.clip(idle_minutes / 120.0, 0.0, 5.0)),
            float(np.clip(sample.get('battery_level', 1.0), 0.0, 1.0)),
            float(np.clip(aligned_time / max(aligned_length, 1.0), 0.0, 1.0)),
            float(np.clip(sample.get('num_requests', 0.0) / max(float(self.max_requests), 1.0), 0.0, 5.0)),
            float(np.clip(sample.get('request_value', 0.0) / 100.0, 0.0, 5.0)),
            float(np.clip(sample.get('surge_value', 0.0) / 50.0, 0.0, 5.0)),
            float(np.clip(sample.get('trip_distance_km', 0.0) / 50.0, 0.0, 5.0)),
            float(np.clip(sample.get('trip_duration_epochs', 0.0) / max(float(self.episode_length), 1.0), 0.0, 5.0)),
        ]

    def _expand_external_zone_dist(self, external_dist, batch_size: int, device=None):
        if external_dist is None:
            return None
        target_device = device or self.device
        if isinstance(external_dist, (list, tuple)) and len(external_dist) > 0 and any(item is None for item in external_dist):
            if self._num_zones <= 0:
                return None
            rows = []
            for item in external_dist:
                if item is None:
                    rows.append([1.0 / self._num_zones] * self._num_zones)
                else:
                    rows.append(item)
            external_dist = rows
        if isinstance(external_dist, torch.Tensor):
            tensor = external_dist.to(target_device, dtype=torch.float32)
        else:
            tensor = torch.tensor(external_dist, dtype=torch.float32, device=target_device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1)
        elif tensor.dim() == 2 and tensor.size(0) == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor

    def _get_env_external_bayes_inputs(self, batch_size: int, device=None):
        if not hasattr(self, 'env') or self.env is None:
            return None, None
        target_device = device or self.device
        external_prior = self._expand_external_zone_dist(
            getattr(self.env, '_bayes_external_prior', None),
            batch_size,
            target_device,
        )
        external_posterior = self._expand_external_zone_dist(
            getattr(self.env, '_bayes_external_posterior', None),
            batch_size,
            target_device,
        )
        return external_prior, external_posterior

    def _get_env_bayes_state_dist(self, batch_size: int, device=None):
        if not hasattr(self, 'env') or self.env is None:
            return None
        state_dist = getattr(self.env, '_bayes_state_posterior', None)
        if state_dist is None and hasattr(self.env, 'refresh_bayes_state_distribution'):
            state_dist = self.env.refresh_bayes_state_distribution()
        return self._expand_external_zone_dist(state_dist, batch_size, device or self.device)

    def _merge_external_prior(self, time_zone_dist: torch.Tensor = None,
                              external_prior_dist: torch.Tensor = None):
        if external_prior_dist is None:
            return time_zone_dist
        if time_zone_dist is None:
            merged = external_prior_dist
        else:
            merged = time_zone_dist * external_prior_dist
        merged = merged / merged.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return merged

    def _build_context_tensors_from_features(self, prior_features, batch_size: int = 1):
        if prior_features is None:
            return None, None
        if len(prior_features) == 0:
            return None, None
        prior_tensor, prior_mask = self._build_padded_prior_context([prior_features] * batch_size)
        return prior_tensor, prior_mask

    def _normalize_bayes_role(self, bayes_role):
        if bayes_role is None:
            return None
        role = str(bayes_role).strip().lower()
        if role in {'leader', 'follower'}:
            return role
        return None

    def _get_default_bayes_role(self):
        env_role = None
        if hasattr(self, 'env') and self.env is not None:
            env_role = self._normalize_bayes_role(getattr(self.env, '_bayes_context_role', None))
        return env_role or 'leader'

    def _expand_bayes_roles(self, bayes_role, batch_size: int):
        if isinstance(bayes_role, torch.Tensor):
            bayes_role = bayes_role.detach().cpu().tolist()
        if isinstance(bayes_role, (list, tuple)):
            roles = [self._normalize_bayes_role(role) or self._get_default_bayes_role() for role in bayes_role]
            if len(roles) != batch_size:
                raise ValueError(f"Expected {batch_size} bayes roles, got {len(roles)}")
            return roles
        resolved_role = self._normalize_bayes_role(bayes_role) or self._get_default_bayes_role()
        return [resolved_role] * batch_size

    def _get_bayes_modules(self, bayes_role=None):
        resolved_role = self._normalize_bayes_role(bayes_role) or self._get_default_bayes_role()
        if resolved_role == 'follower':
            return (
                resolved_role,
                self.time_zone_dist_predictor_follower,
                self.zone_dist_predictor_follower,
                self.time_zone_dist_optimizer_follower,
                self.zone_dist_optimizer_follower,
            )
        return (
            'leader',
            self.time_zone_dist_predictor_leader,
            self.zone_dist_predictor_leader,
            self.time_zone_dist_optimizer_leader,
            self.zone_dist_optimizer_leader,
        )

    def _should_use_monte_carlo_posterior(self, bayes_role=None):
        resolved_role = self._normalize_bayes_role(bayes_role) or self._get_default_bayes_role()
        if resolved_role != 'leader':
            return False
        if not hasattr(self, 'env') or self.env is None:
            return False
        decision_mode = str(getattr(self.env, 'decision_mode', '') or '').strip().lower()
        leader_is_ev = getattr(self.env, '_leader_is_ev', None) is True
        return decision_mode in {'ev_first', 'evfirst'} and leader_is_ev

    def predict_bayes_context(self, current_time: float, prior_features=None,
                              external_prior_dist=None, external_posterior_dist=None,
                              bayes_state_dist=None,
                              bayes_role=None):
        if self.zone_distribution_mode == "none":
            return None, None
        resolved_role = self._normalize_bayes_role(bayes_role) or self._get_default_bayes_role()
        if self._should_use_monte_carlo_posterior(resolved_role):
            external_posterior_dist = self._estimate_monte_carlo_posterior(
                current_time=current_time,
                vehicle_type=1,
                bayes_role='leader',
            )
        hour_norm_tensor = self._get_hour_norm_tensor(current_time)
        prior_tensor, prior_mask = self._build_context_tensors_from_features(prior_features, batch_size=1)
        time_zone_dist = self._get_time_zone_dist_tensor(hour_norm_tensor, bayes_role=resolved_role)
        likelihood_logits = self._get_likelihood_zone_logits_tensor(hour_norm_tensor, prior_tensor, prior_mask, bayes_role=resolved_role)
        external_prior = self._expand_external_zone_dist(external_prior_dist, 1, self.device)
        external_posterior = self._expand_external_zone_dist(external_posterior_dist, 1, self.device)
        state_posterior = self._expand_external_zone_dist(bayes_state_dist, 1, self.device)
        if state_posterior is None:
            state_posterior = self._get_env_bayes_state_dist(1, self.device)
        effective_prior = self._merge_external_prior(time_zone_dist, external_prior)
        posterior = self._combine_zone_dist_tensors(
            time_zone_dist,
            likelihood_logits,
            external_prior_dist=external_prior,
            external_posterior_dist=external_posterior,
            bayes_state_dist=state_posterior,
        )
        prior_output = None if effective_prior is None else effective_prior.squeeze(0).detach().cpu().tolist()
        posterior_output = None if posterior is None else posterior.squeeze(0).detach().cpu().tolist()
        return prior_output, posterior_output

    def _get_prior_features_tensor(self, batch_size: int):
        """Build prior_features tensor from env cache for cross-attention.

        Returns:
            torch.Tensor [batch_size, N, 5] or None if no prior features cached.
        """
        prior_tensor, _ = self._get_prior_context_tensors(batch_size)
        return prior_tensor
    


    def _simulate_random_trajectory(self, experience: Experience):
        """Simulate a random trajectory from the given experience by taking random feasible actions."""
        if not experience.feasible_actions_all_agents:
            return
        for feasible_actions in experience.feasible_actions_all_agents:
            if not feasible_actions:
                continue
            random_action = random.choice(feasible_actions)
            # Simulate the effect of this action on the experience (this is a placeholder - actual implementation would depend on environment dynamics)
            # For example, we could update the experience's state, time, and reward based on the action taken.
            # This is a simplified example and should be replaced with actual simulation logic.
            experience.current_time += 1  # Increment time as an example
            experience.cumulative_reward += random_action.reward if hasattr(random_action, 'reward') else 0

    def monte_carlo_sample(self, experience: Experience = None, num_samples: int = 10,
                           current_time: float = None, vehicle_type: int = 1,
                           bayes_role: str = 'leader') -> List[dict]:
        """Sample historical experiences at the same t-step for Monte Carlo posterior estimation."""
        sample_time = current_time
        if sample_time is None and experience is not None:
            sample_time = getattr(experience, 'current_time', None)
        if sample_time is None and experience is not None:
            sample_time = getattr(experience, 'time', None)
        if sample_time is None:
            return []

        normalized_role = self._normalize_bayes_role(bayes_role) or 'leader'
        tolerance = 1e-6
        matched_history = []
        for hist_exp in self.experience_buffer:
            if not isinstance(hist_exp, dict):
                continue
            if hist_exp.get('vehicle_type', None) != vehicle_type:
                continue
            hist_role = self._normalize_bayes_role(hist_exp.get('bayes_role', None)) or 'leader'
            if hist_role != normalized_role:
                continue
            hist_time = hist_exp.get('current_time', None)
            if hist_time is None or abs(float(hist_time) - float(sample_time)) > tolerance:
                continue
            if hist_exp.get('target_zoneid', None) is None:
                continue
            matched_history.append(hist_exp)

        if len(matched_history) < num_samples:
            return []
        if len(matched_history) == num_samples:
            return list(matched_history)
        return random.sample(matched_history, num_samples)

    def _uniform_zone_distribution(self):
        if self._num_zones <= 0:
            return None
        return [1.0 / self._num_zones] * self._num_zones

    def _stored_distribution_zone_index(self, sample: dict, key: str = 'target_zoneid') -> int | None:
        explicit_key = f"{key}_index" if key.endswith('zoneid') else f"{key}_zone_index"
        zone_idx = sample.get(explicit_key, None)
        if zone_idx is None and key == 'target_zoneid':
            zone_idx = sample.get('target_zone_index', None)
        if zone_idx is not None:
            zone_idx = int(zone_idx)
            return zone_idx if 0 <= zone_idx < self._num_zones else None

        zone_id = sample.get(key, None)
        if zone_id is None:
            return None
        zone_id = int(zone_id)
        if 1 <= zone_id <= self._num_zones:
            return zone_id - 1
        if 0 <= zone_id < self._num_zones:
            return zone_id
        return None

    def _estimate_monte_carlo_posterior(self, current_time: float, vehicle_type: int = 1,
                                        bayes_role: str = 'leader', num_samples: int = 10):
        samples = self.monte_carlo_sample(
            num_samples=num_samples,
            current_time=current_time,
            vehicle_type=vehicle_type,
            bayes_role=bayes_role,
        )
        if len(samples) < num_samples:
            return self._uniform_zone_distribution()

        zone_counts = np.zeros(self._num_zones, dtype=np.float32)
        for sample in samples:
            zone_idx = self._stored_distribution_zone_index(sample)
            if zone_idx is not None:
                zone_counts[zone_idx] += 1.0

        total = float(zone_counts.sum())
        if total <= 0.0:
            return self._uniform_zone_distribution()
        return (zone_counts / total).tolist()



    def _build_padded_prior_context(self, prior_list, prior_dim: int = 5):
        """Pad variable-length prior feature lists and build a validity mask."""
        max_n = 0
        for pf in prior_list:
            if pf is not None and len(pf) > 0:
                max_n = max(max_n, len(pf))

        if max_n == 0:
            return None, None

        padded = []
        masks = []
        for pf in prior_list:
            if pf is not None and len(pf) > 0:
                tensor = torch.tensor(pf, dtype=torch.float32, device=self.device)
                mask = torch.ones(tensor.size(0), dtype=torch.bool, device=self.device)
                if tensor.size(0) < max_n:
                    pad_rows = max_n - tensor.size(0)
                    tensor = torch.cat(
                        [tensor, torch.zeros(pad_rows, prior_dim, device=self.device)],
                        dim=0,
                    )
                    mask = torch.cat(
                        [mask, torch.zeros(pad_rows, dtype=torch.bool, device=self.device)],
                        dim=0,
                    )
            else:
                tensor = torch.zeros(max_n, prior_dim, device=self.device)
                mask = torch.zeros(max_n, dtype=torch.bool, device=self.device)
            padded.append(tensor)
            masks.append(mask)

        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    def _get_prior_context_tensors(self, batch_size: int):
        """Build prior_features and validity mask from env cache for cross-attention."""
        if not hasattr(self, 'env') or self.env is None:
            return None, None
        feats = getattr(self.env, '_prior_features_for_posterior', None)
        if feats is None or len(feats) == 0:
            return None, None
        # feats is List[List[float]] with shape [N, 5]
        t = torch.tensor(feats, dtype=torch.float32, device=self.device)  # [N, 5]
        mask = torch.ones(t.size(0), dtype=torch.bool, device=self.device)  # [N]
        # Replicate for every element in the batch
        return (
            t.unsqueeze(0).expand(batch_size, -1, -1),
            mask.unsqueeze(0).expand(batch_size, -1),
        )

    def _get_time_zone_dist_tensor(self, hour_norm_tensor: torch.Tensor, bayes_role=None):
        """Compute global time-dependent prior zone distribution.

        Args:
            hour_norm_tensor: [B, 1] hour_of_day / 24.0, in [0, 1].
        Returns:
            torch.Tensor [B, num_zones] zone distribution probabilities.
        """
        if self.zone_distribution_mode == "none":
            return None
        batch_size = hour_norm_tensor.size(0)
        roles = self._expand_bayes_roles(bayes_role, batch_size)
        probs = torch.full(
            (batch_size, self._num_zones),
            1.0 / self._num_zones,
            dtype=torch.float32,
            device=hour_norm_tensor.device,
        )
        for role_name in ('leader', 'follower'):
            role_indices = [idx for idx, role in enumerate(roles) if role == role_name]
            if not role_indices:
                continue
            _, predictor, _, _, _ = self._get_bayes_modules(role_name)
            if predictor is None:
                continue
            index_tensor = torch.tensor(role_indices, dtype=torch.long, device=hour_norm_tensor.device)
            with torch.no_grad():
                predictor.eval()
                _, role_probs = predictor(hour_norm_tensor.index_select(0, index_tensor))
            probs.index_copy_(0, index_tensor, role_probs)
        return probs

    def _get_likelihood_zone_logits_tensor(self, hour_norm_tensor: torch.Tensor,
                                           prior_tensor: torch.Tensor,
                                           prior_mask: torch.Tensor = None,
                                           bayes_role=None):
        """Compute time-conditioned likelihood logits for strict Bayes fusion.

        Args:
            hour_norm_tensor: [B, 1] hour_of_day / 24.0, in [0, 1].
            prior_tensor: [B, N, 5] leader prior features, or None.
        Returns:
            torch.Tensor [B, num_zones] logits or None if unavailable.
        """
        if not self.encoder:
            return None
        if prior_tensor is None or prior_tensor.size(1) == 0:
            return None
        batch_size = hour_norm_tensor.size(0)
        roles = self._expand_bayes_roles(bayes_role, batch_size)
        has_predictor = any(self._get_bayes_modules(role)[2] is not None for role in roles)
        if not has_predictor:
            return None
        logits = torch.zeros((batch_size, self._num_zones), dtype=torch.float32, device=hour_norm_tensor.device)
        for role_name in ('leader', 'follower'):
            role_indices = [idx for idx, role in enumerate(roles) if role == role_name]
            if not role_indices:
                continue
            _, _, predictor, _, _ = self._get_bayes_modules(role_name)
            if predictor is None:
                continue
            index_tensor = torch.tensor(role_indices, dtype=torch.long, device=hour_norm_tensor.device)
            with torch.no_grad():
                predictor.eval()
                role_logits, _ = predictor(
                    prior_tensor.index_select(0, index_tensor),
                    hour_norm_tensor.index_select(0, index_tensor),
                    prior_mask.index_select(0, index_tensor) if prior_mask is not None else None,
                )
            logits.index_copy_(0, index_tensor, role_logits)
        return logits

    def _compute_bayesian_log_posterior(self, prior_logits: torch.Tensor,
                                        likelihood_logits: torch.Tensor):
        """Compute strict Bayes posterior in log-space.

        The time branch defines the prior p(z|t). The likelihood branch outputs
        unnormalized log-likelihood scores proportional to p(c|z,t). The fused
        posterior is computed as

            log p(z|t,c) = log p(z|t) + log p(c|z,t) - log Z.
        """
        prior_log_probs = torch.log_softmax(prior_logits, dim=1)
        return torch.log_softmax(prior_log_probs + likelihood_logits, dim=1)

    def _combine_zone_dist_tensors(self, time_zone_dist: torch.Tensor = None,
                                   likelihood_logits: torch.Tensor = None,
                                   external_prior_dist: torch.Tensor = None,
                                   external_posterior_dist: torch.Tensor = None,
                                   bayes_state_dist: torch.Tensor = None):
        """Combine time prior and likelihood branch under strict Bayes rule.

        If both are present, treat time_zone_dist as p(z|t) and likelihood_logits
        as unnormalized log p(c|z,t), then compute the normalized posterior.
        If only one input is available, return its normalized zone distribution.
        """
        if bayes_state_dist is not None:
            return bayes_state_dist / bayes_state_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)
        time_zone_dist = self._merge_external_prior(time_zone_dist, external_prior_dist)
        if external_posterior_dist is not None:
            external_logits = torch.log(external_posterior_dist.clamp_min(1e-8))
            likelihood_logits = external_logits if likelihood_logits is None else (likelihood_logits + external_logits)
        if likelihood_logits is None:
            return time_zone_dist
        if time_zone_dist is None:
            return torch.softmax(likelihood_logits, dim=1)
        prior_log_probs = torch.log(time_zone_dist.clamp_min(1e-8))
        fused_log_probs = torch.log_softmax(prior_log_probs + likelihood_logits, dim=1)
        return torch.exp(fused_log_probs)

    def _get_combined_zone_dist_tensor(self, hour_norm_tensor: torch.Tensor,
                                       prior_tensor: torch.Tensor = None,
                                       prior_mask: torch.Tensor = None,
                                       external_prior_dist: torch.Tensor = None,
                                       external_posterior_dist: torch.Tensor = None,
                                       bayes_state_dist: torch.Tensor = None,
                                       bayes_role=None):
        """Get fused zone distribution used by the value network."""
        if self.zone_distribution_mode == "none":
            return None
        if bayes_state_dist is None:
            bayes_state_dist = self._get_env_bayes_state_dist(
                hour_norm_tensor.size(0),
                hour_norm_tensor.device,
            )
        time_zone_dist = self._get_time_zone_dist_tensor(hour_norm_tensor, bayes_role=bayes_role)
        likelihood_logits = self._get_likelihood_zone_logits_tensor(hour_norm_tensor, prior_tensor, prior_mask, bayes_role=bayes_role)
        if external_prior_dist is None and external_posterior_dist is None:
            env_external_prior, env_external_posterior = self._get_env_external_bayes_inputs(
                hour_norm_tensor.size(0),
                hour_norm_tensor.device,
            )
            external_prior_dist = env_external_prior
            external_posterior_dist = env_external_posterior
        return self._combine_zone_dist_tensors(
            time_zone_dist,
            likelihood_logits,
            external_prior_dist=external_prior_dist,
            external_posterior_dist=external_posterior_dist,
            bayes_state_dist=bayes_state_dist,
        )

    def _get_hour_norm_tensor(self, current_time: float) -> torch.Tensor:
        """Convert raw current_time to hour_of_day / 24.0 tensor [1, 1].

        Uses env.get_hour_of_day() if available, else falls back to linear mapping.
        """
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_hour_of_day'):
            hour = self.env.get_hour_of_day(current_time)
        else:
            hour = (current_time / self.episode_length * 24.0) if self.episode_length > 0 else 0.0
        return torch.tensor([[hour / 24.0]], dtype=torch.float32).to(self.device)

    def _get_aligned_time_scalar(self, current_time: float) -> tuple[float, float]:
        aligned_episode_length = float(getattr(self, 'aligned_inference_episode_length', self.episode_length))
        if aligned_episode_length <= 0:
            aligned_episode_length = float(self.episode_length) if self.episode_length > 0 else 1.0
        aligned_time_offset = float(getattr(self, 'aligned_inference_time_offset', 0.0))
        aligned_current_time = float(current_time) + aligned_time_offset
        aligned_current_time = min(max(0.0, aligned_current_time), aligned_episode_length)
        return aligned_current_time, aligned_episode_length

    def _get_aligned_time_array(self, current_times) -> tuple[np.ndarray, float]:
        aligned_episode_length = float(getattr(self, 'aligned_inference_episode_length', self.episode_length))
        if aligned_episode_length <= 0:
            aligned_episode_length = float(self.episode_length) if self.episode_length > 0 else 1.0
        aligned_time_offset = float(getattr(self, 'aligned_inference_time_offset', 0.0))
        aligned_current_times = np.asarray(current_times, dtype=np.float32) + np.float32(aligned_time_offset)
        aligned_current_times = np.clip(aligned_current_times, 0.0, aligned_episode_length)
        return aligned_current_times, aligned_episode_length

    def _build_hour_norm_batch(self, raw_current_times) -> torch.Tensor:
        """Convert a list/sequence of raw current_time values to [B, 1] hour_norm tensor.

        Uses env.get_hour_of_day() when available.
        """
        has_env_hour = (hasattr(self, 'env') and self.env is not None
                        and hasattr(self.env, 'get_hour_of_day'))
        hours = []
        for ct in raw_current_times:
            if has_env_hour:
                hours.append(self.env.get_hour_of_day(ct) / 24.0)
            else:
                hours.append(ct / self.episode_length if self.episode_length > 0 else 0.0)
        return torch.tensor(hours, dtype=torch.float32).unsqueeze(1).to(self.device)

    def _get_dropout_state_values(self, vehicle_id=-1, satisfaction=None, salary_ratio=None, dropout_probability=None):
        if salary_ratio is None:
            vehicle = None
            if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'vehicles'):
                vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle is not None and vehicle.get('type') == 1:
                base_salary = max(float(getattr(self.env, 'ev_basesalary', 0.0)), 1e-6)
                salary_ratio = vehicle.get('daily_salary', 0.0) / base_salary
            else:
                salary_ratio = 0.0
        return (float(np.clip(salary_ratio, 0.0, 5.0)),)

    def _dropout_state_tensor(self, vehicle_id=-1, satisfaction=None, salary_ratio=None, dropout_probability=None, device=None):
        target_device = device if device is not None else self.device
        return torch.tensor([self._get_dropout_state_values(vehicle_id, satisfaction, salary_ratio, dropout_probability)], dtype=torch.float32, device=target_device)

    def _batch_dropout_state_tensor(self, batch_inputs, device=None, prefix=''):
        target_device = device if device is not None else self.device
        rows = []
        for inp in batch_inputs:
            rows.append(self._get_dropout_state_values(
                inp.get('vehicle_id', -1),
                inp.get(f'{prefix}dropout_satisfaction'),
                inp.get(f'{prefix}dropout_salary_ratio'),
                inp.get(f'{prefix}dropout_probability'),
            ))
        return torch.tensor(rows, dtype=torch.float32, device=target_device)

    def _batch_dropout_state_tensor_from_vehicle_ids(self, vehicle_ids, device=None):
        target_device = device if device is not None else self.device
        rows = [self._get_dropout_state_values(int(vehicle_id)) for vehicle_id in vehicle_ids]
        return torch.tensor(rows, dtype=torch.float32, device=target_device)

    def batch_get_mixed_q_values(
        self,
        *,
        vehicle_ids,
        vehicle_locations,
        target_locations,
        current_times,
        other_vehicles,
        num_requests,
        battery_levels,
        request_values,
        target_distances,
        target_zoneids,
        vehicle_idle_times,
        action_type_ids,
        post_action_distances=None,
        post_action_durations=None,
        post_action_zoneids=None,
        post_action_locations=None,
        target_station_ids=None,
        request_ids=None,
        rejection_probabilities=None,
    ):
        if len(vehicle_ids) == 0:
            return []

        batch_size = len(vehicle_ids)
        acceptance = self.rejection_for_live_edges(vehicle_ids, action_type_ids, request_ids, rejection_probabilities)
        action_type_ids = np.asarray(action_type_ids, dtype=np.int64)
        vehicle_locations = np.asarray(vehicle_locations, dtype=np.int64)
        target_locations = np.asarray(target_locations, dtype=np.int64)
        current_times = np.asarray(current_times, dtype=np.float32)
        other_vehicles = np.asarray(other_vehicles, dtype=np.float32)
        num_requests = np.asarray(num_requests, dtype=np.float32)
        battery_levels = np.asarray(battery_levels, dtype=np.float32)
        request_values = np.asarray(request_values, dtype=np.float32)
        target_distances = np.asarray(target_distances, dtype=np.float32)
        target_zoneids = np.asarray(target_zoneids, dtype=np.int64)
        vehicle_idle_times = np.asarray(vehicle_idle_times, dtype=np.float32)
        if post_action_distances is None:
            post_action_distances = np.zeros(batch_size, dtype=np.float32)
        if post_action_durations is None:
            post_action_durations = np.zeros(batch_size, dtype=np.float32)
        if post_action_zoneids is None:
            post_action_zoneids = np.zeros(batch_size, dtype=np.int64)
        post_action_distances = np.asarray(post_action_distances, dtype=np.float32)
        post_action_durations = np.asarray(post_action_durations, dtype=np.float32)
        post_action_zoneids = np.asarray(post_action_zoneids, dtype=np.int64)

        safe_vehicle_locations = np.clip(vehicle_locations, 0, self.num_locations - 1)
        safe_target_locations = np.clip(target_locations, 0, self.num_locations - 1)
        path_target_locations = np.where(
            action_type_ids == 1,
            safe_vehicle_locations,
            safe_target_locations,
        )

        path_locations = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)
        path_locations[:, 0] = torch.as_tensor(safe_vehicle_locations + 1, dtype=torch.long, device=self.device)
        path_locations[:, 1] = torch.as_tensor(path_target_locations + 1, dtype=torch.long, device=self.device)

        path_delays = torch.zeros((batch_size, 3, 1), dtype=torch.float32, device=self.device)
        assign_mask = action_type_ids == 2
        aligned_current_times, aligned_episode_length = self._get_aligned_time_array(current_times)
        if np.any(assign_mask):
            assign_delays = np.maximum(0.0, (aligned_episode_length - aligned_current_times[assign_mask]) / aligned_episode_length)
            path_delays[torch.as_tensor(assign_mask, dtype=torch.bool, device=self.device), 1, 0] = torch.as_tensor(
                assign_delays, dtype=torch.float32, device=self.device
            )
        charge_mask = action_type_ids == 3
        if np.any(charge_mask):
            path_delays[torch.as_tensor(charge_mask, dtype=torch.bool, device=self.device), 1, 0] = 0.8
        idle_mask = action_type_ids == 1
        if np.any(idle_mask):
            path_delays[torch.as_tensor(idle_mask, dtype=torch.bool, device=self.device), 1, 0] = 0.05

        current_time_tensor = torch.as_tensor(
            (aligned_current_times / aligned_episode_length).reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        other_agents_tensor = torch.as_tensor(
            (np.minimum(other_vehicles, self.num_vehicles) / self.num_vehicles).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        num_requests_tensor = torch.as_tensor(
            (np.minimum(num_requests, self.max_requests) / self.max_requests).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        battery_tensor = torch.as_tensor(battery_levels.reshape(-1, 1), dtype=torch.float32, device=self.device)
        request_value_tensor = torch.as_tensor((request_values / 100.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
        target_distance_tensor = torch.as_tensor(target_distances.reshape(-1, 1), dtype=torch.float32, device=self.device)
        target_zoneid_tensor = torch.as_tensor(
            np.clip(target_zoneids, 0, self._num_zones).reshape(-1, 1), dtype=torch.long, device=self.device
        )
        vehicle_idle_time_tensor = torch.as_tensor(vehicle_idle_times.reshape(-1, 1), dtype=torch.float32, device=self.device)
        action_type_tensor = torch.as_tensor(action_type_ids.reshape(-1, 1), dtype=torch.long, device=self.device)
        dropout_state_tensor = self._batch_dropout_state_tensor_from_vehicle_ids(vehicle_ids, device=self.device)
        post_action_distance_tensor = torch.as_tensor(post_action_distances.reshape(-1, 1), dtype=torch.float32, device=self.device)
        post_action_duration_tensor = torch.as_tensor(
            (np.minimum(post_action_durations, self.episode_length) / max(float(self.episode_length), 1.0)).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        post_action_zoneid_tensor = torch.as_tensor(
            np.clip(post_action_zoneids, 0, self._num_zones).reshape(-1, 1),
            dtype=torch.long,
            device=self.device,
        )
        prior_tensor = None
        prior_mask = None
        time_zone_dist_tensor = None
        if self.encoder or self.zone_distribution_mode != "none":
            prior_tensor, prior_mask = self._get_prior_context_tensors(batch_size)
            hour_norm_batch = self._build_hour_norm_batch(current_times)
            time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_batch, prior_tensor, prior_mask)

        with torch.no_grad():
            batch_q_values = self.network(
                path_locations=path_locations,
                rejection_probability=torch.as_tensor(acceptance, device=self.device).unsqueeze(1),
                human_response_mask=torch.as_tensor(self.response_masks_for_live_edges(vehicle_ids, action_type_ids), device=self.device).unsqueeze(1),
                path_delays=path_delays,
                current_time=current_time_tensor,
                other_agents=other_agents_tensor,
                num_requests=num_requests_tensor,
                battery_level=battery_tensor,
                request_value=request_value_tensor,
                target_distance=target_distance_tensor,
                target_zoneid=target_zoneid_tensor,
                action_type=action_type_tensor,
                vehicle_idle_time=vehicle_idle_time_tensor,
                vehicle_type=None,
                dropout_state_features=dropout_state_tensor,
                post_action_distance=post_action_distance_tensor,
                post_action_duration=post_action_duration_tensor,
                post_action_zoneid=post_action_zoneid_tensor,
                prior_features=prior_tensor,
                prior_mask=prior_mask,
                time_zone_dist=time_zone_dist_tensor,
            )

        return batch_q_values.detach().cpu().numpy().reshape(-1).tolist()

    def get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                   target_location: int, current_time: float = 0.0, 
                   other_vehicles: int = 0, num_requests: int = 0, 
                   battery_level: float = 1.0, request_value: float = 0.0) -> float:
        """
        Neural network-based Q-value calculation using PyTorchPathBasedNetwork
        现在支持vehicle_id、battery_level、request_value、action_type以及🆕 target location和zone_id信息
        """
        # 将action_type字符串转换为数值编码
        action_type = str(action_type)
        if self.acceptance_input_enabled:
            kind = 2 if action_type.startswith('assign') else (3 if action_type.startswith('charge') else 1)
            target_id = int(action_type.split('_', 1)[1]) if kind == 2 and '_' in action_type else -1
            return self._acceptance_batch_assignment([dict(
                vehicle_id=vehicle_id, target_id=target_id, vehicle_location=vehicle_location,
                target_location=target_location, current_time=current_time,
                other_vehicles=other_vehicles, num_requests=num_requests,
                battery_level=battery_level, request_value=request_value,
            )], action_id=kind)[0]
        if action_type == 'idle' or action_type == 'reloc' or action_type.startswith('reloc'):
            action_type_id = 1
        elif action_type.startswith('assign'):
            action_type_id = 2
        elif action_type.startswith('charge'):
            action_type_id = 3
        else:
            action_type_id = 2  # 默认为assign
        
        # 从Environment中获取车辆类型（需要从外部传入或者推断）
        # 假设vehicle_id为偶数是EV，奇数是AEV（简化处理）
        # 实际应用中应该从环境或配置中获取
        vehicle_type_id = self._vehicle_type_id(vehicle_id)
        
        # 🆕 计算target location的曼哈顿距离并归一化
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, '_manhattan_distance_loc'):
            manhattan_distance = self.env._manhattan_distance_loc(vehicle_location, target_location)
        else:
            # Fallback: 手动计算
            grid_size = self.grid_size if hasattr(self, 'grid_size') else int(math.sqrt(self.num_locations))
            vehicle_x = vehicle_location % grid_size
            vehicle_y = vehicle_location // grid_size
            target_x = target_location % grid_size
            target_y = target_location // grid_size
            manhattan_distance = abs(vehicle_x - target_x) + abs(vehicle_y - target_y)
        normalized_distance = manhattan_distance
        
        # 🆕 从environment获取target location的zone_id
        target_zoneid = 0  # 默认值
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_zone_id'):
            target_zoneid = _env_zone_index(self.env, target_location)
        
        # 使用支持battery、request_value、target_distance和target_zoneid的输入准备方法
        inputs = self._prepare_network_input_with_battery(
            vehicle_location, target_location, current_time, 
            other_vehicles, num_requests, action_type, battery_level, request_value,
            normalized_distance, target_zoneid  # 🆕 传入target信息
        )
        
        # 处理返回的输入（包含battery、request_value、target_distance和target_zoneid）
        if len(inputs) == 9:  # 🆕 完整输入：包含所有特征
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor, value_tensor, distance_tensor, zoneid_tensor = inputs
        elif len(inputs) == 7:  # 向后兼容：包含battery和request_value
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor, value_tensor = inputs
            distance_tensor = torch.tensor([[normalized_distance]], dtype=torch.float32).to(self.device)
            zoneid_tensor = torch.tensor([[target_zoneid]], dtype=torch.long).to(self.device)
        elif len(inputs) == 6:  # 只包含battery
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor = inputs
            value_tensor = torch.tensor([[request_value]], dtype=torch.float32).to(self.device)
            distance_tensor = torch.tensor([[normalized_distance]], dtype=torch.float32).to(self.device)
            zoneid_tensor = torch.tensor([[target_zoneid]], dtype=torch.long).to(self.device)
        else:  # 不包含battery（向后兼容）
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = inputs
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
            value_tensor = torch.tensor([[request_value]], dtype=torch.float32).to(self.device)
            distance_tensor = torch.tensor([[normalized_distance]], dtype=torch.float32).to(self.device)
            zoneid_tensor = torch.tensor([[target_zoneid]], dtype=torch.long).to(self.device)
        
        # 获取车辆的idle时间（归一化）
        vehicle_idle_time = 0.0
        if hasattr(self, 'env') and self.env is not None:
            vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle:
                raw_idle_time = vehicle.get('idle_timer', 0)
                # 归一化idle时间：假设最大idle时间为10个时间单位
                vehicle_idle_time = raw_idle_time
        
        # 创建vehicle和action相关的tensors
        action_type_tensor = torch.tensor([[action_type_id]], dtype=torch.long).to(self.device)
        vehicle_idle_time_tensor = torch.tensor([[vehicle_idle_time]], dtype=torch.float32).to(self.device)
        vehicle_type_tensor = torch.tensor([[vehicle_type_id]], dtype=torch.long).to(self.device)
        dropout_state_tensor = self._dropout_state_tensor(vehicle_id)
        
        # Forward pass through network
        self.network.eval()
        prior_tensor, prior_mask = self._get_prior_context_tensors(1)
        hour_norm_tensor = self._get_hour_norm_tensor(current_time)
        time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_tensor, prior_tensor, prior_mask)
        with torch.no_grad():
            q_value = self.network(
                path_locations=path_locations,
                path_delays=path_delays,
                current_time=time_tensor,
                other_agents=others_tensor,
                num_requests=requests_tensor,
                battery_level=battery_tensor,
                request_value=value_tensor,
                target_distance=distance_tensor,      # 🆕 传入target距离
                target_zoneid=zoneid_tensor,          # 🆕 传入target zone_id
                action_type=action_type_tensor,
                vehicle_idle_time=vehicle_idle_time_tensor,
                vehicle_type=vehicle_type_tensor,
                dropout_state_features=dropout_state_tensor,
                prior_features=prior_tensor,
                time_zone_dist=time_zone_dist_tensor
            )
            
            # Apply clipping to prevent extreme Q-values that can dominate the objective
            raw_q_value = float(q_value.item())
            
            # Clip Q-values to reasonable range to prevent optimization instability
            # This ensures Q-values don't overwhelm request values in the objective function
            # clipped_q_value = max(-50.0, min(50.0, raw_q_value))
            
            #if abs(raw_q_value - clipped_q_value) > 1e-6:  # Only log if actual clipping occurred
                #print(f"Q-value clipped: {raw_q_value:.3f} -> {clipped_q_value:.3f} for action {action_type}")
            
            return raw_q_value
    
    def _prepare_network_input(self, vehicle_location: int, target_location: int, 
                              current_time: float, other_vehicles: int, num_requests: int,
                              action_type: str):
        """Prepare input tensors for the neural network"""
        # Create path sequence: current location -> target location
        path_locations = torch.zeros(1, 3, dtype=torch.long)  # batch_size=1, seq_len=3
        path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
        
        # Set path: current -> target -> end (with boundary checking)
        # Handle coordinate tuples or integer indices
        def _convert_location_to_index(location):
            if isinstance(location, tuple) and len(location) == 2:
                # Convert coordinate tuple to location index
                x, y = location
                grid_size = int(self.num_locations ** 0.5)  # Assuming square grid
                return y * grid_size + x
            elif isinstance(location, int):
                return location
            else:
                # Fallback for unexpected types (silent handling)
                if location is None:
                    return 0
                return 0
        
        # Convert locations to indices and ensure they are within valid range [0, num_locations-1]
        safe_vehicle_location = max(0, min(_convert_location_to_index(vehicle_location), self.num_locations - 1))
        safe_target_location = max(0, min(_convert_location_to_index(target_location), self.num_locations - 1))
        
        # Debug: Log if we had to clamp any values
        # if vehicle_location != safe_vehicle_location or target_location != safe_target_location:
        #     print(f"WARNING: Clamped location indices - vehicle: {vehicle_location}->{safe_vehicle_location}, target: {target_location}->{safe_target_location}, max_allowed: {self.num_locations-1}")
        
        path_locations[0, 0] = safe_vehicle_location + 1  # +1 because 0 is padding
        path_locations[0, 1] = safe_target_location + 1
        path_locations[0, 2] = 0  # End token
        
        # Set delays based on action type
        aligned_current_time, aligned_episode_length = self._get_aligned_time_scalar(current_time)
        if action_type.startswith('assign'):
            # Passenger service - delays based on urgency
            path_delays[0, 0, 0] = 0.0  # No delay at current location
            path_delays[0, 1, 0] = max(0.0, (aligned_episode_length - aligned_current_time) / aligned_episode_length)  # Normalized urgency
        elif action_type.startswith('charge'):
            # Charging action - charging time penalty
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.8  # High delay for charging (opportunity cost)
        else:
            # Movement or idle
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.1  # Small delay for movement
        
        # Normalize time (0-1 range)
        time_tensor = torch.tensor([[aligned_current_time / aligned_episode_length]], dtype=torch.float32)
        
        # Normalize other metrics
        others_tensor = torch.tensor([[min(other_vehicles, self.num_vehicles) / self.num_vehicles]], dtype=torch.float32)
        requests_tensor = torch.tensor([[min(num_requests, self.max_requests) / self.max_requests]], dtype=torch.float32)
        

        
        # Move to device
        return (path_locations.to(self.device), 
                path_delays.to(self.device),
                time_tensor.to(self.device),
                others_tensor.to(self.device),
                requests_tensor.to(self.device))
    
    def validate_normalization_params(self):
        """验证归一化参数的合理性"""
        print("=== Normalization Parameters Validation ===")
        print(f"Grid size: {self.grid_size}")
        print(f"Number of vehicles: {self.num_vehicles}")
        print(f"Episode length: {self.episode_length}")
        print(f"Max requests: {self.max_requests}")
        print(f"Number of locations: {self.num_locations}")
        
        # 检查参数合理性
        issues = []
        if self.episode_length <= 0:
            issues.append("Episode length must be positive")
        if self.num_vehicles <= 0:
            issues.append("Number of vehicles must be positive")
        if self.max_requests <= 0:
            issues.append("Max requests must be positive")
            
        if issues:
            print("⚠️ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All normalization parameters are valid")
        print("=" * 45)
    
    def _prepare_network_input_with_battery(self, vehicle_location: int, target_location: int, 
                                           current_time: float, other_vehicles: int, 
                                           num_requests: int, action_type: str, 
                                           battery_level: float = 1.0, request_value: float = 0.0,
                                           target_distance: float = None, target_zoneid: int = None):
        """
        Prepare input tensors for the neural network including battery, request value, and 🆕 target information
        
        Args:
            vehicle_location: 车辆当前位置
            target_location: 目标位置
            current_time: 当前时间
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            action_type: 动作类型
            battery_level: 电池电量 (0-1)
            request_value: 请求价值 (只对assign动作有效)
            target_distance: 🆕 到target的曼哈顿距离（已归一化）
            target_zoneid: 🆕 target location的zone ID
        """
        # 验证和修正输入值 - 处理None或无效值（静默处理）
        if vehicle_location is None:
            vehicle_location = 0  # 默认值
        if target_location is None:
            target_location = vehicle_location if vehicle_location is not None else 0  # 使用当前位置作为默认值
        
        # 确保位置值在有效范围内
        vehicle_location = int(vehicle_location) if isinstance(vehicle_location, (int, float)) else 0
        target_location = int(target_location) if isinstance(target_location, (int, float)) else vehicle_location
        if target_zoneid is not None:
            target_zoneid = max(0, min(int(target_zoneid), self._num_zones))
        
        # 根据动作类型选择合适的输入准备方法
        if action_type == 'idle':
            # 对于idle状态，处理目标位置为当前位置
            path_locations = torch.zeros(1, 3, dtype=torch.long)  # batch_size=1, seq_len=3
            path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
            
            # 设置路径：当前位置 -> 当前位置（表示停留）-> 结束 (with boundary checking)
            # Ensure indices are within valid range [0, num_locations-1]
            safe_vehicle_location = max(0, min(vehicle_location, self.num_locations - 1))
            
            path_locations[0, 0] = safe_vehicle_location + 1  # +1 because 0 is padding
            path_locations[0, 1] = safe_vehicle_location + 1  # 同样的位置表示idle
            path_locations[0, 2] = 0  # End token
            
            # 设置延迟 - idle状态的延迟模式
            path_delays[0, 0, 0] = 0.0  # 当前位置无延迟
            path_delays[0, 1, 0] = 0.05  # idle的小延迟（等待成本）
            path_delays[0, 2, 0] = 0.0  # 结束位置无延迟
            
            # 归一化时间 (0-1 range)
            aligned_current_time, aligned_episode_length = self._get_aligned_time_scalar(current_time)
            time_tensor = torch.tensor([[aligned_current_time / aligned_episode_length]], dtype=torch.float32)
            
            # 归一化其他指标
            others_tensor = torch.tensor([[min(other_vehicles, self.num_vehicles) / self.num_vehicles]], dtype=torch.float32)
            requests_tensor = torch.tensor([[min(num_requests, self.max_requests) / self.max_requests]], dtype=torch.float32)
            

            
            # 归一化电池电量
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32)
            
            # 归一化请求价值 (对idle动作，request_value应该为0)
            value_tensor = torch.tensor([[request_value / 100.0]], dtype=torch.float32)  # 假设最大价值100
            
            # 🆕 Target信息 (对idle动作，target_distance和target_zoneid为0)
            # 归一化distance: 假设最大曼哈顿距离为20（10x10网格的对角线距离约18）
            normalized_distance = (target_distance if target_distance is not None else 0.0) 
            distance_tensor = torch.tensor([[normalized_distance]], dtype=torch.float32)
            zoneid_tensor = torch.tensor([[target_zoneid if target_zoneid is not None else 0]], dtype=torch.long)
            
            # Move to device
            return (path_locations.to(self.device), 
                    path_delays.to(self.device),
                    time_tensor.to(self.device),
                    others_tensor.to(self.device),
                    requests_tensor.to(self.device),
                    battery_tensor.to(self.device),
                    value_tensor.to(self.device),
                    distance_tensor.to(self.device),      # 🆕
                    zoneid_tensor.to(self.device))        # 🆕
        else:
            # 对于非idle动作，使用标准方法并添加battery和request_value信息
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = self._prepare_network_input(
                vehicle_location, target_location, current_time, 
                other_vehicles, num_requests, action_type
            )
            
            # 添加battery信息
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
            
            # 添加request_value信息 (归一化)
            normalized_value = request_value / 100.0 if action_type.startswith('assign') else 0.0
            value_tensor = torch.tensor([[normalized_value]], dtype=torch.float32).to(self.device)
            
            # 🆕 添加target信息 - 归一化distance
            # 归一化distance: 假设最大曼哈顿距离为20（10x10网格的对角线距离约18）
            normalized_distance = (target_distance if target_distance is not None else 0.0) 
            distance_tensor = torch.tensor([[normalized_distance]], dtype=torch.float32).to(self.device)
            zoneid_tensor = torch.tensor([[target_zoneid if target_zoneid is not None else 0]], dtype=torch.long).to(self.device)
            
            return (path_locations, path_delays, time_tensor, 
                   others_tensor, requests_tensor, battery_tensor, value_tensor,
                   distance_tensor, zoneid_tensor)  # 🆕
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_reject: int, target_location: int, 
                              current_time: float = 0.0, other_vehicles: int = 0, 
                              num_requests: int = 0, battery_level: float = 1.0,
                              request_value: float = 0.0, pickup_dist: float = None, 
                              pick_zone: int = None) -> float:
        """
        Enhanced Q-value for vehicle assignment to request using neural network
        现在包含更丰富的上下文信息和优化的计算逻辑，以及EV拒绝学习机制
        """
        # 基础Q值计算
        base_q_value = self.get_q_value(vehicle_id, f"assign_{target_id}", 
                                       vehicle_location, target_location, current_time, 
                                       other_vehicles, num_requests, battery_level, request_value)
        
        # 为EV车辆添加距离惩罚和拒绝风险评估
        # if hasattr(self, 'env') and self.env is not None:
        #     vehicle = self.env.vehicles.get(vehicle_id)
        #     if vehicle and vehicle.get('type') == 1:  # EV车辆
        #         # 计算到接客点的距离（假设target_id对应pickup位置）
        #         grid_size = self.env.grid_size if hasattr(self.env, 'grid_size') else int(math.sqrt(max(vehicle_location, target_reject)) + 1)
        #         distance = self._calculate_manhattan_distance(vehicle_location, target_reject, grid_size)
        #         print(f"Vehicle {vehicle_id} (EV) distance to request {target_id}: {distance}")
        #         # 距离惩罚：距离越远，Q值越低
        #         distance_penalty = distance * 0.15  # 可调节的距离惩罚因子
                
        #         # 拒绝风险惩罚：基于历史经验学习的拒绝概率
        #         rejection_penalty = self._calculate_rejection_risk_penalty(vehicle_id, distance)
                
        #         base_q_value = base_q_value - distance_penalty - rejection_penalty
                
        
        # 返回调整后的Q值
        return base_q_value 
    
    def batch_get_assignment_q_value(self, batch_inputs, multi_gpu_devices=None):
        """
        批量计算多个vehicle-request对的Q值，支持多GPU并行
        
        Args:
            batch_inputs: List of input dictionaries, each containing:
                - vehicle_id, target_id, vehicle_location, target_reject, target_location,
                - current_time, other_vehicles, num_requests, battery_level, request_value
            multi_gpu_devices: List of GPU device strings (e.g., ['cuda:0', 'cuda:1'])
                
        Returns:
            List of Q-values corresponding to each input
        """
        if not batch_inputs:
            return []
        
        # 🚀 多GPU并行处理
        if self.acceptance_input_enabled:
            return self._acceptance_batch_assignment(batch_inputs)
        if multi_gpu_devices and len(multi_gpu_devices) > 1 and all('cuda' in d for d in multi_gpu_devices):
            return self._multi_gpu_batch_process(batch_inputs, multi_gpu_devices)
        
        # 单GPU/CPU处理
        return self._single_device_batch_process(batch_inputs)
    
    def _acceptance_batch_assignment(self, rows, action_id=2):
        self.network.eval()
        return self.batch_get_mixed_q_values(
            vehicle_ids=[r['vehicle_id'] for r in rows], request_ids=[r.get('target_id', -1) for r in rows],
            vehicle_locations=[r['vehicle_location'] for r in rows], target_locations=[r['target_location'] for r in rows],
            current_times=[r.get('current_time', 0.0) for r in rows],
            other_vehicles=[r.get('other_vehicles', 0) for r in rows], num_requests=[r.get('num_requests', 0) for r in rows],
            battery_levels=[r.get('battery_level', 1.0) for r in rows], request_values=[r.get('request_value', 0.0) for r in rows],
            target_distances=[r.get('pickup_dist') or self.env._manhattan_distance_loc(
                r['vehicle_location'], r['target_location']) for r in rows],
            target_zoneids=[r.get('pick_zone') or _env_zone_index(self.env, r['target_location']) for r in rows],
            vehicle_idle_times=[r.get('vehicle_idle_time', self.env.vehicles[r['vehicle_id']].get('idle_timer', 0.0)) for r in rows],
            action_type_ids=[action_id] * len(rows),
            post_action_distances=[r.get('post_action_distance') or 0.0 for r in rows],
            post_action_durations=[r.get('post_action_duration') or 0.0 for r in rows],
            post_action_zoneids=[r.get('post_action_zoneid') or 0 for r in rows],
        )

    def extra_checkpoint_state(self):
        return {"ev_response": self.acceptance_checkpoint_state()}

    def load_extra_checkpoint_state(self, state):
        self.load_acceptance_checkpoint_state(state)

    def _multi_gpu_batch_process(self, batch_inputs, gpu_devices):
        """
        真正的多GPU并行处理 - 将数据分割到不同GPU上独立计算，增加总显存容量
        
        策略：为每个GPU创建独立的模型副本，并行处理各自的数据块
        """
        import copy
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_size = len(batch_inputs)
        num_gpus = len([d for d in gpu_devices if 'cuda' in d])
        
        if num_gpus <= 1:
            return self._single_device_batch_process(batch_inputs)
        
        print(f"🚀 多GPU并行模式: {num_gpus} GPUs, {total_size} 个任务")
        
        # 显示每个GPU的显存
        for i, device in enumerate(gpu_devices):
            if 'cuda' in device:
                gpu_id = int(device.split(':')[1])
                props = torch.cuda.get_device_properties(gpu_id)
                free_mem = torch.cuda.memory_reserved(gpu_id) - torch.cuda.memory_allocated(gpu_id)
                total_mem = props.total_memory / (1024**3)
                print(f"  GPU {gpu_id}: {props.name}, 总显存 {total_mem:.1f}GB")
        
        # 将数据分割到各个GPU
        chunk_size = (total_size + num_gpus - 1) // num_gpus
        gpu_chunks = []
        for i, device in enumerate(gpu_devices):
            if 'cuda' not in device:
                continue
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_size)
            if start_idx < total_size:
                gpu_chunks.append({
                    'device': device,
                    'gpu_id': int(device.split(':')[1]),
                    'data': batch_inputs[start_idx:end_idx],
                    'start_idx': start_idx
                })
        
        # 保存原始状态
        original_device = self.device
        original_state_dict = copy.deepcopy(self.network.state_dict())
        
        # 为每个GPU创建独立的模型副本
        gpu_models = {}
        for chunk_info in gpu_chunks:
            device = torch.device(chunk_info['device'])
            gpu_id = chunk_info['gpu_id']
            
            # 创建模型副本并移动到对应GPU
            model_copy = copy.deepcopy(self.network)
            model_copy.load_state_dict(original_state_dict)
            model_copy = model_copy.to(device)
            model_copy.eval()
            gpu_models[gpu_id] = {
                'model': model_copy,
                'device': device
            }
            print(f"  ✓ GPU {gpu_id} 模型副本已创建")
        
        # 定义在单个GPU上处理数据的函数
        def process_on_gpu(chunk_info, model_info):
            gpu_id = chunk_info['gpu_id']
            device = model_info['device']
            model = model_info['model']
            data = chunk_info['data']
            
            try:
                results = []
                # 在这个GPU上分批处理
                local_batch_size = 128  # 每个GPU的本地批大小
                
                for i in range(0, len(data), local_batch_size):
                    mini_batch = data[i:i+local_batch_size]
                    
                    # 准备张量并放到对应GPU
                    batch_tensors = self._prepare_batch_tensors_for_device(mini_batch, device, model)
                    
                    if batch_tensors is None:
                        # 如果准备失败，使用简单fallback
                        for inp in mini_batch:
                            results.append(0.0)
                        continue
                    
                    # 在对应GPU上计算
                    with torch.no_grad():
                        with torch.cuda.device(device):
                            q_values = model(
                                path_locations=batch_tensors['path_locations'],
                                path_delays=batch_tensors['path_delays'],
                                current_time=batch_tensors['current_time'],
                                other_agents=batch_tensors['other_agents'],
                                num_requests=batch_tensors['num_requests'],
                                battery_level=batch_tensors['battery_level'],
                                request_value=batch_tensors['request_value'],
                                target_distance=batch_tensors['target_distance'],
                                target_zoneid=batch_tensors['target_zoneid'],
                                action_type=batch_tensors['action_type'],
                                vehicle_idle_time=batch_tensors['vehicle_idle_time'],
                                vehicle_type=None,
                                prior_features=batch_tensors.get('prior_features'),
                                prior_mask=batch_tensors.get('prior_mask'),
                                time_zone_dist=batch_tensors.get('time_zone_dist')
                            )
                    
                    results.extend(q_values.cpu().numpy().flatten().tolist())
                    
                    # 清理GPU缓存
                    del batch_tensors
                    if i % (local_batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                
                return gpu_id, results, None
                
            except Exception as e:
                return gpu_id, None, str(e)
        
        # 并行在多个GPU上处理
        all_results = [None] * total_size
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for chunk_info in gpu_chunks:
                gpu_id = chunk_info['gpu_id']
                model_info = gpu_models[gpu_id]
                future = executor.submit(process_on_gpu, chunk_info, model_info)
                futures[future] = chunk_info
            
            for future in as_completed(futures):
                chunk_info = futures[future]
                gpu_id, results, error = future.result()
                
                if error:
                    print(f"  ❌ GPU {gpu_id} 处理失败: {error}")
                    # Fallback: 在主GPU上处理这部分
                    try:
                        self.network.to(original_device)
                        self.device = original_device
                        results = self._single_device_batch_process(chunk_info['data'])
                    except:
                        results = [0.0] * len(chunk_info['data'])
                
                # 将结果放回正确位置
                start_idx = chunk_info['start_idx']
                for i, r in enumerate(results):
                    all_results[start_idx + i] = r
                
                print(f"  ✅ GPU {gpu_id} 完成 {len(results)} 个任务")
        
        # 清理GPU模型副本
        for gpu_id, model_info in gpu_models.items():
            del model_info['model']
        del gpu_models
        torch.cuda.empty_cache()
        
        # 恢复原始状态
        self.network.to(original_device)
        self.device = original_device
        
        print(f"  🎯 多GPU并行完成: {sum(1 for r in all_results if r is not None)} 个结果")
        
        return all_results
    
    def _prepare_batch_tensors_for_device(self, batch_inputs, device, model):
        """为指定设备准备批量张量"""
        try:
            batch_size = len(batch_inputs)
            
            # 收集数据
            path_locs_list = []
            path_delays_list = []
            current_times_list = []
            other_agents_list = []
            num_requests_list = []
            battery_levels_list = []
            request_values_list = []
            target_distances_list = []
            target_zoneids_list = []
            vehicle_idle_times_list = []
            
            for inp in batch_inputs:
                # 准备网络输入 - 返回的是元组，不是字典
                # 返回格式: (path_locations, path_delays, time_tensor, others_tensor, 
                #            requests_tensor, battery_tensor, value_tensor, 
                #            distance_tensor, zoneid_tensor)
                network_input = self._prepare_network_input_with_battery(
                    inp['vehicle_location'], 
                    inp['target_location'],
                    inp.get('current_time', 0.0),
                    inp.get('other_vehicles', 0),
                    inp.get('num_requests', 0),
                    "assign",
                    inp.get('battery_level', 1.0),
                    inp.get('request_value', 0.0),
                    inp.get('pickup_dist', 0.0),
                    inp.get('pick_zone', 0)
                )
                
                # 解包元组 (9个元素)
                (path_locations, path_delays, time_tensor, others_tensor, 
                 requests_tensor, battery_tensor, value_tensor, 
                 distance_tensor, zoneid_tensor) = network_input
                
                path_locs_list.append(path_locations.squeeze(0))
                path_delays_list.append(path_delays.squeeze(0))
                current_times_list.append(time_tensor.squeeze(0))
                other_agents_list.append(others_tensor.squeeze(0))
                num_requests_list.append(requests_tensor.squeeze(0))
                battery_levels_list.append(battery_tensor.squeeze(0))
                request_values_list.append(value_tensor.squeeze(0))
                target_distances_list.append(distance_tensor.squeeze(0))
                target_zoneids_list.append(zoneid_tensor.squeeze(0))
                vehicle_idle_times_list.append(torch.tensor([inp.get('vehicle_idle_time', 0.0)]))
            
            # 堆叠为批量张量并放到指定设备
            result = {
                'path_locations': torch.stack(path_locs_list).to(device),
                'path_delays': torch.stack(path_delays_list).to(device),
                'current_time': torch.stack(current_times_list).to(device),
                'other_agents': torch.stack(other_agents_list).to(device),
                'num_requests': torch.stack(num_requests_list).to(device),
                'battery_level': torch.stack(battery_levels_list).to(device),
                'request_value': torch.stack(request_values_list).to(device),
                'target_distance': torch.stack(target_distances_list).to(device),
                'target_zoneid': torch.stack(target_zoneids_list).to(device),
                'action_type': torch.full((batch_size, 1), 2, dtype=torch.long).to(device),
                'vehicle_idle_time': torch.stack(vehicle_idle_times_list).to(device),
            }
            # Add prior features for cross-attention if available
            prior_tensor = self._get_prior_features_tensor(batch_size)
            if prior_tensor is not None:
                result['prior_features'] = prior_tensor.to(device)
            # Add time-dependent zone distribution (use real hour_of_day)
            # inp['current_time'] is raw sim step; convert to hour_of_day / 24.0
            hour_norms = []
            for inp in batch_inputs:
                ct = inp.get('current_time', 0.0)
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_hour_of_day'):
                    h = self.env.get_hour_of_day(ct) / 24.0
                else:
                    h = ct / self.episode_length if self.episode_length > 0 else 0.0
                hour_norms.append([h])
            hour_norm_tensor = torch.tensor(hour_norms, dtype=torch.float32).to(device)
            time_zone_dist = self._get_combined_zone_dist_tensor(hour_norm_tensor, prior_tensor)
            if time_zone_dist is not None:
                result['time_zone_dist'] = time_zone_dist.to(device)
            return result
        except Exception as e:
            print(f"  ⚠️ 张量准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cpu_fallback_batch_process(self, batch_inputs):
        """CPU批处理fallback"""
        print(f"  🔄 切换到CPU处理 {len(batch_inputs)} 个任务...")
        
        # 临时切换到CPU
        original_device = self.device
        self.device = torch.device('cpu')
        self.network.to(self.device)
        
        try:
            results = []
            # 使用较小的批次避免内存问题
            mini_batch_size = 16
            for i in range(0, len(batch_inputs), mini_batch_size):
                mini_batch = batch_inputs[i:i+mini_batch_size]
                
                # 逐个处理以确保稳定性
                for input_data in mini_batch:
                    clean_input = {
                        'vehicle_id': input_data['vehicle_id'],
                        'target_id': input_data['target_id'],
                        'vehicle_location': input_data['vehicle_location'],
                        'target_location': input_data['target_location'],
                        'target_reject': input_data.get('target_reject', 0),
                        'current_time': input_data.get('current_time', 0.0),
                        'other_vehicles': input_data.get('other_vehicles', 0),
                        'num_requests': input_data.get('num_requests', 0),
                        'battery_level': input_data.get('battery_level', 1.0),
                        'request_value': input_data.get('request_value', 0.0),
                        'pickup_dist': input_data.get('pickup_dist', None),
                        'pick_zone': input_data.get('pick_zone', None)
                    }
                    q_val = self.get_assignment_q_value(**clean_input)
                    results.append(q_val)
            
            return results
            
        finally:
            # 恢复原设备
            self.device = original_device
            self.network.to(self.device)
    
    def _single_device_batch_process(self, batch_inputs):
        """单设备批处理（原有逻辑）"""
        # 准备批量数据
        batch_size = len(batch_inputs)
        
        # 收集所有输入数据
        vehicle_ids = []
        target_ids = []
        vehicle_locations = []
        target_locations = []
        current_times = []
        other_vehicles_list = []
        num_requests_list = []
        battery_levels = []
        request_values = []
        pickupdistances = []
        pickzoneids = []
        vehicle_idle_times = []
        for input_data in batch_inputs:
            vehicle_ids.append(input_data['vehicle_id'])
            target_ids.append(input_data['target_id'])
            vehicle_locations.append(input_data['vehicle_location'])
            target_locations.append(input_data['target_location'])
            current_times.append(input_data.get('current_time', 0.0))
            other_vehicles_list.append(input_data.get('other_vehicles', 0))
            num_requests_list.append(input_data.get('num_requests', 0))
            battery_levels.append(input_data.get('battery_level', 1.0))
            request_values.append(input_data.get('request_value', 0.0))
            pickupdistances.append(input_data.get('pickup_dist', 0.0))
            pickzoneids.append(input_data.get('pick_zone', 0))
            vehicle_idle_times.append(input_data.get('vehicle_idle_time', 0.0))
        # 批量准备神经网络输入
        try:
            batch_network_inputs = []
            batch_vehicle_idle_times = []  # 收集idle_time用于批处理
            for i in range(batch_size):
                action_type = "assign"  # assign动作
                # 🆕 计算target distance（如果未提供）
                target_distance = pickupdistances[i]
                if target_distance == 0.0:
                    if hasattr(self, 'env') and self.env is not None and hasattr(self.env, '_manhattan_distance_loc'):
                        manhattan_dist = self.env._manhattan_distance_loc(vehicle_locations[i], target_locations[i])
                    elif hasattr(self, 'grid_size'):
                        # Fallback: 手动计算
                        grid_size = self.grid_size
                        v_x = vehicle_locations[i] % grid_size
                        v_y = vehicle_locations[i] // grid_size
                        t_x = target_locations[i] % grid_size
                        t_y = target_locations[i] // grid_size
                        manhattan_dist = abs(v_x - t_x) + abs(v_y - t_y)
                    else:
                        manhattan_dist = 0
                    target_distance = manhattan_dist
                
                # 🆕 获取target zone_id（如果未提供）
                target_zoneid = pickzoneids[i]
                if target_zoneid == 0 and hasattr(self, 'env') and self.env is not None:
                    if hasattr(self.env, 'get_zone_id'):
                        target_zoneid = _env_zone_index(self.env, target_locations[i])
                
                network_input = self._prepare_network_input_with_battery(
                    vehicle_locations[i], target_locations[i], current_times[i],
                    other_vehicles_list[i], num_requests_list[i], action_type,
                    battery_levels[i], request_values[i],
                    target_distance, target_zoneid  # 🆕 传入target信息
                )
                batch_network_inputs.append(network_input)
                batch_vehicle_idle_times.append(vehicle_idle_times[i])
            
            # 批量转换为张量
            batch_tensors = self._batch_prepare_tensors(batch_network_inputs, batch_vehicle_idle_times)
            dropout_state_tensor = self._batch_dropout_state_tensor(batch_inputs)
            
            # 使用神经网络进行批量前向传播
            prior_tensor, prior_mask = self._get_prior_context_tensors(batch_size)
            hour_norm_batch = self._build_hour_norm_batch(current_times)
            time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_batch, prior_tensor, prior_mask)
            with torch.no_grad():
                batch_q_values = self.network(
                    path_locations=batch_tensors['path_locations'],
                    path_delays=batch_tensors['path_delays'],
                    current_time=batch_tensors['current_time'],
                    other_agents=batch_tensors['other_agents'],
                    num_requests=batch_tensors['num_requests'],
                    battery_level=batch_tensors['battery_level'],
                    request_value=batch_tensors['request_value'],
                    target_distance=batch_tensors['target_distance'],
                    target_zoneid=batch_tensors['target_zoneid'],
                    action_type=torch.full((batch_size, 1), 2, dtype=torch.long).to(self.device),  # 2=assign
                    vehicle_idle_time=batch_tensors['vehicle_idle_time'],
                    vehicle_type=None,
                    dropout_state_features=dropout_state_tensor,
                    prior_features=prior_tensor,
                    prior_mask=prior_mask,
                    time_zone_dist=time_zone_dist_tensor
                )

            q_values = batch_q_values.cpu().numpy().flatten().tolist()
            
            return q_values
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️  CUDA out of memory in batch processing, falling back to CPU...")
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 使用CPU分批处理，减小批次大小
                results = []
                batch_size = 16  # 减小批次大小以节省内存
                for i in range(0, len(batch_inputs), batch_size):
                    mini_batch = batch_inputs[i:i+batch_size]
                    for input_data in mini_batch:
                        # 传递所有必需的参数
                        clean_input = {
                            'vehicle_id': input_data['vehicle_id'],
                            'target_id': input_data['target_id'],
                            'vehicle_location': input_data['vehicle_location'],
                            'target_location': input_data['target_location'],
                            'target_reject': input_data.get('target_reject', 0),
                            'current_time': input_data.get('current_time', 0.0),
                            'other_vehicles': input_data.get('other_vehicles', 0),
                            'num_requests': input_data.get('num_requests', 0),
                            'battery_level': input_data.get('battery_level', 1.0),
                            'request_value': input_data.get('request_value', 0.0),
                            'pickup_dist': input_data.get('pickup_dist', None),
                            'pick_zone': input_data.get('pick_zone', None)
                        }
                        q_val = self.get_assignment_q_value(**clean_input)
                        results.append(q_val)
                return results
            else:
                raise
        except Exception as e:
            print(f"❌ Batch Q-value calculation failed: {e}")
            # 回退到单独计算，传递所有必需参数
            results = []
            for input_data in batch_inputs:
                clean_input = {
                    'vehicle_id': input_data['vehicle_id'],
                    'target_id': input_data['target_id'],
                    'vehicle_location': input_data['vehicle_location'],
                    'target_location': input_data['target_location'],
                    'target_reject': input_data.get('target_reject', 0),
                    'current_time': input_data.get('current_time', 0.0),
                    'other_vehicles': input_data.get('other_vehicles', 0),
                    'num_requests': input_data.get('num_requests', 0),
                    'battery_level': input_data.get('battery_level', 1.0),
                    'request_value': input_data.get('request_value', 0.0),
                    'pickup_dist': input_data.get('pickup_dist', None),
                    'pick_zone': input_data.get('pick_zone', None)
                }
                results.append(self.get_assignment_q_value(**clean_input))
            return results
    
    def _batch_prepare_tensors(self, batch_network_inputs, batch_vehicle_idle_times=None):
        """
        将批量网络输入转换为适合批量处理的张量
        
        Args:
            batch_network_inputs: 批量网络输入列表
            batch_vehicle_idle_times: 批量车辆idle时间列表（可选）
        """
        batch_size = len(batch_network_inputs)
        
        # 初始化批量张量
        batch_tensors = {}
        
        # 获取第一个输入的维度信息
        first_input = batch_network_inputs[0]
        if len(first_input) >= 9:  # 🆕 包含battery, request_value, target_distance, target_zoneid
            path_locations_list = []
            path_delays_list = []
            current_time_list = []
            other_agents_list = []
            num_requests_list = []
            battery_level_list = []
            request_value_list = []
            target_distance_list = []  # 🆕
            target_zoneid_list = []    # 🆕
            
            for network_input in batch_network_inputs:
                path_locations, path_delays, current_time, other_agents, num_requests, battery_level, request_value, target_distance, target_zoneid = network_input
                path_locations_list.append(path_locations.squeeze(0))
                path_delays_list.append(path_delays.squeeze(0))
                current_time_list.append(current_time.squeeze(0))
                other_agents_list.append(other_agents.squeeze(0))
                num_requests_list.append(num_requests.squeeze(0))
                battery_level_list.append(battery_level.squeeze(0))
                request_value_list.append(request_value.squeeze(0))
                target_distance_list.append(target_distance.squeeze(0))  # 🆕
                target_zoneid_list.append(target_zoneid.squeeze(0))      # 🆕
            
            # 堆叠为批量张量
            batch_tensors['path_locations'] = torch.stack(path_locations_list)
            batch_tensors['path_delays'] = torch.stack(path_delays_list)
            batch_tensors['current_time'] = torch.stack(current_time_list)
            batch_tensors['other_agents'] = torch.stack(other_agents_list)
            batch_tensors['num_requests'] = torch.stack(num_requests_list)
            batch_tensors['battery_level'] = torch.stack(battery_level_list)
            batch_tensors['request_value'] = torch.stack(request_value_list)
            batch_tensors['target_distance'] = torch.stack(target_distance_list)  # 🆕
            batch_tensors['target_zoneid'] = torch.stack(target_zoneid_list)      # 🆕
            
            # 添加vehicle_idle_time
            if batch_vehicle_idle_times is not None:
                batch_tensors['vehicle_idle_time'] = torch.tensor(
                    batch_vehicle_idle_times, dtype=torch.float32
                ).unsqueeze(1).to(self.device)
            else:
                batch_tensors['vehicle_idle_time'] = torch.zeros(batch_size, 1).to(self.device)
        elif len(first_input) >= 7:  # 向后兼容：只有battery和request_value
            path_locations_list = []
            path_delays_list = []
            current_time_list = []
            other_agents_list = []
            num_requests_list = []
            battery_level_list = []
            request_value_list = []
            
            for network_input in batch_network_inputs:
                path_locations, path_delays, current_time, other_agents, num_requests, battery_level, request_value = network_input
                path_locations_list.append(path_locations.squeeze(0))
                path_delays_list.append(path_delays.squeeze(0))
                current_time_list.append(current_time.squeeze(0))
                other_agents_list.append(other_agents.squeeze(0))
                num_requests_list.append(num_requests.squeeze(0))
                battery_level_list.append(battery_level.squeeze(0))
                request_value_list.append(request_value.squeeze(0))
            
            # 堆叠为批量张量
            batch_tensors['path_locations'] = torch.stack(path_locations_list)
            batch_tensors['path_delays'] = torch.stack(path_delays_list)
            batch_tensors['current_time'] = torch.stack(current_time_list)
            batch_tensors['other_agents'] = torch.stack(other_agents_list)
            batch_tensors['num_requests'] = torch.stack(num_requests_list)
            batch_tensors['battery_level'] = torch.stack(battery_level_list)
            batch_tensors['request_value'] = torch.stack(request_value_list)
            # 创建默认的target信息
            batch_tensors['target_distance'] = torch.zeros(batch_size, 1).to(self.device)
            batch_tensors['target_zoneid'] = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
            # 添加vehicle_idle_time
            if batch_vehicle_idle_times is not None:
                batch_tensors['vehicle_idle_time'] = torch.tensor(
                    batch_vehicle_idle_times, dtype=torch.float32
                ).unsqueeze(1).to(self.device)
            else:
                batch_tensors['vehicle_idle_time'] = torch.zeros(batch_size, 1).to(self.device)
        else:
            # 向后兼容处理
            raise ValueError("Insufficient input dimensions for batch processing")
        
        return batch_tensors
        
    def _calculate_context_adjustment(self, vehicle_id: int, vehicle_location: int, 
                                    target_location: int, battery_level: float,
                                    request_value: float, other_vehicles: int, 
                                    num_requests: int, current_time: float) -> float:
        """
        计算基于上下文的Q值调整因子
        考虑车辆类型、电池状态、竞争环境、请求价值等因素
        """
        adjustment = 0.0
        
        # 1. 电池电量对分配的影响
        if battery_level < 0.3:  # 低电量时
            # 计算到充电站的距离影响
            grid_size = int(math.sqrt(max(vehicle_location, target_location)) + 1)
            distance_to_target = self._calculate_manhattan_distance(vehicle_location, target_location, grid_size)
            # 距离越远，电量越低，Q值调整越负
            battery_penalty = -0.2 * (0.3 - battery_level) * (distance_to_target / 10.0)
            adjustment += battery_penalty
            
        # 2. 请求价值对分配的影响
        if request_value > 0:
            # 高价值请求获得奖励
            value_bonus = min(0.1 * (request_value / 50.0), 0.5)  # 最大奖励0.5
            adjustment += value_bonus
            
        # 3. 竞争环境的影响
        if other_vehicles > 0 and num_requests > 0:
            competition_ratio = other_vehicles / max(num_requests, 1)
            if competition_ratio > 1.0:  # 车辆多于请求
                # 竞争激烈时，距离近的分配获得更多奖励
                grid_size = int(math.sqrt(max(vehicle_location, target_location)) + 1)
                distance = self._calculate_manhattan_distance(vehicle_location, target_location, grid_size)
                distance_bonus = max(0, 0.2 - 0.02 * distance)  # 距离越近奖励越高
                adjustment += distance_bonus
                
        # 4. 时间因素的影响（紧急请求）
        # 假设current_time可以反映请求的紧急程度
        if current_time > 0:
            time_factor = min(current_time / 100.0, 1.0)  # 时间标准化
            urgency_bonus = 0.1 * time_factor  # 时间越长越紧急
            adjustment += urgency_bonus
            
        # 5. 车辆类型的影响
        vehicle_type_id = self._vehicle_type_id(vehicle_id)
        if vehicle_type_id == 2:  # AEV类型车辆
            # AEV在某些情况下可能有优势
            aev_bonus = 0.05 if battery_level > 0.7 else -0.05
            adjustment += aev_bonus
            
        return adjustment
    def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, target_location: int,
                        battery_level: float, current_time: float = 0.0, 
                        other_vehicles: int = 0, num_requests: int = 0) -> float:
        """
        Get Q-value for idle action with random movement
        Idle action involves moving to a random nearby location, not staying in place
        """
        import random
        
        # Convert location index to coordinates for random target generation
        current_x = vehicle_location % self.grid_size
        current_y = vehicle_location // self.grid_size
        

        
        # Use the generated random target location for Q-value calculation
        return self.get_q_value(vehicle_id, "idle", vehicle_location, target_location, 
                               current_time, other_vehicles, num_requests, battery_level)

    def get_waiting_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = 0.0, 
                        other_vehicles: int = 0, num_requests: int = 0) -> float:
        """
        Get Q-value for waiting action (staying in place)
        Unlike idle action, waiting means the vehicle stays at the current location
        """
        # For waiting action, target location equals current location (no movement)
        return self.get_q_value(
            vehicle_id=vehicle_id,
            action_type="idle",
            vehicle_location=vehicle_location,
            target_location=vehicle_location,
            current_time=current_time,
            other_vehicles=other_vehicles,
            num_requests=num_requests,
            battery_level=battery_level,
        )


    def batch_get_charging_q_value(self, batch_inputs):
        """
        批量计算充电动作的Q值
        
        Args:
            batch_inputs: List of input dictionaries, each containing:
                - vehicle_id, station_id, vehicle_location, station_location,
                - current_time, other_vehicles, num_requests, battery_level
                
        Returns:
            List of Q-values
        """
        if not batch_inputs:
            return []
        
        batch_size = len(batch_inputs)
        
        try:
            batch_network_inputs = []
            batch_vehicle_idle_times = []
            
            for input_data in batch_inputs:
                vehicle_location = input_data['vehicle_location']
                station_location = input_data['station_location']
                current_time = input_data.get('current_time', 0.0)
                other_vehicles = input_data.get('other_vehicles', 0)
                num_requests = input_data.get('num_requests', 0)
                battery_level = input_data.get('battery_level', 1.0)
                vehicle_idle_time = input_data.get('vehicle_idle_time', 0.0)
                
                # 计算距离
                target_distance = 0.0
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, '_manhattan_distance_loc'):
                    target_distance = self.env._manhattan_distance_loc(vehicle_location, station_location)
                
                # 获取zone_id
                target_zoneid = 0
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_zone_id'):
                    target_zoneid = _env_zone_index(self.env, station_location)
                
                network_input = self._prepare_network_input_with_battery(
                    vehicle_location, station_location, current_time,
                    other_vehicles, num_requests, "charge",
                    battery_level, 0.0,  # request_value=0 for charging
                    target_distance, target_zoneid
                )
                batch_network_inputs.append(network_input)
                batch_vehicle_idle_times.append(vehicle_idle_time)
            
            # 批量转换为张量
            batch_tensors = self._batch_prepare_tensors(batch_network_inputs, batch_vehicle_idle_times)
            
            # 批量前向传播
            prior_tensor, prior_mask = self._get_prior_context_tensors(batch_size)
            raw_cts_charge = [inp.get('current_time', 0.0) for inp in batch_inputs]
            hour_norm_batch = self._build_hour_norm_batch(raw_cts_charge)
            time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_batch, prior_tensor, prior_mask)
            with torch.no_grad():
                batch_q_values = self.network(
                    path_locations=batch_tensors['path_locations'],
                    path_delays=batch_tensors['path_delays'],
                    current_time=batch_tensors['current_time'],
                    other_agents=batch_tensors['other_agents'],
                    num_requests=batch_tensors['num_requests'],
                    battery_level=batch_tensors['battery_level'],
                    request_value=batch_tensors['request_value'],
                    target_distance=batch_tensors['target_distance'],
                    target_zoneid=batch_tensors['target_zoneid'],
                    action_type=torch.full((batch_size, 1), 0, dtype=torch.long).to(self.device),  # 0=charge
                    vehicle_idle_time=batch_tensors['vehicle_idle_time'],
                    vehicle_type=None,
                    prior_features=prior_tensor,
                    prior_mask=prior_mask,
                    time_zone_dist=time_zone_dist_tensor
                )
            
            return batch_q_values.cpu().numpy().flatten().tolist()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️  CUDA out of memory, clearing cache and using CPU fallback...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"Batch charging Q-value calculation failed: {e}")
            return [self.get_charging_q_value(
                input_data['vehicle_id'], input_data['station_id'],
                input_data['vehicle_location'], input_data['station_location'],
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0), input_data.get('battery_level', 1.0)
            ) for input_data in batch_inputs]
        except Exception as e:
            print(f"❌ Batch charging Q-value calculation failed: {e}")
            return [self.get_charging_q_value(
                input_data['vehicle_id'], input_data['station_id'],
                input_data['vehicle_location'], input_data['station_location'],
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0), input_data.get('battery_level', 1.0)
            ) for input_data in batch_inputs]


    def batch_get_idle_q_value(self, batch_inputs):
        """
        批量计算重定位动作的Q值
        
        Args:
            batch_inputs: List of input dictionaries, each containing:
                - vehicle_id, vehicle_location, target_location,
                - current_time, other_vehicles, num_requests, battery_level
                
        Returns:
            List of Q-values
        """
        if not batch_inputs:
            return []
        
        batch_size = len(batch_inputs)
        
        try:
            batch_network_inputs = []
            batch_vehicle_idle_times = []
            
            for input_data in batch_inputs:
                vehicle_location = input_data['vehicle_location']
                target_location = input_data['target_location']
                current_time = input_data.get('current_time', 0.0)
                other_vehicles = input_data.get('other_vehicles', 0)
                num_requests = input_data.get('num_requests', 0)
                battery_level = input_data.get('battery_level', 1.0)
                vehicle_idle_time = input_data.get('vehicle_idle_time', 0.0)
                
                # 计算距离
                target_distance = 0.0
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, '_manhattan_distance_loc'):
                    target_distance = self.env._manhattan_distance_loc(vehicle_location, target_location)
                
                # 获取zone_id
                target_zoneid = 0
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_zone_id'):
                    target_zoneid = _env_zone_index(self.env, target_location)
                
                network_input = self._prepare_network_input_with_battery(
                    vehicle_location, target_location, current_time,
                    other_vehicles, num_requests, "idle",
                    battery_level, 0.0,  # request_value=0 for idle
                    target_distance, target_zoneid
                )
                batch_network_inputs.append(network_input)
                batch_vehicle_idle_times.append(vehicle_idle_time)
            
            # 批量转换为张量
            batch_tensors = self._batch_prepare_tensors(batch_network_inputs, batch_vehicle_idle_times)
            
            # 批量前向传播
            prior_tensor, prior_mask = self._get_prior_context_tensors(batch_size)
            raw_cts_idle = [inp.get('current_time', 0.0) for inp in batch_inputs]
            hour_norm_batch = self._build_hour_norm_batch(raw_cts_idle)
            time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_batch, prior_tensor, prior_mask)
            with torch.no_grad():
                batch_q_values = self.network(
                    path_locations=batch_tensors['path_locations'],
                    path_delays=batch_tensors['path_delays'],
                    current_time=batch_tensors['current_time'],
                    other_agents=batch_tensors['other_agents'],
                    num_requests=batch_tensors['num_requests'],
                    battery_level=batch_tensors['battery_level'],
                    request_value=batch_tensors['request_value'],
                    target_distance=batch_tensors['target_distance'],
                    target_zoneid=batch_tensors['target_zoneid'],
                    action_type=torch.full((batch_size, 1), 1, dtype=torch.long).to(self.device),  # 1=idle
                    vehicle_idle_time=batch_tensors['vehicle_idle_time'],
                    vehicle_type=None,
                    prior_features=prior_tensor,
                    prior_mask=prior_mask,
                    time_zone_dist=time_zone_dist_tensor
                )
            
            return batch_q_values.cpu().numpy().flatten().tolist()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️  CUDA out of memory, clearing cache and using CPU fallback...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"Batch idle Q-value calculation failed: {e}")
            return [self.get_idle_q_value(
                input_data['vehicle_id'],
                input_data['vehicle_location'], input_data['target_location'],
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0), input_data.get('battery_level', 1.0)
            ) for input_data in batch_inputs]
        except Exception as e:
            print(f"❌ Batch idle Q-value calculation failed: {e}")
            return [self.get_idle_q_value(
                input_data['vehicle_id'],
                input_data['vehicle_location'], input_data['target_location'],
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0), input_data.get('battery_level', 1.0)
            ) for input_data in batch_inputs]


    def batch_get_waiting_q_value(self, batch_inputs):
        """
        批量计算等待动作的Q值
        
        Args:
            batch_inputs: List of input dictionaries, each containing:
                - vehicle_id, vehicle_location,
                - current_time, other_vehicles, num_requests, battery_level
                
        Returns:
            List of Q-values
        """
        if not batch_inputs:
            return []
        
        batch_size = len(batch_inputs)
        
        try:
            batch_network_inputs = []
            batch_vehicle_idle_times = []
            
            for input_data in batch_inputs:
                vehicle_location = input_data['vehicle_location']
                current_time = input_data.get('current_time', 0.0)
                other_vehicles = input_data.get('other_vehicles', 0)
                num_requests = input_data.get('num_requests', 0)
                battery_level = input_data.get('battery_level', 1.0)
                vehicle_idle_time = input_data.get('vehicle_idle_time', 0.0)
                
                # 等待动作：target_location = vehicle_location
                target_distance = 0.0
                target_zoneid = 0
                if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'get_zone_id'):
                    target_zoneid = _env_zone_index(self.env, vehicle_location)
                
                network_input = self._prepare_network_input_with_battery(
                    vehicle_location, vehicle_location, current_time,
                    other_vehicles, num_requests, "wait",
                    battery_level, 0.0,  # request_value=0 for waiting
                    target_distance, target_zoneid
                )
                batch_network_inputs.append(network_input)
                batch_vehicle_idle_times.append(vehicle_idle_time)
            
            # 批量转换为张量
            batch_tensors = self._batch_prepare_tensors(batch_network_inputs, batch_vehicle_idle_times)
            
            # 批量前向传播
            prior_tensor, prior_mask = self._get_prior_context_tensors(batch_size)
            raw_cts_wait = [inp.get('current_time', 0.0) for inp in batch_inputs]
            hour_norm_batch = self._build_hour_norm_batch(raw_cts_wait)
            time_zone_dist_tensor = self._get_combined_zone_dist_tensor(hour_norm_batch, prior_tensor, prior_mask)
            with torch.no_grad():
                batch_q_values = self.network(
                    path_locations=batch_tensors['path_locations'],
                    path_delays=batch_tensors['path_delays'],
                    current_time=batch_tensors['current_time'],
                    other_agents=batch_tensors['other_agents'],
                    num_requests=batch_tensors['num_requests'],
                    battery_level=batch_tensors['battery_level'],
                    request_value=batch_tensors['request_value'],
                    target_distance=batch_tensors['target_distance'],
                    target_zoneid=batch_tensors['target_zoneid'],
                    action_type=torch.full((batch_size, 1), 3, dtype=torch.long).to(self.device),  # 3=wait
                    vehicle_idle_time=batch_tensors['vehicle_idle_time'],
                    vehicle_type=None,
                    prior_features=prior_tensor,
                    prior_mask=prior_mask,
                    time_zone_dist=time_zone_dist_tensor
                )
            
            return batch_q_values.cpu().numpy().flatten().tolist()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️  CUDA out of memory, clearing cache and using CPU fallback...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"Batch waiting Q-value calculation failed: {e}")
            return [self.get_waiting_q_value(
                input_data['vehicle_id'],
                input_data['vehicle_location'],
                input_data.get('battery_level', 1.0),
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0)
            ) for input_data in batch_inputs]
        except Exception as e:
            print(f"❌ Batch waiting Q-value calculation failed: {e}")
            return [self.get_waiting_q_value(
                input_data['vehicle_id'],
                input_data['vehicle_location'],
                input_data.get('battery_level', 1.0),
                input_data.get('current_time', 0.0), input_data.get('other_vehicles', 0),
                input_data.get('num_requests', 0)
            ) for input_data in batch_inputs]



    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int,
                           current_time: float = 0.0, other_vehicles: int = 0,
                           num_requests: int = 0, battery_level: float = 1.0) -> float:
        """
        Get Q-value for vehicle charging decision using neural network
        现在支持battery_level参数
        """
        return self.get_q_value(vehicle_id, f"charge_{station_id}",
                               vehicle_location, station_location, current_time,
                               other_vehicles, num_requests, battery_level)
    
    def _calculate_rejection_risk_penalty(self, vehicle_id: int, distance: float) -> float:
        """
        基于神经网络从历史拒绝经验学习的拒绝概率计算惩罚
        使用训练好的神经网络预测拒绝概率，而不是固定的数学公式
        """
        distance = float(distance or 0.0)

        def _fallback_penalty() -> float:
            distance_factor = 0.1
            fallback_prob = min(0.9, 1 - math.exp(-distance * distance_factor))
            return fallback_prob * 2.0

        # 获取车辆信息
        if hasattr(self, 'env') and self.env is not None:
            vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle is None:
                return 0.0
            
            battery_level = vehicle.get('battery', 1.0)
            vehicle_type = vehicle.get('type', 1)
            current_time = self.env.current_time if hasattr(self.env, 'current_time') else 0
            num_requests = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
            pickup_time_minutes = 0.0
            avg_velocity = float(getattr(self.env, 'average_velocity_kmph', 0.0) or 0.0)
            if avg_velocity > 0:
                pickup_time_minutes = distance / avg_velocity * 60.0
            vehicle_idle_time = float(vehicle.get('idle_timer', 0.0))
        else:
            # 回退到默认值
            battery_level = 1.0
            vehicle_type = self._vehicle_type_id(vehicle_id)
            current_time = 0
            num_requests = 0
            pickup_time_minutes = 0.0
            vehicle_idle_time = 0.0
        
        # 只为EV车辆计算拒绝风险，AEV不拒绝
        if vehicle_type != 1:  # 不是EV
            return 0.0

        if (
            not getattr(self, 'rejection_predictor_trained', False)
            or len(self.rejection_buffer) < self.rejection_min_train_samples
        ):
            return _fallback_penalty()

        sample = {
            'pickup_distance_km': distance,
            'pickup_time_minutes': pickup_time_minutes,
            'vehicle_idle_time': vehicle_idle_time,
            'battery_level': battery_level,
            'current_time': current_time,
            'num_requests': num_requests,
            'request_value': 0.0,
            'surge_value': 0.0,
            'trip_distance_km': 0.0,
            'trip_duration_epochs': 0.0,
            'vehicle_type': vehicle_type,
            'was_rejected': False,
        }
        features = torch.tensor([self._rejection_feature_vector(sample)], dtype=torch.float32).to(self.device)

        self.rejection_predictor.eval()
        with torch.no_grad():
            rejection_prob = self.rejection_predictor(features).item()
        
        return float(rejection_prob) * 3.0
    
    def _calculate_manhattan_distance(self, location1: int, location2: int, grid_size: int = None) -> float:
        """
        计算两个位置之间的曼哈顿距离
        
        Args:
            location1: 位置1（网格索引）
            location2: 位置2（网格索引）
            grid_size: 网格大小，如果未提供则使用self.grid_size
            
        Returns:
            曼哈顿距离
        """
        if grid_size is None:
            grid_size = self.grid_size
            
        # 将位置索引转换为坐标
        x1, y1 = location1 % grid_size, location1 // grid_size
        x2, y2 = location2 % grid_size, location2 // grid_size
        
        # 计算曼哈顿距离
        distance = abs(x1 - x2) + abs(y1 - y2)
        
        return float(distance)
    
    def store_experience(self, vehicle_id: int, action_type: str, vehicle_location: int,
                        target_location: int, current_time: float, reward: float,
                        next_vehicle_location: int, next_target_location: int, battery_level: float = 1.0, 
                        next_battery_level: float = 1.0, other_vehicles: int = 0, 
                        num_requests: int = 0, request_value: float = 0.0,
                        next_action_type: str = None, next_request_value: float = 0.0,
                        dur_time: float = 1.0, is_system_done: bool = False, vehicle_idle_time: float = 0.0,next_vehicle_idle_time: float = 0.0,
                        is_vehicle_done: bool = False, ev_dropout_after_action: bool = False, dropout_penalty: float = 0.0,
                        dropout_satisfaction: float = 0.0, dropout_salary_ratio: float = 0.0, dropout_probability: float = 0.0,
                        next_dropout_satisfaction: float = 0.0, next_dropout_salary_ratio: float = 0.0, next_dropout_probability: float = 0.0,
                        next_prior_features = None, next_hour_of_day: float = None,
                        was_rejected: bool = False,
                        post_action_location: int = None,
                        post_action_distance: float = None,
                        post_action_duration: float = None,
                        post_action_zoneid: int = None,
                        next_post_action_distance: float = None,
                        next_post_action_duration: float = None,
                        next_post_action_zoneid: int = None,
                        next_candidate_actions = None, **feature_snapshots):
        """
        Store experience for training - 现在支持vehicle_id、battery、request_value、idle_time、持续时间和系统结束状态信息
        
        Args:
            vehicle_id: 车辆ID
            action_type: 动作类型
            vehicle_location: 车辆当前位置
            target_location: 目标位置
            current_time: 当前时间
            reward: 获得的奖励
            next_vehicle_location: 下一状态的车辆位置
            battery_level: 当前电池电量 (默认1.0为向后兼容)
            next_battery_level: 下一状态的电池电量 (默认1.0为向后兼容)
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            request_value: 请求价值 (只对assign动作有效，默认0.0)
            next_action_type: 下一个动作类型 (车辆完成当前动作后根据ILP分配的动作标签)
            dur_time: 动作持续时间 (默认1.0为向后兼容)
            is_system_done: 整个系统是否结束 (默认False为向后兼容)
            vehicle_idle_time: 车辆idle时间 (归一化0-1，默认0.0)
        """
        vehicle_info = self.env.vehicles.get(vehicle_id, {}) if hasattr(self, 'env') and hasattr(self.env, 'vehicles') else {}
        if bool(getattr(self.env, 'evaluatemode', False)):
            return
        vehicle_type = int(vehicle_info.get('type', self._vehicle_type_id(vehicle_id)))
        
        # 🔧 确保location是int类型，如果是tuple则转换为location ID
        def ensure_location_id(loc):
            """确保location是int ID而不是tuple坐标"""
            if loc is None:
                return 0  # 默认值
            if isinstance(loc, tuple):
                # 如果是(x, y)坐标，转换为location ID
                x, y = loc
                return int(y * self.env.grid_size + x)
            return int(loc)
        
        # 确保 target_location 和 next_target_location 不为 None
        if target_location is None:
            target_location = 0
        if next_target_location is None:
            next_target_location = 0
        
        vehicle_location = ensure_location_id(vehicle_location)
        target_location = ensure_location_id(target_location)
        next_vehicle_location = ensure_location_id(next_vehicle_location)
        next_target_location = ensure_location_id(next_target_location)
        reject_dist = self.env._manhattan_distance_loc(vehicle_location, target_location) if hasattr(self.env, '_manhattan_distance_loc') else 0
        #if action_type.startswith('assign') and reward<-10:
            #print(f"❌🚫 EV REJECTION - Distance: {reject_dist} | Vehicle: {vehicle_id} | Reward: {reward:.2f}")
        #if action_type.startswith('assign') and reward>10 and self.env.vehicles[vehicle_id]['type']==1:
            #print(f"✅ EV ASSIGNED - Distance: {reject_dist} | Vehicle: {vehicle_id} | Reward: {reward:.2f} | Battery: {battery_level:.2f}")
        current_hour_of_day = self.env.get_hour_of_day(current_time) if hasattr(self.env, 'get_hour_of_day') else (current_time / self.episode_length * 24.0 if self.episode_length > 0 else 0.0)
        if next_hour_of_day is None:
            next_raw_time = current_time + dur_time
            next_hour_of_day = self.env.get_hour_of_day(next_raw_time) if hasattr(self.env, 'get_hour_of_day') else (next_raw_time / self.episode_length * 24.0 if self.episode_length > 0 else 0.0)
        action_context = self._replay_collection_context
        action_metadata = getattr(action_context, "metadata", None)
        request_obj = (
            getattr(action_metadata, "request_snapshot", None)
            if action_metadata is not None
            else None
        )
        rejection_feature_vector = None
        if request_obj is None and action_type.startswith('assign') and hasattr(self, 'env') and self.env is not None:
            try:
                request_id = int(str(action_type).split('_', 1)[1])
                request_obj = getattr(self.env, 'active_requests', {}).get(request_id)
            except (IndexError, TypeError, ValueError):
                request_obj = None
        if request_obj is not None:
            pickup_location = int(getattr(request_obj, 'pickup', target_location))
            dropoff_location = int(getattr(request_obj, 'dropoff', target_location))
            pickup_distance = float(self.env.get_distance_km(vehicle_location, pickup_location)) if hasattr(self.env, 'get_distance_km') else 0.0
            trip_distance = float(self.env.get_distance_km(pickup_location, dropoff_location)) if hasattr(self.env, 'get_distance_km') else 0.0
            pickup_duration = float(self.env.get_travel_time(vehicle_location, pickup_location)) if hasattr(self.env, 'get_travel_time') else 0.0
            trip_duration = float(getattr(request_obj, 'travel_time', 0.0))
            post_action_location = dropoff_location if post_action_location is None else post_action_location
            post_action_distance = pickup_distance + trip_distance if post_action_distance is None else post_action_distance
            post_action_duration = pickup_duration + trip_duration if post_action_duration is None else post_action_duration
            rejection_sample = self._build_rejection_sample(
                vehicle_id,
                vehicle_location,
                pickup_location,
                current_time,
                request=request_obj,
                was_rejected=was_rejected,
            )
            rejection_feature_vector = self._rejection_feature_vector(rejection_sample)
        if post_action_location is None:
            post_action_location = target_location
        post_action_location = ensure_location_id(post_action_location)
        if post_action_distance is None:
            post_action_distance = self.env._manhattan_distance_loc(vehicle_location, post_action_location) if hasattr(self.env, '_manhattan_distance_loc') else 0.0
        if post_action_duration is None:
            post_action_duration = dur_time
        if post_action_zoneid is None:
            post_action_zoneid = _env_zone_index(self.env, post_action_location)
        target_zone_index = _env_distribution_zone_index(self.env, target_location)
        next_target_zone_index = _env_distribution_zone_index(self.env, next_target_location)

        experience = {
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,  # 添加车辆类型
            'vehicle_idle_time': vehicle_idle_time,  # 添加车辆idle时间
            'next_vehicle_idle_time': next_vehicle_idle_time,  # 添加下一状态车辆idle时间
            'action_type': action_type,
            'vehicle_location': vehicle_location,
            'target_location': target_location,
            'battery_level': battery_level,  # 添加当前电池电量
            'current_time': current_time,
            'target_distance': self.env._manhattan_distance_loc(vehicle_location, target_location),
            'pickup_dist': self.env._manhattan_distance_loc(vehicle_location, target_location),  # 添加pickup_dist用于统计
            'target_zoneid': _env_zone_index(self.env, target_location),
            'target_zone_index': target_zone_index,
            'reward': reward,
            'next_vehicle_location': next_vehicle_location,
            'next_battery_level': next_battery_level,  # 添加下一状态电池电量
            'next_action_type': next_action_type if next_action_type is not None else action_type,  # 添加下一动作类型，默认为当前动作类型
            'other_vehicles': other_vehicles,
            'num_requests': num_requests,
            'request_value': request_value,  # 添加请求价值信息
            'was_rejected': bool(was_rejected),
            'next_request_value': next_request_value,  # 下一状态请求价值
            'next_target_distance': self.env._manhattan_distance_loc(next_vehicle_location, next_target_location),
            'next_target_zoneid': _env_zone_index(self.env, next_target_location),
            'next_target_zone_index': next_target_zone_index,
            'post_action_location': post_action_location,
            'post_action_distance': float(post_action_distance),
            'post_action_duration': float(post_action_duration),
            'post_action_zoneid': int(post_action_zoneid or 0),
            'rejection_feature_vector': rejection_feature_vector,
            'rejection_label': 1.0 if was_rejected else 0.0,
            'next_post_action_distance': float(next_post_action_distance or 0.0),
            'next_post_action_duration': float(next_post_action_duration or 0.0),
            'next_post_action_zoneid': int(next_post_action_zoneid or 0),
            'next_candidate_actions': list(next_candidate_actions or []),
            'is_idle': str(action_type) == 'idle' or str(action_type).startswith('reloc'),  # 自动标记idle/reloc状态
            'dur_time': dur_time,  # 添加动作持续时间
            'is_system_done': is_system_done,
            'is_vehicle_done': is_vehicle_done,
            'ev_dropout_after_action': ev_dropout_after_action,
            'dropout_penalty': dropout_penalty,
            'dropout_satisfaction': dropout_satisfaction,
            'dropout_salary_ratio': dropout_salary_ratio,
            'dropout_probability': dropout_probability,
            'next_dropout_satisfaction': next_dropout_satisfaction,
            'next_dropout_salary_ratio': next_dropout_salary_ratio,
            'next_dropout_probability': next_dropout_probability,
            # Prior features + zone dist target for TimeZoneDistributionPredictor training
            'prior_features': getattr(self.env, '_prior_features_for_posterior', None),
            'zone_dist_target': getattr(self.env, '_prior_zone_dist_target', None),
            'is_posterior': getattr(self.env, '_prior_features_for_posterior', None) is not None,
            'external_prior_zone_dist': getattr(self.env, '_bayes_external_prior', None),
            'external_posterior_zone_dist': getattr(self.env, '_bayes_external_posterior', None),
            'bayes_state_zone_dist': getattr(self.env, '_bayes_state_posterior', None),
            'bayes_role': getattr(self.env, '_bayes_context_role', None),
            # Real-world hour (0-24) for TimeZoneDistributionPredictor
            'hour_of_day': current_hour_of_day,
            'next_prior_features': next_prior_features,
            'next_hour_of_day': next_hour_of_day,
        }
        experience.update(feature_snapshots)
        if action_metadata is not None:
            experience.update({
                'schema_version': 1,
                'transition_id': action_metadata.transition_id,
                'stage_id': int(action_metadata.stage_id),
                'acceptance_outcome': action_metadata.acceptance_outcome,
                'residual_category': action_metadata.residual_category,
                'state_snapshot': action_metadata.state_snapshot,
                'feasible_graph_snapshot': action_metadata.feasible_graph_snapshot,
                'residual_state_snapshot': action_metadata.residual_state_snapshot,
                'next_state_snapshot': action_metadata.next_state_snapshot,
                'joint_action_snapshot': action_metadata.joint_action_snapshot,
            })
            experience.update(action_metadata.extras)
        experience.setdefault('mode', getattr(self.env, 'decision_mode', 'integrated'))
        experience.setdefault('recourse_variant', getattr(self.env, 'recourse_variant', 'legacy'))
        experience.setdefault('solver_backend', getattr(self.env, 'mcmf_backend', 'unknown'))
        if request_obj is not None:
            experience['request_id'] = request_obj.request_id
        q, mask = self.response_from_experience(experience)
        experience.update(rejection_probability=q, human_response_mask=mask,
                          response_model_hash=self.response_model_hash, response_schema_version=3)
        if self.acceptance_input_enabled:
            next_context = getattr(action_context, 'next_action', None)
            next_metadata = getattr(next_context, 'metadata', None)
            next_request = getattr(next_metadata, 'request_snapshot', None)
            next_request_id = getattr(next_context, 'request_id', None)
            if next_request is not None:
                next_request_id = next_request.request_id
            if next_request_id is not None:
                experience['next_request_id'] = int(next_request_id)
            if next_metadata is not None and next_metadata.state_snapshot is not None:
                experience['next_state_snapshot'] = next_metadata.state_snapshot
            q_next, mask_next = self.response_from_experience(experience, next_state=True)
            experience.update(next_rejection_probability=q_next, next_human_response_mask=mask_next)
        self.experience_buffer.append(experience)
        self._replay_collection_context = None
        previous_total_seen = getattr(self, 'total_experiences_seen', max(0, len(self.experience_buffer) - 1))
        self.total_experiences_seen = previous_total_seen + 1
    
    def store_idle_experience(self, vehicle_id: int, vehicle_location: int, 
                            battery_level: float, current_time: float, reward: float,
                            next_vehicle_location: int, next_battery_level: float,
                            other_vehicles: int = 0, num_requests: int = 0, request_value: float = 0.0):
        """
        Store idle experience for training - 专门为idle动作存储经验
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 当前电池电量
            current_time: 当前时间
            reward: 获得的奖励
            next_vehicle_location: 下一状态的车辆位置
            next_battery_level: 下一状态的电池电量
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            request_value: 请求价值 (idle时为0.0)
        """
        target_zone_index = _env_distribution_zone_index(self.env, vehicle_location) if hasattr(self, 'env') else None
        experience = {
            'vehicle_id': vehicle_id,
            'action_type': 'idle',
            'vehicle_location': vehicle_location,
            'target_location': vehicle_location,  # idle时目标位置就是当前位置
            'battery_level': battery_level,
            'current_time': current_time,
            'target_zoneid': _env_zone_index(self.env, vehicle_location) if hasattr(self, 'env') else 0,
            'target_zone_index': target_zone_index,
            'reward': reward,
            'next_vehicle_location': next_vehicle_location,
            'next_battery_level': next_battery_level,
            'next_target_zoneid': _env_zone_index(self.env, next_vehicle_location) if hasattr(self, 'env') else 0,
            'next_target_zone_index': _env_distribution_zone_index(self.env, next_vehicle_location) if hasattr(self, 'env') else None,
            'other_vehicles': other_vehicles,
            'num_requests': num_requests,
            'request_value': request_value,  # 添加请求价值信息（idle时为0）
            'is_idle': True,  # 标记这是一个idle经验
            'zone_dist_target': getattr(self.env, '_prior_zone_dist_target', None) if hasattr(self, 'env') and self.env else None,
            'bayes_state_zone_dist': getattr(self.env, '_bayes_state_posterior', None) if hasattr(self, 'env') and self.env else None,
            # Real-world hour (0-24) for TimeZoneDistributionPredictor
            'hour_of_day': self.env.get_hour_of_day(current_time) if hasattr(self, 'env') and self.env and hasattr(self.env, 'get_hour_of_day') else (current_time / self.episode_length * 24.0 if self.episode_length > 0 else 0.0),
        }
        self.experience_buffer.append(experience)
        previous_total_seen = getattr(self, 'total_experiences_seen', max(0, len(self.experience_buffer) - 1))
        self.total_experiences_seen = previous_total_seen + 1
    
    def analyze_experience_data(self):
        """分析经验缓冲区中的奖励分布和动作类型统计"""
        if len(self.experience_buffer) < 100:
            return None
            
        experiences = list(self.experience_buffer)
        
        # 奖励分析
        rewards = [exp['reward'] for exp in experiences]
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        neutral_rewards = [r for r in rewards if r == 0]
        
        reward_stats = {
            'total_count': len(rewards),
            'positive_count': len(positive_rewards),
            'negative_count': len(negative_rewards),
            'neutral_count': len(neutral_rewards),
            'positive_ratio': len(positive_rewards) / len(rewards),
            'negative_ratio': len(negative_rewards) / len(rewards),
            'neutral_ratio': len(neutral_rewards) / len(rewards),
            'mean_reward': np.mean(rewards),
            'mean_positive': np.mean(positive_rewards) if positive_rewards else 0,
            'mean_negative': np.mean(negative_rewards) if negative_rewards else 0,
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
        
        # 动作类型分析
        action_types = [exp['action_type'] for exp in experiences]
        assign_actions = [exp for exp in experiences if exp['action_type'].startswith('assign')]
        charge_actions = [exp for exp in experiences if exp['action_type'].startswith('charge')]
        idle_actions = [exp for exp in experiences if exp['action_type'] == 'idle']
        ev_assign_positive_actions = [exp for exp in assign_actions if exp.get('vehicle_type') == 1 and exp['reward'] > 10]
        ev_assign_negative_actions = [exp for exp in assign_actions if exp.get('vehicle_type') == 1 and exp['reward'] < 10]
        
        action_stats = {
            'assign_count': len(assign_actions),
            'charge_count': len(charge_actions), 
            'idle_count': len(idle_actions),
            'assign_ratio': len(assign_actions) / len(experiences),
            'assign_mean_reward': np.mean([exp['reward'] for exp in assign_actions]) if assign_actions else 0,
            'charge_mean_reward': np.mean([exp['reward'] for exp in charge_actions]) if charge_actions else 0,
            'idle_mean_reward': np.mean([exp['reward'] for exp in idle_actions]) if idle_actions else 0,
            'assign_positive_ratio': len([exp for exp in assign_actions if exp['reward'] > 0]) / len(assign_actions) if assign_actions else 0,
            'charge_positive_ratio': len([exp for exp in charge_actions if exp['reward'] > 0]) / len(charge_actions) if charge_actions else 0,
            'idle_positive_ratio': len([exp for exp in idle_actions if exp['reward'] > 0]) / len(idle_actions) if idle_actions else 0,
            'ev_assign_positive_count': len(ev_assign_positive_actions),
            'ev_assign_negative_count': len(ev_assign_negative_actions),
            'ev_assign_positive_ratio': len(ev_assign_positive_actions) / len(assign_actions) if assign_actions else 0,
        }
        buffer_stats = {
            'current_size': len(experiences),
            'capacity': self.experience_buffer.maxlen,
            'total_seen': getattr(self, 'total_experiences_seen', len(experiences)),
        }
        
        return {
            'reward_stats': reward_stats,
            'action_stats': action_stats,
            'buffer_stats': buffer_stats,
        }
    
    def store_rejection_experience(self, vehicle_id: int, request_id: int, vehicle_location: int,
                                 pickup_location: int, current_time: float, distance: float,
                                 rejection_reason: str = "distance",
                                 rejection_sample: dict = None):
        #print("Storing rejection experience...")
        """
        存储EV拒绝订单的负面经验，用于训练避免分配给EV远距离或容易被拒绝的订单
        
        Args:
            vehicle_id: 拒绝订单的EV车辆ID
            request_id: 被拒绝的请求ID
            vehicle_location: 车辆位置
            pickup_location: 接客位置
            current_time: 当前时间
            distance: 距离（主要的拒绝因素）
            rejection_reason: 拒绝原因
        """
        request_obj = None
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'active_requests'):
            request_obj = self.env.active_requests.get(request_id)
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, '_calculate_rejection_reward'):
            distance_penalty = self.env._calculate_rejection_reward(
                vehicle_id,
                request_obj,
                pickup_location=pickup_location,
                vehicle_location=vehicle_location,
            )
        else:
            distance_penalty = -1.0 - (distance * 0.2)

        experience_target_location = int(pickup_location)
        request_value = float(getattr(request_obj, 'final_value', 0.0))

        if self.acceptance_input_enabled:
            # The canonical ServiceAction replay already stores this reward
            # with its pre-offer snapshot. Keep the auxiliary label, but do
            # not create a duplicate, post-response TD transition here.
            if not bool(getattr(self.env, 'evaluatemode', False)):
                sample = dict(rejection_sample) if rejection_sample is not None else self._build_rejection_sample(
                    vehicle_id, vehicle_location, pickup_location, current_time,
                    distance=distance, request=request_obj, was_rejected=True,
                )
                sample['was_rejected'] = True
                self.rejection_buffer.append(sample)
            return

        vehicle = self.env.vehicles.get(vehicle_id) if hasattr(self, 'env') and self.env is not None else None
        if vehicle is not None:
            battery_level = vehicle.get('battery', 1.0)
            num_requests = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
            other_vehicles = len([
                v for v in self.env.vehicles.values()
                if v.get('assigned_request') is not None or v.get('passenger_onboard') is not None
            ]) if hasattr(self.env, 'vehicles') else 0
            self.store_experience(
                vehicle_id=vehicle_id,
                action_type=f'assign_{request_id}',
                vehicle_location=vehicle_location,
                target_location=experience_target_location,
                current_time=current_time,
                reward=distance_penalty,
                next_vehicle_location=vehicle_location,
                next_target_location=vehicle_location,
                battery_level=battery_level,
                next_battery_level=battery_level,
                other_vehicles=other_vehicles,
                num_requests=num_requests,
                request_value=request_value,
                next_action_type='idle',
                next_request_value=0.0,
                dur_time=1.0,
                is_system_done=getattr(self.env, 'done', False),
                vehicle_idle_time=vehicle.get('idle_timer', 0),
                next_vehicle_idle_time=vehicle.get('idle_timer', 0),
                was_rejected=True,
            )

            sample = dict(rejection_sample) if rejection_sample is not None else (
                self._build_rejection_sample(
                    vehicle_id,
                    vehicle_location,
                    pickup_location,
                    current_time,
                    distance=distance,
                    request=request_obj,
                    was_rejected=True,
                )
            )
            sample['was_rejected'] = True
            self.rejection_buffer.append(sample)
    
    def store_acceptance_experience(self, vehicle_id: int, request_id: int, vehicle_location: int,
                                  pickup_location: int, current_time: float, distance: float,
                                  rejection_sample: dict = None):
        """
        存储EV接受订单的正面经验，用于训练拒绝概率预测器
        这与拒绝经验形成对比，帮助网络学习接受/拒绝的边界
        """
        if hasattr(self, 'env') and self.env is not None:
            vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle is not None and vehicle.get('type') == 1:  # 只存储EV的接受数据
                battery_level = vehicle.get('battery', 1.0)
                num_requests = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
                
                request_obj = getattr(self.env, 'active_requests', {}).get(request_id) if hasattr(self.env, 'active_requests') else None
                sample = dict(rejection_sample) if rejection_sample is not None else (
                    self._build_rejection_sample(
                        vehicle_id,
                        vehicle_location,
                        pickup_location,
                        current_time,
                        distance=distance,
                        request=request_obj,
                        was_rejected=False,
                    )
                )
                sample['was_rejected'] = False
                self.rejection_buffer.append(sample)
                
                if self.debug_mode:
                    print(f"Stored acceptance experience: EV {vehicle_id} accepted request {request_id} "
                          f"(distance={distance:.2f})")
        

    
    def train_rejection_predictor(self, batch_size=64, num_epochs=10):
        """
        训练拒绝概率预测神经网络
        使用存储的拒绝和接受数据进行监督学习
        
        Args:
            batch_size: 批次大小
            num_epochs: 训练轮数
        """
        if self.debug_mode:
            print("Training rejection predictor...")
            print(f"Rejection buffer size: {len(self.rejection_buffer)}")
        if len(self.rejection_buffer) < batch_size:
            return None  # 数据不足，无法训练
        
        rejected_data = [sample for sample in self.rejection_buffer if sample.get('was_rejected', False)]
        accepted_data = [sample for sample in self.rejection_buffer if not sample.get('was_rejected', False)]
        if not rejected_data or not accepted_data:
            return None
        class_count = min(len(rejected_data), len(accepted_data))
        max_per_class = max(batch_size, 1) * 8
        class_count = min(class_count, max_per_class)
        data = random.sample(rejected_data, class_count) + random.sample(accepted_data, class_count)
        random.shuffle(data)
        
        # 分离特征和标签
        features = []
        labels = []
        
        for sample in data:
            features.append(self._rejection_feature_vector(sample))
            labels.append(1.0 if sample['was_rejected'] else 0.0)
        
        # 转换为张量
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 多轮训练
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                self.rejection_optimizer.zero_grad()
                
                # 前向传播
                predictions = self.rejection_predictor(batch_X).squeeze(-1)  # 只移除最后一个维度
                # 确保维度匹配
                if predictions.dim() == 0:  # 如果是标量，添加一个维度
                    predictions = predictions.unsqueeze(0)
                if batch_y.dim() == 0:  # 如果batch_y是标量，添加一个维度
                    batch_y = batch_y.unsqueeze(0)
                
                loss = self.rejection_criterion(predictions, batch_y)
                
                # 反向传播
                loss.backward()
                self.rejection_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            total_loss += epoch_loss
            total_batches += num_batches
            
            if self.debug_mode and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch+1}/{num_epochs}: avg_loss={avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        
        self.rejection_predictor_trained = True
        self.rejection_training_losses.append(avg_loss)
        if self.debug_mode:
            print(f"Rejection predictor training complete: {len(data)} balanced samples, {num_epochs} epochs, final avg_loss={avg_loss:.4f}")
        
        return avg_loss

    def _rejection_predictor_diagnostics(self, max_samples: int = 256):
        rejected_data = [sample for sample in self.rejection_buffer if sample.get('was_rejected', False)]
        accepted_data = [sample for sample in self.rejection_buffer if not sample.get('was_rejected', False)]
        stats = {
            'buffer_rejected': len(rejected_data),
            'buffer_accepted': len(accepted_data),
            'trained': bool(getattr(self, 'rejection_predictor_trained', False)),
        }
        if (
            not stats['trained']
            or not rejected_data
            or not accepted_data
            or not hasattr(self, 'rejection_predictor')
            or self.rejection_predictor is None
        ):
            return stats

        per_class = min(max(1, max_samples // 2), len(rejected_data), len(accepted_data))
        data = random.sample(rejected_data, per_class) + random.sample(accepted_data, per_class)
        labels = torch.tensor(
            [1.0 if sample.get('was_rejected', False) else 0.0 for sample in data],
            dtype=torch.float32,
            device=self.device,
        )
        features = torch.tensor(
            [self._rejection_feature_vector(sample) for sample in data],
            dtype=torch.float32,
            device=self.device,
        )
        self.rejection_predictor.eval()
        with torch.no_grad():
            preds = self.rejection_predictor(features).squeeze(-1)
            loss = self.rejection_criterion(preds, labels).item()
            pred_labels = (preds >= 0.5).float()
            accuracy = (pred_labels == labels).float().mean().item()
            rejected_mask = labels == 1.0
            accepted_mask = labels == 0.0

        stats.update({
            'loss': float(loss),
            'accuracy': float(accuracy),
            'pred_rejected_mean': float(preds[rejected_mask].mean().item()) if rejected_mask.any() else 0.0,
            'pred_accepted_mean': float(preds[accepted_mask].mean().item()) if accepted_mask.any() else 0.0,
            'samples': len(data),
        })
        return stats

    def predict_rejection_probability(self, vehicle_id, request_id, vehicle_location, pickup_location, current_time, request=None):
        """
        使用训练好的rejection_predictor预测EV拒绝某个请求的概率
        
        Args:
            vehicle_id: 车辆ID
            request_id: 请求ID
            vehicle_location: 车辆当前位置
            pickup_location: 接客位置
            current_time: 当前时间
            
        Returns:
            float: 拒绝概率 (0-1之间)
        """
        if not hasattr(self, 'rejection_predictor') or self.rejection_predictor is None:
            return 0.0
        if not getattr(self, 'rejection_predictor_trained', False):
            return 0.0
        if request is None and hasattr(self, 'env') and self.env is not None:
            request = getattr(self.env, 'active_requests', {}).get(request_id)
        sample = self._build_rejection_sample(
            vehicle_id,
            vehicle_location,
            pickup_location,
            current_time,
            request=request,
            was_rejected=False,
        )
        features = torch.tensor([self._rejection_feature_vector(sample)], dtype=torch.float32).to(self.device)
        
        # 预测
        self.rejection_predictor.eval()
        with torch.no_grad():
            rejection_prob = self.rejection_predictor(features).item()
        
        return rejection_prob
    
    def get_rejection_statistics(self):
        """获取拒绝经验的统计信息"""
        rejection_experiences = [exp for exp in self.experience_buffer if exp.get('is_rejection', False)]
        
        if not rejection_experiences:
            return None
            
        distances = [exp['rejection_distance'] for exp in rejection_experiences]
        rewards = [exp['reward'] for exp in rejection_experiences]
        
        return {
            'total_rejections': len(rejection_experiences),
            'avg_rejection_distance': np.mean(distances),
            'max_rejection_distance': np.max(distances),
            'min_rejection_distance': np.min(distances),
            'avg_rejection_penalty': np.mean(rewards),
            'rejection_ratio': len(rejection_experiences) / len(self.experience_buffer) if self.experience_buffer else 0
        }
    
    def _advanced_sample(self, batch_size: int, method: str = "balanced"):
        """
        简化的采样策略：只保留balanced和importance采样
        """
        experiences = list(self.experience_buffer)
        
        if method == "importance":
            return self._importance_sampling(experiences, batch_size)
        elif method == "action_balanced":
            return self._action_balanced_sample(batch_size)
        else:
            return self._balanced_sample(batch_size)
    
    def _importance_sampling(self, experiences, batch_size: int):
        """
        重要性采样：根据经验的重要性权重进行采样
        重要性基于：TD误差、奖励稀有性、动作类型稀有性
        """
        if len(experiences) == 0:
            return []
        
        # 计算每个经验的重要性权重
        weights = []
        action_counts = {'idle': 0, 'assign': 0, 'charge': 0}
        reward_values = [exp['reward'] for exp in experiences]
        
        # 统计动作类型分布
        for exp in experiences:
            action_type = exp['action_type']
            if action_type == 'idle':
                action_counts['idle'] += 1
            elif action_type.startswith('assign'):
                action_counts['assign'] += 1
            elif action_type.startswith('charge'):
                action_counts['charge'] += 1
        
        total_experiences = len(experiences)
        reward_std = np.std(reward_values) if len(reward_values) > 1 else 1.0
        
        for i, exp in enumerate(experiences):
            # 1. 动作稀有性权重
            action_type = exp['action_type']
            if action_type == 'idle':
                action_rarity = total_experiences / max(1, action_counts['idle'])
            elif action_type.startswith('assign'):
                action_rarity = total_experiences / max(1, action_counts['assign'])
            elif action_type.startswith('charge'):
                action_rarity = total_experiences / max(1, action_counts['charge'])
            else:
                action_rarity = 1.0
            
            # 2. 奖励稀有性权重
            reward = exp['reward']
            reward_rarity = abs(reward) / (reward_std + 1e-8)
            
            # 3. 时间权重（最近的经验更重要）
            time_weight = 0.5 + 0.5 * (i / max(1, len(experiences) - 1))
            
            # 4. 如果是高价值assign动作，给予额外权重
            if action_type.startswith('assign') and reward > 10:
                assign_bonus = 2.0
            else:
                assign_bonus = 1.0
            
            # 组合权重
            total_weight = action_rarity * reward_rarity * time_weight * assign_bonus
            weights.append(max(0.1, total_weight))  # 最小权重防止0权重
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 根据权重采样
        indices = np.random.choice(len(experiences), size=min(batch_size, len(experiences)), 
                                 replace=False, p=weights)
        
        sampled_experiences = [experiences[i] for i in indices]
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            print(f"📊 Importance sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}")
        
        return sampled_experiences
    
    def _thompson_sampling(self, experiences, batch_size: int):
        """
        Thompson采样：基于贝叶斯优化的探索-利用平衡
        为每种动作类型维护一个Beta分布
        """
        if len(experiences) == 0:
            return []
        
        # 为每种动作类型维护成功/失败计数
        action_stats = {
            'idle': {'success': 1, 'failure': 1},      # 先验参数
            'assign': {'success': 1, 'failure': 1},
            'charge': {'success': 1, 'failure': 1}
        }
        
        # 更新统计数据
        for exp in experiences:
            action_type = exp['action_type']
            reward = exp['reward']
            
            if action_type == 'idle':
                key = 'idle'
            elif action_type.startswith('assign'):
                key = 'assign'
            elif action_type.startswith('charge'):
                key = 'charge'
            else:
                continue
            
            # 定义成功的标准
            if reward > 0:
                action_stats[key]['success'] += 1
            else:
                action_stats[key]['failure'] += 1
        
        # 从Beta分布采样获得每种动作的期望回报
        action_expectations = {}
        for action_type, stats in action_stats.items():
            # Beta分布采样
            alpha = stats['success']
            beta = stats['failure']
            expectation = np.random.beta(alpha, beta)
            action_expectations[action_type] = expectation
        
        print(f"🎲 Thompson sampling expectations: {action_expectations}")
        
        # 基于期望回报分配采样概率
        total_expectation = sum(action_expectations.values())
        if total_expectation > 0:
            sampling_probs = {k: v/total_expectation for k, v in action_expectations.items()}
        else:
            sampling_probs = {k: 1.0/3 for k in action_expectations.keys()}
        
        # 分别从每种动作类型中采样
        sampled_experiences = []
        for action_type, prob in sampling_probs.items():
            target_count = int(batch_size * prob)
            
            # 找到该动作类型的所有经验
            if action_type == 'idle':
                type_experiences = [exp for exp in experiences if exp['action_type'] == 'idle']
            elif action_type == 'assign':
                type_experiences = [exp for exp in experiences if exp['action_type'].startswith('assign')]
            elif action_type == 'charge':
                type_experiences = [exp for exp in experiences if exp['action_type'].startswith('charge')]
            
            if type_experiences and target_count > 0:
                actual_count = min(target_count, len(type_experiences))
                sampled = random.sample(type_experiences, actual_count)
                sampled_experiences.extend(sampled)
        
        # 如果采样不足，随机补充
        remaining = batch_size - len(sampled_experiences)
        if remaining > 0:
            remaining_experiences = [exp for exp in experiences if exp not in sampled_experiences]
            if remaining_experiences:
                additional = random.sample(remaining_experiences, min(remaining, len(remaining_experiences)))
                sampled_experiences.extend(additional)
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 10000 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            print(f"📊 Thompson sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}")
        
        return sampled_experiences
    
    def _prioritized_sampling(self, experiences, batch_size: int):
        """
        优先经验回放：基于TD误差的优先级采样
        优先级 = |TD误差| + 动作价值 + 探索奖励
        """
        if len(experiences) == 0:
            return []
        
        priorities = []
        
        for exp in experiences:
            # 1. 基于奖励的基础优先级
            reward = exp['reward']
            base_priority = abs(reward) + 1e-6  # 避免0优先级
            
            # 2. 动作类型奖励
            action_type = exp['action_type']
            if action_type.startswith('assign'):
                action_bonus = 2.0  # assign动作更重要
            elif action_type.startswith('charge'):
                action_bonus = 1.5  # charge动作中等重要
            else:  # idle
                action_bonus = 1.0
            
            # 3. 稀有动作奖励
            rarity_bonus = 1.0
            if action_type.startswith('assign') and reward > 10:
                rarity_bonus = 3.0  # 高价值assign动作
            elif action_type.startswith('charge') and reward > 0:
                rarity_bonus = 2.0  # 有正回报的charge动作
            
            # 组合优先级
            priority = base_priority * action_bonus * rarity_bonus
            priorities.append(priority)
        
        # 转换为概率分布
        priorities = np.array(priorities)
        
        # 使用alpha参数控制优先级强度
        alpha = 0.6  # 0表示均匀采样，1表示纯优先级采样
        priorities = priorities ** alpha
        
        # 归一化
        probabilities = priorities / np.sum(priorities)
        
        # 采样
        indices = np.random.choice(len(experiences), size=min(batch_size, len(experiences)), 
                                 replace=False, p=probabilities)
        
        sampled_experiences = [experiences[i] for i in indices]
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            avg_priority = np.mean([priorities[i] for i in indices])
            print(f"📊 Prioritized sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}, Avg Priority={avg_priority:.3f}")
        
        return sampled_experiences
        """
        平衡采样策略：确保正样本和负样本的比例均衡
        
        Args:
            batch_size: 批次大小
            
        Returns:
            均衡采样的经验列表
        """
    def _balanced_sample(self, batch_size: int):
        experiences = list(self.experience_buffer)
        
        # 根据奖励将经验分为正样本和负样本
        positive_samples = []  # 正奖励样本
        negative_samples = []  # 负奖励样本
        neutral_samples = []   # 接近零的奖励样本
        reward_threshold = 0
        reward_threshold_positive = 1.0   # 正样本阈值 - 只有明显的正奖励
        reward_threshold_negative = -0.1  # 负样本阈值 - 包含大部分负奖励
        
        for exp in experiences:
            reward = exp['reward']
            if reward > reward_threshold_positive:
                positive_samples.append(exp)
            elif reward < reward_threshold_negative:
                negative_samples.append(exp)
            else:
                neutral_samples.append(exp)
        
        # 计算采样比例
        total_positive = len(positive_samples)
        total_negative = len(negative_samples)
        total_neutral = len(neutral_samples)
        
        if total_positive == 0 and total_negative == 0:
            # 如果没有明确的正负样本，使用随机采样
            return random.sample(experiences, min(batch_size, len(experiences)))
        
        # 计算期望的采样数量 - 优先保证正负样本均衡
        if total_positive > 0 and total_negative > 0:
            # 有正负样本时，采用平衡策略
            positive_count = min(batch_size // 3, total_positive)  # 1/3 正样本
            negative_count = min(batch_size // 3, total_negative)  # 1/3 负样本
            neutral_count = min(batch_size - positive_count - negative_count, total_neutral)  # 剩余为中性样本
        elif total_positive > 0:
            # 只有正样本时
            positive_count = min(batch_size // 2, total_positive)
            negative_count = 0
            neutral_count = min(batch_size - positive_count, total_neutral)
        else:
            # 只有负样本时
            positive_count = 0
            negative_count = min(batch_size // 2, total_negative)
            neutral_count = min(batch_size - negative_count, total_neutral)
        
        # 执行采样
        sampled_batch = []
        
        if positive_count > 0:
            sampled_batch.extend(random.sample(positive_samples, positive_count))
        
        if negative_count > 0:
            sampled_batch.extend(random.sample(negative_samples, negative_count))
        
        if neutral_count > 0:
            sampled_batch.extend(random.sample(neutral_samples, neutral_count))
        
        # 如果采样数量不足，从所有样本中补充
        remaining_needed = batch_size - len(sampled_batch)
        if remaining_needed > 0:
            sampled_ids = {id(exp) for exp in sampled_batch}
            remaining_experiences = [exp for exp in experiences if id(exp) not in sampled_ids]
            if remaining_experiences:
                additional_samples = random.sample(
                    remaining_experiences, 
                    min(remaining_needed, len(remaining_experiences))
                )
                sampled_batch.extend(additional_samples)
        
        # 打印采样统计信息（每100步打印一次）
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            pos_in_batch = sum(1 for exp in sampled_batch if exp['reward'] > reward_threshold)
            neg_in_batch = sum(1 for exp in sampled_batch if exp['reward'] < reward_threshold)
            neu_in_batch = len(sampled_batch) - pos_in_batch - neg_in_batch
            
            print(f"  📊 Balanced sampling: Pos={pos_in_batch}, Neg={neg_in_batch}, Neutral={neu_in_batch}")
            print(f"     Buffer stats: Pos={total_positive}, Neg={total_negative}, Neutral={total_neutral}")
        
        return sampled_batch


    def _assign_evbalanced_sample(self, batch_size: int):
        experiences = list(self.experience_buffer)






    def _action_balanced_sample(self, batch_size: int, ifEV = False):
        """
        基于动作类型的平衡采样，解决动作分布不平衡问题
        确保request、charge、reloc动作在AEV训练batch中按6:2:2进入
        """
        experiences = list(self.experience_buffer)
        if not experiences:
            return []

        def _action_name(exp):
            return str(exp.get('action_type', ''))

        def _is_reloc(exp):
            action_type = _action_name(exp)
            if action_type == 'reloc' or action_type.startswith('reloc'):
                return True
            if action_type == 'idle':
                for key in ('target_distance', 'post_action_distance'):
                    try:
                        if float(exp.get(key, 0.0) or 0.0) > 1e-8:
                            return True
                    except (TypeError, ValueError):
                        continue
            return False

        # 按动作类型分类
        assign_samples = [exp for exp in experiences if _action_name(exp).startswith('assign')]
        charge_samples = [exp for exp in experiences if _action_name(exp).startswith('charge')]
        reloc_samples = [exp for exp in experiences if _is_reloc(exp)]
        idle_samples = [
            exp for exp in experiences
            if _action_name(exp) == 'idle' and not _is_reloc(exp)
        ]

        total_assign = len(assign_samples)
        total_charge = len(charge_samples)
        total_reloc = len(reloc_samples)
        total_idle = len(idle_samples)

        if self.training_step % 100 == 0:  # 减少打印频率
            print(
                "  🔍 Action distribution in buffer: "
                f"Assign={total_assign}, Charge={total_charge}, Reloc={total_reloc}, Wait={total_idle}"
            )

        sampled_batch = []

        def _extend_unique(pool, count):
            if count <= 0 or not pool:
                return 0
            sampled_ids = {id(exp) for exp in sampled_batch}
            candidates = [exp for exp in pool if id(exp) not in sampled_ids]
            take = min(int(count), len(candidates))
            if take > 0:
                sampled_batch.extend(random.sample(candidates, take))
            return take

        def _extend_balanced(pool, count):
            if count <= 0 or not pool:
                return 0
            count = int(count)
            if len(pool) >= count:
                sampled_batch.extend(random.sample(pool, count))
            else:
                sampled_batch.extend(random.choices(pool, k=count))
            return count

        if ifEV:
            assign_pos_samples = [
                exp for exp in assign_samples
                if exp.get('vehicle_type') == 1 and not exp.get('was_rejected', False)
            ]
            assign_neg_samples = [
                exp for exp in assign_samples
                if exp.get('vehicle_type') == 1 and exp.get('was_rejected', False)
            ]
            if assign_pos_samples and assign_neg_samples:
                pos_count = batch_size // 2
                neg_count = batch_size - pos_count
                _extend_balanced(assign_pos_samples, pos_count)
                _extend_balanced(assign_neg_samples, neg_count)
            else:
                _extend_unique(assign_samples, min(batch_size, len(assign_samples)))
        else:
            target_assign = int(round(batch_size * 0.60))
            target_charge = int(round(batch_size * 0.20))
            target_reloc = max(0, batch_size - target_assign - target_charge)
            _extend_unique(assign_samples, target_assign)
            _extend_unique(charge_samples, target_charge)
            _extend_unique(reloc_samples, target_reloc)

        # 如果采样数量不足，从剩余经验中补充
        remaining_needed = batch_size - len(sampled_batch)
        if remaining_needed > 0:
            sampled_ids = {id(exp) for exp in sampled_batch}
            if ifEV:
                remaining_experiences = [exp for exp in assign_samples if id(exp) not in sampled_ids]
            else:
                prioritized_remaining = assign_samples + charge_samples + reloc_samples + idle_samples
                seen_remaining = set()
                remaining_experiences = []
                for exp in prioritized_remaining + experiences:
                    exp_id = id(exp)
                    if exp_id in sampled_ids or exp_id in seen_remaining:
                        continue
                    seen_remaining.add(exp_id)
                    remaining_experiences.append(exp)
            if remaining_experiences:
                additional_samples = random.sample(
                    remaining_experiences, 
                    min(remaining_needed, len(remaining_experiences))
                )
                sampled_batch.extend(additional_samples)
        
        # 打印采样结果统计
        if self.training_step % 100 == 0:  # 每100步打印详细信息
            final_assign = sum(1 for exp in sampled_batch if _action_name(exp).startswith('assign'))
            final_charge = sum(1 for exp in sampled_batch if _action_name(exp).startswith('charge'))
            final_reloc = sum(1 for exp in sampled_batch if _is_reloc(exp))
            final_idle = sum(1 for exp in sampled_batch if _action_name(exp) == 'idle' and not _is_reloc(exp))
            denom = max(1, len(sampled_batch))
            print(
                "  📊 Action-balanced batch: "
                f"Assign={final_assign}, Charge={final_charge}, Reloc={final_reloc}, Wait={final_idle}"
            )
            print(
                "     Target ratios achieved: "
                f"Assign={final_assign/denom:.1%}, Charge={final_charge/denom:.1%}, Reloc={final_reloc/denom:.1%}"
            )
            if ifEV:
                ev_assign = [
                    exp for exp in sampled_batch
                    if exp.get('vehicle_type') == 1 and _action_name(exp).startswith('assign')
                ]
                ev_rejected = sum(1 for exp in ev_assign if exp.get('was_rejected', False))
                ev_accepted = len(ev_assign) - ev_rejected
                print(
                    "     EV request labels: "
                    f"accepted={ev_accepted}, rejected={ev_rejected}"
                )
        
        return sampled_batch

    def _action_type_id(self, action_type: str) -> int:
        action_type = str(action_type)
        if action_type == 'idle' or action_type == 'reloc' or action_type.startswith('reloc'):
            return 1
        if action_type.startswith('charge'):
            return 3
        return 2

    def _double_dqn_next_values(
        self,
        batch,
        fallback_next_values: torch.Tensor,
        next_time_zone_dist: torch.Tensor = None,
        next_prior_features: torch.Tensor = None,
        next_prior_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Estimate max_a Q(s', a) from stored next-state candidate action snapshots."""
        candidate_rows = []
        owners = []
        for batch_idx, exp in enumerate(batch):
            candidates = exp.get('next_candidate_actions') or []
            if not candidates:
                continue
            next_time = float(exp.get('current_time', 0.0)) + float(exp.get('dur_time', 1.0))
            for candidate in candidates:
                target_location = int(candidate.get('target_location', exp.get('next_target_location', exp.get('next_vehicle_location', 0))))
                action_type = candidate.get('action_type', exp.get('next_action_type', 'idle'))
                inputs = self._prepare_network_input_with_battery(
                    exp.get('next_vehicle_location', exp.get('vehicle_location', 0)),
                    target_location,
                    next_time,
                    candidate.get('other_vehicles', exp.get('other_vehicles', 0)),
                    candidate.get('num_requests', exp.get('num_requests', 0)),
                    action_type,
                    exp.get('next_battery_level', 1.0),
                    candidate.get('request_value', 0.0),
                    candidate.get('target_distance', exp.get('next_target_distance', 0.0)),
                    candidate.get('target_zoneid', exp.get('next_target_zoneid', 0)),
                )
                (
                    path_locations,
                    path_delays,
                    time_tensor,
                    others_tensor,
                    requests_tensor,
                    battery_tensor,
                    value_tensor,
                    distance_tensor,
                    zoneid_tensor,
                ) = inputs
                candidate_rows.append({
                    'rejection_probability': self.rejection_from_experience(exp, candidate, next_state=True),
                    'human_response_mask': self.response_mask_from_experience(exp, candidate, next_state=True),
                    'path_locations': path_locations.squeeze(0),
                    'path_delays': path_delays.squeeze(0),
                    'current_time': time_tensor.squeeze(0),
                    'other_agents': others_tensor.squeeze(0),
                    'num_requests': requests_tensor.squeeze(0),
                    'battery_level': battery_tensor.squeeze(0),
                    'request_value': value_tensor.squeeze(0),
                    'target_distance': distance_tensor.squeeze(0),
                    'target_zoneid': zoneid_tensor.squeeze(0),
                    'action_type_id': self._action_type_id(action_type),
                    'vehicle_idle_time': float(candidate.get('vehicle_idle_time', exp.get('next_vehicle_idle_time', 0.0))),
                    'vehicle_type': int(exp.get('vehicle_type', 1)),
                    'post_action_distance': float(candidate.get('post_action_distance', 0.0)),
                    'post_action_duration': float(candidate.get('post_action_duration', 0.0)),
                    'post_action_zoneid': int(candidate.get('post_action_zoneid', 0) or 0),
                })
                owners.append(batch_idx)

        if not candidate_rows:
            self._last_ddqn_candidate_stats = {
                'owner_count': 0,
                'candidate_count': 0,
                'batch_size': len(batch),
                'avg_candidates': 0.0,
            }
            return fallback_next_values

        owner_tensor = torch.tensor(owners, dtype=torch.long, device=self.device)
        cand_path_locations = torch.stack([row['path_locations'] for row in candidate_rows])
        cand_path_delays = torch.stack([row['path_delays'] for row in candidate_rows])
        cand_current_time = torch.stack([row['current_time'] for row in candidate_rows])
        cand_other_agents = torch.stack([row['other_agents'] for row in candidate_rows])
        cand_num_requests = torch.stack([row['num_requests'] for row in candidate_rows])
        cand_battery_levels = torch.stack([row['battery_level'] for row in candidate_rows])
        cand_request_values = torch.stack([row['request_value'] for row in candidate_rows])
        cand_target_distances = torch.stack([row['target_distance'] for row in candidate_rows])
        cand_target_zoneids = torch.stack([row['target_zoneid'] for row in candidate_rows])
        cand_action_types = torch.tensor([row['action_type_id'] for row in candidate_rows], dtype=torch.long, device=self.device).unsqueeze(1)
        cand_idle_times = torch.tensor([row['vehicle_idle_time'] for row in candidate_rows], dtype=torch.float32, device=self.device).unsqueeze(1)
        cand_vehicle_types = torch.tensor([row['vehicle_type'] for row in candidate_rows], dtype=torch.long, device=self.device).unsqueeze(1)
        cand_acceptance = torch.tensor([row['rejection_probability'] for row in candidate_rows], dtype=torch.float32, device=self.device).unsqueeze(1)
        cand_response_mask = torch.tensor([row['human_response_mask'] for row in candidate_rows], dtype=torch.float32, device=self.device).unsqueeze(1)
        cand_post_distances = torch.tensor([row['post_action_distance'] for row in candidate_rows], dtype=torch.float32, device=self.device).unsqueeze(1)
        cand_post_durations = torch.tensor(
            [min(row['post_action_duration'], self.episode_length) / max(float(self.episode_length), 1.0) for row in candidate_rows],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        cand_post_zoneids = torch.tensor([row['post_action_zoneid'] for row in candidate_rows], dtype=torch.long, device=self.device).unsqueeze(1)
        cand_dropout_states = torch.zeros(len(candidate_rows), 1, dtype=torch.float32, device=self.device)
        cand_time_zone_dist = next_time_zone_dist.index_select(0, owner_tensor) if next_time_zone_dist is not None else None
        cand_prior_features = next_prior_features.index_select(0, owner_tensor) if next_prior_features is not None else None
        cand_prior_mask = next_prior_mask.index_select(0, owner_tensor) if next_prior_mask is not None else None

        with torch.no_grad():
            self.network.eval()
            self.target_network.eval()
            online_values = self.network(
                rejection_probability=cand_acceptance,
                human_response_mask=cand_response_mask,
                path_locations=cand_path_locations,
                path_delays=cand_path_delays,
                current_time=cand_current_time,
                other_agents=cand_other_agents,
                num_requests=cand_num_requests,
                battery_level=cand_battery_levels,
                request_value=cand_request_values,
                target_distance=cand_target_distances,
                target_zoneid=cand_target_zoneids,
                action_type=cand_action_types,
                vehicle_idle_time=cand_idle_times,
                vehicle_type=cand_vehicle_types,
                dropout_state_features=cand_dropout_states,
                post_action_distance=cand_post_distances,
                post_action_duration=cand_post_durations,
                post_action_zoneid=cand_post_zoneids,
                prior_features=cand_prior_features,
                prior_mask=cand_prior_mask,
                time_zone_dist=cand_time_zone_dist,
            ).squeeze(1)
            target_values = self.target_network(
                rejection_probability=cand_acceptance,
                human_response_mask=cand_response_mask,
                path_locations=cand_path_locations,
                path_delays=cand_path_delays,
                current_time=cand_current_time,
                other_agents=cand_other_agents,
                num_requests=cand_num_requests,
                battery_level=cand_battery_levels,
                request_value=cand_request_values,
                target_distance=cand_target_distances,
                target_zoneid=cand_target_zoneids,
                action_type=cand_action_types,
                vehicle_idle_time=cand_idle_times,
                vehicle_type=cand_vehicle_types,
                dropout_state_features=cand_dropout_states,
                post_action_distance=cand_post_distances,
                post_action_duration=cand_post_durations,
                post_action_zoneid=cand_post_zoneids,
                prior_features=cand_prior_features,
                prior_mask=cand_prior_mask,
                time_zone_dist=cand_time_zone_dist,
            ).squeeze(1)

        ddqn_values = fallback_next_values.clone().squeeze(1)
        unique_owners = owner_tensor.unique()
        for owner in unique_owners:
            mask = owner_tensor == owner
            local_online = online_values[mask]
            local_target = target_values[mask]
            best_local_idx = int(torch.argmax(local_online).item())
            ddqn_values[int(owner.item())] = local_target[best_local_idx]
        self._last_ddqn_candidate_stats = {
            'owner_count': int(unique_owners.numel()),
            'candidate_count': int(len(candidate_rows)),
            'batch_size': int(len(batch)),
            'avg_candidates': float(len(candidate_rows) / max(1, int(unique_owners.numel()))),
            'online_mean': float(online_values.mean().item()),
            'target_mean': float(target_values.mean().item()),
        }
        return ddqn_values.unsqueeze(1)

    def train_step(self, batch_size: int = 64, tau: float = 0.02,ifEV=False):  # 软更新系数，稍大可让target Q更快跟随online Q
        """Perform one training step using stored experiences with proper DQN algorithm"""
        if len(self.experience_buffer) < batch_size   :  # Wait for more experiences
            return 0.0
        

        batch = self._action_balanced_sample(batch_size,ifEV)

        
        # 如果平衡采样失败，回退到随机采样
        if not batch or len(batch) < batch_size // 2:
            batch = random.sample(list(self.experience_buffer), min(batch_size, len(self.experience_buffer)))

        # 统计批次中拒绝和接受订单的distance（AEV版本）
        reject_distances = []
        accept_distances = []
        reject_details = []  # 存储拒绝订单的详细信息
        accept_details = []  # 存储接受订单的详细信息
        
        for exp in batch:
            if 'pickup_dist' in exp:
                dist = exp['pickup_dist']
                reward = exp['reward']
                action_type = exp.get('action_type', 'unknown')
                
                if reward < 0:  # 拒绝订单（负奖励）
                    reject_distances.append(dist)
                    reject_details.append({
                        'dist': dist,
                        'reward': reward,
                        'action': action_type,
                        'vehicle_id': exp.get('vehicle_id', -1)
                    })
                else:  # 接受订单（正奖励）
                    accept_distances.append(dist)
                    accept_details.append({
                        'dist': dist,
                        'reward': reward,
                        'action': action_type,
                        'vehicle_id': exp.get('vehicle_id', -1)
                    })

        current_states = []
        next_states = []
        rewards = []
        


        for exp in batch:
            # Current state - 使用支持battery和request_value的输入准备方法
            current_battery = exp.get('battery_level', 0.5)  # 向后兼容
            current_request_value = exp.get('request_value', 0.0)  # 提取请求价值
            current_target_distance = exp.get('target_distance', 0)  # 当前目标距离
            current_target_zoneid = exp.get('target_zoneid', 0)  # 当前目标区域ID
            current_post_action_distance = exp.get('post_action_distance', 0.0)
            current_post_action_duration = exp.get('post_action_duration', 0.0)
            current_post_action_zoneid = exp.get('post_action_zoneid', 0)
            current_inputs = self._prepare_network_input_with_battery(
                exp['vehicle_location'], exp['target_location'], exp['current_time'], 
                exp['other_vehicles'], exp['num_requests'], exp['action_type'], 
                current_battery, current_request_value, current_target_distance, current_target_zoneid
            )
            
            # 处理返回的输入（现在包含battery、request_value、target_distance、target_zoneid）
            if len(current_inputs) == 9:  # 🆕 完整输入：包含所有特征
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor, current_value_tensor, current_distance_tensor, current_zoneid_tensor = current_inputs
            elif len(current_inputs) == 7:  # 向后兼容：包含battery和request_value
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor, current_value_tensor = current_inputs
                current_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                current_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            elif len(current_inputs) == 6:  # 包含battery但没有request_value
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor = current_inputs
                current_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                current_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                current_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            else:  # 不包含battery和request_value（向后兼容）
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor = current_inputs
                current_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                current_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                current_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                current_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            
            current_states.append({
                'path_locations': current_path_locations.squeeze(0),
                'path_delays': current_path_delays.squeeze(0),
                'current_time': current_time_tensor.squeeze(0),
                'other_agents': current_others_tensor.squeeze(0),
                'num_requests': current_requests_tensor.squeeze(0),
                'battery_level': current_battery_tensor.squeeze(0),  # 添加battery信息
                'request_value': current_value_tensor.squeeze(0),  # 添加request_value信息
                'target_distance': current_distance_tensor.squeeze(0),  # 🆕 添加target_distance
                'target_zoneid': current_zoneid_tensor.squeeze(0),  # 🆕 添加target_zoneid
                'post_action_distance': float(current_post_action_distance),
                'post_action_duration': float(current_post_action_duration),
                'post_action_zoneid': int(current_post_action_zoneid or 0),
                'action_type': exp['action_type'],  # 添加action_type信息
                'vehicle_id': exp['vehicle_id'],    # 添加vehicle_id信息
                'vehicle_idle_time': exp.get('vehicle_idle_time', 0.0),  # 添加当前状态的vehicle_idle_time
                'vehicle_type': exp.get('vehicle_type', 1),  # 添加vehicle_type信息（向后兼容）
                'dropout_satisfaction': exp.get('dropout_satisfaction', 0.0),
                'dropout_salary_ratio': exp.get('dropout_salary_ratio', 0.0),
                'dropout_probability': exp.get('dropout_probability', 0.0)
            })
            
            # Next state (for target calculation) - 使用支持battery和request_value的输入准备方法
            next_battery = exp.get('next_battery_level', 1.0)  # 向后兼容
            next_request_value = exp.get('next_request_value', 0.0)  # 下一状态请求价值
            next_action_type = exp.get('next_action_type', exp['action_type'])  # 获取下一个动作类型，如果没有则使用当前动作类型作为备用
            next_target_location = exp.get('next_target_location', exp['target_location'])
            next_target_distance = exp.get('next_target_distance', 0)  # 下一状态目标距离
            next_target_zoneid = exp.get('next_target_zoneid', 0)  # 下一状态目标区域ID
            next_post_action_distance = exp.get('next_post_action_distance', 0.0)
            next_post_action_duration = exp.get('next_post_action_duration', 0.0)
            next_post_action_zoneid = exp.get('next_post_action_zoneid', 0)
            next_state_time = exp['current_time'] + exp.get('dur_time', 1.0)
            next_inputs = self._prepare_network_input_with_battery(
                exp['next_vehicle_location'], next_target_location,
                next_state_time, exp['other_vehicles'], exp['num_requests'],
                next_action_type, next_battery, next_request_value, next_target_distance, next_target_zoneid
            )
            
            
            # 处理next state的返回值（现在包含battery、request_value、target_distance、target_zoneid）
            if len(next_inputs) == 9:  # 🆕 完整输入：包含所有特征
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor, next_value_tensor, next_distance_tensor, next_zoneid_tensor = next_inputs
            elif len(next_inputs) == 7:  # 向后兼容：包含battery和request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor, next_value_tensor = next_inputs
                next_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                next_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            elif len(next_inputs) == 6:  # 包含battery但没有request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor = next_inputs
                next_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                next_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                next_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            else:  # 不包含battery和request_value（向后兼容）
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor = next_inputs
                next_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                next_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                next_distance_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                next_zoneid_tensor = torch.tensor([[0]], dtype=torch.long).to(self.device)
            
            next_states.append({
                'path_locations': next_path_locations.squeeze(0),
                'path_delays': next_path_delays.squeeze(0),
                'current_time': next_time_tensor.squeeze(0),
                'other_agents': next_others_tensor.squeeze(0),
                'num_requests': next_requests_tensor.squeeze(0),
                'battery_level': next_battery_tensor.squeeze(0),  # 添加battery信息
                'request_value': next_value_tensor.squeeze(0),  # 添加request_value信息
                'target_distance': next_distance_tensor.squeeze(0),  # 🆕 添加target_distance
                'target_zoneid': next_zoneid_tensor.squeeze(0),  # 🆕 添加target_zoneid
                'post_action_distance': float(next_post_action_distance),
                'post_action_duration': float(next_post_action_duration),
                'post_action_zoneid': int(next_post_action_zoneid or 0),
                'action_type': next_action_type,  # 使用下一个动作类型而不是当前动作类型
                'vehicle_id': exp['vehicle_id'],    # 添加vehicle_id信息
                'vehicle_type': exp.get('vehicle_type', 1),  # 添加vehicle_type信息（向后兼容）
                'next_vehicle_idle_time': exp.get('next_vehicle_idle_time', 0.0),  # 添加下一状态的vehicle_idle_time
                'next_dropout_satisfaction': exp.get('next_dropout_satisfaction', 0.0),
                'next_dropout_salary_ratio': exp.get('next_dropout_salary_ratio', 0.0),
                'next_dropout_probability': exp.get('next_dropout_probability', 0.0)
            })
            
            rewards.append(exp['reward'])
        
        # Stack batch inputs for current states
        current_batch_path_locations = torch.stack([state['path_locations'] for state in current_states])
        current_batch_path_delays = torch.stack([state['path_delays'] for state in current_states])
        current_batch_current_time = torch.stack([state['current_time'] for state in current_states])
        current_batch_other_agents = torch.stack([state['other_agents'] for state in current_states])
        current_batch_num_requests = torch.stack([state['num_requests'] for state in current_states])
        current_batch_battery_levels = torch.stack([state['battery_level'] for state in current_states])  # 添加battery批处理
        current_batch_request_values = torch.stack([state['request_value'] for state in current_states])  # 添加request_value批处理
        current_batch_target_distances = torch.stack([state['target_distance'] for state in current_states])  # 🆕 添加target_distance批处理
        current_batch_target_zoneids = torch.stack([state['target_zoneid'] for state in current_states])  # 🆕 添加target_zoneid批处理
        current_batch_post_action_distances = torch.tensor([state['post_action_distance'] for state in current_states], dtype=torch.float32).unsqueeze(1).to(self.device)
        current_batch_post_action_durations = torch.tensor(
            [min(state['post_action_duration'], self.episode_length) / max(float(self.episode_length), 1.0) for state in current_states],
            dtype=torch.float32,
        ).unsqueeze(1).to(self.device)
        current_batch_post_action_zoneids = torch.tensor([state['post_action_zoneid'] for state in current_states], dtype=torch.long).unsqueeze(1).to(self.device)
        
        # Convert action_type strings to tensors for current states
        current_action_types = []
        current_vehicle_idle_times = []
        current_vehicle_types = []
        for state in current_states:
            action_type_str = state['action_type']
            action_type_str = str(action_type_str)
            if action_type_str == 'idle' or action_type_str == 'reloc' or action_type_str.startswith('reloc'):
                action_type_id = 1
            elif action_type_str.startswith('assign'):
                action_type_id = 2
            elif action_type_str.startswith('charge'):
                action_type_id = 3
            else:
                action_type_id = 2  # 默认为assign
            current_action_types.append(action_type_id)
            current_vehicle_idle_times.append(state['vehicle_idle_time'])  # idle时间是连续值
            current_vehicle_types.append(state['vehicle_type'])
        
        current_batch_action_types = torch.tensor(current_action_types, dtype=torch.long).to(self.device)
        current_batch_vehicle_idle_times = torch.tensor(current_vehicle_idle_times, dtype=torch.float32).unsqueeze(1).to(self.device)
        current_batch_vehicle_types = torch.tensor(current_vehicle_types, dtype=torch.long).to(self.device)
        current_batch_dropout_states = self._batch_dropout_state_tensor(current_states)
        
        # Stack batch inputs for next states
        next_batch_path_locations = torch.stack([state['path_locations'] for state in next_states])
        next_batch_path_delays = torch.stack([state['path_delays'] for state in next_states])
        next_batch_current_time = torch.stack([state['current_time'] for state in next_states])
        next_batch_other_agents = torch.stack([state['other_agents'] for state in next_states])
        next_batch_num_requests = torch.stack([state['num_requests'] for state in next_states])
        next_batch_battery_levels = torch.stack([state['battery_level'] for state in next_states])  # 添加next states的battery批处理
        next_batch_request_values = torch.stack([state['request_value'] for state in next_states])  # 添加next states的request_value批处理
        next_batch_target_distances = torch.stack([state['target_distance'] for state in next_states])  # 🆕 添加next states的target_distance批处理
        next_batch_target_zoneids = torch.stack([state['target_zoneid'] for state in next_states])  # 🆕 添加next states的target_zoneid批处理
        next_batch_post_action_distances = torch.tensor([state['post_action_distance'] for state in next_states], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_batch_post_action_durations = torch.tensor(
            [min(state['post_action_duration'], self.episode_length) / max(float(self.episode_length), 1.0) for state in next_states],
            dtype=torch.float32,
        ).unsqueeze(1).to(self.device)
        next_batch_post_action_zoneids = torch.tensor([state['post_action_zoneid'] for state in next_states], dtype=torch.long).unsqueeze(1).to(self.device)
        # Convert action_type strings to tensors for next states

        next_vehicle_idle_times = []
        next_vehicle_types = []
        next_action_types = []
        for state in next_states:
            action_type_str = state['action_type']
            next_vehicle_idle_times.append(state['next_vehicle_idle_time'])  # 使用下一状态的idle时间
            next_vehicle_types.append(state['vehicle_type'])
            action_type_str = str(action_type_str)
            if action_type_str == 'idle' or action_type_str == 'reloc' or action_type_str.startswith('reloc'):
                action_type_id = 1
            elif action_type_str.startswith('assign'):
                action_type_id = 2
            elif action_type_str.startswith('charge'):
                action_type_id = 3
            else:
                action_type_id = 2  # 默认为assign
            next_action_types.append(action_type_id)


        next_batch_vehicle_idle_times = torch.tensor(next_vehicle_idle_times, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_batch_vehicle_types = torch.tensor(next_vehicle_types, dtype=torch.long).to(self.device)
        next_batch_action_types = torch.tensor(next_action_types, dtype=torch.long).to(self.device)
        next_batch_dropout_states = self._batch_dropout_state_tensor(next_states, prefix='next_')

        # ==================== Build prior_features tensor from experience buffer ====================
        batch_prior_features = None
        batch_prior_mask = None
        if hasattr(self, 'network') and getattr(self.network, 'encoder', False):
            prior_list = [exp.get('prior_features', None) for exp in batch]
            batch_prior_features, batch_prior_mask = self._build_padded_prior_context(prior_list)

        # ==================== Train TimeZoneDistributionPredictor (time-conditioned prior) ====================
        # Aggregate zone_dist_target per time-bin, update EMA, then train against smoothed targets.
        time_zone_dist_loss_value = 0.0
        # Step 1: collect raw targets and group by time bin
        role_bin_raw_targets = {'leader': {}, 'follower': {}}  # {role: {time_bin_int: list of zone_dist arrays}}
        for exp in batch:
            zdt = exp.get('zone_dist_target', None)
            if zdt is not None:
                role_name = self._normalize_bayes_role(exp.get('bayes_role', None)) or 'leader'
                hour = exp.get('hour_of_day', None)
                if hour is not None:
                    norm_t = hour / 24.0
                else:
                    ct = exp.get('current_time', 0.0)
                    norm_t = ct / self.episode_length if self.episode_length > 0 else 0.0
                time_bin = int(norm_t * (self._tz_num_time_bins - 1))
                time_bin = max(0, min(self._tz_num_time_bins - 1, time_bin))
                role_bin_raw_targets.setdefault(role_name, {}).setdefault(time_bin, []).append(zdt)

        # Step 2: update EMA for each observed time bin
        time_zone_role_losses = []
        for role_name in ('leader', 'follower'):
            bin_raw_targets = role_bin_raw_targets.get(role_name, {})
            if not bin_raw_targets:
                continue
            role_ema = self._tz_ema.setdefault(role_name, {})
            for tbin, targets in bin_raw_targets.items():
                batch_mean = np.mean(targets, axis=0)
                if tbin not in role_ema:
                    role_ema[tbin] = np.array(batch_mean, dtype=np.float32)
                else:
                    alpha = self._tz_ema_alpha
                    role_ema[tbin] = (1 - alpha) * role_ema[tbin] + alpha * np.array(batch_mean, dtype=np.float32)

            _, time_predictor, _, time_optimizer, _ = self._get_bayes_modules(role_name)
            if time_predictor is None or time_optimizer is None:
                continue

            batch_hour_norms_for_tz = []
            batch_tz_targets = []
            for tbin in bin_raw_targets.keys():
                ema_target = role_ema[tbin]
                ema_sum = ema_target.sum()
                if ema_sum > 0:
                    ema_target = ema_target / ema_sum
                else:
                    ema_target = np.ones(self._num_zones, dtype=np.float32) / self._num_zones
                norm_t = (tbin + 0.5) / self._tz_num_time_bins
                batch_hour_norms_for_tz.append(norm_t)
                batch_tz_targets.append(ema_target.tolist())

            tz_time_tensor = torch.tensor(batch_hour_norms_for_tz, dtype=torch.float32).unsqueeze(1).to(self.device)
            tz_target_tensor = torch.tensor(batch_tz_targets, dtype=torch.float32).to(self.device)

            time_predictor.train()
            tz_logits, _ = time_predictor(tz_time_tensor)
            tz_log_probs = torch.log_softmax(tz_logits, dim=1)
            tz_loss = torch.nn.functional.kl_div(tz_log_probs, tz_target_tensor, reduction='batchmean')

            time_optimizer.zero_grad()
            tz_loss.backward()
            torch.nn.utils.clip_grad_norm_(time_predictor.parameters(), max_norm=5.0)
            time_optimizer.step()
            time_zone_role_losses.append(tz_loss.item())

        if time_zone_role_losses:
            time_zone_dist_loss_value = float(np.mean(time_zone_role_losses))
            self.time_zone_dist_losses.append(time_zone_dist_loss_value)

        # ==================== Train strict-Bayes fused posterior ====================
        zone_dist_loss_value = 0.0
        if self.zone_distribution_mode == "bayes" and batch_prior_features is not None:
            fused_role_groups = {
                'leader': {
                    'targets': [],
                    'prior_features': [],
                    'prior_masks': [],
                    'hour_norms': [],
                    'external_priors': [],
                    'external_posteriors': [],
                },
                'follower': {
                    'targets': [],
                    'prior_features': [],
                    'prior_masks': [],
                    'hour_norms': [],
                    'external_priors': [],
                    'external_posteriors': [],
                },
            }
            for idx, exp in enumerate(batch):
                pf = exp.get('prior_features', None)
                zdt = exp.get('zone_dist_target', None)
                if pf is None or len(pf) == 0 or zdt is None:
                    continue
                role_name = self._normalize_bayes_role(exp.get('bayes_role', None)) or 'leader'
                role_group = fused_role_groups[role_name]
                role_group['prior_features'].append(batch_prior_features[idx])
                role_group['prior_masks'].append(batch_prior_mask[idx])
                role_group['targets'].append(zdt)
                hour = exp.get('hour_of_day', None)
                if hour is not None:
                    role_group['hour_norms'].append(hour / 24.0)
                else:
                    ct = exp.get('current_time', 0.0)
                    role_group['hour_norms'].append(ct / self.episode_length if self.episode_length > 0 else 0.0)
                role_group['external_priors'].append(exp.get('external_prior_zone_dist', None))
                role_group['external_posteriors'].append(exp.get('external_posterior_zone_dist', None))

            zone_role_losses = []
            for role_name in ('leader', 'follower'):
                role_group = fused_role_groups[role_name]
                if not role_group['targets']:
                    continue
                _, time_predictor, likelihood_predictor, time_optimizer, likelihood_optimizer = self._get_bayes_modules(role_name)
                if (
                    time_predictor is None or likelihood_predictor is None
                    or time_optimizer is None or likelihood_optimizer is None
                ):
                    continue

                pf_tensor = torch.stack(role_group['prior_features'], dim=0)
                pf_mask_tensor = torch.stack(role_group['prior_masks'], dim=0)
                zt_tensor = torch.tensor(role_group['targets'], dtype=torch.float32, device=self.device)
                hour_tensor = torch.tensor(role_group['hour_norms'], dtype=torch.float32, device=self.device).unsqueeze(1)
                external_prior_tensor = self._expand_external_zone_dist(role_group['external_priors'], len(role_group['targets']), self.device)
                external_posterior_tensor = self._expand_external_zone_dist(role_group['external_posteriors'], len(role_group['targets']), self.device)

                time_predictor.train()
                likelihood_predictor.train()
                prior_logits, _ = time_predictor(hour_tensor)
                likelihood_logits, _ = likelihood_predictor(pf_tensor, hour_tensor, pf_mask_tensor)
                prior_probs = torch.softmax(prior_logits, dim=1)
                fused_probs = self._combine_zone_dist_tensors(
                    prior_probs,
                    likelihood_logits,
                    external_prior_dist=external_prior_tensor,
                    external_posterior_dist=external_posterior_tensor,
                )
                fused_log_probs = torch.log(fused_probs.clamp_min(1e-8))
                zone_dist_loss = torch.nn.functional.kl_div(fused_log_probs, zt_tensor, reduction='batchmean')

                time_optimizer.zero_grad()
                likelihood_optimizer.zero_grad()
                zone_dist_loss.backward()
                torch.nn.utils.clip_grad_norm_(time_predictor.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(likelihood_predictor.parameters(), max_norm=5.0)
                time_optimizer.step()
                likelihood_optimizer.step()
                zone_role_losses.append(zone_dist_loss.item())

            if zone_role_losses:
                zone_dist_loss_value = float(np.mean(zone_role_losses))
                self.zone_dist_losses.append(zone_dist_loss_value)

        # ==================== Compute time_zone_dist for current & next states ====================
        # Build hour_norm tensors from experiences' hour_of_day
        current_hour_norms = []
        next_hour_norms = []
        for exp in batch:
            hour = exp.get('hour_of_day', None)
            next_hour = exp.get('next_hour_of_day', None)
            if hour is not None:
                current_hour_norms.append(hour / 24.0)
                next_hour_norms.append((next_hour / 24.0) if next_hour is not None else (hour / 24.0))
            else:
                ct = exp.get('current_time', 0.0)
                current_hour_norms.append(ct / self.episode_length if self.episode_length > 0 else 0.0)
                next_hour_norms.append((ct + exp.get('dur_time', 1.0)) / self.episode_length if self.episode_length > 0 else 0.0)
        current_hour_norm_tensor = torch.tensor(current_hour_norms, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_hour_norm_tensor = torch.tensor(next_hour_norms, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_prior_list = [exp.get('next_prior_features', None) for exp in batch]
        next_batch_prior_features, next_batch_prior_mask = self._build_padded_prior_context(next_prior_list)
        batch_external_priors = self._expand_external_zone_dist(
            [exp.get('external_prior_zone_dist', None) for exp in batch],
            batch_size,
            self.device,
        )
        batch_external_posteriors = self._expand_external_zone_dist(
            [exp.get('external_posterior_zone_dist', None) for exp in batch],
            batch_size,
            self.device,
        )
        batch_bayes_state_values = [exp.get('bayes_state_zone_dist', None) for exp in batch]
        batch_bayes_state_dist = None
        if any(item is not None for item in batch_bayes_state_values):
            batch_bayes_state_dist = self._expand_external_zone_dist(
                batch_bayes_state_values,
                batch_size,
                self.device,
            )
        batch_bayes_roles = [exp.get('bayes_role', None) for exp in batch]
        current_time_zone_dist = self._get_combined_zone_dist_tensor(
            current_hour_norm_tensor,
            batch_prior_features,
            batch_prior_mask,
            batch_external_priors,
            batch_external_posteriors,
            batch_bayes_state_dist,
            bayes_role=batch_bayes_roles,
        )
        next_time_zone_dist = self._get_combined_zone_dist_tensor(
            next_hour_norm_tensor,
            next_batch_prior_features,
            next_batch_prior_mask,
            bayes_state_dist=batch_bayes_state_dist,
            bayes_role=batch_bayes_roles,
        )

        # Current Q-values (with gradients) - 现在包含所有特征信息
        self.network.train()
        current_q_values = self.network(
            rejection_probability=self.rejection_tensor(batch),
            human_response_mask=self.response_mask_tensor(batch),
            path_locations=current_batch_path_locations,
            path_delays=current_batch_path_delays,
            current_time=current_batch_current_time,
            other_agents=current_batch_other_agents,
            num_requests=current_batch_num_requests,
            battery_level=current_batch_battery_levels,
            request_value=current_batch_request_values,
            target_distance=current_batch_target_distances,
            target_zoneid=current_batch_target_zoneids,
            action_type=current_batch_action_types.unsqueeze(1),
            vehicle_idle_time=current_batch_vehicle_idle_times,
            vehicle_type=current_batch_vehicle_types.unsqueeze(1),
            dropout_state_features=current_batch_dropout_states,
            post_action_distance=current_batch_post_action_distances,
            post_action_duration=current_batch_post_action_durations,
            post_action_zoneid=current_batch_post_action_zoneids,
            prior_features=batch_prior_features,
            prior_mask=batch_prior_mask,
            time_zone_dist=current_time_zone_dist
        )
        
        # Next Q-values using target network (without gradients) - 现在包含所有特征信息
        with torch.no_grad():
            self.target_network.eval()
            next_q_values = self.target_network(
                rejection_probability=self.rejection_tensor(batch, next_state=True),
                human_response_mask=self.response_mask_tensor(batch, next_state=True),
                path_locations=next_batch_path_locations,
                path_delays=next_batch_path_delays,
                current_time=next_batch_current_time,
                other_agents=next_batch_other_agents,
                num_requests=next_batch_num_requests,
                battery_level=next_batch_battery_levels,
                request_value=next_batch_request_values,
                target_distance=next_batch_target_distances,
                target_zoneid=next_batch_target_zoneids,
                action_type=next_batch_action_types.unsqueeze(1),
                vehicle_idle_time=next_batch_vehicle_idle_times,
                vehicle_type=next_batch_vehicle_types.unsqueeze(1),
                dropout_state_features=next_batch_dropout_states,
                post_action_distance=next_batch_post_action_distances,
                post_action_duration=next_batch_post_action_durations,
                post_action_zoneid=next_batch_post_action_zoneids,
                prior_features=next_batch_prior_features,
                prior_mask=next_batch_prior_mask,
                time_zone_dist=next_time_zone_dist
            )
            next_q_values = self._double_dqn_next_values(
                batch,
                next_q_values,
                next_time_zone_dist=next_time_zone_dist,
                next_prior_features=next_batch_prior_features,
                next_prior_mask=next_batch_prior_mask,
            )
        
        # Calculate TD targets without normalization
        gamma = 0.90  # 降低折扣因子，减少未来idle惩罚对当前决策的影响
        
        # 奖励缩放：将奖励值缩放到更合理的范围，避免数值过大导致Q值不稳定
        reward_scale = float(getattr(self, 'reward_scale', 0.1))  # 将奖励缩放10倍，使得10-50的订单价值变成1-5
        self.reward_scale = reward_scale
        rewards_tensor = torch.tensor([r * reward_scale for r in rewards], dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Extract duration times and system done flags from batch
        dur_times = [exp.get('dur_time', 1.0) for exp in batch]
        is_done_flags = [exp.get('is_system_done', False) or exp.get('is_vehicle_done', False) for exp in batch]
        rejected_flags = [bool(exp.get('was_rejected', False)) for exp in batch]
        
        dur_times_tensor = torch.tensor(dur_times, dtype=torch.float32).to(self.device).unsqueeze(1)
        is_done_tensor = torch.tensor([1.0 if not done else 0.0 for done in is_done_flags], 
                                     dtype=torch.float32).to(self.device).unsqueeze(1)
        rejected_tensor = torch.tensor([1.0 if rejected else 0.0 for rejected in rejected_flags],
                                      dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Calculate target Q-values with duration-adjusted discount and terminal state handling
        with torch.no_grad():
            # EV rejection is a within-epoch realization, not a terminal
            # state.  Only genuine environment/vehicle termination masks the
            # continuation target.
            bootstrap_mask = is_done_tensor
            target_q_values = rewards_tensor + (gamma ** dur_times_tensor) * next_q_values * bootstrap_mask
            
            # 添加数值稳定性检查
            if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
                print(f"WARNING: Invalid target Q-values detected!")
                print(f"  Rewards range: [{rewards_tensor.min():.3f}, {rewards_tensor.max():.3f}]")
                print(f"  Next Q-values range: [{next_q_values.min():.3f}, {next_q_values.max():.3f}]")
                return 0.0
        if self.training_step % 20 == 0:
            next_request_values = [float(exp.get('next_request_value', 0.0) or 0.0) for exp in batch]
            next_nonzero = sum(1 for value in next_request_values if abs(value) > 1e-8)
            candidate_counts = [len(exp.get('next_candidate_actions') or []) for exp in batch]
            candidate_states = sum(1 for count in candidate_counts if count > 0)
            max_candidate_values = [
                max((float(candidate.get('request_value', 0.0) or 0.0) for candidate in (exp.get('next_candidate_actions') or [])), default=0.0)
                for exp in batch
            ]
            ddqn_stats = getattr(self, '_last_ddqn_candidate_stats', {})
            print(
                f"[{self.debug_name}] ADPNextValue: next_request_nonzero={next_nonzero}/{len(batch)} "
                f"mean_abs_next_request={np.mean(np.abs(next_request_values)) if next_request_values else 0.0:.3f} "
                f"candidate_states={candidate_states}/{len(batch)} avg_candidates={np.mean(candidate_counts) if candidate_counts else 0.0:.2f} "
                f"mean_max_candidate_request={np.mean(max_candidate_values) if max_candidate_values else 0.0:.3f} "
                f"ddqn_owner={ddqn_stats.get('owner_count', 0)}/{ddqn_stats.get('batch_size', len(batch))} "
                f"next_q_mean={next_q_values.mean().item():.3f} next_q_std={next_q_values.std().item():.3f}",
                flush=True,
            )
	            
        # 添加数值稳定性检查
        if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
            print(f"WARNING: Invalid current Q-values detected!")
            return 0.0
            
        # Compute loss with raw values
        loss = self.loss_fn(current_q_values, target_q_values)
        loss_value = loss.item()  # Define loss_value immediately after loss computation
        
        # 每100个training step或loss异常时，打印详细的loss分析（AEV版本）
        # if self.training_step % 200 == 0 or loss_value > 100.0:
        #     with torch.no_grad():
        #         td_errors = (current_q_values - target_q_values).abs()
        #         print(f"\n📈 AEV Loss Analysis at step {self.training_step}:")
        #         print(f"   Loss: {loss_value:.4f}")
        #         print(f"   Current Q: mean={current_q_values.mean().item():.2f}, "
        #               f"std={current_q_values.std().item():.2f}, "
        #               f"range=[{current_q_values.min().item():.2f}, {current_q_values.max().item():.2f}]")
        #         print(f"   Target Q:  mean={target_q_values.mean().item():.2f}, "
        #               f"std={target_q_values.std().item():.2f}, "
        #               f"range=[{target_q_values.min().item():.2f}, {target_q_values.max().item():.2f}]")
        #         print(f"   Rewards:   mean={rewards_tensor.mean().item():.2f}, "
        #               f"range=[{rewards_tensor.min().item():.2f}, {rewards_tensor.max().item():.2f}]")
        #         print(f"   TD Error:  mean={td_errors.mean().item():.2f}, max={td_errors.max().item():.2f}")
                
        #         # 统计批次中各类动作的数量和奖励
        #         action_stats = {}
        #         for exp in batch:
        #             action = exp['action_type']
        #             if action not in action_stats:
        #                 action_stats[action] = {'count': 0, 'rewards': []}
        #             action_stats[action]['count'] += 1
        #             action_stats[action]['rewards'].append(exp['reward'])
                
        #         print(f"   Batch composition:")
        #         for action, stats in action_stats.items():
        #             pick_updist = [exp['pickup_dist'] for exp in batch if exp['action_type'] == action and 'pickup_dist' in exp]
        #             avg_reward = sum(stats['rewards']) / len(stats['rewards'])
        #             if pick_updist:
        #                 avg_dist = sum(pick_updist) / len(pick_updist)
        #                 print(f"      {action}: n={stats['count']}, avg_reward={avg_reward:.2f}, "
        #                       f"avg_pickup_dist={avg_dist:.2f} (min={min(pick_updist):.1f}, max={max(pick_updist):.1f})")
        #             else:
        #                 print(f"      {action}: n={stats['count']}, avg_reward={avg_reward:.2f}, pickup_dist=N/A")
        
        # 检查损失是否异常
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected: {loss_value}")
            return 0.0
        
        # Additional safety check
        if not loss.requires_grad:
            print("WARNING: Loss does not require gradients!")
            return 0.0
        
        # Backpropagation with gradient monitoring
        self.optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradients for debugging
        total_grad_norm = 0.0
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        # 添加梯度裁剪防止训练崩溃（特别是处理样本不平衡时）
        # 当梯度>5时裁剪，避免loss spike
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Update learning rate scheduler based on loss
        self.scheduler.step(loss_value)
        
        # Update target network periodically (key DQN component)
        
        if self.training_step % self.target_update_frequency == 0:
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            print(f"🔄 [{self.debug_name}] Target network soft-updated at step {self.training_step} with tau={tau}")
        
        # Record training metrics
        self.training_losses.append(loss_value)
        
        # Record Q-values statistics
        with torch.no_grad():
            scale = target_q_values.detach().abs().clamp_min(1.0)
            normalized_td_loss = F.mse_loss(
                current_q_values.detach() / scale,
                target_q_values.detach() / scale,
            ).item()
            normalized_abs = torch.abs(current_q_values.detach() - target_q_values.detach()) / scale
            raw_abs = torch.abs(current_q_values.detach() - target_q_values.detach())
            if not hasattr(self, 'normalized_td_losses'):
                self.normalized_td_losses = []
            if not hasattr(self, 'td_error_history'):
                self.td_error_history = []
            self.normalized_td_losses.append(float(normalized_td_loss))
            self.td_error_history.append({
                'normalized_td_loss': float(normalized_td_loss),
                'normalized_td_abs_mean': float(normalized_abs.mean().item()),
                'normalized_td_abs_max': float(normalized_abs.max().item()),
                'td_abs_mean': float(raw_abs.mean().item()),
                'td_abs_max': float(raw_abs.max().item()),
                'td_bias_mean': float((current_q_values.detach() - target_q_values.detach()).mean().item()),
                'target_scale_mean': float(scale.mean().item()),
            })
            q_mean = current_q_values.mean().item()
            q_std = current_q_values.std().item()
            q_max = current_q_values.max().item()
            q_min = current_q_values.min().item()
            self.q_values_history.append({
                'mean': q_mean, 'std': q_std, 'max': q_max, 'min': q_min,
                'normalized_td_loss': float(normalized_td_loss),
            })
        
        self.training_step += 1

        if (
            ifEV
            and self.training_step % 20 == 0
            and len(self.rejection_buffer) >= self.rejection_min_train_samples
        ):
            self.train_rejection_predictor(
                batch_size=min(128, len(self.rejection_buffer)),
                num_epochs=1,
            )
        
        # Print training progress occasionally  
        if self.training_step % 100 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            
            time_zone_loss_avg = sum(self.time_zone_dist_losses[-100:]) / max(1, len(self.time_zone_dist_losses[-100:])) if self.time_zone_dist_losses else 0.0
            print(f"[{self.debug_name}] Training step {self.training_step}: Loss={loss_value:.4f}, Q_mean={q_mean:.4f}, Q_std={q_std:.4f}, Q_range=[{q_min:.4f}, {q_max:.4f}], LR={current_lr:.6f}")
            print(f"  [{self.debug_name}] Gradient norm: {total_grad_norm:.4f}, No normalization - using raw Q-values")
            print(f"  [{self.debug_name}] TimeZonePredictor: loss_avg_100={time_zone_loss_avg:.4f}")
            if ifEV:
                ev_assign_samples = [
                    exp for exp in batch
                    if exp.get('vehicle_type') == 1 and str(exp.get('action_type', '')).startswith('assign')
                ]
                ev_rejected = sum(1 for exp in ev_assign_samples if exp.get('was_rejected', False))
                print(f"  [{self.debug_name}] EV assign batch labels: accepted={len(ev_assign_samples) - ev_rejected}, rejected={ev_rejected}")
                reject_diag = self._rejection_predictor_diagnostics()
                recent_reject_losses = self.rejection_training_losses[-100:]
                avg_reject_loss = sum(recent_reject_losses) / max(1, len(recent_reject_losses)) if recent_reject_losses else 0.0
                print(
                    f"  [{self.debug_name}] RejectClassifier: loss_avg_100={avg_reject_loss:.4f}, "
                    f"diag_loss={reject_diag.get('loss', 0.0):.4f}, "
                    f"acc={reject_diag.get('accuracy', 0.0):.3f}, "
                    f"pred_rej={reject_diag.get('pred_rejected_mean', 0.0):.3f}, "
                    f"pred_acc={reject_diag.get('pred_accepted_mean', 0.0):.3f}, "
                    f"buffer accepted={reject_diag.get('buffer_accepted', 0)}, "
                    f"rejected={reject_diag.get('buffer_rejected', 0)}"
                )
        
        return loss_value

    def train_step_supervised(self, env, num_samples: int = 64):
        """
        Supervised training: minimize MSE between optimizer-induced option values (labels)
        and network predictions for the same actions, aligning with src_2's "planner-in-the-loop" idea.

        This does NOT use optimizer to assign vehicles in the environment step; it only calls the
        optimizer here to get a one-shot assignment to define labels. If optimizer is unavailable,
        it falls back to a heuristic assignment.
        """
        import torch
        import random
        from collections import deque

        # Sanity: environment must provide vehicles, requests, and evaluators
        if env is None or not hasattr(env, 'vehicles'):
            return 0.0

        # 1) Build a pool of candidate idle vehicles similar to simulate_motion
        vehicles_to_rebalance = []
        for vehicle_id, vehicle in env.vehicles.items():
            if ((vehicle.get('assigned_request') is None and
                 vehicle.get('passenger_onboard') is None and
                 vehicle.get('charging_station') is None and
                 vehicle.get('idle_target') is None) or
                (vehicle.get('battery', 1.0) <= getattr(env, 'rebalance_battery_threshold', 0.5)) or
                vehicle.get('is_stationary', False)):
                vehicles_to_rebalance.append(vehicle_id)

        if not vehicles_to_rebalance:
            return 0.0

        # 2) Get an assignment mapping using optimizer (preferred) or heuristic fallback
        assignments = {}
        try:
            if not hasattr(env, 'gurobi_optimizer'):
                from src.GurobiOptimizer import GurobiOptimizer
                env.gurobi_optimizer = GurobiOptimizer(env)
            assignments = env.gurobi_optimizer.optimize_vehicle_rebalancing_reject(vehicles_to_rebalance)
        except Exception:
            # Heuristic fallback without modifying optimizer implementation
            try:
                available_requests = list(getattr(env, 'active_requests', {}).values())
                charging_stations = [st for st in getattr(env, 'charging_manager').stations.values() if st.available_slots > 0] if hasattr(env, 'charging_manager') else []
                assignments = env.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance, available_requests, charging_stations)  # type: ignore
            except Exception:
                return 0.0

        if not assignments:
            return 0.0

        # 3) Build supervised mini-batch from assignments (action -> label)
        #    Label = env.evaluate_service_option / evaluate_charging_option (option completion value)
        #    Match the network inputs for those actions
        inputs_list = []
        labels_list = []

        # Helper: pack a single sample
        def _append_sample(vehicle_id: int, action_type: str, veh_loc: int, tgt_loc: int,
                           current_time: float, battery: float, request_value: float, label: float):
            # Raw counts for normalization inside helper
            other_vehicles_raw = max(0, sum(1 for v in env.vehicles.values() if v.get('assigned_request') is None and v.get('passenger_onboard') is None and v.get('charging_station') is None) - 1)
            num_requests_raw = len(getattr(env, 'active_requests', {}))

            # Prepare full set of inputs with battery/request value; tensors are already normalized as in other code paths
            path_locations_b, path_delays_b, time_b, others_b, requests_b, battery_b, value_b, _, _ = self._prepare_network_input_with_battery(
                veh_loc, tgt_loc, current_time, other_vehicles_raw, num_requests_raw, action_type, battery, request_value
            )
            # Package tensors in expected dict form
            sample = {
                'path_locations': path_locations_b,
                'path_delays': path_delays_b,
                'current_time': time_b,
                'other_agents': others_b,
                'num_requests': requests_b,
                'battery_level': battery_b,
                'request_value': value_b,
                'action_type': torch.tensor([[self._action_type_id(action_type)]], dtype=torch.long, device=self.device),
                'vehicle_id': torch.tensor([[vehicle_id + 1]], dtype=torch.long, device=self.device),
                'vehicle_type': torch.tensor([[self._vehicle_type_id(vehicle_id)]], dtype=torch.long, device=self.device)
            }
            inputs_list.append(sample)
            kind = self._action_type_id(action_type)
            rid = int(action_type.split('_', 1)[1]) if kind == 2 else -1
            sample['rejection_probability'] = torch.as_tensor(
                self.rejection_for_live_edges([vehicle_id], [kind], [rid]), device=self.device
            ).unsqueeze(1)
            sample['human_response_mask'] = torch.as_tensor(
                self.response_masks_for_live_edges([vehicle_id], [kind]), device=self.device).unsqueeze(1)
            labels_list.append(label)

        # Fill samples from assignments
        for vehicle_id, target in assignments.items():
            vehicle = env.vehicles.get(vehicle_id)
            if not vehicle:
                continue
            veh_loc = vehicle.get('location', 0)
            battery = vehicle.get('battery', 1.0)

            # Service assignment
            if target and hasattr(target, 'pickup') and hasattr(target, 'dropoff'):
                try:
                    label_val = env.evaluate_service_option(vehicle_id, target)
                except Exception:
                    label_val = 0.0
                # Use pickup as target_location for the assign action
                tgt_loc = getattr(target, 'pickup', veh_loc)
                req_val = getattr(target, 'final_value', getattr(target, 'value', 0.0))
                _append_sample(vehicle_id, f"assign_{getattr(target, 'request_id', 0)}", veh_loc, tgt_loc, env.current_time, battery, req_val, label_val)
            # Charging assignment
            elif target and hasattr(target, 'id') and hasattr(target, 'location'):
                try:
                    label_val = env.evaluate_charging_option(vehicle_id, target)
                except Exception:
                    label_val = 0.0
                tgt_loc = getattr(target, 'location', veh_loc)
                _append_sample(vehicle_id, f"charge_{getattr(target, 'id', 0)}", veh_loc, tgt_loc, env.current_time, battery, 0.0, label_val)
            else:
                # Idle/no-op: optionally skip, to focus on meaningful supervised labels
                continue

        if not inputs_list:
            return 0.0
        
        # Downsample to num_samples if necessary
        if len(inputs_list) > num_samples:
            idxs = random.sample(range(len(inputs_list)), num_samples)
            inputs_list = [inputs_list[i] for i in idxs]
            labels_list = [labels_list[i] for i in idxs]

        # 4) Batch the tensors and train with MSE
        def _stack(key):
            return torch.cat([s[key] for s in inputs_list], dim=0)

        self.network.train()
        _current_time_stacked = _stack('current_time')
        # Compute hour_norm for TimeZoneDistributionPredictor
        # All samples in this batch share the same env.current_time
        _hour_norm_supervised = self._get_hour_norm_tensor(env.current_time).expand(len(inputs_list), -1)
        preds = self.network(
            rejection_probability=_stack('rejection_probability'),
            human_response_mask=_stack('human_response_mask'),
            path_locations=_stack('path_locations'),
            path_delays=_stack('path_delays'),
            current_time=_current_time_stacked,
            other_agents=_stack('other_agents'),
            num_requests=_stack('num_requests'),
            battery_level=_stack('battery_level'),
            request_value=_stack('request_value'),
            action_type=_stack('action_type'),
            vehicle_idle_time=_stack('vehicle_idle_time') if 'vehicle_idle_time' in inputs_list[0] else None,
            vehicle_type=_stack('vehicle_type'),
            prior_features=None,  # supervised training: no prior info available
            time_zone_dist=self._get_time_zone_dist_tensor(_hour_norm_supervised)
        )

        labels_tensor = torch.tensor(labels_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        loss = self.loss_fn(preds, labels_tensor)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)  # 从10.0改为5.0，与AEV一致
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        self.training_step += 1

        # Track Q stats
        with torch.no_grad():
            self.q_values_history.append({
                'mean': preds.mean().item(),
                'std': preds.std().item(),
                'max': preds.max().item(),
                'min': preds.min().item()
            })

        return loss_value
    
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Compatibility method for Experience-based interface"""
        # This is a simplified implementation for compatibility
        return []
    
    def update(self, *args, **kwargs):
        """Update the neural network"""
        if len(self.experience_buffer) > 100:
            loss = self.train_step()
            self.add_to_logs('training_loss', loss, self.training_step)
            self.training_step += 1
    
    def remember(self, experience: Experience):
        """简化的经验存储，依赖Environment进行筛选"""
        try:
            # 从Experience中提取相关信息
            for agent_id, actions_info in experience.action_to_take_all_agents.items():
                if not actions_info:
                    continue
                
                action, reward = actions_info[0] if len(actions_info) > 0 else (None, 0.0)
                if action is None:
                    continue
                
                # 获取当前状态信息
                current_state = experience.current_states.get(agent_id)
                next_state = experience.next_states.get(agent_id) if hasattr(experience, 'next_states') else None
                
                if current_state is None:
                    continue
                
                # 创建简化的经验记录
                enhanced_experience = {
                    'vehicle_id': agent_id,
                    'vehicle_location': getattr(current_state, 'location', 0),
                    'target_location': 0,  # 将从action中提取
                    'current_time': experience.current_time,
                    'reward': reward,
                    'next_vehicle_location': getattr(next_state, 'location', 0) if next_state else 0,
                    'other_vehicles': len(experience.current_states) - 1,
                    'num_requests': len(getattr(experience, 'active_requests', [])),
                    'battery_level': getattr(current_state, 'battery', 1.0),
                    'next_battery_level': getattr(next_state, 'battery', 1.0) if next_state else 1.0,
                    'request_value': 0.0,
                    'action_type': 'idle',  # 默认值
                }
                
                # 根据action类型更新相关信息
                if hasattr(action, 'requests') and action.requests:
                    # Service action
                    request = list(action.requests)[0]
                    enhanced_experience['target_location'] = getattr(request, 'pickup', 0)
                    enhanced_experience['request_value'] = getattr(request, 'final_value', 0.0)
                    enhanced_experience['action_type'] = 'assign'
                elif hasattr(action, 'charging_station_id'):
                    # Charging action
                    enhanced_experience['action_type'] = 'charge'
                    enhanced_experience['target_location'] = getattr(action, 'charging_station_id', 0)
                else:
                    # Idle action
                    enhanced_experience['action_type'] = 'idle'
                    if hasattr(action, 'target_coords'):
                        target_x, target_y = action.target_coords
                        grid_size = int(math.sqrt(enhanced_experience['vehicle_location']) + 1)
                        enhanced_experience['target_location'] = target_y * grid_size + target_x
                
                # 直接存储，依赖Environment进行预筛选
                self.experience_buffer.append(enhanced_experience)
                    
                # 定期进行训练
                if len(self.experience_buffer) % 50 == 0:
                    self.train_step(batch_size=32)
                        
        except Exception as e:
            print(f"Warning: Error in simplified remember method: {e}")
            pass
    
    def _evaluate_assignment_quality(self, action, reward: float) -> float:
        """
        重新定义分配质量评估：专注于订单完成能力
        成功的assignment = 能够完成整个服务流程的分配
        """
        # 1. 最高优先级：实际完成了订单（获得了final_value奖励）
        if reward >= 15:  # 完成订单的典型奖励范围
            return 1.0  # 完美质量 - 这是我们最想学习的经验
        
        # 2. 高优先级：部分完成但有正向进展
        elif reward >= 5:  # 可能完成了pickup但还未dropoff
            return 0.8  # 高质量 - 展示了完成能力
        
        # 3. 中等优先级：成功分配但还在执行中
        elif reward > 0:  # 成功分配，正在执行
            return 0.6  # 中等质量 - 有潜力完成
        
        # 4. 低优先级：分配被拒绝或失败
        elif reward == 0:  # 分配失败或被拒绝
            return 0.2  # 低质量 - 可以学习为什么失败
        
        # 5. 负面案例：电池耗尽、无法完成等
        else:  # 负奖励 - 电池耗尽、乘客滞留等
            return 0.0  # 零质量 - 避免学习这类经验
    
    def _analyze_competitive_context(self, experience: Experience) -> float:
        """分析竞争环境上下文"""
        num_vehicles = len(experience.current_states) if hasattr(experience, 'current_states') else 1
        num_requests = len(getattr(experience, 'active_requests', []))
        
        if num_requests == 0:
            return 0.0  # 无请求环境
        
        competition_ratio = num_vehicles / num_requests
        if competition_ratio > 2.0:
            return 1.0  # 高竞争
        elif competition_ratio > 1.0:
            return 0.6  # 中等竞争
        else:
            return 0.2  # 低竞争

    def _assess_order_completion_potential(self, action, current_state, reward: float) -> float:
        """
        评估订单完成潜力：预测这个分配决策能否成功完成订单
        """
        # 基础完成潜力评估
        completion_potential = 0.0
        
        # 1. 电池充足度对完成潜力的影响
        battery_level = getattr(current_state, 'battery', 1.0)
        if battery_level > 0.5:
            completion_potential += 0.4  # 高电量 = 高完成潜力
        elif battery_level > 0.3:
            completion_potential += 0.2  # 中等电量 = 中等完成潜力
        else:
            completion_potential += 0.0  # 低电量 = 低完成潜力
        
        # 2. 如果是assignment action，考虑距离因素
        if hasattr(action, 'requests') and action.requests:
            request = list(action.requests)[0]
            pickup_location = getattr(request, 'pickup', 0)
            current_location = getattr(current_state, 'location', 0)
            
            # 简化的距离计算（假设grid_size=40）
            grid_size = 40
            pickup_x, pickup_y = pickup_location % grid_size, pickup_location // grid_size
            current_x, current_y = current_location % grid_size, current_location // grid_size
            distance = abs(pickup_x - current_x) + abs(pickup_y - current_y)
            
            # 距离越近，完成潜力越高
            if distance <= 3:
                completion_potential += 0.3  # 很近
            elif distance <= 6:
                completion_potential += 0.2  # 较近
            elif distance <= 10:
                completion_potential += 0.1  # 中等距离
            # 远距离不加分
        
        # 3. 实际奖励反馈的完成潜力
        if reward >= 15:  # 已完成订单
            completion_potential = 1.0  # 确定完成
        elif reward >= 5:  # 部分完成
            completion_potential = max(completion_potential, 0.8)
        elif reward > 0:  # 正在执行
            completion_potential = max(completion_potential, 0.6)
        
        return min(1.0, completion_potential)
    
    def _is_order_completion_valuable_experience(self, experience: dict) -> bool:
        """
        严格控制experience存储：只存储关键决策点
        - 完成订单的experience（最终收益）
        - 充电决策的experience  
        - idle移动决策的experience
        - 排除pickup/dropoff执行过程中的experience
        """
        reward = experience['reward']
        action_type = experience['action_type']
        assignment_quality = experience['assignment_quality']
        
        # 1. 【最高优先级】完成订单的experience - 这是最终的成功决策结果
        if reward >= 15 and assignment_quality >= 0.8:
            print(f"✓ Storing COMPLETED ORDER experience: reward={reward}, vehicle={experience['vehicle_id']}")
            return True
        
        # 2. 【充电决策】- 电池管理的关键决策点
        if action_type.startswith('charge'):
            battery_level = experience.get('battery_level', 1.0)
            # 只存储真正需要充电的决策（低电量）
            if battery_level < 0.5:
                print(f"✓ Storing CHARGING decision experience: battery={battery_level}, vehicle={experience['vehicle_id']}")
                return True
            return False
        
        # 3. 【Idle决策】- 空闲状态的移动决策
        if action_type == 'idle' or action_type == 'reloc' or action_type.startswith('reloc'):
            # 存储所有idle决策，因为这些是重要的定位决策
            return True
        
        # 4. 【初始assignment决策】- 只存储刚开始分配的决策，不存储执行过程
        if action_type.startswith('assign'):
            # 只存储真正的分配决策时刻（高质量或负面教训）
            if assignment_quality >= 0.6:  # 成功的分配决策
                print(f"✓ Storing SUCCESSFUL assignment decision: quality={assignment_quality}, reward={reward}")
                return True
            elif assignment_quality == 0.0 and reward <= 0:  # 失败的分配决策（学习教训）
                print(f"✓ Storing FAILED assignment decision for learning: quality={assignment_quality}, reward={reward}")
                return True
            else:
                # 排除执行过程中的中间状态（pickup进行中、dropoff进行中等）
                return False
        
        # 5. 其他情况：不存储
        return False
    
    def plot_training_metrics(self, save_path: str = None):
        """Plot training losses and Q-values over time"""
        import matplotlib.pyplot as plt
        
        if not self.training_losses:
            print("No training data to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot training loss
        ax1.plot(self.training_losses, label='Training Loss', color='red', alpha=0.7)
        ax1.set_title('Neural Network Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Q-values mean
        if self.q_values_history:
            q_means = [q['mean'] for q in self.q_values_history]
            q_stds = [q['std'] for q in self.q_values_history]
            
            ax2.plot(q_means, label='Q-value Mean', color='blue', alpha=0.7)
            ax2.fill_between(range(len(q_means)), 
                           [m - s for m, s in zip(q_means, q_stds)],
                           [m + s for m, s in zip(q_means, q_stds)],
                           alpha=0.2, color='blue', label='±1 Std')
            ax2.set_title('Q-Values Statistics')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Q-Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot Q-values standard deviation
            ax3.plot(q_stds, label='Q-value Std Dev', color='green', alpha=0.7)
            ax3.set_title('Q-Values Standard Deviation')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Standard Deviation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {save_path}")
        else:
            plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
            print("Training metrics plot saved to: results/training_metrics.png")
        
        plt.show()
        return fig


class TimeZoneDistributionPredictor(nn.Module):
    """Time-period-dependent zone distribution predictor.

    Maps real-world hour_of_day (0-24, normalised to 0-1 via /24) to a zone-level
    action distribution.  This captures the *global* (not per-agent) time-varying
    demand / action pattern: e.g., morning-rush vs. night have different zone
    selection logics.

    Trained with KL-divergence against empirical zone distributions collected
    per time step across *all* agents.

    Input convention:
        - NYC env: ``env.get_hour_of_day() / 24.0`` → true hour normalised.
        - Grid env: ``current_time / episode_length`` → linear 24h mapping.
    """

    def __init__(self, num_zones: int, hidden_dim: int = 64, num_time_bins: int = 24):
        super().__init__()
        self.num_zones = num_zones
        self.num_time_bins = num_time_bins

        # Learnable time-bin embedding (one bin ≈ one hour of the day)
        self.time_bin_embedding = nn.Embedding(num_time_bins, hidden_dim)

        # MLP: time_continuous (1-d) + time_bin_embedding → zone distribution
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_zones),
        )

    def forward(self, hour_norm: torch.Tensor):
        """
        Args:
            hour_norm: [B, 1]  hour_of_day / 24.0, in [0, 1].
        Returns:
            zone_logits: [B, num_zones]
            zone_probs:  [B, num_zones]
        """
        # Discretise into hourly bins
        bin_ids = (hour_norm.squeeze(-1) * (self.num_time_bins - 1)).long().clamp(0, self.num_time_bins - 1)
        bin_embed = self.time_bin_embedding(bin_ids)          # [B, hidden_dim]
        x = torch.cat([hour_norm, bin_embed], dim=-1)         # [B, 1 + hidden_dim]
        logits = self.net(x)                                   # [B, num_zones]
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class LikelihoodZonePredictor(nn.Module):
    """Predict unnormalized log-likelihood scores from time and leader context.
    This branch is used for strict Bayes fusion:

        p(z | t, c) \\propto p(z | t) p(c | z, t)

    The network outputs zone-wise evidence scores that are treated as
    unnormalized log p(c | z, t).
    """

    def __init__(self, prior_dim: int, hidden_dim: int, num_zones: int, num_time_bins: int = 24):
        super().__init__()
        self.num_zones = num_zones
        self.num_time_bins = num_time_bins
        self.num_location_bins = 100
        self.soc_bin_boundaries = (0.25, 0.50, 0.75)
        self.idle_bin_boundaries = (0.05, 0.15, 0.30, 0.60)
        self.num_action_bins = 4

        self.location_embedding = nn.Embedding(self.num_location_bins, 16)
        self.soc_embedding = nn.Embedding(len(self.soc_bin_boundaries) + 1, 8)
        self.idle_embedding = nn.Embedding(len(self.idle_bin_boundaries) + 1, 8)
        self.target_embedding = nn.Embedding(self.num_location_bins, 16)
        self.action_embedding = nn.Embedding(self.num_action_bins, 8)

        mapped_prior_dim = 16 + 8 + 8 + 16 + 8

        self.prior_encoder = nn.Sequential(
            nn.Linear(mapped_prior_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.time_bin_embedding = nn.Embedding(num_time_bins, hidden_dim)
        self.zone_head = nn.Sequential(
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_zones),
        )

    def _bucketize_feature(self, values: torch.Tensor, boundaries):
        boundary_tensor = torch.tensor(boundaries, dtype=values.dtype, device=values.device)
        return torch.bucketize(values, boundary_tensor)

    def _map_prior_features(self, prior_features: torch.Tensor):
        loc_norm = prior_features[..., 0].clamp(0.0, 1.0)
        soc = prior_features[..., 1].clamp(0.0, 1.0)
        idle_time = prior_features[..., 2].clamp(min=0.0)
        tgt_norm = prior_features[..., 3].clamp(0.0, 1.0)
        action_type = prior_features[..., 4].round().long().clamp(0, self.num_action_bins - 1)

        loc_ids = (loc_norm * (self.num_location_bins - 1)).long().clamp(0, self.num_location_bins - 1)
        soc_ids = self._bucketize_feature(soc, self.soc_bin_boundaries)
        idle_ids = self._bucketize_feature(idle_time, self.idle_bin_boundaries)
        tgt_ids = (tgt_norm * (self.num_location_bins - 1)).long().clamp(0, self.num_location_bins - 1)

        loc_embed = self.location_embedding(loc_ids)
        soc_embed = self.soc_embedding(soc_ids)
        idle_embed = self.idle_embedding(idle_ids)
        tgt_embed = self.target_embedding(tgt_ids)
        action_embed = self.action_embedding(action_type)
        return torch.cat([loc_embed, soc_embed, idle_embed, tgt_embed, action_embed], dim=-1)

    def forward(self, prior_features: torch.Tensor = None, hour_norm: torch.Tensor = None,
                prior_mask: torch.Tensor = None):
        if prior_features is None or prior_features.size(1) == 0:
            batch_size = 1 if prior_features is None else prior_features.size(0)
            device = 'cpu' if prior_features is None else prior_features.device
            zeros = torch.zeros(batch_size, self.num_zones, device=device)
            probs = torch.ones(batch_size, self.num_zones, device=device) / self.num_zones
            return zeros, probs

        if hour_norm is None:
            hour_norm = torch.zeros(prior_features.size(0), 1, device=prior_features.device)

        if prior_mask is None:
            prior_mask = torch.ones(prior_features.size(0), prior_features.size(1), dtype=torch.bool, device=prior_features.device)
        prior_mask_f = prior_mask.unsqueeze(-1).float()

        mapped_features = self._map_prior_features(prior_features)
        encoded = self.prior_encoder(mapped_features)
        pooled = (encoded * prior_mask_f).sum(dim=1) / prior_mask_f.sum(dim=1).clamp_min(1.0)
        bin_ids = (hour_norm.squeeze(-1) * (self.num_time_bins - 1)).long().clamp(0, self.num_time_bins - 1)
        bin_embed = self.time_bin_embedding(bin_ids)
        fused = torch.cat([hour_norm, pooled, bin_embed], dim=-1)
        logits = self.zone_head(fused)
        probs = torch.softmax(logits, dim=1)
        return logits, probs


class CrossAttentionEncoder(nn.Module):
    """Cross-Attention encoder for sequential decision-making (sequential_encode.md).

    Maps a variable number of prior agents' state-action pairs into a fixed-dim
    context vector ``E_dist`` via scaled dot-product cross-attention.

    Prior inputs are **detached** to prevent gradient flow back into the leader's
    feature pathway (bidirectional gradient detachment).
    """

    def __init__(self, prior_dim: int, posterior_dim: int, d_k: int, num_zones: int):
        """
        Args:
            prior_dim:     dim of each prior's feature vector (state ⊕ action).
            posterior_dim: dim of the posterior's state feature vector.
            d_k:           latent / output dimensionality.
            num_zones:     number of discrete zones (kept for interface compat).
        """
        super().__init__()
        self.d_k = d_k
        self.num_zones = num_zones

        self.W_Q = nn.Linear(posterior_dim, d_k)
        self.W_K = nn.Linear(prior_dim, d_k)
        self.W_V = nn.Linear(prior_dim, d_k)

    def forward(self, posterior_state: torch.Tensor,
                prior_features: torch.Tensor = None,
                prior_mask: torch.Tensor = None):
        """
        Args:
            posterior_state: [B, posterior_dim]  - one posterior per batch element.
            prior_features:  [B, N, prior_dim]  - N priors.  None or N==0
                             triggers zero-padding (Phase 1 / leader mode).

        Returns:
            context: [B, d_k]        - E_dist (zero if no priors).
        """
        B = posterior_state.size(0)
        device = posterior_state.device

        if prior_features is None or prior_features.size(1) == 0:
            return torch.zeros(B, self.d_k, device=device)

        # Gradient detach on prior inputs
        prior_features = prior_features.detach()
        if prior_mask is None:
            prior_mask = torch.ones(B, prior_features.size(1), dtype=torch.bool, device=device)

        Q = self.W_Q(posterior_state)                       # [B, d_k]
        K = self.W_K(prior_features)                        # [B, N, d_k]
        V = self.W_V(prior_features)                        # [B, N, d_k]

        # Scaled dot-product attention
        scores = torch.bmm(K, Q.unsqueeze(2)).squeeze(2)    # [B, N]
        scores = scores / (self.d_k ** 0.5)
        scores = scores.masked_fill(~prior_mask, -1e9)
        alpha = torch.softmax(scores, dim=1)                 # [B, N]
        alpha = alpha * prior_mask.float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)

        context = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # [B, d_k]

        return context


class PyTorchPathBasedNetwork(nn.Module):
    """PyTorch implementation of path-based neural network with action type and vehicle embedding"""
    
    def __init__(self, 
                 num_locations: int,
                 num_vehicles: int,
                 max_capacity: int,
                 embedding_dim: int = 100,
                 lstm_hidden: int = 200,
                 dense_hidden: int = 300,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 num_zones: int = 4,
                 encoder: bool = True,
                 iftransformer: bool = False):
        super(PyTorchPathBasedNetwork, self).__init__()
        
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        self.context_dim = embedding_dim // 2  # d_k for cross-attention
        self.encoder = encoder  # 是否启用sequential encoder
        self.iftransformer = bool(iftransformer)
        self.num_zones = int(max(1, num_zones))
        
        # Location embedding layer
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Vehicle idle time feature processing layer (替代原来的vehicle ID embedding)
        # 输入: idle_time (归一化后的连续值 0-1)
        self.vehicle_idle_time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),  # 将idle时间映射到embedding空间
            nn.ELU(),
            nn.Dropout(0.1)
        )
        
        # Vehicle type embedding layer (EV vs AEV)
        # 0: unknown, 1: EV, 2: AEV
        self.vehicle_type_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=embedding_dim // 4,
            padding_idx=0
        )
        
        # 🆕 Target location embedding layer (for pickup/dropoff/charge station locations)
        # 将target location编码为embedding，与path location共享相同的embedding空间
        self.target_location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim // 4,
            padding_idx=0
        )
        
        # 🆕 Zone ID embedding layer (for geographical zones)
        self.zone_embedding = nn.Embedding(
            num_embeddings=self.num_zones + 1,
            embedding_dim=embedding_dim // 8,
            padding_idx=0
        )

        # Action type embedding layer
        # 0: padding, 1: idle, 2: assign, 3: charge
        self.action_type_embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=embedding_dim // 2,
            padding_idx=0
        )
        
        # Initialize with pretrained embeddings if available
        if pretrained_embeddings is not None:
            self.location_embedding.weight.data.copy_(pretrained_embeddings)
            self.location_embedding.weight.requires_grad = False
        
        # LSTM for path processing
        self.path_lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # embedding + delay
            hidden_size=lstm_hidden,
            batch_first=True
        )
        if self.iftransformer:
            path_attention_heads = 4 if embedding_dim % 4 == 0 else 1
            self.path_attention_input = nn.Linear(embedding_dim + 1, embedding_dim)
            self.path_self_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=path_attention_heads,
                dropout=0.1,
                batch_first=True
            )
            self.path_attention_norm = nn.LayerNorm(embedding_dim)
            self.path_attention_ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim * 2, embedding_dim),
            )
            self.path_attention_ffn_norm = nn.LayerNorm(embedding_dim)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ELU()
        )
        
        # Context embedding for action-specific features
        self.context_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim // 2),  # battery + request_value
            nn.ELU(),
            
        )
        
        # Vehicle-specific feature processing
        self.vehicle_feature_embedding = nn.Sequential(
            nn.Linear(embedding_dim // 4 + embedding_dim // 4, embedding_dim // 2),  # vehicle_id + vehicle_type
            nn.ELU(),
            
        )
        
        # 🆕 Target feature processing (target_location + distance + zone)
        # 输入: target_location_embed (dim//4) + distance (1) + zone_embed (dim//8)
        self.target_feature_embedding = nn.Sequential(
            nn.Linear(embedding_dim // 4 + 1 + embedding_dim // 8, embedding_dim // 2),
            nn.ELU(),
            
        )

        # Request outcome features let assign actions expose where and when the
        # vehicle will be after service, without replacing pickup as the action target.
        self.action_outcome_embedding = nn.Sequential(
            nn.Linear(embedding_dim // 8 + 3, embedding_dim // 2),
            nn.ELU(),
        )
        
        # Cross-attention encoder for sequential decision (sequential_encode.md)
        # prior_dim=5: [location_norm, battery, idle_time, target_location_norm, action_type_id]
        # posterior_dim=5: same features for the current agent
        if self.encoder:
            self.seq_encoder = CrossAttentionEncoder(
                prior_dim=5, posterior_dim=5,
                d_k=self.context_dim, num_zones=self.num_zones
            )

        # State embedding layers - 包含所有特征
        state_input_dim = (lstm_hidden + embedding_dim + 2 +      # path + time + other_agents + num_requests
                          embedding_dim // 2 +                    # action_type_embedding
                          embedding_dim // 2 +                    # context_embedding (battery + request_value)
                          embedding_dim // 2 +                    # vehicle_feature_embedding (vehicle_id + type)
                          embedding_dim // 2 +                    # target_feature_embedding (target + distance + zone)
                          embedding_dim // 2 +                    # action_outcome_embedding (post-action zone/time/distance)
                          1 +                                      # EV salary-ratio state
                          self.num_zones)                         # time_zone_dist (global time-dependent)
        if self.encoder:
            state_input_dim += self.context_dim  # cross-attention context E_dist
        
        self.state_embedding = nn.Sequential(
            nn.Linear(state_input_dim, dense_hidden),
            nn.ELU(),
            
            nn.Linear(dense_hidden, dense_hidden),
            nn.ELU(),
            
            nn.Linear(dense_hidden, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, 
                path_locations: torch.Tensor,
                path_delays: torch.Tensor,
                current_time: torch.Tensor,
                other_agents: torch.Tensor,
                num_requests: torch.Tensor,
                battery_level: torch.Tensor = None,
                request_value: torch.Tensor = None,
                target_distance: torch.Tensor = None,
                target_zoneid: torch.Tensor = None,
                action_type: torch.Tensor = None,
                vehicle_idle_time: torch.Tensor = None,
                vehicle_type: torch.Tensor = None,
                dropout_state_features: torch.Tensor = None,
                post_action_distance: torch.Tensor = None,
                post_action_duration: torch.Tensor = None,
                post_action_zoneid: torch.Tensor = None,
                prior_features: torch.Tensor = None,
                prior_mask: torch.Tensor = None,
                time_zone_dist: torch.Tensor = None,
                rejection_probability: torch.Tensor = None,
                human_response_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            path_locations: [batch_size, seq_len] - Location IDs in path
            path_delays: [batch_size, seq_len, 1] - Delay information
            current_time: [batch_size, 1] - Current time
            other_agents: [batch_size, 1] - Number of other agents nearby
            num_requests: [batch_size, 1] - Number of current requests
            battery_level: [batch_size, 1] - Battery level (0-1), optional
            request_value: [batch_size, 1] - Request value (0-1), optional
            target_distance: [batch_size, 1] - 🆕 Manhattan distance to target (normalized 0-1), optional
            target_zoneid: [batch_size, 1] - 🆕 Zone ID of target location (0-100), optional
            action_type: [batch_size, 1] - Action type (1=idle, 2=assign, 3=charge), optional
            vehicle_idle_time: [batch_size, 1] - 🆕 Vehicle idle time (normalized 0-1), optional
            vehicle_type: [batch_size, 1] - Vehicle type (1=EV, 2=AEV), optional
            post_action_distance: [batch_size, 1] - total distance until the vehicle reaches the post-action zone.
            post_action_duration: [batch_size, 1] - normalized duration until that post-action state.
            post_action_zoneid: [batch_size, 1] - zone after the action completes, e.g. request dropoff.
            prior_features: [batch_size, N, 5] - Prior agents' state-action features (optional).
                            None or empty → zero-padded context (Phase 1 / leader).
            prior_mask: [batch_size, N] - Valid prior-agent mask for padded prior features.
            time_zone_dist: [batch_size, num_zones] - Global time-dependent zone distribution (optional).
                            None → zero contribution.
        """
        batch_size = path_locations.size(0)
        
        # 🆕 提取target location (假设是path的最后一个非零位置)
        # 对于assign动作，target是pickup location；对于charge动作，target是station location
        mask = (path_locations != 0).long()
        seq_lengths = mask.sum(dim=1)  # [batch_size]
        target_location_ids = torch.zeros(batch_size, dtype=torch.long, device=path_locations.device)
        for i in range(batch_size):
            if seq_lengths[i] > 0:
                target_location_ids[i] = path_locations[i, seq_lengths[i]-1]
        
        # 🆕 处理target distance
        if target_distance is None:
            target_distance = torch.zeros(batch_size, 1).to(path_locations.device)
        
        # 🆕 处理target zone ID
        if target_zoneid is None:
            target_zoneid = torch.zeros(batch_size, 1, dtype=torch.long).to(path_locations.device)
        else:
            target_zoneid = target_zoneid.long()

        if post_action_distance is None:
            post_action_distance = torch.zeros(batch_size, 1, device=path_locations.device)
        else:
            post_action_distance = post_action_distance.to(path_locations.device, dtype=torch.float32)
        if post_action_duration is None:
            post_action_duration = torch.zeros(batch_size, 1, device=path_locations.device)
        else:
            post_action_duration = post_action_duration.to(path_locations.device, dtype=torch.float32)
        if post_action_zoneid is None:
            post_action_zoneid = torch.zeros(batch_size, 1, dtype=torch.long, device=path_locations.device)
        else:
            post_action_zoneid = post_action_zoneid.to(path_locations.device).long()
        post_action_zoneid = post_action_zoneid.clamp(0, self.num_zones)
        
        # Get location embeddings
        location_embeds = self.location_embedding(path_locations)  # [batch_size, seq_len, embedding_dim]
        
        # Create mask for padding
        padding_mask = (path_locations == 0)  # [batch_size, seq_len]
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply mask to delays  
        masked_delays = path_delays * mask
        
        # Combine location embeddings with delays
        path_input = torch.cat([location_embeds, masked_delays], dim=-1)  # [batch_size, seq_len, embedding_dim + 1]
        if self.iftransformer:
            # Self-attention lets each path token see the full route before sequential aggregation.
            path_tokens = self.path_attention_input(path_input)  # [batch_size, seq_len, embedding_dim]
            safe_padding_mask = padding_mask.clone()
            empty_paths = safe_padding_mask.all(dim=1)
            if empty_paths.any():
                safe_padding_mask[empty_paths, 0] = False
            attn_out, _ = self.path_self_attention(
                path_tokens,
                path_tokens,
                path_tokens,
                key_padding_mask=safe_padding_mask,
                need_weights=False,
            )
            path_tokens = self.path_attention_norm(path_tokens + attn_out)
            path_tokens = self.path_attention_ffn_norm(path_tokens + self.path_attention_ffn(path_tokens))
            path_tokens = path_tokens * mask
            path_input = torch.cat([path_tokens, masked_delays], dim=-1)

            lstm_out, _ = self.path_lstm(path_input)  # [batch_size, seq_len, lstm_hidden]
            last_indices = (seq_lengths.clamp(min=1) - 1).view(-1, 1, 1)
            last_indices = last_indices.expand(-1, 1, lstm_out.size(-1))
            path_representation = lstm_out.gather(dim=1, index=last_indices).squeeze(1)
            if empty_paths.any():
                path_representation = path_representation.masked_fill(empty_paths.unsqueeze(1), 0.0)
        else:
            # Legacy path encoder, kept for old checkpoint compatibility.
            _, (hidden, _) = self.path_lstm(path_input)  # [batch_size, seq_len, lstm_hidden]
            path_representation = hidden[-1]  # [batch_size, lstm_hidden]
        
        # Process time
        time_embed = self.time_embedding(current_time)  # [batch_size, embedding_dim]
        
        # Handle battery level - 如果没有提供battery_level，使用默认值1.0
        if battery_level is None:
            battery_level = torch.ones(current_time.size()).to(current_time.device)
        
        # Handle request value - 如果没有提供request_value，使用默认值0.0
        if request_value is None:
            request_value = torch.zeros(current_time.size()).to(current_time.device)
        
        # Handle vehicle_idle_time - 如果没有提供，使用默认值0.0（刚开始工作，没有idle时间）
        if vehicle_idle_time is None:
            vehicle_idle_time = torch.zeros(current_time.size()).to(current_time.device)
        
        # Handle vehicle_type - 如果没有提供，默认为EV (1)
        if vehicle_type is None:
            vehicle_type = torch.ones(current_time.size(), dtype=torch.long).to(current_time.device)

        if dropout_state_features is None:
            dropout_state_features = torch.zeros(batch_size, 1, device=path_locations.device)
        elif dropout_state_features.dim() == 1:
            dropout_state_features = dropout_state_features.unsqueeze(1)
        if dropout_state_features.size(1) > 1:
            dropout_state_features = dropout_state_features[:, 1:2]
        dropout_state_features = dropout_state_features.to(path_locations.device, dtype=torch.float32)
        
        # Handle action type - 如果没有提供action_type，尝试从路径推断
        if action_type is None:
            # 从路径模式推断action type
            # idle: 路径中第一个位置 == 第二个位置
            # assign/charge: 路径中第一个位置 != 第二个位置
            is_idle = (path_locations[:, 0] == path_locations[:, 1])  # 保持为布尔张量
            action_type = torch.where(is_idle, 
                                    torch.ones(is_idle.size(), dtype=torch.long, device=is_idle.device),  # idle = 1
                                    torch.full(is_idle.size(), 2, dtype=torch.long, device=is_idle.device))  # assign/charge = 2
            action_type = action_type.unsqueeze(1)  # [batch_size, 1]
        
        # Get embeddings
        action_embed = self.action_type_embedding(action_type.squeeze(1))  # [batch_size, embedding_dim//2]
        
        # Process idle time (连续值，不是embedding)
        vehicle_idle_time_embed = self.vehicle_idle_time_embedding(vehicle_idle_time)  # [batch_size, embedding_dim//4]
        vehicle_type_embed = self.vehicle_type_embedding(vehicle_type.squeeze(1))  # [batch_size, embedding_dim//4]
        
        # Process context features (battery + request_value)
        context_features = torch.cat([battery_level, request_value], dim=1)  # [batch_size, 2]
        context_embed = self.context_embedding(context_features)  # [batch_size, embedding_dim//2]
        
        # Process vehicle features (vehicle_idle_time + vehicle_type)
        vehicle_features = torch.cat([vehicle_idle_time_embed, vehicle_type_embed], dim=1)  # [batch_size, embedding_dim//2]
        vehicle_embed = self.vehicle_feature_embedding(vehicle_features)  # [batch_size, embedding_dim//2]
        
        # 🆕 Process target features (target_location + distance + zone)
        target_loc_embed = self.target_location_embedding(target_location_ids)  # [batch_size, embedding_dim//4]
        target_zone_embed = self.zone_embedding(target_zoneid.squeeze(1))  # [batch_size, embedding_dim//8]
        target_features = torch.cat([target_loc_embed, target_distance, target_zone_embed], dim=1)  # [batch_size, dim//4 + 1 + dim//8]
        target_embed = self.target_feature_embedding(target_features)  # [batch_size, embedding_dim//2]

        post_action_zone_embed = self.zone_embedding(post_action_zoneid.squeeze(1))
        post_action_type = action_type.to(path_locations.device, dtype=torch.float32) / 2.0
        outcome_features = torch.cat(
            [post_action_zone_embed, post_action_distance, post_action_duration, post_action_type],
            dim=1,
        )
        outcome_embed = self.action_outcome_embedding(outcome_features)
        
        # Cross-attention context (sequential_encode.md)
        if self.encoder:
            # Build posterior_state: 5-dim [location_norm, battery, idle_time, target_norm, action_float]
            loc_norm = path_locations[:, 0:1].float() / max(self.num_locations, 1)  # [B,1]
            tgt_norm = target_location_ids.unsqueeze(1).float() / max(self.num_locations, 1)  # [B,1]
            act_float = action_type.float() if action_type.dim() == 2 else action_type.unsqueeze(1).float()  # [B,1]
            posterior_state_5 = torch.cat([loc_norm, battery_level, vehicle_idle_time, tgt_norm, act_float], dim=1)  # [B,5]
            seq_context = self.seq_encoder(posterior_state_5, prior_features, prior_mask)  # [B, d_k]
        
        # Global time-dependent zone distribution
        if time_zone_dist is None:
            num_zones = self.seq_encoder.num_zones if self.encoder else self.num_zones
            time_zone_dist = torch.zeros(batch_size, num_zones, device=path_locations.device)

        # Combine all features
        feature_list = [
            path_representation,     # [batch_size, lstm_hidden]
            time_embed,             # [batch_size, embedding_dim]
            other_agents,           # [batch_size, 1]
            num_requests,           # [batch_size, 1]
            action_embed,           # [batch_size, embedding_dim//2]
            context_embed,          # [batch_size, embedding_dim//2]
            vehicle_embed,          # [batch_size, embedding_dim//2]
            target_embed,           # [batch_size, embedding_dim//2]
            outcome_embed,          # [batch_size, embedding_dim//2]
            dropout_state_features, # [batch_size, 1]
            time_zone_dist,         # [batch_size, num_zones]  (global time-dependent)
        ]
        if self.encoder:
            feature_list.append(seq_context)   # [batch_size, context_dim]  (E_dist or zeros)
        if getattr(self, 'acceptance_input_enabled', False):
            if rejection_probability is None or human_response_mask is None:
                if bool(((action_type == 2) & (vehicle_type == 1)).any()):
                    raise ValueError("EV service network input is missing q_reject or response mask")
                rejection_probability = torch.zeros_like(current_time)
                human_response_mask = torch.zeros_like(current_time)
            mask = human_response_mask.to(current_time)
            eligible = ((action_type == 2) & (vehicle_type == 1)).float()
            if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask > eligible):
                raise ValueError('Invalid human response mask')
            q = rejection_probability.to(current_time)
            if not torch.isfinite(q).all() or torch.any((q < 0) | (q > 1)) or torch.any(q * (1 - mask) != 0):
                raise ValueError('Invalid rejection probability/mask pair')
            feature_list.extend([q, mask])
        combined_features = torch.cat(feature_list, dim=1)  # [batch_size, total_features]
        
        # Get final value prediction
        value = self.state_embedding(combined_features)  # [batch_size, 1]
        
        return value


class PyTorchNeuralNetworkBased(PyTorchValueFunction):
    """PyTorch implementation of neural network-based value function"""
    
    def __init__(self,
                 envt: Environment,
                 num_locations: int,
                 max_capacity: int,
                 load_model_path: str = '',
                 log_dir: str = 'logs/nn_based',
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 batch_size_fit: int = 32,
                 batch_size_predict: int = 1024,
                 target_update_tau: float = 0.001,
                 device: str = 'cpu'):
        
        super().__init__(log_dir, device)
        
        # Environment and hyperparameters
        self.envt = envt
        self.num_locations = num_locations
        self.max_capacity = max_capacity
        self.gamma = gamma
        self.batch_size_fit = batch_size_fit
        self.batch_size_predict = batch_size_predict
        self.target_update_tau = target_update_tau
        
        # Initialize networks
        self.value_network = self._init_network().to(self.device)
        self.target_network = self._init_network().to(self.device)
        
        # Load pretrained model if specified
        if load_model_path and PathlibPath(load_model_path).exists():
            self.load_model(load_model_path)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.value_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        buffer_size = max(int(1e6 / getattr(envt, 'NUM_AGENTS', 10)), 10000)
        self.replay_buffer = PyTorchReplayBuffer(buffer_size, str(self.device))
        
    def _init_network(self) -> PyTorchPathBasedNetwork:
        """Initialize the neural network"""
        # Try to load pretrained embeddings
        pretrained_embeddings = None
        if hasattr(self.envt, 'DATA_DIR'):
            embedding_path = PathlibPath(self.envt.DATA_DIR) / 'embedding_weights.pkl'
            if embedding_path.exists():
                try:
                    with open(embedding_path, 'rb') as f:
                        weights = pickle.load(f)
                    pretrained_embeddings = torch.FloatTensor(weights[0])
                except Exception as e:
                    logging.warning(f"Failed to load pretrained embeddings: {e}")
        
        return PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            max_capacity=self.max_capacity,
            pretrained_embeddings=pretrained_embeddings
        )
    
    def _format_input_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Format experiences into network input tensors"""
        batch_size = len(experiences)
        max_seq_len = self.max_capacity * 2 + 1
        
        # Initialize tensors
        path_locations = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        path_delays = torch.zeros(batch_size, max_seq_len, 1, dtype=torch.float32)
        current_times = torch.zeros(batch_size, 1, dtype=torch.float32)
        other_agents = torch.zeros(batch_size, 1, dtype=torch.float32)
        num_requests = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        for i, experience in enumerate(experiences):
            # Extract time information
            if hasattr(experience, 'time'):
                normalized_time = self._normalize_time(experience.time)
                current_times[i, 0] = normalized_time
            
            # Extract request count
            if hasattr(experience, 'num_requests'):
                normalized_requests = experience.num_requests / getattr(self.envt, 'NUM_AGENTS', 10)
                num_requests[i, 0] = normalized_requests
            
            # Process agents (simplified - take first agent for demo)
            if hasattr(experience, 'agents') and len(experience.agents) > 0:
                agent = experience.agents[0]
                
                # Extract path information
                if hasattr(agent, 'path') and hasattr(agent.path, 'request_order'):
                    self._extract_path_features(agent, path_locations[i], path_delays[i])
                
                # Count other agents (simplified)
                other_agents[i, 0] = len(experience.agents) / getattr(self.envt, 'NUM_AGENTS', 10)
        
        return {
            'path_locations': path_locations.to(self.device),
            'path_delays': path_delays.to(self.device),
            'current_time': current_times.to(self.device),
            'other_agents': other_agents.to(self.device),
            'num_requests': num_requests.to(self.device)
        }
    
    def _normalize_time(self, time: float) -> float:
        """Normalize time to [0, 1] range"""
        start_time = getattr(self.envt, 'START_EPOCH', 0)
        end_time = getattr(self.envt, 'STOP_EPOCH', 86400)
        return (time - start_time) / (end_time - start_time) if end_time > start_time else 0.0
    
    def _extract_path_features(self, agent: LearningAgent, 
                              path_locations: torch.Tensor, 
                              path_delays: torch.Tensor):
        """Extract path features from agent"""
        # Add current location
        if hasattr(agent, 'position') and hasattr(agent.position, 'next_location'):
            path_locations[0] = agent.position.next_location + 1
            path_delays[0, 0] = 1.0
        
        # Add path nodes (simplified extraction)
        if hasattr(agent, 'path') and hasattr(agent.path, 'request_order'):
            for idx, node in enumerate(agent.path.request_order):
                if idx >= len(path_locations) - 1:
                    break
                
                # Extract location and delay information
                try:
                    location, deadline = agent.path.get_info(node)
                    visit_time = getattr(node, 'expected_visit_time', 0)
                    
                    path_locations[idx + 1] = location + 1
                    # Normalize delay
                    max_delay = getattr(Request, 'MAX_DROPOFF_DELAY', 3600)
                    normalized_delay = (deadline - visit_time) / max_delay
                    path_delays[idx + 1, 0] = normalized_delay
                except Exception:
                    # Handle cases where path info extraction fails
                    break
    
    def get_value(self, experiences: List[Experience], 
                 use_target: bool = False) -> List[List[Tuple[Action, float]]]:
        """Get value estimates for experiences"""
        if not experiences:
            return []
        
        # Format input batch
        inputs = self._format_input_batch(experiences)
        
        # Get network predictions
        network = self.target_network if use_target else self.value_network
        network.eval()
        
        with torch.no_grad():
            values = network(**inputs)  # [batch_size, 1]
            values = values.cpu().numpy().flatten()
        
        # Format output to match expected interface
        scored_actions_all_agents = []
        value_idx = 0
        
        for experience in experiences:
            if hasattr(experience, 'feasible_actions_all_agents'):
                for feasible_actions in experience.feasible_actions_all_agents:
                    scored_actions = []
                    for action in feasible_actions:
                        # Get immediate reward
                        immediate_reward = self._get_immediate_reward(action)
                        
                        # Add discounted future value
                        future_value = values[value_idx] if value_idx < len(values) else 0.0
                        total_value = immediate_reward + self.gamma * future_value
                        
                        scored_actions.append((action, total_value))
                    
                    scored_actions_all_agents.append(scored_actions)
                    value_idx += 1
        
        return scored_actions_all_agents
    
    def _get_immediate_reward(self, action: Action) -> float:
        """Get immediate reward for an action"""
        if hasattr(self.envt, 'get_reward'):
            return self.envt.get_reward(action)
        elif hasattr(action, 'requests'):
            return sum([getattr(req, 'value', 0) for req in action.requests])
        else:
            return 0.0
    
    def remember(self, experience: Experience):
        """Store experience in replay buffer"""
        self.replay_buffer.add(experience)
    
    def update(self, central_agent: CentralAgent, num_samples: int = 3):
        """Update value function using sampled experiences"""
        # Check if enough experiences for training
        min_samples = max(self.batch_size_fit, 100)
        if len(self.replay_buffer) < min_samples:
            return
        
        # Sample experiences
        experiences, weights, indices = self.replay_buffer.sample(num_samples)
        if not experiences:
            return
        
        # Prepare training data
        self.value_network.train()
        
        # Get current value predictions
        inputs = self._format_input_batch(experiences)
        current_values = self.value_network(**inputs)
        
        # Get target values using target network
        target_values = self._compute_target_values(experiences, central_agent)
        
        # Compute loss with importance sampling
        weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        td_errors = F.mse_loss(current_values, target_values, reduction='none')
        weighted_loss = (td_errors * weights_tensor).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 修复梯度裁剪：从1.0增加到10.0，避免过度裁剪
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target()
        
        # Update replay buffer priorities
        priorities = td_errors.detach().cpu().numpy().flatten() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities.tolist())
        
        # Log training statistics
        self.add_to_logs('loss', weighted_loss.item(), self.training_step)
        self.add_to_logs('mean_value', current_values.mean().item(), self.training_step)
        self.training_step += 1
    
    def _compute_target_values(self, experiences: List[Experience], 
                              central_agent: CentralAgent) -> torch.Tensor:
        """Compute target values for training"""
        target_values = []
        
        for experience in experiences:
            # Get next state values using target network
            next_state_values = self.get_value([experience], use_target=True)
            
            # Use central agent to get optimal actions and their values
            if next_state_values and hasattr(central_agent, 'choose_actions'):
                try:
                    optimal_actions = central_agent.choose_actions(
                        next_state_values, 
                        is_training=False
                    )
                    target_value = sum([score for _, score in optimal_actions]) / len(optimal_actions)
                except Exception:
                    target_value = 0.0
            else:
                target_value = 0.0
            
            target_values.append(target_value)
        
        return torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
    
    def _soft_update_target(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.value_network.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + 
                (1 - self.target_update_tau) * target_param.data
            )
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'value_network': self.value_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'num_vehicles': self.value_network.num_vehicles  # 保存车辆数量
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state with automatic embedding expansion for different vehicle numbers"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # 获取checkpoint中的车辆数量
        saved_num_vehicles = checkpoint.get('num_vehicles', None)
        current_num_vehicles = self.value_network.num_vehicles
        
        # 如果车辆数量不同，需要扩展vehicle_embedding层
        if saved_num_vehicles is not None and saved_num_vehicles != current_num_vehicles:
            print(f"⚠️  Vehicle number mismatch: trained with {saved_num_vehicles} vehicles, "
                  f"loading to {current_num_vehicles} vehicles")
            print(f"🔧 Automatically expanding vehicle embedding layer...")
            
            # 扩展value_network的vehicle_embedding
            checkpoint['value_network'] = self._expand_vehicle_embedding(
                checkpoint['value_network'], 
                saved_num_vehicles, 
                current_num_vehicles,
                'vehicle_embedding.weight'
            )
            
            # 扩展target_network的vehicle_embedding
            checkpoint['target_network'] = self._expand_vehicle_embedding(
                checkpoint['target_network'],
                saved_num_vehicles,
                current_num_vehicles, 
                'vehicle_embedding.weight'
            )
            
            print(f"✓ Vehicle embedding expanded from {saved_num_vehicles+1} to {current_num_vehicles+1} embeddings")
        
        # 加载模型
        self.value_network.load_state_dict(checkpoint['value_network'], strict=False)
        self.target_network.load_state_dict(checkpoint['target_network'], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except (ValueError, RuntimeError) as e:
            print(f"⚠️  Optimizer state incompatible with current network; using fresh optimizer. ({e})")
        self.training_step = checkpoint.get('training_step', 0)
        
    def _expand_vehicle_embedding(self, state_dict, old_num_vehicles, new_num_vehicles, key):
        """扩展vehicle_embedding层以支持更多车辆"""
        if key not in state_dict:
            return state_dict
            
        old_embedding = state_dict[key]  # Shape: (old_num_vehicles+1, embedding_dim)
        old_num_embeddings, embedding_dim = old_embedding.shape
        new_num_embeddings = new_num_vehicles + 1
        
        if new_num_embeddings <= old_num_embeddings:
            # 新环境车辆数更少或相等，直接截取
            state_dict[key] = old_embedding[:new_num_embeddings, :]
        else:
            # 新环境车辆数更多，需要扩展
            # 创建新的embedding矩阵
            new_embedding = torch.zeros(new_num_embeddings, embedding_dim)
            
            # 复制已有的embedding权重
            new_embedding[:old_num_embeddings, :] = old_embedding
            
            # 为新的vehicle_id初始化embedding（使用已有embedding的均值+小随机扰动）
            if old_num_embeddings > 1:  # 排除padding
                existing_mean = old_embedding[1:, :].mean(dim=0)
                existing_std = old_embedding[1:, :].std(dim=0)
                for i in range(old_num_embeddings, new_num_embeddings):
                    # 使用均值+随机扰动初始化
                    new_embedding[i, :] = existing_mean + torch.randn(embedding_dim) * existing_std * 0.1
            
            state_dict[key] = new_embedding
            
        return state_dict


# Compatibility aliases for existing code
ValueFunction = PyTorchValueFunction
RewardPlusDelay = PyTorchRewardPlusDelay
ImmediateReward = lambda: PyTorchRewardPlusDelay(delay_coefficient=0)
NeuralNetworkBased = PyTorchNeuralNetworkBased
PathBasedNN = PyTorchNeuralNetworkBased


def main():
    """Test the PyTorch value function implementation"""
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy environment for testing
    class DummyEnvironment:
        NUM_AGENTS = 5
        NUM_LOCATIONS = 100
        MAX_CAPACITY = 4
        START_EPOCH = 0
        STOP_EPOCH = 86400
        DATA_DIR = 'data/'
        
        def get_reward(self, action):
            return random.uniform(0, 10)
    
    env = DummyEnvironment()
    
    # Test PyTorch value function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    value_function = PyTorchNeuralNetworkBased(
        envt=env,
        num_locations=env.NUM_LOCATIONS,
        max_capacity=env.MAX_CAPACITY,
        device=device
    )
    
    print("PyTorch Value Function initialized successfully!")
    print(f"Network parameters: {sum(p.numel() for p in value_function.value_network.parameters())}")
    print(f"Device: {device}")


# =============================================================================
# DQN Implementation for Benchmark Comparison
# =============================================================================

class DQNActionNetwork(nn.Module):
    """
    Deep Q-Network for action selection in vehicle dispatch
    Provides a benchmark for comparison with ILP-ADP approach
    """
    def __init__(self, state_dim=64, action_dim=32, hidden_dim=128, device='cpu'):
        super(DQNActionNetwork, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature encoders for different input modalities
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim//2),  # vehicle_id, type, location, battery, etc.
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        
        self.request_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim//2),  # pickup, dropoff, time, value, etc.
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim//4),  # num_vehicles, num_requests, time, etc.
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//8)
        )
        
        # Main DQN network (Dueling architecture)
        total_feature_dim = hidden_dim//4 + hidden_dim//4 + hidden_dim//8
        
        self.feature_layer = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dueling DQN: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        self.to(device)
    
    def forward(self, vehicle_features, request_features, global_features):
        """
        Forward pass through the DQN
        
        Args:
            vehicle_features: Tensor of vehicle state features
            request_features: Tensor of request features
            global_features: Tensor of global environment features
        
        Returns:
            Q-values for all possible actions
        """
        # Encode different feature types
        vehicle_encoded = self.vehicle_encoder(vehicle_features)
        request_encoded = self.request_encoder(request_features)
        global_encoded = self.global_encoder(global_features)
        
        # Concatenate all features
        combined_features = torch.cat([vehicle_encoded, request_encoded, global_encoded], dim=-1)
        
        # Main feature processing
        features = self.feature_layer(combined_features)
        
        # Dueling streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (dueling DQN formula)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values
    
    def get_action(self, vehicle_features, request_features, global_features, epsilon=0.0):
        """
        Select action using epsilon-greedy policy
        
        Args:
            vehicle_features: Vehicle state features
            request_features: Request features
            global_features: Global environment features
            epsilon: Exploration probability
        
        Returns:
            Selected action index and Q-values
        """
        if random.random() < epsilon:
            # Random action for exploration
            action = random.randint(0, self.action_dim - 1)
            with torch.no_grad():
                q_values = self.forward(vehicle_features, request_features, global_features)
            return action, q_values
        else:
            # Greedy action selection
            with torch.no_grad():
                q_values = self.forward(vehicle_features, request_features, global_features)
                action = q_values.argmax(dim=-1).item()
            return action, q_values


class DQNExperienceReplay:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for vehicle dispatch decision making
    Serves as benchmark comparison for ILP-ADP approach
    """
    def __init__(self, state_dim=64, action_dim=32, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1000,
                 target_update=100, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0
        
        # Networks
        self.policy_net = DQNActionNetwork(state_dim, action_dim, device=device)
        self.target_net = DQNActionNetwork(state_dim, action_dim, device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = DQNExperienceReplay(capacity=10000)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_q_values': [],
            'losses': []
        }
    
    def get_epsilon(self):
        """Calculate current epsilon for exploration"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 math.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon
    
    def select_action(self, vehicle_features, request_features, global_features, training=True, force_idle_constraint=False):
        """
        Select action for given state
        
        Args:
            vehicle_features: Vehicle state tensor
            request_features: Request features tensor  
            global_features: Global environment tensor
            training: Whether in training mode (affects exploration)
            force_idle_constraint: Whether to prioritize idle actions due to constraint
        
        Returns:
            action: Selected action index
            q_values: Q-values for all actions
        """
        epsilon = self.get_epsilon() if training else 0.0
        self.steps_done += 1
        
        # 如果有idle约束，修改动作选择策略
        if force_idle_constraint:
            # 获取所有Q值
            action, q_values = self.policy_net.get_action(vehicle_features, request_features, 
                                            global_features, epsilon=0.0)  # 不使用随机探索
            
            # 识别idle/wait动作的索引（通常在动作空间的末尾）
            # 假设动作空间：[assign_0, assign_1, ..., assign_9, rebalance_0, ..., rebalance_9, charge_0, ..., charge_4, wait_0, wait_1, idle]
            idle_action_indices = list(range(28, 32))  # wait和idle动作的索引范围
            
            # 计算idle动作的平均Q值
            idle_q_values = [q_values[i].item() if i < len(q_values) else -1000 for i in idle_action_indices]
            best_idle_idx = idle_action_indices[idle_q_values.index(max(idle_q_values))]
            
            # 使用高概率选择idle动作，但仍保留一定的探索性
            if training and epsilon > 0:
                # 80%概率选择最佳idle动作，20%概率正常选择
                if torch.rand(1).item() < 0.8:
                    return best_idle_idx, q_values
            else:
                # 非训练模式下直接选择最佳idle动作
                return best_idle_idx, q_values
        
        # 正常的动作选择
        return self.policy_net.get_action(vehicle_features, request_features, 
                                        global_features, epsilon)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=32):
        """
        Perform one training step
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            loss: Training loss value
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        batch_states = {
            'vehicle': torch.stack([s['vehicle'] for s in states]).to(self.device),
            'request': torch.stack([s['request'] for s in states]).to(self.device),
            'global': torch.stack([s['global'] for s in states]).to(self.device)
        }
        
        batch_next_states = {
            'vehicle': torch.stack([s['vehicle'] for s in next_states]).to(self.device),
            'request': torch.stack([s['request'] for s in next_states]).to(self.device),
            'global': torch.stack([s['global'] for s in next_states]).to(self.device)
        }
        
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(
            batch_states['vehicle'], 
            batch_states['request'], 
            batch_states['global']
        ).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(
                batch_next_states['vehicle'],
                batch_next_states['request'], 
                batch_next_states['global']
            ).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Store training stats
        self.training_stats['losses'].append(loss.item())
        avg_q = current_q_values.mean().item()
        self.training_stats['avg_q_values'].append(avg_q)
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'steps_done': self.steps_done
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.steps_done = checkpoint['steps_done']


def create_dqn_state_features(environment, vehicle_id, current_time=0.0):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get vehicle information
    vehicle = environment.vehicles.get(vehicle_id, {})
    vehicle_location = vehicle.get('location', 0)
    vehicle_type = vehicle.get('type', 1)
    battery_level = vehicle.get('battery', 1.0)
    is_idle = vehicle.get('idle', True)
    
    # Vehicle features: [id, type, location, battery, is_idle, x_coord, y_coord, capacity]
    vehicle_features = torch.tensor([
        vehicle_id / 100.0,  # Normalized vehicle ID
        vehicle_type / 2.0,  # Normalized vehicle type (1 or 2)
        vehicle_location / float(environment.NUM_LOCATIONS if hasattr(environment, 'NUM_LOCATIONS') else 50),
        battery_level,
        1.0 if is_idle else 0.0,
        (vehicle_location % 10) / 10.0,  # X coordinate (assuming grid layout)
        (vehicle_location // 10) / 10.0,  # Y coordinate
        environment.MAX_CAPACITY / 10.0 if hasattr(environment, 'MAX_CAPACITY') else 1.0
    ], dtype=torch.float32).to(device)
    
    # Request features (using active requests if available)
    active_requests = getattr(environment, 'active_requests', None)
    pickup_loc = 0
    dropoff_loc = 0
    request_value = 0.0
    request_time = current_time
    distance = 0
    urgency = 0.0

    # Convert to a list if it's a dict; environment.active_requests is often a dict {id: Request}
    if active_requests:
        if isinstance(active_requests, dict):
            req_list = list(active_requests.values())
        else:
            req_list = list(active_requests)

        if len(req_list) > 0:
            # Use the first available request as a simple representative feature
            req = req_list[0]
            pickup_loc = getattr(req, 'pickup', 0)
            dropoff_loc = getattr(req, 'dropoff', 0)
            # Prefer final_value if available, otherwise fallback to value
            request_value = getattr(req, 'final_value', getattr(req, 'value', 1.0))
            # We don't track a per-request creation time here; default to current_time
            request_time = current_time
            distance = abs(int(pickup_loc) - int(vehicle_location)) if isinstance(vehicle_location, (int, float)) else 0
            # Simple urgency proxy (no timestamp available): constant mid-level urgency
            urgency = 0.5
    
    # Request features: [pickup, dropoff, value, time, distance, urgency]
    request_features = torch.tensor([
        pickup_loc / float(environment.NUM_LOCATIONS if hasattr(environment, 'NUM_LOCATIONS') else 50),
        dropoff_loc / float(environment.NUM_LOCATIONS if hasattr(environment, 'NUM_LOCATIONS') else 50),
        request_value / 100.0,  # Normalized request value
        (current_time % 1440) / 1440.0,  # Normalized time (assuming daily cycle)
        distance / float(environment.NUM_LOCATIONS if hasattr(environment, 'NUM_LOCATIONS') else 50),
        urgency
    ], dtype=torch.float32).to(device)
    
    # Global features: [num_vehicles, num_requests, current_time, avg_battery]
    num_vehicles = len(environment.vehicles) if hasattr(environment, 'vehicles') else 1
    num_requests = len(active_requests)
    avg_battery = sum(v.get('battery', 1.0) for v in environment.vehicles.values()) / num_vehicles if hasattr(environment, 'vehicles') and environment.vehicles else 1.0
    
    global_features = torch.tensor([
        num_vehicles / 100.0,  # Normalized number of vehicles
        num_requests / 50.0,   # Normalized number of requests
        (current_time % 1440) / 1440.0,  # Normalized time
        avg_battery
    ], dtype=torch.float32).to(device)
    
    return {
        'vehicle': vehicle_features,
        'request': request_features,
        'global': global_features
    }


if __name__ == "__main__":
    main()
