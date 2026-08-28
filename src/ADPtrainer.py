"""
ADP Training Module - 电动车充电优化训练器
"""
import time
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import time
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
import re
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from src.ChargingIntegrationVisualization import ChargingIntegrationVisualization
# 导入配置管理器
from config.config_manager import ConfigManager, get_config, get_training_config, get_sampling_config

# 导入核心组件
from .Environment import ChargingIntegratedEnvironment
from .ValueFunction_pytorch_bayes import PyTorchChargingValueFunction
from .Action import Action, ChargingAction, ServiceAction
from .Request import Request
from .charging_station import ChargingStationManager, ChargingStation
from .charging_wait_metrics import aggregate_wait_metrics
from .CentralAgent import CentralAgent
from .SpatialVisualization import SpatialVisualization
from .recourse.critics import (
    enforce_critic_identity,
    uses_shared_critic,
    wire_recourse_critics,
)
from .recourse.training import training_readiness


class ADPTrainer:
    """ADP训练器类 - 负责电动车充电优化的强化学习训练"""

    @staticmethod
    def _should_train_value_function(adpvalue: float, trainnetwork: bool) -> bool:
        """Return whether ADP value functions should receive gradient updates.

        Assignment is the policy used to collect transitions, not a requirement
        for training the value function.  In particular, heuristic assignment
        can rank actions with the current Q estimates and train from the
        resulting transitions in exactly the same way as the exact solvers.
        """
        return float(adpvalue) > 0.0 and bool(trainnetwork)

    @staticmethod
    def _resolve_checkpoint_assign_tag(
        assignmentgurobi: bool,
        load_checkpoint_assign_tag: str | None = None,
    ) -> str:
        """Resolve the assignment backend used to train a checkpoint."""
        if load_checkpoint_assign_tag is None:
            return "gurobi" if assignmentgurobi else "heu"
        if load_checkpoint_assign_tag not in {"gurobi", "heu"}:
            raise ValueError(
                "load_checkpoint_assign_tag must be one of: gurobi, heu"
            )
        return load_checkpoint_assign_tag
    
    def __init__(self, config_manager=None, log_file="logs/output.log"):
        """
        初始化训练器
        
        Args:
            config_manager: 配置管理器实例
            log_file: 日志文件路径，默认为logs/output.log
        """
        self.config_manager = config_manager or ConfigManager()
        self.training_config = self.config_manager.get_training_config()
        self.env_config = self.config_manager.get_environment_config()
        self.sampling_config = self.config_manager.get_sampling_config()
        self.adp_value = self.training_config.get('adp_value', 0)
        self.assignmentgurobi = self.training_config.get('assignmentgurobi', True)
        # 训练状态  
        self.env = None
        self.batch_size = self.training_config.get('batch_size', 256)
        self.value_function = None
        self.training_history = {
            'episode_rewards': [],
            'training_losses': [],
            'q_values': [],
            'exploration_rates': []
        }
        
        # 配置日志输出
        self.log_file = log_file
        self._setup_logging()
        
        self.logger.info("🚀 ADPTrainer初始化完成")
        self.logger.info(f"   - 配置加载: {self.config_manager.config_path}")
        self.logger.info(f"   - 日志文件: {self.log_file}")
    
    def _setup_logging(self):
        """配置日志输出到文件和控制台"""
        # 创建日志目录
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger('ADPTrainer')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的handlers
        self.logger.handlers = []
        
        # 文件handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(message)s')  # 控制台输出包含时间
        console_handler.setFormatter(console_formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _flush_logger_handlers(self):
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    
    def setup_environment(self, num_vehicles=None, num_stations=None, simulation_period=50, days_per_week=7, episode_length=None, daily_drop_off=False):
        """
        设置训练环境
        
        Args:
            num_vehicles: 车辆数量，默认从配置获取
            num_stations: 充电站数量，默认从配置获取
        """
        num_vehicles = num_vehicles or self.env_config.get('max_vehicles', 40)
        num_stations = num_stations or self.env_config.get('max_charging_stations', 12)
        
        episode_days = None if episode_length is None else max(1, int(np.ceil(episode_length / max(simulation_period, 1))))
        self.env = ChargingIntegratedEnvironment(
            num_vehicles=num_vehicles,
            num_stations=num_stations,
            simulation_period=simulation_period,
            days_per_week=days_per_week,
            episode_days=episode_days,
            daily_drop_off=daily_drop_off,
        )
        
        print(f"✓ 环境设置完成: {num_vehicles}辆车, {num_stations}个充电站")
        return self.env
    
    def setup_value_function(self):
        """
        设置价值函数
        
        Args:
            adp_value: ADP参数值
            use_neural_network: 是否使用神经网络
        """
        use_neural_network = self.adp_value > 0
        if use_neural_network and self.adp_value > 0:
            network_config = self.config_manager.get_network_config()
            
            self.value_function = PyTorchChargingValueFunction(
                grid_size=self.env.grid_size,
                num_vehicles=self.env.num_vehicles,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                episode_length=self.env.episode_length,
                max_requests=1000,
            )
            
            # 设置价值函数到环境
            self.env.set_value_function(self.value_function)
            
            print(f"✓ 神经网络价值函数初始化完成")
            print(f"   - 网络参数数量: {sum(p.numel() for p in self.value_function.network.parameters())}")
            print(f"   - 设备: {self.value_function.device}")
        else:
            self.value_function = None
            print(f"✓ 不使用神经网络 (ADP={self.adp_value})")

        return self.value_function

    @staticmethod
    def _set_random_seeds(seed: int = 42) -> None:
        """统一设置随机种子，保证可复现。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"✓ Random seeds set to {seed} for all generators (Python, NumPy, PyTorch)")

    
    
    @staticmethod
    def find_latest_checkpoint(
        checkpoint_dir: str,
        prefer_best: bool = False,
        prefer_best_loss: bool = False,
    ) -> str:
        """返回指定目录下 episode 号最大的检查点路径。
        默认优先选择 latest；当 prefer_best=True 时优先选择 best reward；
        当 prefer_best_loss=True 时优先选择 best loss。
        在同一标签下优先选择 full_state (包含完整训练状态)，
        若不存在则回退到 network_only。
        """
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            return None

        prefix_groups = [
            ("full_state_episode_", "network_episode_"),
            ("best_full_state_episode_", "best_network_episode_"),
            ("best_loss_full_state_episode_", "best_loss_network_episode_"),
        ]
        if prefer_best_loss:
            prefix_groups = [prefix_groups[2], prefix_groups[1], prefix_groups[0]]
        elif prefer_best:
            prefix_groups.reverse()
            prefix_groups = [prefix_groups[1], prefix_groups[2], prefix_groups[0]]

        for prefixes in prefix_groups:
            for prefix in prefixes:
                pattern = re.compile(re.escape(prefix) + r"(\d+)\.pth$")
                latest_path = None
                latest_episode = -1
                for file_path in dir_path.glob(f"{prefix}*.pth"):
                    match = pattern.match(file_path.name)
                    if not match:
                        continue
                    episode_num = int(match.group(1))
                    if episode_num > latest_episode:
                        latest_episode = episode_num
                        latest_path = file_path
                if latest_path is not None:
                    return str(latest_path)

        return None

    @staticmethod
    def _checkpoint_identity(checkpoint_path: str) -> dict:
        """Read the lightweight identity used to keep EV/AEV checkpoints paired."""
        path = Path(checkpoint_path)
        match = re.search(r"(?:^|_)(episode_)(\d+)\.pth$", path.name)
        episode = int(match.group(2)) if match else -1
        tag = "latest"
        for candidate in ("best_combined_loss", "best_loss", "best_ev", "best_aev", "best"):
            if path.name.startswith(f"{candidate}_"):
                tag = candidate
                break
        identity = {"episode": episode, "checkpoint_tag": tag, "pair_id": None}
        if "full_state" not in path.name:
            return identity
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            identity["episode"] = int(payload.get("episode", episode))
            identity["checkpoint_tag"] = str(payload.get("checkpoint_tag", tag))
            identity["pair_id"] = payload.get("checkpoint_pair_id")
            identity["combined_reward"] = payload.get("combined_reward")
            identity["training_run_id"] = payload.get("training_run_id")
        return identity

    @classmethod
    def find_checkpoint_pair(
        cls,
        ev_checkpoint_dir: str,
        aev_checkpoint_dir: str,
        prefer_best: bool = False,
        prefer_best_loss: bool = False,
    ) -> tuple[str | None, str | None]:
        """Return EV/AEV checkpoints from exactly the same tag and episode.

        New checkpoints additionally carry a shared pair id.  Legacy checkpoints
        remain loadable when their tag and episode match, but independently
        selected EV-best and AEV-best files can never be combined here.
        """
        groups = [
            ("latest", "full_state_episode_", "network_episode_"),
            ("best", "best_full_state_episode_", "best_network_episode_"),
            ("best_loss", "best_loss_full_state_episode_", "best_loss_network_episode_"),
        ]
        if prefer_best_loss:
            groups = [groups[2], groups[1], groups[0]]
        elif prefer_best:
            groups = [groups[1], groups[0], groups[2]]

        ev_root = Path(ev_checkpoint_dir)
        aev_root = Path(aev_checkpoint_dir)
        if not ev_root.is_dir() or not aev_root.is_dir():
            return None, None

        def candidates(root: Path, full_prefix: str, network_prefix: str) -> dict[int, Path]:
            selected: dict[int, Path] = {}
            for prefix in (network_prefix, full_prefix):
                pattern = re.compile(re.escape(prefix) + r"(\d+)\.pth$")
                for path in root.glob(f"{prefix}*.pth"):
                    match = pattern.match(path.name)
                    if match:
                        # full-state files are visited second and take priority.
                        selected[int(match.group(1))] = path
            return selected

        for _tag, full_prefix, network_prefix in groups:
            ev_candidates = candidates(ev_root, full_prefix, network_prefix)
            aev_candidates = candidates(aev_root, full_prefix, network_prefix)
            for episode in sorted(set(ev_candidates) & set(aev_candidates), reverse=True):
                ev_path = ev_candidates[episode]
                aev_path = aev_candidates[episode]
                ev_identity = cls._checkpoint_identity(str(ev_path))
                aev_identity = cls._checkpoint_identity(str(aev_path))
                if (
                    ev_identity["episode"] != aev_identity["episode"]
                    or ev_identity["checkpoint_tag"] != aev_identity["checkpoint_tag"]
                ):
                    continue
                ev_pair_id = ev_identity.get("pair_id")
                aev_pair_id = aev_identity.get("pair_id")
                if ev_pair_id is not None or aev_pair_id is not None:
                    if not ev_pair_id or ev_pair_id != aev_pair_id:
                        continue
                return str(ev_path), str(aev_path)
        return None, None
    
    
    @staticmethod
    def _save_q_network_checkpoint(
        value_function,
        episode: int,
        checkpoint_dir: str = "checkpoints/q_networks",
        checkpoint_tag: str = "latest",
        checkpoint_metadata: dict | None = None,
    ) -> dict:
        """保存Q-network检查点，并按标签保留最近一次保存。"""
        import os
        import glob

        if value_function is None:
            print("❌ Value function为空，无法保存")
            return {}

        os.makedirs(checkpoint_dir, exist_ok=True)

        file_prefixes = {
            "latest": {
                'full_state': "full_state_episode_",
                'network_only': "network_episode_",
            },
            "best": {
                'full_state': "best_full_state_episode_",
                'network_only': "best_network_episode_",
            },
            "best_loss": {
                'full_state': "best_loss_full_state_episode_",
                'network_only': "best_loss_network_episode_",
            },
            "best_combined_loss": {
                'full_state': "best_combined_loss_full_state_episode_",
                'network_only': "best_combined_loss_network_episode_",
            },
            "best_ev": {
                'full_state': "best_ev_full_state_episode_",
                'network_only': "best_ev_network_episode_",
            },
            "best_aev": {
                'full_state': "best_aev_full_state_episode_",
                'network_only': "best_aev_network_episode_",
            },
        }
        prefixes = file_prefixes.get(checkpoint_tag)
        if prefixes is None:
            print(f"❌ 未知checkpoint_tag: {checkpoint_tag}")
            return {}

        # 删除同标签的旧检查点文件，只保留该标签下最新的
        old_files = glob.glob(os.path.join(checkpoint_dir, f"{prefixes['full_state']}*.pth"))
        old_files.extend(glob.glob(os.path.join(checkpoint_dir, f"{prefixes['network_only']}*.pth")))
        for old_file in old_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"⚠️ 删除旧检查点失败: {e}")

        paths = {
            'full_state': os.path.join(checkpoint_dir, f"{prefixes['full_state']}{episode}.pth"),
            'network_only': os.path.join(checkpoint_dir, f"{prefixes['network_only']}{episode}.pth"),
        }

        try:
            # 保存完整状态
            save_dict = {
                    'episode': episode,
                    'checkpoint_tag': checkpoint_tag,
                    'training_step': getattr(value_function, 'training_step', 0),
                    'network_state_dict': getattr(value_function.network, 'state_dict', lambda: {})(),
                    'target_network_state_dict': getattr(value_function.target_network, 'state_dict', lambda: {})(),
                    'optimizer_state_dict': getattr(value_function.optimizer, 'state_dict', lambda: {})(),
                    'epsilon': getattr(value_function, 'epsilon', 0.1),
                    'training_losses': list(getattr(value_function, 'training_losses', [])),
                    'normalized_td_losses': list(getattr(value_function, 'normalized_td_losses', [])),
                    'td_error_history': list(getattr(value_function, 'td_error_history', [])),
                    'q_values_history': list(getattr(value_function, 'q_values_history', [])),
            }
            if checkpoint_metadata:
                save_dict.update(dict(checkpoint_metadata))
            # Save time_zone_dist_predictor if it exists
            if (
                getattr(value_function, 'time_zone_dist_predictor', None) is not None
            ):
                save_dict['time_zone_dist_predictor_state_dict'] = value_function.time_zone_dist_predictor.state_dict()
                if getattr(value_function, 'time_zone_dist_optimizer', None) is not None:
                    save_dict['time_zone_dist_optimizer_state_dict'] = value_function.time_zone_dist_optimizer.state_dict()
            if getattr(value_function, 'rejection_predictor', None) is not None:
                save_dict['rejection_predictor_state_dict'] = value_function.rejection_predictor.state_dict()
                save_dict['rejection_predictor_trained'] = bool(
                    getattr(value_function, 'rejection_predictor_trained', False)
                )
                save_dict['rejection_training_losses'] = list(
                    getattr(value_function, 'rejection_training_losses', [])
                )
                if getattr(value_function, 'rejection_optimizer', None) is not None:
                    save_dict['rejection_optimizer_state_dict'] = value_function.rejection_optimizer.state_dict()
            if hasattr(value_function, 'extra_checkpoint_state'):
                extra_state = value_function.extra_checkpoint_state()
                if extra_state:
                    save_dict['extra_value_function_state'] = extra_state
            for role in ('leader', 'follower'):
                time_predictor = getattr(value_function, f'time_zone_dist_predictor_{role}', None)
                time_optimizer = getattr(value_function, f'time_zone_dist_optimizer_{role}', None)
                if time_predictor is not None:
                    save_dict[f'time_zone_dist_predictor_{role}_state_dict'] = time_predictor.state_dict()
                    if time_optimizer is not None:
                        save_dict[f'time_zone_dist_optimizer_{role}_state_dict'] = time_optimizer.state_dict()
                zone_predictor = getattr(value_function, f'zone_dist_predictor_{role}', None)
                zone_optimizer = getattr(value_function, f'zone_dist_optimizer_{role}', None)
                if zone_predictor is not None:
                    save_dict[f'zone_dist_predictor_{role}_state_dict'] = zone_predictor.state_dict()
                    if zone_optimizer is not None:
                        save_dict[f'zone_dist_optimizer_{role}_state_dict'] = zone_optimizer.state_dict()
            torch.save(save_dict, paths['full_state'])
            
            # 单独保存主网络权重
            torch.save(
                getattr(value_function.network, 'state_dict', lambda: {})(),
                paths['network_only']
            )

            print(f"✓ 价值网络{checkpoint_tag}检查点已保存到 {checkpoint_dir}")
            print(f"  - 完整状态: {os.path.basename(paths['full_state'])}")
            print(f"  - 网络权重: {os.path.basename(paths['network_only'])}")
            artifacts = getattr(value_function, "checkpoint_artifact_paths", [])
            artifacts.extend(paths.values())
            value_function.checkpoint_artifact_paths = list(dict.fromkeys(artifacts))
            return paths
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
            return {}

    @staticmethod
    def _analyze_vehicle_visit_patterns(env) -> dict:
        """统计车辆访问热点信息（从原测试脚本迁移）。"""
        vehicle_visit_stats = {}
        hotspots = [
            (env.grid_size // 4, env.grid_size // 4),
            (3 * env.grid_size // 4, env.grid_size // 4),
            (env.grid_size // 2, 3 * env.grid_size // 4),
        ]

        for vehicle_id, vehicle in env.vehicles.items():
            position_history = env.vehicle_position_history.get(vehicle_id, [])
            if not position_history:
                current_coords = vehicle['coordinates']
                location_counts = {str(current_coords): 1}
            else:
                location_counts = {}
                for entry in position_history:
                    coords_str = str(entry['coords'])
                    location_counts[coords_str] = location_counts.get(coords_str, 0) + 1

            if location_counts:
                most_visited_location = max(location_counts, key=location_counts.get)
                most_visited_coords = eval(most_visited_location)
                visit_count = location_counts[most_visited_location]
                unique_locations = len(location_counts)
                total_visits = sum(location_counts.values())
                diversity_score = unique_locations / total_visits if total_visits > 0 else 0

                avg_distance_from_hotspots = 0
                hotspot_visits = 0
                for location_str, count in location_counts.items():
                    coords = eval(location_str)
                    min_distance_to_hotspot = min(
                        abs(coords[0] - hx) + abs(coords[1] - hy) for hx, hy in hotspots
                    )
                    avg_distance_from_hotspots += min_distance_to_hotspot * count
                    if min_distance_to_hotspot <= 2:
                        hotspot_visits += count

                total_counts = sum(location_counts.values()) or 1
                avg_distance_from_hotspots /= total_counts

                vehicle_visit_stats[vehicle_id] = {
                    'most_visited_location': most_visited_coords,
                    'visit_count': visit_count,
                    'unique_locations': unique_locations,
                    'total_visits': total_visits,
                    'diversity_score': diversity_score,
                    'hotspot_visits': hotspot_visits,
                    'avg_distance_from_hotspots': avg_distance_from_hotspots,
                }

        # 汇总统计
        all_vehicles_data = {}
        for episode_idx, episode_history in getattr(env, 'episode_vehicle_visits', {}).items():
            for vehicle_id, vehicle_data in episode_history.items():
                if vehicle_id not in all_vehicles_data:
                    all_vehicles_data[vehicle_id] = {
                        'total_unique_locations': 0,
                        'total_visits': 0,
                        'episodes_count': 0,
                    }
                all_vehicles_data[vehicle_id]['total_unique_locations'] += vehicle_data['unique_locations']
                all_vehicles_data[vehicle_id]['total_visits'] += vehicle_data['total_visits']
                all_vehicles_data[vehicle_id]['episodes_count'] += 1

        if all_vehicles_data:
            avg_unique_locations = np.mean(
                [data['total_unique_locations'] / data['episodes_count'] for data in all_vehicles_data.values()]
            )
            avg_visits_per_episode = np.mean(
                [data['total_visits'] / data['episodes_count'] for data in all_vehicles_data.values()]
            )
            print("\n🚛 车辆移动性分析:")
            print(f"   平均每episode访问的不同位置数: {avg_unique_locations:.1f}")
            print(f"   平均每episode总访问次数: {avg_visits_per_episode:.1f}")
            mobility_scores = {
                vid: data['total_unique_locations'] / data['episodes_count'] for vid, data in all_vehicles_data.items()
            }
            most_mobile = max(mobility_scores, key=mobility_scores.get)
            least_mobile = min(mobility_scores, key=mobility_scores.get)
            print(f"   最活跃车辆: Vehicle {most_mobile} ({mobility_scores[most_mobile]:.1f} 个不同位置/episode)")
            print(f"   最不活跃车辆: Vehicle {least_mobile} ({mobility_scores[least_mobile]:.1f} 个不同位置/episode)")

        return vehicle_visit_stats

    @staticmethod
    def _save_episode_stats_to_excel(
        env,
        episode_stats,
        results_dir,
        vehicle_visit_stats=None,
        transportation_mode='integrated',
        zone_distribution_mode='bayes',
    ):
        """保存episode统计到Excel（从原测试脚本迁移）。"""
        if not episode_stats:
            print("⚠ No episode statistics to save")
            return None, None

        zone_history_rows = []
        period_dropout_rows = []
        hourly_completed_rows = []
        cleaned_episode_stats = []
        for row in episode_stats:
            cleaned_row = dict(row)
            zone_total_history = cleaned_row.pop('zone_vehicle_count_history', None)
            zone_ev_history = cleaned_row.pop('zone_ev_count_history', None)
            zone_aev_history = cleaned_row.pop('zone_aev_count_history', None)
            period_dropout_counts = cleaned_row.pop('period_dropout_counts', None)
            hourly_completed_orders = cleaned_row.pop('hourly_completed_orders', None)

            if zone_total_history:
                max_steps = max(
                    len(zone_total_history or []),
                    len(zone_ev_history or []),
                    len(zone_aev_history or []),
                )
                for step_idx in range(max_steps):
                    total_counts = zone_total_history[step_idx] if step_idx < len(zone_total_history or []) else []
                    ev_counts = zone_ev_history[step_idx] if step_idx < len(zone_ev_history or []) else []
                    aev_counts = zone_aev_history[step_idx] if step_idx < len(zone_aev_history or []) else []
                    zone_count = max(len(total_counts), len(ev_counts), len(aev_counts))
                    for zone_id in range(zone_count):
                        zone_history_rows.append({
                            'episode_number': cleaned_row.get('episode_number'),
                            'time_step': step_idx,
                            'zone_id': zone_id,
                            'total_vehicles': total_counts[zone_id] if zone_id < len(total_counts) else 0,
                            'ev_vehicles': ev_counts[zone_id] if zone_id < len(ev_counts) else 0,
                            'aev_vehicles': aev_counts[zone_id] if zone_id < len(aev_counts) else 0,
                        })

            if period_dropout_counts:
                for period_idx, dropout_count in enumerate(period_dropout_counts, start=1):
                    period_dropout_rows.append({
                        'episode_number': cleaned_row.get('episode_number'),
                        'period_index': period_idx,
                        'dropoff_driver_count': dropout_count,
                    })

            if hourly_completed_orders:
                for hourly_row in hourly_completed_orders:
                    hourly_completed_rows.append({
                        'episode_number': cleaned_row.get('episode_number'),
                        'completed_date': hourly_row.get('completed_date'),
                        'completed_hour': hourly_row.get('completed_hour'),
                        'completed_orders': hourly_row.get('completed_orders', 0),
                        'completed_ev_orders': hourly_row.get('completed_ev_orders', 0),
                        'completed_aev_orders': hourly_row.get('completed_aev_orders', 0),
                    })

            cleaned_episode_stats.append(cleaned_row)

        df = pd.DataFrame(cleaned_episode_stats)
        num_ev = env.ev_num_vehicles
        adpvalue = getattr(env, 'adp_value', 1.0)
        demand_pattern = "intense" if getattr(env, 'use_intense_requests', True) else "random"
        charging_penalty = getattr(env, 'charging_penalty', 2.0)
        unserved_penalty = getattr(env, 'unserved_penalty', 1.5)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = (
            f"episode_statistics_adp_{transportation_mode}_{adpvalue}_demand{demand_pattern}_{env.heuristic_battery_threshold}_{num_ev}_{timestamp}_{env.knownreject}_{zone_distribution_mode}_{env.battery_first}.xlsx"
        )
        excel_path = results_dir / excel_filename
        spatial_image_path = results_dir / f"spatial_analysis_adp{adpvalue}_demand{demand_pattern}_{timestamp}.png"

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Episode_Statistics', index=False)
                if zone_history_rows:
                    pd.DataFrame(zone_history_rows).to_excel(writer, sheet_name='Zone_Vehicle_Counts', index=False)
                if period_dropout_rows:
                    pd.DataFrame(period_dropout_rows).to_excel(writer, sheet_name='Period_Dropoffs', index=False)
                if hourly_completed_rows:
                    pd.DataFrame(hourly_completed_rows).to_excel(writer, sheet_name='Hourly_Completed_Orders', index=False)
                adp_config_data = {
                    'Parameter': [
                        'ADP_Value',
                        'Demand_Pattern',
                        'Zone_Scope',
                        'Only_Manhattan_Zones',
                        'Distribution_Mode',
                        'Random_Seed',
                        'GAT_Neighbour_Number',
                        'Known_Reject',
                        'If_Reject',
                        'Charging_Penalty',
                        'Unserved_Penalty',
                        'Grid_Size',
                        'Number_of_Vehicles',
                        'Number_of_Stations',
                        'Episode_Length',
                        'Request_Generation_Rate',
                        'Vehicle_Types',
                        'Hotspot_Configuration',
                    ],
                    'Value': [
                        adpvalue,
                        demand_pattern,
                        'manhattan' if getattr(env, 'ifonlymanhatten', False) else 'full_nyc',
                        bool(getattr(env, 'ifonlymanhatten', False)),
                        zone_distribution_mode,
                        getattr(env, 'initial_random_seed', 'N/A'),
                        getattr(
                            getattr(getattr(env, 'value_function', None), 'graph_encoder', None),
                            'neighbour_number',
                            0,
                        ),
                        bool(getattr(env, 'knownreject', False)),
                        bool(getattr(env, 'ifreject', False)),
                        charging_penalty,
                        unserved_penalty,
                        env.grid_size,
                        env.num_vehicles,
                        env.num_stations,
                        env.episode_length,
                        getattr(env, 'request_rate', 'N/A'),
                        getattr(env, 'vehicle_types', 'EV/AEV'),
                        getattr(env, 'hotspot_locations', 'default'),
                    ],
                }
                pd.DataFrame(adp_config_data).to_excel(writer, sheet_name='ADP_Config', index=False)

                if vehicle_visit_stats:
                    visit_rows = []
                    for vid, stats in vehicle_visit_stats.items():
                        visit_rows.append(
                            {
                                'Vehicle_ID': vid,
                                'Most_Visited_Location': stats.get('most_visited_location'),
                                'Visit_Count': stats.get('visit_count'),
                                'Unique_Locations': stats.get('unique_locations'),
                                'Total_Visits': stats.get('total_visits'),
                                'Diversity_Score': stats.get('diversity_score'),
                                'Hotspot_Visits': stats.get('hotspot_visits'),
                                'Avg_Distance_From_Hotspots': stats.get('avg_distance_from_hotspots'),
                            }
                        )
                    pd.DataFrame(visit_rows).to_excel(writer, sheet_name='Vehicle_Visit_Patterns', index=False)

            # 生成空间可视化
            if hasattr(env, 'visualize_spatial_distribution'):
                env.visualize_spatial_distribution(save_path=spatial_image_path)
            print(f"✓ 统计结果已保存到 {excel_path}")
        except Exception as e:
            print(f"❌ 保存统计时出错: {e}")
            excel_path, spatial_image_path = None, None

        return excel_path, spatial_image_path

    def _load_network_state_dict_compatible(self, module, state_dict, label):
        compatible_loader = getattr(module, 'load_compatible_state_dict', None)
        if callable(compatible_loader):
            compatible_loader(state_dict)
            print(f"✓ {label} 使用模块兼容加载器", flush=True)
            return
        current_state = module.state_dict()
        compatible_state = {}
        skipped_keys = []
        inserted_input_index = getattr(module, 'inserted_input_index', None)
        for key, value in state_dict.items():
            if key in current_state and hasattr(value, 'shape') and current_state[key].shape == value.shape:
                compatible_state[key] = value
            elif (
                inserted_input_index is not None
                and key in current_state
                and hasattr(value, 'shape')
                and value.ndim == 2
                and current_state[key].ndim == 2
                and value.shape[0] == current_state[key].shape[0]
                and value.shape[1] + 1 == current_state[key].shape[1]
            ):
                expanded = torch.zeros_like(current_state[key])
                index = int(inserted_input_index)
                expanded[:, :index] = value[:, :index]
                expanded[:, index + 1:] = value[:, index:]
                compatible_state[key] = expanded
            else:
                skipped_keys.append(key)
        module.load_state_dict(compatible_state, strict=False)
        if skipped_keys:
            preview = ', '.join(skipped_keys[:5])
            suffix = '...' if len(skipped_keys) > 5 else ''
            print(f"⚠ {label} 跳过 {len(skipped_keys)} 个不兼容权重: {preview}{suffix}", flush=True)

    def load_checkpoint(self, value_function, checkpoint_path):
        try:
            if not os.path.exists(checkpoint_path):
                print(f"❌ Checkpoint文件不存在: {checkpoint_path}", flush=True)
                return False
            
            print(f"📂 加载checkpoint: {checkpoint_path}", flush=True)
            artifacts = getattr(value_function, "checkpoint_artifact_paths", [])
            artifacts.append(str(checkpoint_path))
            value_function.checkpoint_artifact_paths = list(dict.fromkeys(artifacts))
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=value_function.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    checkpoint_path, map_location=value_function.device
                )
            
            if hasattr(value_function, 'load_acceptance_checkpoint_state'):
                value_function.load_acceptance_checkpoint_state(
                    checkpoint.get('extra_value_function_state', {})
                )
            # 选择要加载的state_dict
            # if 'target_network_state_dict' in checkpoint:
            #     state_dict_to_load = checkpoint['target_network_state_dict']
            #     network_type = 'target_network'
            # elif 'network_state_dict' in checkpoint:
            #     state_dict_to_load = checkpoint['network_state_dict']
            #     network_type = 'network'
            # else:
            #     print(f"❌ Checkpoint中没有找到网络权重")
            #     return False
            
            # Handle both full_state dict and network-only state_dict
            if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
                # full_state checkpoint
                self._load_network_state_dict_compatible(value_function.network, checkpoint['network_state_dict'], 'network')
                print(f"✓ 成功加载网络权重 (full_state)", flush=True)
                # Load time_zone_dist_predictor
                if hasattr(value_function, 'time_zone_dist_predictor') and 'time_zone_dist_predictor_state_dict' in checkpoint:
                    value_function.time_zone_dist_predictor.load_state_dict(checkpoint['time_zone_dist_predictor_state_dict'])
                    print(f"✓ 成功加载time_zone_dist_predictor权重", flush=True)
                else:
                    print(f"⚠ checkpoint中无time_zone_dist_predictor权重，使用随机初始化", flush=True)
                if hasattr(value_function, 'time_zone_dist_optimizer') and 'time_zone_dist_optimizer_state_dict' in checkpoint:
                    value_function.time_zone_dist_optimizer.load_state_dict(checkpoint['time_zone_dist_optimizer_state_dict'])
                for role in ('leader', 'follower'):
                    time_predictor = getattr(value_function, f'time_zone_dist_predictor_{role}', None)
                    time_optimizer = getattr(value_function, f'time_zone_dist_optimizer_{role}', None)
                    zone_predictor = getattr(value_function, f'zone_dist_predictor_{role}', None)
                    zone_optimizer = getattr(value_function, f'zone_dist_optimizer_{role}', None)

                    split_time_key = f'time_zone_dist_predictor_{role}_state_dict'
                    split_time_opt_key = f'time_zone_dist_optimizer_{role}_state_dict'
                    split_zone_key = f'zone_dist_predictor_{role}_state_dict'
                    split_zone_opt_key = f'zone_dist_optimizer_{role}_state_dict'

                    if time_predictor is not None:
                        if split_time_key in checkpoint:
                            time_predictor.load_state_dict(checkpoint[split_time_key])
                            print(f"✓ 成功加载time_zone_dist_predictor_{role}权重", flush=True)
                        elif 'time_zone_dist_predictor_state_dict' in checkpoint:
                            time_predictor.load_state_dict(checkpoint['time_zone_dist_predictor_state_dict'])
                            print(f"✓ 使用legacy time_zone_dist_predictor权重初始化{role}", flush=True)
                    if time_optimizer is not None:
                        if split_time_opt_key in checkpoint:
                            time_optimizer.load_state_dict(checkpoint[split_time_opt_key])
                        elif 'time_zone_dist_optimizer_state_dict' in checkpoint:
                            time_optimizer.load_state_dict(checkpoint['time_zone_dist_optimizer_state_dict'])

                    if zone_predictor is not None:
                        if split_zone_key in checkpoint:
                            zone_predictor.load_state_dict(checkpoint[split_zone_key])
                            print(f"✓ 成功加载zone_dist_predictor_{role}权重", flush=True)
                        elif 'zone_dist_predictor_state_dict' in checkpoint:
                            zone_predictor.load_state_dict(checkpoint['zone_dist_predictor_state_dict'])
                            print(f"✓ 使用legacy zone_dist_predictor权重初始化{role}", flush=True)
                    if zone_optimizer is not None:
                        if split_zone_opt_key in checkpoint:
                            zone_optimizer.load_state_dict(checkpoint[split_zone_opt_key])
                        elif 'zone_dist_optimizer_state_dict' in checkpoint:
                            zone_optimizer.load_state_dict(checkpoint['zone_dist_optimizer_state_dict'])
                # Also load target network if available
                if 'target_network_state_dict' in checkpoint:
                    self._load_network_state_dict_compatible(value_function.target_network, checkpoint['target_network_state_dict'], 'target_network')
                if (
                    hasattr(value_function, 'optimizer')
                    and 'optimizer_state_dict' in checkpoint
                ):
                    try:
                        value_function.optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict']
                        )
                        print("✓ 成功加载主optimizer状态", flush=True)
                    except (ValueError, RuntimeError) as e:
                        print(
                            "⚠ 主optimizer状态与当前alpha接口不兼容，"
                            f"使用新optimizer: {e}",
                            flush=True,
                        )
                for metric_name in (
                    'training_losses',
                    'normalized_td_losses',
                    'td_error_history',
                    'q_values_history',
                ):
                    if metric_name in checkpoint:
                        setattr(value_function, metric_name, list(checkpoint.get(metric_name, [])))
                if getattr(value_function, 'rejection_predictor', None) is not None and 'rejection_predictor_state_dict' in checkpoint:
                    value_function.rejection_predictor.load_state_dict(checkpoint['rejection_predictor_state_dict'])
                    value_function.rejection_predictor_trained = bool(
                        checkpoint.get('rejection_predictor_trained', False)
                    )
                    value_function.rejection_training_losses = list(
                        checkpoint.get('rejection_training_losses', [])
                    )
                    print(f"✓ 成功加载rejection_predictor权重", flush=True)
                if getattr(value_function, 'rejection_optimizer', None) is not None and 'rejection_optimizer_state_dict' in checkpoint:
                    try:
                        value_function.rejection_optimizer.load_state_dict(checkpoint['rejection_optimizer_state_dict'])
                    except (ValueError, RuntimeError) as e:
                        print(f"⚠ rejection_optimizer状态不兼容，使用新optimizer: {e}", flush=True)
                if hasattr(value_function, 'load_extra_checkpoint_state') and 'extra_value_function_state' in checkpoint:
                    value_function.load_extra_checkpoint_state(checkpoint['extra_value_function_state'])
                    print(f"✓ 成功加载extra value_function状态", flush=True)
            else:
                # network-only checkpoint (legacy)
                self._load_network_state_dict_compatible(value_function.network, checkpoint, 'network_only')
                print(f"✓ 成功加载网络权重 (network_only)", flush=True)
                print(f"⚠ network_only checkpoint不含完整训练状态，使用随机初始化", flush=True)
            
            # 设置为评估模式
            # value_function.network.eval()
            
            # 显示checkpoint信息
            if 'episode' in checkpoint:
                print(f"  - Episode: {checkpoint['episode']}", flush=True)
            if 'training_step' in checkpoint:
                if hasattr(value_function, 'training_step'):
                    value_function.training_step = checkpoint['training_step']
                print(f"  - Training step: {checkpoint['training_step']}", flush=True)
            if 'buffer_size' in checkpoint:
                print(f"  - Buffer size: {checkpoint['buffer_size']}", flush=True)
            
            return True
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
        print(f"✓ Random seeds set to {seed} for all generators (Python, NumPy, PyTorch)", flush=True)
    

    
    def run_charging_integration_test(
        self,
        adpvalue: float,
        num_episodes: int,
        use_intense_requests: bool,
        assignmentgurobi: bool,
        batch_size: int = 256,
        checkpoint_replay: str = "recent",
        checkpoint_replay_recent: int = 5_000,
        num_vehicles: int = 50,
        num_ev: int = 25,
        heuristic_battery_threshold: float = 0.5,
        transportation_mode: str = "integrated",
        start_training_episode: int = 2,
        usemcmf: bool = True,
        mcmf_use_gpu: bool = False,
        mcmf_solver: str = None,
        mcmf_backend: str = "auto",
        mcmf_strict: bool = True,
        mcmf_cost_scale: int = 10_000,
        mcmf_graph_reduction: bool = True,
        mcmf_verify: bool = False,
        useauction: bool = False,
        auction_use_gpu: bool = False,
        auction_epsilon: float = 1e-3,
        auction_max_rounds: int = None,
        auction_top_k: int = None,
        knownreject: bool = False,
        ifloadgingValueFunction: bool = True,
        trainnetwork: bool = False,
        gurobi_network: bool = True,
        gurobi_network_lp: bool = True,
        evaluatemode: bool = False,
        random_seed: int = 42,
        record_time: bool = False,
        multi_gpu_devices: list = None,  # 🚀 添加多GPU支持
        grid_size: int = 20,
        num_stations: int = 5,
        station_capacity: int = 10,
        station_queue_capacity: int = 0,
        charge_duration: int = 2,
        aev_initial_battery_scale: float = 1.0,
        critical_charging_battery: float = 0.15,
        simulation_period: int = 50,
        days_per_week: int = 7,
        episode_length: int = None,
        encoder: bool = True,
        zone_distribution_mode: str = None,
        battery_first: bool = False,
        daily_drop_off: bool = False,
        iftest_aev: bool = False,
        aev_test_service_time: float = None,
        aev_test_request_rate_scale: float = 1.0,
        aev_test_request_generation_rate_override: float = None,
        charging_wait_penalty_per_step: float = 1.0,
        synthetic_demand_profile: str = "predictive",
        synthetic_demand_scale: float = 1.0,
        randomize_training_vehicle_states: bool = True,
        post_demand_q_weight: float = 0.0,
        post_demand_head_lr_multiplier: float = 10.0,
        masac_target_entropy_ratio: float = 0.9,
        save_checkpoints: bool = True,
        save_results: bool = True,
        checkpoint_selection: str = None,
        disable_queue_predictor: bool = False,
        disable_post_demand_predictor: bool = False,
        checkpoint_scenario_suffix: str = "",
        load_checkpoint_assign_tag: str | None = "gurobi",
        recourse_variant: str = "legacy",
        rejection_logit_shift: float = 0.0,
        common_random_numbers: bool = False,
        state_variant: str = "joint_state_separate_critics",
        learner_variant: str = "legacy",
        ev_acceptance_feature: str = "off",
        ev_acceptance_model: str | None = None,
        ev_response_anchor: str = 'auto',
        ev_response_critic_input: str = 'q_mask',
    ):
        """完整迁移的集成测试（含神经网络训练与梯度更新）。"""

        def log_progress(message: str):
            self.logger.info(message)
            self._flush_logger_handlers()

        self.logger.info("="*80)
        self.logger.info("=== Starting Enhanced Charging Behavior Integration Test ===")
        self.logger.info("="*80)
        self.logger.info(f"📊 Configuration Summary:")
        self.logger.info(f"   Mode: {transportation_mode.upper()}")
        self.logger.info(f"   Demand: {'INTENSE' if use_intense_requests else 'RANDOM'}")
        self.logger.info(f"   ADP: {adpvalue}")
        if recourse_variant != "legacy" and transportation_mode != "evfirst":
            raise ValueError("R0--R4 recourse variants require transportation_mode='evfirst'")
        valid_state_variants = {
            "joint_state_shared_critic",
            "joint_state_separate_critics",
            "fleet_local_separate_critics",
            "fleet_local_shared_critic",
        }
        if state_variant not in valid_state_variants:
            raise ValueError(f"unknown state variant: {state_variant}")
        shared_critic = uses_shared_critic(state_variant)
        if useauction:
            usemcmf = True

        assignment_label = "AUCTION" if usemcmf and useauction else ("MCMF" if usemcmf else ("GUROBI" if assignmentgurobi else "HEURISTIC"))
        self.logger.info(f"   Assignment: {assignment_label}")
        if usemcmf:
            solver_label = "auction" if useauction else (mcmf_solver or "legacy")
            backend_label = mcmf_backend if solver_label == "exact" else (
                "GPU" if mcmf_use_gpu else "CPU"
            )
            self.logger.info(
                f"   MCMF solver: {solver_label} (backend={backend_label}, "
                f"strict={mcmf_strict})"
            )
        if useauction:
            self.logger.info(f"   Auction solver: {'GPU' if auction_use_gpu else 'CPU'}")
        self.logger.info(f"   Episodes: {num_episodes}")
        self.logger.info(f"   Vehicles: {num_vehicles} (EV: {num_ev}, AEV: {num_vehicles - num_ev})")
        self.logger.info(f"   Batch size: {batch_size}")
        self.logger.info(f"   Start training episode: {start_training_episode}")
        self.logger.info(f"   Daily drop-off mechanism: {daily_drop_off}")
        self.logger.info(f"   Charging wait penalty per step: {charging_wait_penalty_per_step}")
        self.logger.info(f"   Synthetic demand profile: {synthetic_demand_profile}")
        self.logger.info(f"   Synthetic demand scale: {synthetic_demand_scale:g}")
        self.logger.info(
            "   Randomize training vehicle states: "
            f"{randomize_training_vehicle_states}"
        )
        self.logger.info(f"   Charging stations: count={num_stations}, capacity={station_capacity}, duration={charge_duration} steps")
        self.logger.info(
            "   Synthetic queue configuration: "
            f"waiting-room={station_queue_capacity} per station, "
            f"AEV initial battery scale={aev_initial_battery_scale:g}, "
            f"critical SoC={critical_charging_battery:g}"
        )
        self.logger.info(
            "   Checkpoint scenario suffix: "
            f"{checkpoint_scenario_suffix or '<legacy>'}"
        )
        self.logger.info(f"   AEV test capacity mode: {iftest_aev}")
        self.logger.info(f"   AEV test service time: {aev_test_service_time}")
        self.logger.info(f"   AEV test request rate scale: {aev_test_request_rate_scale}")
        self.logger.info(f"   AEV fixed request generation rate: {aev_test_request_generation_rate_override}")
        self.logger.info("="*80)

        episode_days = None if episode_length is None else max(1, int(np.ceil(episode_length / max(simulation_period, 1))))

        effective_zone_distribution_mode = zone_distribution_mode or ("bayes" if encoder else "none")
        if learner_variant == "legacy" and effective_zone_distribution_mode in {"integrated_directq", "optimization_anchored_residual"}:
            learner_variant = effective_zone_distribution_mode
        encoder_enabled = effective_zone_distribution_mode in {"bayes", "bayes_simple"}

        def get_distribution_suffix() -> str:
            if effective_zone_distribution_mode == "bayes":
                suffix = ""
            elif effective_zone_distribution_mode == "bayes_simple":
                suffix = "_bayes_simple"
            elif effective_zone_distribution_mode == "none":
                suffix = "_none"
            elif effective_zone_distribution_mode == "masac_queue_length":
                suffix = "_masac_queue_length"
            elif effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature":
                suffix = "_st_masac_gat_former2_queue_feature"
            elif effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature_greedy_alpha":
                suffix = "_st_masac_gat_former2_queue_feature_greedy_alpha"
            elif effective_zone_distribution_mode == "st_masac_gat_former2_queue_feature_fixed_alpha":
                suffix = "_st_masac_gat_former2_queue_feature_fixed_alpha"
            elif effective_zone_distribution_mode == "masac_baseline":
                suffix = "_masac_baseline"
            elif effective_zone_distribution_mode == "st_masac_gat_post_demand_direct":
                suffix = "_st_masac_gat_post_demand_direct"
            elif effective_zone_distribution_mode == "st_masac_gat_queue_demand_gurobi":
                suffix = "_st_masac_gat_queue_demand_gurobi"
            elif effective_zone_distribution_mode in {"integrated_directq", "optimization_anchored_residual"}:
                suffix = "_" + effective_zone_distribution_mode
            elif effective_zone_distribution_mode == "standard_masac_gat":
                suffix = "_standard_masac_gat"
            elif effective_zone_distribution_mode == "standard_masac_gat_total_q":
                suffix = "_standard_masac_gat_total_q"
            elif effective_zone_distribution_mode == "standard_masac_gat_greedy_alpha":
                suffix = "_standard_masac_gat_greedy_alpha"
            elif effective_zone_distribution_mode == "standard_masac_gat_fixed_alpha":
                suffix = "_standard_masac_gat_fixed_alpha"
            else:
                suffix = "_noenc"
            return suffix + str(checkpoint_scenario_suffix or "")

        self._set_random_seeds(seed=random_seed)

        env_init_start = time.time()
        log_progress(
            f"Initializing environment: vehicles={num_vehicles}, ev={num_ev}, stations={num_stations}, grid={grid_size}"
        )
        env = ChargingIntegratedEnvironment(
            num_vehicles=num_vehicles,
            num_stations=num_stations,
            station_capacity=station_capacity,
            station_queue_capacity=station_queue_capacity,
            charge_duration=charge_duration,
            aev_initial_battery_scale=aev_initial_battery_scale,
            critical_charging_battery=critical_charging_battery,
            ev_num_vehicles=num_ev,
            grid_size=grid_size,
            heuristic_battery_threshold=heuristic_battery_threshold,
            use_intense_requests=use_intense_requests,
            assignmentgurobi=assignmentgurobi,
            usemcmf=usemcmf,
            useauction=useauction,
            mcmf_solver=mcmf_solver,
            mcmf_backend=mcmf_backend,
            mcmf_strict=mcmf_strict,
            mcmf_cost_scale=mcmf_cost_scale,
            mcmf_graph_reduction=mcmf_graph_reduction,
            mcmf_verify=mcmf_verify,
            auction_use_gpu=auction_use_gpu,
            auction_epsilon=auction_epsilon,
            auction_max_rounds=auction_max_rounds,
            auction_top_k=auction_top_k,
            knownreject=knownreject,
            gurobi_network=gurobi_network,
            gurobi_network_lp=gurobi_network_lp,
            random_seed=random_seed,
            record_time=record_time,
            multi_gpu_devices=multi_gpu_devices,  # 🚀 传递多GPU设备列表
            simulation_period=simulation_period,
            days_per_week=days_per_week,
            episode_days=episode_days,
            battery_first=battery_first,
            daily_drop_off=daily_drop_off,
            iftest_aev=iftest_aev,
            aev_test_service_time=aev_test_service_time,
            aev_test_request_rate_scale=aev_test_request_rate_scale,
            aev_test_request_generation_rate_override=aev_test_request_generation_rate_override,
            charging_wait_penalty_per_step=charging_wait_penalty_per_step,
            synthetic_demand_profile=synthetic_demand_profile,
            synthetic_demand_scale=synthetic_demand_scale,
        )
        env.configure_recourse_experiment(
            recourse_variant,
            rejection_logit_shift=rejection_logit_shift,
            common_random_numbers=common_random_numbers,
        )
        env.state_variant = str(state_variant)
        env.learner_variant = str(learner_variant)
        from src.acceptance_features import configure_acceptance_feature
        configure_acceptance_feature(env, ev_acceptance_feature, ev_acceptance_model,
                                     anchor=ev_response_anchor, critic_input=ev_response_critic_input)
        if ev_acceptance_feature == "predicted":
            from src.acceptance_features import acceptance_checkpoint_suffix
            checkpoint_scenario_suffix += acceptance_checkpoint_suffix(ev_acceptance_feature, ev_acceptance_model,
                anchor=ev_response_anchor, critic_input=ev_response_critic_input)
        env.mcmf_use_gpu = bool(mcmf_use_gpu)
        env.use_cuda_ssp = bool(mcmf_use_gpu)
        env.useauction = bool(useauction or getattr(env, 'ifsolveauctioncuda', False))
        env.mcmf_solver = "auction" if env.useauction else mcmf_solver
        env.mcmf_backend = mcmf_backend
        env.mcmf_strict = bool(mcmf_strict)
        env.mcmf_cost_scale = int(mcmf_cost_scale)
        env.mcmf_graph_reduction = bool(mcmf_graph_reduction)
        env.mcmf_verify = bool(mcmf_verify)
        env.auction_use_gpu = bool(auction_use_gpu)
        env.auction_epsilon = float(auction_epsilon)
        env.auction_max_rounds = auction_max_rounds
        env.auction_top_k = auction_top_k
        env.evaluatemode = evaluatemode
        env.randomize_vehicle_initial_state = bool(
            randomize_training_vehicle_states and trainnetwork and not evaluatemode
        )
        log_progress(f"Environment initialization complete in {time.time() - env_init_start:.2f}s")
        log_progress(f"✓ Episode length auto-derived as {env.episode_length} steps ({env.episode_days} day(s), simulation_period={env.simulation_period})")
        if env.randomize_vehicle_initial_state:
            log_progress("✓ Training vehicle positions and battery levels vary by episode")
        else:
            log_progress("✓ Vehicle positions and battery levels are fixed across episodes")
        log_progress("✓ Request generation will vary by episode for learning progression")

        use_neural_network = self._should_train_value_function(
            adpvalue,
            trainnetwork,
        )

        if use_neural_network or ifloadgingValueFunction:
            value_function_init_start = time.time()
            log_progress("Initializing AEV value function")
            value_function_extra_kwargs = {}
            if effective_zone_distribution_mode in {
                "st_masac_gat_post_demand_direct",
                "st_masac_gat_queue_demand_gurobi",
                "standard_masac_gat",
                "standard_masac_gat_total_q",
                "standard_masac_gat_greedy_alpha",
                "standard_masac_gat_fixed_alpha",
            }:
                value_function_extra_kwargs.update({
                    "post_demand_q_weight": float(post_demand_q_weight),
                    "post_demand_head_lr_multiplier": float(post_demand_head_lr_multiplier),
                })
            if effective_zone_distribution_mode in {
                "masac_baseline",
                "standard_masac_gat",
                "standard_masac_gat_total_q",
                "standard_masac_gat_greedy_alpha",
                "standard_masac_gat_fixed_alpha",
            }:
                value_function_extra_kwargs["entropy_target_ratio"] = float(
                    masac_target_entropy_ratio
                )
            value_function = PyTorchChargingValueFunction(
                grid_size=env.grid_size,
                num_vehicles=num_vehicles,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                episode_length=env.episode_length,
                max_requests=10000,
                env=env,
                encoder=encoder_enabled,
                zone_distribution_mode=effective_zone_distribution_mode,
                **value_function_extra_kwargs,
            )
            log_progress(f"AEV value function initialized in {time.time() - value_function_init_start:.2f}s")
            ev_value_function_init_start = time.time()
            log_progress("Initializing EV value function")
            value_function_ev = PyTorchChargingValueFunction(
                grid_size=env.grid_size,
                num_vehicles=num_vehicles,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                episode_length=env.episode_length,
                max_requests=10000,
                env=env,
                encoder=encoder_enabled,
                zone_distribution_mode=effective_zone_distribution_mode,
                **value_function_extra_kwargs,
            )
            if (
                transportation_mode == 'aevfirst'
                and effective_zone_distribution_mode == 'bayes_simple'
                and hasattr(value_function_ev, 'disable_follower_zone_distribution_predictor')
            ):
                value_function_ev.disable_follower_zone_distribution_predictor()
            value_function, value_function_ev = enforce_critic_identity(
                value_function,
                value_function_ev,
                state_variant=state_variant,
            )
            for current_value_function in {
                id(value_function): value_function,
                id(value_function_ev): value_function_ev,
            }.values():
                current_value_function.state_variant = str(state_variant)
                current_value_function.learner_variant = str(learner_variant)
                current_value_function.recourse_variant = str(recourse_variant)
                if hasattr(current_value_function, "joint_replay_buffer"):
                    if checkpoint_replay not in {"none", "recent", "full"}:
                        raise ValueError(
                            "checkpoint_replay must be none, recent, or full"
                        )
                    current_value_function.checkpoint_replay = str(
                        checkpoint_replay
                    )
                    current_value_function.checkpoint_replay_recent = max(
                        1, int(checkpoint_replay_recent)
                    )
            env.set_value_function(value_function)
            env.set_value_function_ev(value_function_ev)
            log_progress(f"EV value function initialized in {time.time() - ev_value_function_init_start:.2f}s")
            log_progress(f"Value function initialization complete in {time.time() - value_function_init_start:.2f}s total")
        else:
            value_function = None
            value_function_ev = None

        training_run_id = uuid.uuid4().hex
        resume_episode_offset = 0
        resume_best_reward = float('-inf')
        effective_start_training_episode = (
            0 if ifloadgingValueFunction and trainnetwork else start_training_episode
        )

        if ifloadgingValueFunction:
            checkpoint_loading_start = time.time()
            log_progress("Starting checkpoint discovery and loading")
            load_tag = self._resolve_checkpoint_assign_tag(
                assignmentgurobi,
                load_checkpoint_assign_tag or "gurobi",
            )
            log_progress(
                f"Loading value functions trained with {load_tag.upper()} assignment"
            )
            enc_suffix = get_distribution_suffix()
            if transportation_mode == "integrated":
                evfile = f"checkpoints/q_networks_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}"
                aevfile = f"checkpoints/q_networks_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}"
            elif transportation_mode == "evfirst":
                evfile = f"checkpoints/q_networksevfirst_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}"
                aevfile = f"checkpoints/q_networksevfirst_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}"
            elif transportation_mode == "aevfirst":
                evfile = f"checkpoints/q_networksaevfirst_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}"
                aevfile = f"checkpoints/q_networksaevfirst_{load_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}"

            if checkpoint_selection not in {None, "latest", "best"}:
                raise ValueError(
                    "checkpoint_selection must be one of: latest, best"
                )
            prefer_best_checkpoint = (
                checkpoint_selection == "best"
                if checkpoint_selection is not None
                else bool(evaluatemode)
            )
            ev_checkpoint, aev_checkpoint = self.find_checkpoint_pair(
                evfile,
                aevfile,
                prefer_best=prefer_best_checkpoint,
            )
            log_progress(f"🔍 Searching for latest checkpoints...")
            log_progress(f"   - AEV checkpoint path: {aev_checkpoint or 'Not Found'}")
            log_progress(f"   - EV checkpoint path: {ev_checkpoint or 'Not Found'}")
            if not ev_checkpoint or not aev_checkpoint:
                raise FileNotFoundError(
                    "Requested paired checkpoint load, but no EV/AEV pair with "
                    f"the same tag and episode exists: ev={evfile}, aev={aevfile}"
                )

            ev_identity = self._checkpoint_identity(ev_checkpoint)
            aev_identity = self._checkpoint_identity(aev_checkpoint)
            resume_episode_offset = int(ev_identity.get('episode', 0) or 0)
            loaded_run_id = ev_identity.get('training_run_id')
            if loaded_run_id:
                training_run_id = str(loaded_run_id)
            log_progress(
                "✓ Verified paired checkpoint: "
                f"episode={resume_episode_offset}, "
                f"pair_id={ev_identity.get('pair_id') or '<legacy>'}"
            )

            # Preserve the best combined score across continuation runs even
            # when training resumes from the latest (rather than best) state.
            best_ev_checkpoint, best_aev_checkpoint = self.find_checkpoint_pair(
                evfile,
                aevfile,
                prefer_best=True,
            )
            if best_ev_checkpoint and best_aev_checkpoint:
                best_identity = self._checkpoint_identity(best_ev_checkpoint)
                stored_best = best_identity.get('combined_reward')
                if stored_best is not None:
                    resume_best_reward = float(stored_best)
                    log_progress(
                        "✓ Restored historical combined-best reward: "
                        f"{resume_best_reward:.2f}"
                    )

            if aev_checkpoint:
                aev_checkpoint_load_start = time.time()
                log_progress("Loading AEV checkpoint")
                if not self.load_checkpoint(value_function, aev_checkpoint):
                    raise RuntimeError(
                        "AEV checkpoint load failed; requested model/state/"
                        "learner/recourse configuration was not restored"
                    )
                log_progress(f"AEV checkpoint loaded in {time.time() - aev_checkpoint_load_start:.2f}s")
            else:
                log_progress(f"⚠ 未找到AEV checkpoint，将从头开始训练")
            
            if ev_checkpoint and not shared_critic:
                ev_checkpoint_load_start = time.time()
                log_progress("Loading EV checkpoint")
                if not self.load_checkpoint(value_function_ev, ev_checkpoint):
                    raise RuntimeError(
                        "EV checkpoint load failed; requested model/state/"
                        "learner/recourse configuration was not restored"
                    )
                log_progress(f"EV checkpoint loaded in {time.time() - ev_checkpoint_load_start:.2f}s")
            elif ev_checkpoint and shared_critic:
                log_progress(
                    "Shared-critic variant: verified the EV/AEV checkpoint pair "
                    "and loaded the canonical AEV copy once"
                )
            else:
                log_progress(f"⚠ 未找到EV checkpoint，将从头开始训练")
            log_progress(f"Checkpoint loading complete in {time.time() - checkpoint_loading_start:.2f}s")

        if value_function is not None and value_function_ev is not None:
            for label, current_value_function in (
                ("AEV", value_function),
                ("EV", value_function_ev),
            ):
                if disable_queue_predictor and hasattr(
                    current_value_function, "queue_predictor_trained"
                ):
                    current_value_function.queue_predictor_trained = False
                    log_progress(f"Queue predictor disabled for {label} ablation")
                if disable_post_demand_predictor and hasattr(
                    current_value_function, "post_demand_predictor_trained"
                ):
                    current_value_function.post_demand_predictor_trained = False
                    log_progress(f"Post-demand predictor disabled for {label} ablation")

            value_function, value_function_ev = wire_recourse_critics(
                value_function,
                value_function_ev,
                state_variant=state_variant,
            )
            env.set_value_function(value_function)
            env.set_value_function_ev(value_function_ev)
        
        env.adp_value = adpvalue
        env.assignmentgurobi = assignmentgurobi
        training_frequency = 2
        warmup_steps = 64

        log_progress(f"✓ Initialized environment with {num_vehicles} vehicles and {num_stations} charging stations")
        if use_neural_network:
            log_progress(f"✓ Initialized PyTorchChargingValueFunction with neural network")
            log_progress(f"   - Training assignment backend: {assignment_label}")
            log_progress(f"   - Network parameters: {sum(p.numel() for p in value_function.network.parameters())}")
            log_progress("✓ Optimizer policy used without an extra epsilon action override")
            log_progress(f"   - Training frequency: every {training_frequency} steps after {warmup_steps} warmup steps")
            log_progress(f"   - Using device: {value_function.device}")
        else:
            log_progress(f"✓ Neural network training disabled (ADP={adpvalue}, trainnetwork={trainnetwork})")
            log_progress(f"   - Running without neural network training")

        ev_count = sum(1 for v in env.vehicles.values() if v.get('type') == 1)
        aev_count = sum(1 for v in env.vehicles.values() if v.get('type') == 2)
        log_progress(f"✓ Vehicle distribution: {ev_count} EV vehicles, {aev_count} AEV vehicles")

        results = {
            'Idle_average': [],
            'episode_rewards': [],
            'episode_rewards_aev': [],
            'episode_rewards_ev': [],
            'charging_events': [],
            'episode_detailed_stats': [],
            'episode_rejected_requests': [],
            'episode_recourse_requests': [],
            'episode_lost_requests': [],
            'vehicle_visit_stats': [],
            'battery_levels': [],
            'environment_stats': [],
            'value_function_losses': [],
            'value_function_ev_losses': [],
            'qvalue_losses': [],
            'qvalue_ev_losses': [],
            'drop_off_rates': [],
            'sample_assign_q_values_aev': [],
            'sample_assign_q_values_ev': [],
            'episode_times': [],
            'training_run_id': training_run_id,
            'resume_episode_offset': int(resume_episode_offset),
            'episode_identity_rows': [],
        }

        best_reward = resume_best_reward
        best_reward_ev = float('-inf')
        best_reward_aev = float('-inf')
        original_assignmentgurobi = env.assignmentgurobi
        original_usemcmf = env.usemcmf
        original_useauction = env.useauction
        original_mcmf_solver = getattr(env, 'mcmf_solver', None)
        original_gurobi_network = env.gurobi_network
        original_gurobi_network_lp = env.gurobi_network_lp

        # 打印训练日志表头
        # print("\n" + "="*120)
        # print(f"{'Ep':>4} | {'Reward':>8} | {'Loss(AEV)':>11} | {'Loss(EV)':>11} | {'Total':>6} | {'Accept':>7} | {'Reject':>7} | {'Complete':>9} | {'Battery':>8} | Mode: {transportation_mode.upper()}, Demand: {'INTENSE' if use_intense_requests else 'RANDOM'}")
        # print("="*120)

        for episode in range(num_episodes):
            cumulative_episode_index = resume_episode_offset + episode
            env.cumulative_episode_index = int(cumulative_episode_index)
            env.recourse_run_id = f"synthetic-seed-{int(random_seed)}"
            # Pre-training episodes use the historical pure-heuristic warmup.
            if use_neural_network and episode < effective_start_training_episode:
                env.adp_value = 0
                env.assignmentgurobi = False
                env.usemcmf = False
                env.useauction = False
                env.mcmf_solver = None
                env.gurobi_network = False
                env.gurobi_network_lp = False
                log_progress("Using pure heuristic dispatch during warmup episode")
            elif use_neural_network and episode == effective_start_training_episode:
                if (
                    not ifloadgingValueFunction
                    and getattr(env, 'synthetic_demand_profile', None) == 'predictive'
                ):
                    predictor_pretrain_steps = 500
                    for predictor_name, predictor in (
                        ('AEV', value_function),
                        ('EV', value_function_ev),
                    ):
                        queue_loss = 0.0
                        demand_loss = 0.0
                        for _ in range(predictor_pretrain_steps):
                            if hasattr(predictor, 'train_queue_predictor'):
                                queue_loss = predictor.train_queue_predictor(
                                    batch_size=batch_size
                                )
                            if hasattr(predictor, 'train_post_demand_predictor'):
                                demand_loss = predictor.train_post_demand_predictor(
                                    batch_size=batch_size
                                )
                        log_progress(
                            f"Synthetic predictor pretraining ({predictor_name}, "
                            f"steps={predictor_pretrain_steps}): "
                            f"queue_mse={queue_loss:.4f}, "
                            f"post_demand_mse={demand_loss * 10000.0:.4f}"
                        )
                env.adp_value = adpvalue
                env.assignmentgurobi = original_assignmentgurobi
                env.usemcmf = original_usemcmf
                env.useauction = original_useauction
                env.mcmf_solver = original_mcmf_solver
                env.gurobi_network = original_gurobi_network
                env.gurobi_network_lp = original_gurobi_network_lp
                log_progress("Restoring original assignment mode after pure heuristic warmup")
                log_progress(f"Restoring adp_value={adpvalue} for training phase")

            log_progress(f"Starting episode {episode + 1}/{num_episodes}")
            episode_start = time.time()
            # A continuation run must see new demand and vehicle states instead
            # of replaying episode seeds 32, 33, ... from the previous run.
            episode_seed = 32 + cumulative_episode_index
            log_progress(f"Setting request generation seed to {episode_seed}")
            env.set_request_generation_seed(episode_seed)
            log_progress(f"Request generation seed set to {episode_seed}")

            if env.randomize_vehicle_initial_state:
                env.vehicle_initialization_seed = (
                    int(random_seed) * 1000 + int(cumulative_episode_index)
                )
                log_progress(
                    "Vehicle initialization seed set to "
                    f"{env.vehicle_initialization_seed}"
                )
            else:
                env.vehicle_initialization_seed = env.initial_random_seed
            results['episode_identity_rows'].append({
                'cumulative_episode_index': int(cumulative_episode_index),
                'request_generation_seed': int(episode_seed),
                'vehicle_initialization_seed': int(
                    env.vehicle_initialization_seed
                ),
                'recourse_run_id': str(env.recourse_run_id),
            })

            log_progress(f"Resetting environment for episode {episode + 1}")
            _states = env.reset()
            log_progress(f"Environment reset complete for episode {episode + 1}")
            episode_reward = 0
            episode_reward_aev = 0
            episode_reward_ev = 0
            episode_charging_events = []
            episode_losses = []
            episode_losses_ev = []
            Idle_list = []

            for step in range(env.episode_length):
                if step == 0:
                    log_progress(f"Episode {episode + 1} entering step loop")
                actions = {}
                states_for_training = []
                actions_for_training = []
                current_requests = list(env.active_requests.values())
                if step == 0:
                    log_progress(
                        f"Episode {episode + 1} step 0 starting dispatch with {len(current_requests)} active requests"
                    )
                if transportation_mode == 'integrated':
                    actions, storeactions, storeactions_ev = env.simulate_motion(agents=[], current_requests=current_requests, rebalance=True)
                elif transportation_mode == 'evfirst':
                    actions, storeactions, storeactions_ev = env.simulate_motion_evfirst(agents=[], current_requests=current_requests, rebalance=True)
                elif transportation_mode == 'aevfirst':
                    actions, storeactions, storeactions_ev = env.simulate_motion_aevfirst(agents=[], current_requests=current_requests, rebalance=True)
                if step == 0:
                    log_progress(
                        f"Episode {episode + 1} step 0 dispatch complete with {len(actions)} actions"
                    )
                next_states, rewards, dur_rewards, done, info = env.step(actions, storeactions, storeactions_ev)
                if step == 0:
                    log_progress(
                        f"Episode {episode + 1} step 0 env.step complete with reward {sum(rewards.values()):.2f}"
                    )

                if step % 25 == 0:
                    stats = env.get_stats()
                    active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                    print("whole car number:", len(env.vehicles))
                    vehicle_status_count = {
                        'charging': 0,
                        'onboard': 0,
                        'to_pickup': 0,
                        'to_charge': 0,
                        'idle_moving': 0,
                        'fully_idle': 0,
                    }
                    for vid, v in env.vehicles.items():
                        if v['charging_station'] is not None:
                            status = 'charging'
                        elif v['passenger_onboard'] is not None:
                            status = 'onboard'
                        elif v['assigned_request'] is not None:
                            status = 'to_pickup'
                        elif v.get('charging_target') is not None:
                            status = 'to_charge'
                        elif v.get('idle_target') is not None or v.get('target_location') is not None:
                            status = 'idle_moving'
                        else:
                            status = 'fully_idle'
                        vehicle_status_count[status] += 1
                    step_reward = sum(rewards.values())
                    if 'generated_requests_last_step' in stats:
                        print(
                            f"Step {step}: Active requests: {active_requests}, "
                            f"New requests: {stats.get('generated_requests_last_step', 0)}, "
                            f"Total generated: {stats.get('total_generated_requests', 0)}, "
                            f"Hour: {stats.get('current_real_hour', 0.0):.2f}, "
                            f"Min battery: {stats.get('min_vehicle_battery', 0.0):.3f}, "
                            f"Cannot reach charge: {stats.get('vehicles_unable_to_reach_charging', 0)}, "
                            f"Step reward: {step_reward:.2f}"
                        )
                    else:
                        print(f"Step {step}: Active requests: {active_requests}, Step reward: {step_reward:.2f}")
                    print(
                        "  Vehicle Status: Charging={charging}, Onboard={onboard}, To_pickup={to_pickup}, To_charge={to_charge}, Idle_moving={idle_moving}, Fully_idle={fully_idle}".format(
                            charging=vehicle_status_count['charging'],
                            onboard=vehicle_status_count['onboard'],
                            to_pickup=vehicle_status_count['to_pickup'],
                            to_charge=vehicle_status_count['to_charge'],
                            idle_moving=vehicle_status_count['idle_moving'],
                            fully_idle=vehicle_status_count['fully_idle'],
                        )
                    )
                    print(f"  Total: {sum(vehicle_status_count.values())} vehicles")
                    Idle_list.append(vehicle_status_count['fully_idle'])

                    if use_neural_network and hasattr(value_function, 'training_losses') and value_function.training_losses:
                        recent_loss = value_function.training_losses[-1] if value_function.training_losses else 0.0
                        recent_loss_ev = value_function_ev.training_losses[-1] if value_function_ev.training_losses else 0.0
                        buffer_size = len(value_function.experience_buffer)
                        training_step = value_function.training_step
                        if buffer_size > 0:
                            sample_vehicle_id = list(env.vehicles.keys())[0] if env.vehicles else 0
                            sample_location = list(env.vehicles.values())[0]['location'] if env.vehicles else 0
                            sample_battery = list(env.vehicles.values())[0]['battery'] if env.vehicles else 1.0
                            try:
                                idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_location, sample_battery, current_time=step)
                                assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location + 1, current_time=step, battery_level=sample_battery)
                                charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location + 5, current_time=step, battery_level=sample_battery)
                                print("  Neural Network Status:")
                                print(
                                    f"    Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}, EV Recent loss: {recent_loss_ev:.4f}"
                                )
                                print(
                                    f"    Raw Q-values (no normalization): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}"
                                )
                            except Exception as e:
                                print(f"  Neural Network Status: Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                                print(f"    Error getting sample Q-values: {e}")
                    else:
                        print(f"  Neural Network: {'Not training yet' if use_neural_network else 'Disabled'}")

                aev_training_ready = training_readiness(
                    value_function,
                    ifEV=False,
                    edge_warmup=warmup_steps,
                ).any_ready
                ev_training_ready = training_readiness(
                    value_function_ev,
                    ifEV=True,
                    edge_warmup=warmup_steps,
                ).any_ready
                if use_neural_network and (aev_training_ready or ev_training_ready) and episode >= start_training_episode:
                    if step % training_frequency == 0:
                        if step % 100 == 0:
                            aev_buffer_size = len(value_function.experience_buffer)
                            ev_buffer_size = len(value_function_ev.experience_buffer)
                            print(f"🔄 Training (Episode {episode}): AEV buffer={aev_buffer_size}, EV buffer={ev_buffer_size}")
                        training_loss = (
                            value_function.train_step(batch_size=batch_size, ifEV=False)
                            if aev_training_ready
                            else 0.0
                        )
                        if training_loss > 0:
                            episode_losses.append(training_loss)
                        training_loss_ev = (
                            value_function_ev.train_step(batch_size=batch_size, ifEV=True)
                            if ev_training_ready
                            else 0.0
                        )
                        if training_loss_ev > 0:
                            episode_losses_ev.append(training_loss_ev)
                elif use_neural_network and episode < effective_start_training_episode and step % 100 == 0:
                    aev_buffer_size = len(value_function.experience_buffer)
                    ev_buffer_size = len(value_function_ev.experience_buffer)
                    print(f"📦 Collecting experience (Episode {episode}, no training yet): AEV buffer={aev_buffer_size}, EV buffer={ev_buffer_size}")

                step_reward_total = sum(rewards.values())
                step_reward_ev = sum(
                    reward for vehicle_id, reward in rewards.items()
                    if env.vehicles.get(vehicle_id, {}).get('type') == 1
                )
                step_reward_aev = sum(
                    reward for vehicle_id, reward in rewards.items()
                    if env.vehicles.get(vehicle_id, {}).get('type') == 2
                )

                episode_reward += step_reward_total
                episode_reward_ev += step_reward_ev
                episode_reward_aev += step_reward_aev
                episode_charging_events.extend(info.get('charging_events', []))

            if (
                save_checkpoints
                and use_neural_network
                and episode >= effective_start_training_episode
            ):
                global_episode = resume_episode_offset + episode + 1
                assign_tag = "gurobi" if assignmentgurobi else "heu"
                enc_suffix = get_distribution_suffix()
                if transportation_mode == 'integrated':
                    checkpoint_base = f"checkpoints/q_networks_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}"
                elif transportation_mode == 'evfirst':
                    checkpoint_base = f"checkpoints/q_networksevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}"
                elif transportation_mode == 'aevfirst':
                    checkpoint_base = f"checkpoints/q_networksaevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}"
                else:
                    checkpoint_base = None

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    checkpoint_pair_id = (
                        f"{training_run_id}:combined-best:{global_episode}"
                    )
                    paired_metadata = {
                        'checkpoint_pair_id': checkpoint_pair_id,
                        'training_run_id': training_run_id,
                        'combined_reward': float(episode_reward),
                        'episode_reward_ev': float(episode_reward_ev),
                        'episode_reward_aev': float(episode_reward_aev),
                    }
                    print(f"🏆 New best total reward: {best_reward:.2f} at cumulative episode {global_episode}, saving paired checkpoints...")
                    if checkpoint_base is not None:
                        self._save_q_network_checkpoint(
                            value_function_ev,
                            global_episode,
                            checkpoint_dir=f"{checkpoint_base}_ev{enc_suffix}",
                            checkpoint_tag="best",
                            checkpoint_metadata=paired_metadata,
                        )
                        self._save_q_network_checkpoint(
                            value_function,
                            global_episode,
                            checkpoint_dir=f"{checkpoint_base}_aev{enc_suffix}",
                            checkpoint_tag="best",
                            checkpoint_metadata=paired_metadata,
                        )

                if checkpoint_base is not None and episode_reward_ev > best_reward_ev:
                    best_reward_ev = episode_reward_ev
                    print(f"🏆 New best EV reward: {best_reward_ev:.2f} at episode {episode + 1}, saving EV-only best checkpoint...")
                    self._save_q_network_checkpoint(
                        value_function_ev,
                        global_episode,
                        checkpoint_dir=f"{checkpoint_base}_ev{enc_suffix}",
                        checkpoint_tag="best_ev",
                    )

                if checkpoint_base is not None and episode_reward_aev > best_reward_aev:
                    best_reward_aev = episode_reward_aev
                    print(f"🏆 New best AEV reward: {best_reward_aev:.2f} at episode {episode + 1}, saving AEV-only best checkpoint...")
                    self._save_q_network_checkpoint(
                        value_function,
                        global_episode,
                        checkpoint_dir=f"{checkpoint_base}_aev{enc_suffix}",
                        checkpoint_tag="best_aev",
                    )

                # Keep a recoverable paired continuation point after every
                # completed episode.  This is distinct from the paired
                # combined-best tag and prevents an interrupted long run from
                # falling back to two independently selected checkpoints.
                if checkpoint_base is not None:
                    latest_pair_id = f"{training_run_id}:latest:{global_episode}"
                    latest_metadata = {
                        'checkpoint_pair_id': latest_pair_id,
                        'training_run_id': training_run_id,
                        'combined_reward': float(episode_reward),
                        'episode_reward_ev': float(episode_reward_ev),
                        'episode_reward_aev': float(episode_reward_aev),
                    }
                    self._save_q_network_checkpoint(
                        value_function_ev,
                        global_episode,
                        checkpoint_dir=f"{checkpoint_base}_ev{enc_suffix}",
                        checkpoint_tag="latest",
                        checkpoint_metadata=latest_metadata,
                    )
                    self._save_q_network_checkpoint(
                        value_function,
                        global_episode,
                        checkpoint_dir=f"{checkpoint_base}_aev{enc_suffix}",
                        checkpoint_tag="latest",
                        checkpoint_metadata=latest_metadata,
                    )

            results['Idle_average'].append(sum(Idle_list) / len(Idle_list) if Idle_list else 0)
            results['episode_rewards'].append(episode_reward)
            results['episode_rewards_aev'].append(episode_reward_aev)
            results['episode_rewards_ev'].append(episode_reward_ev)
            results['charging_events'].extend(episode_charging_events)
            results['value_function_losses'].append(np.mean(episode_losses) if episode_losses else 0.0)
            results['value_function_ev_losses'].append(np.mean(episode_losses_ev) if episode_losses_ev else 0.0)
            results['qvalue_losses'].extend(episode_losses)
            results['qvalue_ev_losses'].extend(episode_losses_ev)
            stats = env.get_stats()
            results['active_requests'] = stats['active_requests']
            results['environment_stats'].append(stats)
            results['battery_levels'].append(stats['average_battery'])
            results['completed_requests'] = stats['completed_requests']
            results['avg_requestvalue'] = stats['completed_orders_req']
            episode_stats = env.get_episode_stats()
            episode_stats['episode_number'] = episode + 1
            episode_stats['episode_reward'] = episode_reward
            episode_stats['episode_reward_aev'] = episode_reward_aev
            episode_stats['episode_reward_ev'] = episode_reward_ev
            episode_stats['charging_events_count'] = len(episode_charging_events)
            for vehicle_group, current_value_function in (
                ('aev', value_function),
                ('ev', value_function_ev),
            ):
                queue_losses = list(getattr(current_value_function, 'queue_training_mse_losses', [])) if current_value_function is not None else []
                episode_stats[f'queue_predictor_training_mse_{vehicle_group}'] = (
                    float(queue_losses[-1]) if queue_losses else 0.0
                )
                episode_stats[f'queue_predictor_trained_{vehicle_group}'] = bool(
                    getattr(current_value_function, 'queue_predictor_trained', False)
                ) if current_value_function is not None else False
                post_demand_losses = list(
                    getattr(current_value_function, 'post_demand_training_mse_losses', [])
                ) if current_value_function is not None else []
                episode_stats[f'post_demand_predictor_training_mse_{vehicle_group}'] = (
                    float(post_demand_losses[-1]) if post_demand_losses else 0.0
                )
                episode_stats[f'post_demand_predictor_trained_{vehicle_group}'] = bool(
                    getattr(current_value_function, 'post_demand_predictor_trained', False)
                ) if current_value_function is not None else False
                critic = getattr(current_value_function, 'network', None)
                action_weights = getattr(critic, 'action_weights', None)
                if action_weights is not None:
                    weight_values = action_weights.detach().cpu().numpy().reshape(-1)
                else:
                    weight_values = np.zeros(3, dtype=np.float32)
                for action_name, weight_value in zip(
                    ('reloc', 'request', 'charge'),
                    weight_values,
                ):
                    episode_stats[
                        f'post_demand_weight_{action_name}_{vehicle_group}'
                    ] = float(weight_value)
                network = getattr(current_value_function, 'network', None)
                if network is not None and hasattr(network, 'queue_input_weight_stats'):
                    queue_norm, queue_abs_mean = network.queue_input_weight_stats()
                else:
                    queue_norm, queue_abs_mean = 0.0, 0.0
                episode_stats[f'queue_input_weight_norm_{vehicle_group}'] = float(queue_norm)
                episode_stats[f'queue_input_weight_abs_mean_{vehicle_group}'] = float(queue_abs_mean)
                training_history = list(
                    getattr(current_value_function, 'q_values_history', [])
                ) if current_value_function is not None else []
                latest_training = training_history[-1] if training_history else {}
                replay_with_candidates = int(
                    getattr(
                        current_value_function,
                        'replay_rows_with_candidates',
                        0,
                    ) or 0
                ) if current_value_function is not None else 0
                replay_without_candidates = int(
                    getattr(
                        current_value_function,
                        'replay_rows_without_candidates',
                        0,
                    ) or 0
                ) if current_value_function is not None else 0
                replay_candidate_total = (
                    replay_with_candidates + replay_without_candidates
                )
                alpha_tensor = getattr(current_value_function, 'alpha', None)
                alpha_value = (
                    float(alpha_tensor.detach().item())
                    if hasattr(alpha_tensor, 'detach') else 0.0
                )
                latest_training = {
                    'alpha': alpha_value,
                    'candidate_replay_coverage': (
                        replay_with_candidates / replay_candidate_total
                        if replay_candidate_total else 0.0
                    ),
                    'candidate_replay_rows': replay_with_candidates,
                    'missing_candidate_replay_rows': replay_without_candidates,
                    **latest_training,
                }
                for metric_name in (
                    'alpha',
                    'actor',
                    'critic',
                    'candidate_replay_coverage',
                    'candidate_replay_rows',
                    'missing_candidate_replay_rows',
                ):
                    episode_stats[
                        f'masac_{metric_name}_{vehicle_group}'
                    ] = float(latest_training.get(metric_name, 0.0) or 0.0)
            rebalancing_calls = episode_stats.get('total_rebalancing_calls', 0)
            total_assignments = episode_stats.get('total_rebalancing_assignments', 0)
            avg_assignments = episode_stats.get('avg_rebalancing_assignments_per_call', 0)
            avg_whole = episode_stats.get('avg_rebalancing_assignments_per_whole', 0)
            
            # 计算loss
            avg_loss_aev = np.mean(episode_losses) if episode_losses else 0.0
            avg_loss_ev = np.mean(episode_losses_ev) if episode_losses_ev else 0.0
            
            # 单行格式化输出，便于Linux查看
            print(f"{episode+1:4d} | {episode_reward:8.2f} | AEV={episode_reward_aev:8.2f} | EV={episode_reward_ev:8.2f} | {avg_loss_aev:11.4f} | {avg_loss_ev:11.4f} | {episode_stats['total_orders']:6d} | {episode_stats['accepted_orders']:7d} | {episode_stats['rejected_orders']:7d} | {episode_stats['completed_orders']:9d} | RejectedReq={episode_stats.get('rejected_requests', 0):4d} | RecourseReq={episode_stats.get('recourse_requests', 0):4d} | LostReq={episode_stats.get('lost_requests', 0):4d} | EVComp={episode_stats.get('completed_ev_orders', 0):4d} | AEVComp={episode_stats.get('completed_aev_orders', 0):4d} | AvgWait={episode_stats.get('avg_wait', 0.0):6.2f} | WaitVeh={episode_stats.get('waiting_vehicle_count', 0):4d} | Alpha={episode_stats.get('masac_alpha_aev', 0.0):6.4f} | Actor={episode_stats.get('masac_actor_aev', 0.0):8.4f} | Cand={episode_stats.get('masac_candidate_replay_coverage_aev', 0.0):5.1%} | Online={episode_stats.get('online_vehicles', 0):3d} | Offline={episode_stats.get('offline_vehicles', 0):3d} | {episode_stats['avg_battery_level']:8.2f} | DropOff: {episode_stats.get('drop_off_rate', 0.0):6.3f} | Rebal: {rebalancing_calls}/{total_assignments}")
            
            # 写入日志文件，包含mode和demand信息
            demand_type = "intense" if use_intense_requests else "random"
            self.logger.info(
                f"Episode {episode+1} | Mode: {transportation_mode} | Demand: {demand_type} | "
                f"Reward: {episode_reward:.2f} | Reward(AEV): {episode_reward_aev:.2f} | Reward(EV): {episode_reward_ev:.2f} | Loss(AEV): {avg_loss_aev:.4f} | Loss(EV): {avg_loss_ev:.4f} | "
                f"Accept: {episode_stats['accepted_orders']} | Reject: {episode_stats['rejected_orders']} | "
                f"RejectedRequests: {episode_stats.get('rejected_requests', 0)} | RecourseRequests: {episode_stats.get('recourse_requests', 0)} | LostRequests: {episode_stats.get('lost_requests', 0)} | "
                f"Complete: {episode_stats['completed_orders']} | Complete(EV): {episode_stats.get('completed_ev_orders', 0)} | Complete(AEV): {episode_stats.get('completed_aev_orders', 0)} | "
                f"AvgWait: {episode_stats.get('avg_wait', 0.0):.2f} | WaitingVehicles: {episode_stats.get('waiting_vehicle_count', 0)} | "
                f"Alpha(AEV): {episode_stats.get('masac_alpha_aev', 0.0):.4f} | Actor(AEV): {episode_stats.get('masac_actor_aev', 0.0):.4f} | CandidateCoverage(AEV): {episode_stats.get('masac_candidate_replay_coverage_aev', 0.0):.3f} | "
                f"Online: {episode_stats.get('online_vehicles', 0)} | Offline: {episode_stats.get('offline_vehicles', 0)} | Battery: {episode_stats['avg_battery_level']:.2f} | DropOffRate: {episode_stats.get('drop_off_rate', 0.0):.4f}"
            )

            if use_neural_network:
                episode_stats['neural_network_loss'] = np.mean(episode_losses) if episode_losses else 0.0
                episode_stats['neural_evnetwork_loss'] = np.mean(episode_losses_ev) if episode_losses_ev else 0.0
                episode_stats['neural_network_loss_std'] = np.std(episode_losses) if episode_losses else 0.0
                episode_stats['training_steps_in_episode'] = len(episode_losses)
                follower_value_function = None
                follower_episode_losses = []
                if transportation_mode == 'evfirst':
                    follower_value_function = value_function
                    follower_episode_losses = episode_losses
                elif transportation_mode == 'aevfirst':
                    follower_value_function = value_function_ev
                    follower_episode_losses = episode_losses_ev
                else:
                    follower_value_function = value_function
                    follower_episode_losses = episode_losses

                if getattr(follower_value_function, 'uses_elbo_posterior', False):
                    if hasattr(follower_value_function, 'zone_dist_elbo_losses') and follower_value_function.zone_dist_elbo_losses:
                        episode_stats['elbo_loss'] = np.mean(follower_value_function.zone_dist_elbo_losses[-len(follower_episode_losses):]) if follower_episode_losses else np.mean(follower_value_function.zone_dist_elbo_losses)
                    else:
                        episode_stats['elbo_loss'] = 0.0
                    if hasattr(follower_value_function, 'zone_dist_kl_losses') and follower_value_function.zone_dist_kl_losses:
                        episode_stats['posterior_kl_loss'] = np.mean(follower_value_function.zone_dist_kl_losses[-len(follower_episode_losses):]) if follower_episode_losses else np.mean(follower_value_function.zone_dist_kl_losses)
                    else:
                        episode_stats['posterior_kl_loss'] = 0.0
                else:
                    if hasattr(follower_value_function, 'zone_dist_losses') and follower_value_function.zone_dist_losses:
                        episode_stats['posterior_kl_loss'] = np.mean(follower_value_function.zone_dist_losses[-len(follower_episode_losses):]) if follower_episode_losses else np.mean(follower_value_function.zone_dist_losses)
                    else:
                        episode_stats['posterior_kl_loss'] = 0.0
                    episode_stats['elbo_loss'] = 0.0
                if hasattr(follower_value_function, 'time_zone_dist_losses') and follower_value_function.time_zone_dist_losses:
                    episode_stats['prior_kl_loss'] = np.mean(follower_value_function.time_zone_dist_losses[-len(follower_episode_losses):]) if follower_episode_losses else np.mean(follower_value_function.time_zone_dist_losses)
                else:
                    episode_stats['prior_kl_loss'] = 0.0
                if len(value_function.experience_buffer) > 0:
                    try:
                        aev_vehicle_id = next((vid for vid, vehicle in env.vehicles.items() if vehicle.get('type') == 2), None)
                        ev_vehicle_id = next((vid for vid, vehicle in env.vehicles.items() if vehicle.get('type') == 1), None)

                        if aev_vehicle_id is not None:
                            aev_vehicle = env.vehicles[aev_vehicle_id]
                            aev_location = aev_vehicle['location']
                            aev_battery = aev_vehicle['battery']
                            episode_stats['sample_idle_q_value_aev'] = value_function.get_idle_q_value(aev_vehicle_id, aev_location, aev_location, aev_battery, current_time=env.current_time)
                            episode_stats['sample_assign_q_value_aev'] = value_function.get_q_value(aev_vehicle_id, "assign_1", aev_location, aev_location + 1, current_time=env.current_time, battery_level=aev_battery)
                            episode_stats['sample_charge_q_value_aev'] = value_function.get_q_value(aev_vehicle_id, "charge_1", aev_location, aev_location + 5, current_time=env.current_time, battery_level=aev_battery)
                        else:
                            episode_stats['sample_idle_q_value_aev'] = 0.0
                            episode_stats['sample_assign_q_value_aev'] = 0.0
                            episode_stats['sample_charge_q_value_aev'] = 0.0

                        if ev_vehicle_id is not None:
                            ev_vehicle = env.vehicles[ev_vehicle_id]
                            ev_location = ev_vehicle['location']
                            ev_battery = ev_vehicle['battery']
                            episode_stats['sample_idle_q_value_ev'] = value_function_ev.get_idle_q_value(ev_vehicle_id, ev_location, ev_location, ev_battery, current_time=env.current_time)
                            episode_stats['sample_assign_q_value_ev'] = value_function_ev.get_q_value(ev_vehicle_id, "assign_1", ev_location, ev_location + 1, current_time=env.current_time, battery_level=ev_battery)
                            episode_stats['sample_charge_q_value_ev'] = value_function_ev.get_q_value(ev_vehicle_id, "charge_1", ev_location, ev_location + 5, current_time=env.current_time, battery_level=ev_battery)
                        else:
                            episode_stats['sample_idle_q_value_ev'] = 0.0
                            episode_stats['sample_assign_q_value_ev'] = 0.0
                            episode_stats['sample_charge_q_value_ev'] = 0.0

                        episode_stats['sample_idle_q_value'] = episode_stats['sample_idle_q_value_aev']
                        episode_stats['sample_assign_q_value'] = episode_stats['sample_assign_q_value_aev']
                        episode_stats['sample_charge_q_value'] = episode_stats['sample_charge_q_value_aev']
                    except Exception:
                        episode_stats['sample_idle_q_value'] = 0.0
                        episode_stats['sample_assign_q_value'] = 0.0
                        episode_stats['sample_charge_q_value'] = 0.0
                        episode_stats['sample_idle_q_value_aev'] = 0.0
                        episode_stats['sample_assign_q_value_aev'] = 0.0
                        episode_stats['sample_charge_q_value_aev'] = 0.0
                        episode_stats['sample_idle_q_value_ev'] = 0.0
                        episode_stats['sample_assign_q_value_ev'] = 0.0
                        episode_stats['sample_charge_q_value_ev'] = 0.0
                else:
                    episode_stats['sample_idle_q_value'] = 0.0
                    episode_stats['sample_assign_q_value'] = 0.0
                    episode_stats['sample_charge_q_value'] = 0.0
                    episode_stats['sample_idle_q_value_aev'] = 0.0
                    episode_stats['sample_assign_q_value_aev'] = 0.0
                    episode_stats['sample_charge_q_value_aev'] = 0.0
                    episode_stats['sample_idle_q_value_ev'] = 0.0
                    episode_stats['sample_assign_q_value_ev'] = 0.0
                    episode_stats['sample_charge_q_value_ev'] = 0.0
            else:
                episode_stats['neural_network_loss'] = 0.0
                episode_stats['neural_network_loss_std'] = 0.0
                episode_stats['training_steps_in_episode'] = 0
                episode_stats['sample_idle_q_value'] = 0.0
                episode_stats['sample_assign_q_value'] = 0.0
                episode_stats['sample_charge_q_value'] = 0.0
                episode_stats['sample_idle_q_value_aev'] = 0.0
                episode_stats['sample_assign_q_value_aev'] = 0.0
                episode_stats['sample_charge_q_value_aev'] = 0.0
                episode_stats['sample_idle_q_value_ev'] = 0.0
                episode_stats['sample_assign_q_value_ev'] = 0.0
                episode_stats['sample_charge_q_value_ev'] = 0.0
                episode_stats['elbo_loss'] = 0.0
                episode_stats['posterior_kl_loss'] = 0.0
                episode_stats['prior_kl_loss'] = 0.0

            results['sample_assign_q_values_aev'].append(episode_stats.get('sample_assign_q_value_aev', 0.0))
            results['sample_assign_q_values_ev'].append(episode_stats.get('sample_assign_q_value_ev', 0.0))
            results['drop_off_rates'].append(episode_stats.get('drop_off_rate', 0.0))
            results['episode_rejected_requests'].append(episode_stats.get('rejected_requests', 0))
            results['episode_recourse_requests'].append(episode_stats.get('recourse_requests', 0))
            results['episode_lost_requests'].append(episode_stats.get('lost_requests', 0))
            results['episode_detailed_stats'].append(episode_stats)

            # Record and report per-episode runtime
            episode_time = time.time() - episode_start
            results['episode_times'].append(episode_time)
            episode_stats['episode_time_sec'] = episode_time
            print(f"⏱ Episode {episode+1} time: {episode_time:.2f} s")
            self.logger.info(f"Episode {episode+1} time: {episode_time:.2f} s")

            if 'charging_usage_history' in episode_stats and episode_stats['charging_usage_history']:
                charging_history = episode_stats['charging_usage_history']
                avg_usage = sum(h['vehicles_per_station'] for h in charging_history) / len(charging_history)
                max_usage = max(h['vehicles_per_station'] for h in charging_history)
                min_usage = min(h['vehicles_per_station'] for h in charging_history)
                print(
                    f"  Charging History: {len(charging_history)} time steps, Avg: {avg_usage:.2f}, Max: {max_usage:.2f}, Min: {min_usage:.2f} vehicles/station"
                )

            vehicle_visit_stats = self._analyze_vehicle_visit_patterns(env)
            results['vehicle_visit_stats'].append(vehicle_visit_stats)
            torch.cuda.empty_cache()

        if save_checkpoints and use_neural_network and value_function is not None and value_function_ev is not None:
            episode = num_episodes - 1  # 最后一个episode的索引
            global_episode = resume_episode_offset + episode + 1
            checkpoint_pair_id = f"{training_run_id}:latest:{global_episode}"
            latest_metadata = {
                'checkpoint_pair_id': checkpoint_pair_id,
                'training_run_id': training_run_id,
                'combined_reward': float(results['episode_rewards'][-1]),
                'episode_reward_ev': float(results['episode_rewards_ev'][-1]),
                'episode_reward_aev': float(results['episode_rewards_aev'][-1]),
            }
            assign_tag = "gurobi" if assignmentgurobi else "heu"
            enc_suffix = get_distribution_suffix()
            if transportation_mode == 'integrated':
                self._save_q_network_checkpoint(
                    value_function, global_episode, checkpoint_dir=f"checkpoints/q_networks_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}", checkpoint_metadata=latest_metadata
                )
                self._save_q_network_checkpoint(
                    value_function_ev, global_episode, checkpoint_dir=f"checkpoints/q_networks_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}", checkpoint_metadata=latest_metadata
                )
            elif transportation_mode == 'evfirst':
                self._save_q_network_checkpoint(
                    value_function_ev, global_episode, checkpoint_dir=f"checkpoints/q_networksevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}", checkpoint_metadata=latest_metadata
                )
                self._save_q_network_checkpoint(
                    value_function, global_episode, checkpoint_dir=f"checkpoints/q_networksevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}", checkpoint_metadata=latest_metadata
                )
            elif transportation_mode == 'aevfirst':
                self._save_q_network_checkpoint(
                    value_function, global_episode, checkpoint_dir=f"checkpoints/q_networksaevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_aev{enc_suffix}", checkpoint_metadata=latest_metadata
                )
                self._save_q_network_checkpoint(
                    value_function_ev, global_episode, checkpoint_dir=f"checkpoints/q_networksaevfirst_{assign_tag}_{transportation_mode}_{num_ev}_{use_intense_requests}_ev{enc_suffix}", checkpoint_metadata=latest_metadata
                )









        print("="*120)
        print("\n=== Integration Test Complete ===")
        print(f"📊 Final Summary:")
        print(f"   Total episodes: {num_episodes}")
        print(f"   Average reward: {np.mean(results['episode_rewards']):.2f}")
        print(f"   Total reward: {sum(results['episode_rewards']):.2f}")
        if results['episode_times']:
            avg_time = np.mean(results['episode_times'])
            last_time = results['episode_times'][-1]
            print(f"   Avg episode time: {avg_time:.2f} s")
            print(f"   Last episode time: {last_time:.2f} s")
        if use_neural_network:
            avg_loss_aev = np.mean(results['value_function_losses'])
            avg_loss_ev = np.mean(results['value_function_ev_losses'])
            print(f"   Avg loss (AEV): {avg_loss_aev:.4f}")
            print(f"   Avg loss (EV): {avg_loss_ev:.4f}")
            print(f"   Neural network parameters: {sum(p.numel() for p in value_function.network.parameters())}")
            if results['sample_assign_q_values_aev']:
                print(f"   AEV sample assign Q: {results['sample_assign_q_values_aev'][0]:.4f} -> {results['sample_assign_q_values_aev'][-1]:.4f}")
            if results['sample_assign_q_values_ev']:
                print(f"   EV sample assign Q: {results['sample_assign_q_values_ev'][0]:.4f} -> {results['sample_assign_q_values_ev'][-1]:.4f}")
        else:
            print(f"   Neural network training: DISABLED")

        if results['drop_off_rates']:
            print(f"   Avg drop-off rate: {np.mean(results['drop_off_rates']):.4f}")
        
        # 汇总订单统计
        total_orders = sum(ep['total_orders'] for ep in results['episode_detailed_stats'])
        total_accepted = sum(ep['accepted_orders'] for ep in results['episode_detailed_stats'])
        total_rejected = sum(ep['rejected_orders'] for ep in results['episode_detailed_stats'])
        total_completed = sum(ep['completed_orders'] for ep in results['episode_detailed_stats'])
        print(f"   Orders: Total={total_orders}, Accept={total_accepted}, Reject={total_rejected}, Complete={total_completed}")
        print(f"   Accept rate: {100*total_accepted/total_orders if total_orders > 0 else 0:.1f}%")
        print(f"   Complete rate: {100*total_completed/total_orders if total_orders > 0 else 0:.1f}%")
        if results['episode_detailed_stats']:
            wait_summary = aggregate_wait_metrics(results['episode_detailed_stats'])
            print(
                "   Avg wait among waiting vehicles: "
                f"{wait_summary['avg_wait']:.2f} steps"
            )

        excel_path = None
        spatial_path = None
        if save_results:
            results_dir = Path("results/integrated_tests") if assignmentgurobi else Path("results/integrated_tests_h")
            results_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Results will be saved to: {results_dir}")

            # 获取最后一个episode的vehicle_visit_stats（它是一个字典）
            last_vehicle_visit_stats = results['vehicle_visit_stats'][-1] if results['vehicle_visit_stats'] else None
            excel_path, spatial_path = self._save_episode_stats_to_excel(
                env,
                results['episode_detailed_stats'],
                results_dir,
                last_vehicle_visit_stats,
                transportation_mode,
                effective_zone_distribution_mode,
            )
        results['excel_path'] = excel_path
        results['spatial_image_path'] = spatial_path

        results["optimizer_budget"] = {
            "shared_critic": bool(value_function is value_function_ev),
            "optimizer_steps_total": int(
                sum(
                    getattr(vf, "optimizer_steps_total", 0)
                    for vf in {
                        id(value_function): value_function,
                        id(value_function_ev): value_function_ev,
                    }.values()
                    if vf is not None
                )
            ),
            "optimizer_steps_aev": int(
                getattr(value_function, "optimizer_steps_total", 0)
                if value_function is not None
                else 0
            ),
            "optimizer_steps_ev": int(
                getattr(value_function_ev, "optimizer_steps_total", 0)
                if value_function_ev is not None
                else 0
            ),
            "joint_updates": int(
                sum(
                    getattr(vf, "optimizer_steps_joint", 0)
                    for vf in {
                        id(value_function): value_function,
                        id(value_function_ev): value_function_ev,
                    }.values()
                    if vf is not None
                )
            ),
            "edge_updates": int(
                sum(
                    getattr(vf, "optimizer_steps_edge", 0)
                    for vf in {
                        id(value_function): value_function,
                        id(value_function_ev): value_function_ev,
                    }.values()
                    if vf is not None
                )
            ),
            "queue_updates": int(
                sum(
                    getattr(vf, "optimizer_steps_queue", 0)
                    for vf in {
                        id(value_function): value_function,
                        id(value_function_ev): value_function_ev,
                    }.values()
                    if vf is not None
                )
            ),
        }
        return results, env
    
