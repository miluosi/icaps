from .LearningAgent import LearningAgent
from .Action import Action
from .Request import Request
from .Path import PathNode, RequestInfo
import time
from typing import Type, List, Generator, Tuple, Deque, Dict
import math
from abc import ABCMeta, abstractmethod
from random import choice, randint
from pandas import read_csv
from collections import deque
import gurobipy as gp  # type: ignore
from gurobipy import GRB  # type: ignore
import re
import random
import hashlib
import numpy as np
import math
from .charging_station import ChargingStationManager, ChargingStation
from .charging_metrics import charging_session_metrics
from .Action import Action, ChargingAction, ServiceAction, IdleAction
from src.GurobiOptimizer import GurobiOptimizer
from src.charging_wait_metrics import positive_wait_metrics
from src.qvalue_precision import qvalue_rounding_diagnostics, round_qvalue_matrix
from src.recourse.coordinator import RecourseCoordinator
from src.recourse.lifecycle import RequestLifecycleTracker
from src.recourse.state_snapshot import StateSnapshotBuilder
from src.recourse.target_builder import RecourseTargetBuilder
from src.recourse.types import JointActionSnapshot, RequestSnapshot
import time
class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    REQUEST_HISTORY_SIZE: int = 1000

    def __init__(self, NUM_LOCATIONS: int, MAX_CAPACITY: int, EPOCH_LENGTH: float, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float, DATA_DIR: str):
        # Load environment
        self.NUM_LOCATIONS = NUM_LOCATIONS
        self.MAX_CAPACITY = MAX_CAPACITY
        self.EPOCH_LENGTH = EPOCH_LENGTH
        self.NUM_AGENTS = NUM_AGENTS
        self.START_EPOCH = START_EPOCH
        self.STOP_EPOCH = STOP_EPOCH
        self.DATA_DIR = DATA_DIR
        self.adp_value = 0.5
        self.num_days_trained = 0
        self.recent_request_history: Deque[Request] = deque(maxlen=self.REQUEST_HISTORY_SIZE)
        self.current_time: float = 0.0
        self.idle_vehicle_requirement = 1
        self.idle_penalty = 0.5
        self.charging_penalty = -1.0
        self.chargeincrease_per_epoch = 0.1  # Battery increase per epoch when charging
        self.min_battery_level = 0.2
        self.complete_ratio_reward = 0.5
        # Q-learning components for ADP integration
        self.q_table = {}  # Q-table for state-action values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate

    def _round_assignment_qvalues(self, values):
        """Put every assignment Q matrix on the shared exact-solver grid."""

        rounded = round_qvalue_matrix(
            values,
            int(getattr(self, 'mcmf_cost_scale', 10_000)),
        )
        self.qvalue_precision_last = qvalue_rounding_diagnostics(
            values,
            rounded,
            int(getattr(self, 'mcmf_cost_scale', 10_000)),
        )
        self.qvalue_precision_last["shape"] = tuple(rounded.shape)
        return rounded
    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

    @abstractmethod
    def get_request_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_travel_time(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_next_location(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_initial_states(self, num_agents, is_training):
        raise NotImplementedError

    def simulate_motion(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True) -> None:
        # Move all agents
        agents_to_rebalance: List[Tuple[LearningAgent, float]] = []
        for agent in agents:
            time_remaining: float = self.EPOCH_LENGTH
            time_remaining = self._move_agent(agent, time_remaining)
            # If it has visited all the locations it needs to and has time left, rebalance
            if (time_remaining > 0):
                agents_to_rebalance.append((agent, time_remaining))

        # Update recent_requests list
        self.update_recent_requests(current_requests)

        # Perform Rebalancing
        if (rebalance and agents_to_rebalance):
            rebalancing_targets = self._get_rebalance_targets([agent for agent, _ in agents_to_rebalance])

            # Move cars according to the rebalancing_targets
            for idx, target in enumerate(rebalancing_targets):
                agent, time_remaining = agents_to_rebalance[idx]

                # Insert dummy target
                agent.path.requests.append(RequestInfo(target, False, True))
                agent.path.request_order.append(PathNode(False, 0))  # adds pickup location to 'to-visit' list

                # Move according to dummy target
                self._move_agent(agent, time_remaining)

                # Undo impact of creating dummy target
                agent.path.request_order.clear()
                agent.path.requests.clear()
                agent.path.current_capacity = 0
                agent.path.total_delay = 0

    def _move_agent(self, agent: LearningAgent, time_remaining: float) -> float:
        while(time_remaining >= 0):
            time_remaining -= agent.position.time_to_next_location

            # If we reach an intersection, make a decision about where to go next
            if (time_remaining >= 0):
                # If the intersection is an existing pick-up or drop-off location, update the Agent's path
                if (agent.position.next_location == agent.path.get_next_location()):
                    agent.path.visit_next_location(self.current_time + self.EPOCH_LENGTH - time_remaining)

                # Go to the next location in the path, if it exists
                if (not agent.path.is_empty()):
                    next_location = self.get_next_location(agent.position.next_location, agent.path.get_next_location())
                    agent.position.time_to_next_location = self.get_travel_time(agent.position.next_location, next_location)
                    agent.position.next_location = next_location

                # If no additional locations need to be visited, stop
                else:
                    agent.position.time_to_next_location = 0
                    break
            # Else, continue down the road you're on
            else:
                agent.position.time_to_next_location -= (time_remaining + agent.position.time_to_next_location)

        return time_remaining

    def get_state_representation(self, agent_position: int, target_position: int, 
                                current_time: float) -> str:
        """Get state representation for Q-learning"""
        # Discretize time for state representation
        time_slot = int(current_time // 10)  # 10-minute time slots
        return f"{agent_position}_{target_position}_{time_slot}"
    
    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair"""
        key = f"{state}_{action}"
        return self.q_table.get(key, 0.0)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm"""
        key = f"{state}_{action}"
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state (assuming action is assignment)
        next_q_values = [self.get_q_value(next_state, f"assign_{i}") for i in range(10)]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[key] = new_q
    
    def get_assignment_q_value(self, agent_id: int, target_id: int, 
                              agent_position: int, target_position: int) -> float:
        """Get Q-value for a specific assignment"""
        state = self.get_state_representation(agent_position, target_position, self.current_time)
        action = f"assign_{target_id}"
        return self.get_q_value(state, action)
    
    def _get_rebalance_targets(self, agents: List) -> List:
        """Get rebalancing targets using Gurobi optimization with Q-learning integration"""
        # Get a list of possible targets by sampling from recent_requests
        possible_targets: List[Request] = []
        num_targets = min(500, len(agents))
        for _ in range(num_targets):
            target = choice(self.recent_request_history)
            possible_targets.append(target)

        # Solve an LP to assign each agent to closest possible target
        model = gp.Model()
        model.setParam('OutputFlag', 0)  # Suppress output

        # Define variables, a matrix defining the assignment of agents to targets
        assignments = {}
        for agent_id in range(len(agents)):
            for target_id in range(len(possible_targets)):
                assignments[agent_id, target_id] = model.addVar(vtype=GRB.Binary, 
                                                              name=f'assignment_{agent_id}_{target_id}')

        # Make sure one agent can only be assigned to one target
        for agent_id in range(len(agents)):
            model.addConstr(gp.quicksum(assignments[agent_id, target_id] 
                                      for target_id in range(len(possible_targets))) == 1)

        # Make sure one target can only be assigned to *ratio* agents
        num_fractional_targets = len(agents) - (int(len(agents) / num_targets) * num_targets)
        for target_id in range(len(possible_targets)):
            num_agents_to_target = int(len(agents) / num_targets) + (1 if target_id < num_fractional_targets else 0)
            model.addConstr(gp.quicksum(assignments[agent_id, target_id] 
                                      for agent_id in range(len(agents))) == num_agents_to_target)

        # Define the objective: Combine distance cost and Q-value benefit
        distance_weight = 0.7
        q_weight = 0.3
        
        # Distance cost component
        distance_obj = gp.quicksum(assignments[agent_id, target_id] * 
                                 self.get_travel_time(agents[agent_id].position.next_location, 
                                                    possible_targets[target_id].pickup) 
                                 for target_id in range(len(possible_targets)) 
                                 for agent_id in range(len(agents)))
        
        # Q-value benefit component (negative because we want to maximize benefit)
        q_value_obj = gp.quicksum(assignments[agent_id, target_id] * 
                                (-self.get_assignment_q_value(agent_id, target_id,
                                                            agents[agent_id].position.next_location,
                                                            possible_targets[target_id].pickup))
                                for target_id in range(len(possible_targets)) 
                                for agent_id in range(len(agents)))
        
        # Combined objective function
        obj = distance_weight * distance_obj + q_weight * q_value_obj
        model.setObjective(obj, GRB.MINIMIZE)

        # Solve
        model.optimize()
        assert model.status == GRB.OPTIMAL  # making sure that the model doesn't fail

        # Get the assigned targets
        assigned_targets: List[Request] = []
        for agent_id in range(len(agents)):
            for target_id in range(len(possible_targets)):
                if (assignments[agent_id, target_id].x == 1):
                    assigned_targets.append(possible_targets[target_id])
                    break

        return assigned_targets

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to an agent for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        return sum([request.value for request in action.requests])

    def update_recent_requests(self, recent_requests: List[Request]):
        self.recent_request_history.extend(recent_requests)


class NYEnvironment(Environment):
    """Define an Environment using the cleaned NYC Yellow Cab dataset."""

    NUM_MAX_AGENTS: int = 3000
    NUM_LOCATIONS: int = 4461

    def __init__(self, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float, MAX_CAPACITY, DATA_DIR: str='../data/ny/', EPOCH_LENGTH: float = 60.0):
        super().__init__(NUM_LOCATIONS=self.NUM_LOCATIONS, MAX_CAPACITY=MAX_CAPACITY, EPOCH_LENGTH=EPOCH_LENGTH, NUM_AGENTS=NUM_AGENTS, START_EPOCH=START_EPOCH, STOP_EPOCH=STOP_EPOCH, DATA_DIR=DATA_DIR)
        self.initialise_environment()

    def initialise_environment(self):
        print('Loading Environment...')

        TRAVELTIME_FILE: str = self.DATA_DIR + 'zone_traveltime.csv'
        self.travel_time = read_csv(TRAVELTIME_FILE, header=None).values

        SHORTESTPATH_FILE: str = self.DATA_DIR + 'zone_path.csv'
        self.shortest_path = read_csv(SHORTESTPATH_FILE, header=None).values

        IGNOREDZONES_FILE: str = self.DATA_DIR + 'ignorezonelist.txt'
        self.ignored_zones = read_csv(IGNOREDZONES_FILE, header=None).values.flatten()

        INITIALZONES_FILE: str = self.DATA_DIR + 'taxi_3000_final.txt'
        self.initial_zones = read_csv(INITIALZONES_FILE, header=None).values.flatten()

        assert (self.EPOCH_LENGTH == 60) or (self.EPOCH_LENGTH == 30) or (self.EPOCH_LENGTH == 10)
        self.DATA_FILE_PREFIX: str = "{}files_{}sec/test_flow_5000_".format(self.DATA_DIR, int(self.EPOCH_LENGTH))

    def get_request_batch(self,
                          day: int=2,
                          downsample: float=1) -> Generator[List[Request], None, None]:

        assert 0 < downsample <= 1
        request_id = 0

        def is_in_time_range(current_time):
            current_hour = int(current_time / 3600)
            return True if (current_hour >= self.START_EPOCH / 3600 and current_hour < self.STOP_EPOCH / 3600) else False

        # Open file to read
        with open(self.DATA_FILE_PREFIX + str(day) + '.txt', 'r') as data_file:
            num_batches: int = int(data_file.readline().strip())

            # Defines the 2 possible RE for lines in the data file
            new_epoch_re = re.compile(r'Flows:(\d+)-\d+')
            request_re = re.compile(r'(\d+),(\d+),(\d+)\.0')

            # Parsing rest of the file
            request_list: List[Request] = []
            is_first_epoch = True
            for line in data_file.readlines():
                line = line.strip()

                is_new_epoch = re.match(new_epoch_re, line)
                if (is_new_epoch is not None):
                    if not is_first_epoch:
                        if is_in_time_range(self.current_time):
                            yield request_list
                        request_list.clear()  # starting afresh for new batch
                    else:
                        is_first_epoch = False

                    current_epoch = int(is_new_epoch.group(1))
                    self.current_time = current_epoch * self.EPOCH_LENGTH
                else:
                    request_data = re.match(request_re, line)
                    assert request_data is not None  # Make sure there's nothing funky going on with the formatting

                    num_requests = int(request_data.group(3))
                    for _ in range(num_requests):
                        # Take request according to downsampled rate
                        rand_num = random()
                        if (rand_num <= downsample):
                            source = int(request_data.group(1))
                            destination = int(request_data.group(2))
                            if (source not in self.ignored_zones and destination not in self.ignored_zones and source != destination):
                                    travel_time = self.get_travel_time(source, destination)
                                    request_list.append(Request(request_id, source, destination, self.current_time, travel_time))
                                    request_id += 1

            if is_in_time_range(self.current_time):
                yield request_list

    def get_travel_time(self, source: int, destination: int) -> float:
        return self.travel_time[source, destination]

    def get_next_location(self, source: int, destination: int) -> int:
        return self.shortest_path[source, destination]

    def get_initial_states(self, num_agents: int, is_training: bool) -> List[int]:
        """Give initial states for num_agents agents"""
        if (num_agents > self.NUM_MAX_AGENTS):
            print('Too many agents. Starting with random states.')
            is_training = True

        # If it's training, get random states
        if is_training:
            initial_states = []

            for _ in range(num_agents):
                initial_state = randint(0, self.NUM_LOCATIONS - 1)
                # Make sure it's not an ignored zone
                while (initial_state in self.ignored_zones):
                    initial_state = randint(0, self.NUM_LOCATIONS - 1)

                initial_states.append(initial_state)
        # Else, pick deterministic initial states
        else:
            initial_states = self.initial_zones[:num_agents]

        return initial_states

    def has_valid_path(self, agent: LearningAgent) -> bool:
        """Attempt to check if the request order meets deadline and capacity constraints"""
        def invalid_path_trace(issue: str) -> bool:
            print(issue)
            print('Agent {}:'.format(agent.id))
            print('Requests -> {}'.format(agent.path.requests))
            print('Request Order -> {}'.format(agent.path.request_order))
            print()
            return False

        # Make sure that its current capacity is sensible
        if (agent.path.current_capacity < 0 or agent.path.current_capacity > self.MAX_CAPACITY):
            return invalid_path_trace('Invalid current capacity')

        # Make sure that it visits all the requests that it has accepted
        if (not agent.path.is_complete()):
            return invalid_path_trace('Incomplete path.')

        # Start at global_time and current_capacity
        current_time = self.current_time + agent.position.time_to_next_location
        current_location = agent.position.next_location
        current_capacity = agent.path.current_capacity

        # Iterate over path
        available_delay: float = 0
        for node_idx, node in enumerate(agent.path.request_order):
            next_location, deadline = agent.path.get_info(node)

            # Delay related checks
            travel_time = self.get_travel_time(current_location, next_location)
            if (current_time + travel_time > deadline):
                return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

            current_time += travel_time
            current_location = next_location

            # Updating available delay
            if (node.expected_visit_time != current_time):
                invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
                node.expected_visit_time = current_time

            if (node.is_dropoff):
                available_delay += deadline - node.expected_visit_time

            # Capacity related checks
            if (current_capacity > self.MAX_CAPACITY):
                return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

            if (node.is_dropoff):
                next_capacity = current_capacity - 1
            else:
                next_capacity = current_capacity + 1
            if (node.current_capacity != next_capacity):
                invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
                node.current_capacity = next_capacity
            current_capacity = node.current_capacity

        # Check total_delay
        if (agent.path.total_delay != available_delay):
            invalid_path_trace("(Ignored) Total delay incorrect.")
        agent.path.total_delay = available_delay

        return True



class ChargingIntegratedEnvironment(Environment):
    """
    Integrated charging environment class, inheriting from src.Environment
    """

    def __init__(self, num_vehicles=5, num_stations=3, ev_num_vehicles=None, grid_size=20,heuristic_battery_threshold=0.5, 
                 use_intense_requests=True, assignmentgurobi=True, usemcmf = True, useauction=False,
                 auction_use_gpu=False, auction_epsilon=1e-3, auction_max_rounds=None,
                 auction_top_k=None, mcmf_solver=None, mcmf_backend="auto",
                 mcmf_strict=True, mcmf_cost_scale=10_000,
                 mcmf_graph_reduction=True, mcmf_verify=False,
                 mcmf_fallback_value=None, knownreject = False,
                   gurobi_network=True, gurobi_network_lp=True, record_time = False, random_seed=None, 
                   multi_gpu_devices=None,asign_allrequest = False, simulation_period=50, days_per_week=7, episode_days=None,
                                     battery_first = True, daily_drop_off = True, ifreject=True, ifdropoff=False,
                                                                         iftest_aev=False, aev_test_service_time=None,
                                     aev_test_request_rate_scale=1.0,
                                     aev_test_request_generation_rate_override=None, lowrequest=False,
                                     charging_wait_penalty_per_step=1.0,
                                     synthetic_demand_profile="predictive",
                                     synthetic_demand_scale=1.0,
                                     station_capacity=10, charge_duration=2,
                                     station_queue_capacity=0,
                                     aev_initial_battery_scale=1.0,
                                     critical_charging_battery=0.15):  # Increased grid size
        # Provide required parameters for base class
        NUM_LOCATIONS = grid_size * grid_size  # Total locations in grid
        MAX_CAPACITY = 4  # Maximum capacity per location
        EPOCH_LENGTH = 1.0  # Length of each epoch
        NUM_AGENTS = num_vehicles  # Number of vehicles/agents
        START_EPOCH = 0.0  # Start time
        STOP_EPOCH = 100.0  # Stop time
        DATA_DIR = "data"  # Data directory (not used in this implementation)
        self.use_intense_requests = use_intense_requests
        self.multi_gpu_devices = multi_gpu_devices  # 🚀 多GPU设备列表
        # 设置随机数种子以确保可重复性
        self.initial_random_seed = random_seed  # 保存初始种子用于车辆初始化
        self.request_generation_seed = random_seed  # 请求生成的种子，可以单独设置
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        super().__init__(NUM_LOCATIONS, MAX_CAPACITY, EPOCH_LENGTH, NUM_AGENTS, START_EPOCH, STOP_EPOCH, DATA_DIR)
        self.assignmentgurobi = assignmentgurobi  # Whether to use Gurobi for assignment
        self.num_vehicles = num_vehicles
        self.ev_num_vehicles = ev_num_vehicles if ev_num_vehicles is not None else int(num_vehicles // 2)
        self.num_stations = num_stations
        self.station_capacity = max(1, int(station_capacity))
        # Physical plugs and waiting-room admissions are distinct.  A zero
        # queue capacity preserves the legacy no-AEV-queue behavior; the
        # synthetic queue experiment supplies a small finite waiting room so
        # station congestion is an actual AEV decision variable.
        self.station_queue_capacity = max(0, int(station_queue_capacity))
        self.aev_initial_battery_scale = float(aev_initial_battery_scale)
        if not 0.05 <= self.aev_initial_battery_scale <= 1.0:
            raise ValueError("aev_initial_battery_scale must be in [0.05, 1.0]")
        self.grid_size = grid_size
        self.synthetic_pickup_patience_steps = max(4, self.grid_size // 2)
        self.num_zones = getattr(self, 'num_zones', 4)
        self._prior_zone_dist_target = [1.0 / self.num_zones] * self.num_zones
        self.minimum_charging_level = 0.2  # Minimum battery level before needing to charge
        self.critical_charging_battery = float(critical_charging_battery)
        if not 0.0 < self.critical_charging_battery < 0.5:
            raise ValueError("critical_charging_battery must be in (0, 0.5)")
        self.proactive_charging_max_battery = 0.40
        self.request_priority_margin = 1e-3
        self.queue_forecast_filter_margin = 0.0
        self.queue_forecast_optional_wait_limit = 0.0
        self.queue_forecast_aev_capacity_share = max(
            0.1,
            min(
                1.0,
                0.5
                * (self.num_vehicles - self.ev_num_vehicles)
                / max(1, self.num_vehicles),
            ),
        )
        self.queue_forecast_filtered_actions = 0
        self.queue_forecast_deferred_charges = 0
        self.queue_forecast_reservation_filtered_actions = 0
        self.demand_forecast_filter_margin = 0.01
        self.demand_forecast_filtered_actions = 0
        self.reserve_inbound_charging_capacity = True
        # The raw queue/wait coordinates are learned by the MASAC critic.
        # An external queue-predictor filter would remove congested actions
        # before the policy sees them and make the queue-feature ablation
        # impossible to interpret.
        self.use_queue_forecast_action_filter = False
        # Train and deploy the EV request-assignment value function in the
        # synthetic environment, matching NYC.  EVs still have only request
        # and wait actions here; this controls whether feasible request edges
        # use the learned EV Q score or bypass it with raw request value.
        self.synthetic_ev_myopic_request_q = False
        self._ev_request_q_source_reported = False
        # The EV action matrix has one final outside-action column.  Its
        # synthetic execution semantics are stochastic relocation, so sample
        # the concrete destination once per decision epoch and reuse it for
        # Q evaluation, replay features, and execution.
        self._ev_default_relocation_cache_step = None
        self._ev_default_relocation_targets = {}
        self._ev_default_relocation_probabilities = {}
        # Parameters for reward alignment with Gurobi optimization
        self.charging_penalty_per_session = 2.0
        self.adp_value = 1.0  # Weight for Q-value contribution
        self.unserved_penalty = 50
        self.idle_vehicle_requirement = 1  # Minimum idle vehicles required
        self.charge_duration = max(1, int(charge_duration))
        self.charging_penalty = (
            self.charging_penalty_per_session / float(self.charge_duration)
        )
        self.chargeincrease_per_epoch = 1.0 / float(self.charge_duration)
        self.chargeincrease_whole = self.chargeincrease_per_epoch * self.charge_duration
        self.min_battery_level = 0.2
        self.charging_wait_penalty_per_step = max(0.0, float(charging_wait_penalty_per_step))
        self.charging_wait_penalty_total = 0.0
        self.charging_wait_steps = 0
        self.charging_wait_observations = []
        self._charging_queue_arrivals = {}
        self.charge_finished = 0.0
        self.penalty_reject_requestnum = 3
        self.rejection_loss = []
        self.rejection_pretrained = False
        self.charge_stats = {}
        self.lowrequest = lowrequest
        if synthetic_demand_profile not in {"legacy", "predictive"}:
            raise ValueError(
                "synthetic_demand_profile must be 'legacy' or 'predictive'"
            )
        self.synthetic_demand_profile = synthetic_demand_profile
        self.synthetic_demand_scale = max(0.05, float(synthetic_demand_scale))
        self.current_demand_phase = "legacy"
        self.generated_requests_by_phase = {}
        self.generated_requests_last_step = 0
        self.ifreject = ifreject
        self.ifdropoff = ifdropoff
        self.recourse_variant = "legacy"
        self.rejection_logit_shift = 0.0
        self.common_random_numbers = False
        self.state_variant = "joint_state_shared_critic"
        self.learner_variant = "optimization_anchored_residual"
        self.request_lifecycle = RequestLifecycleTracker()
        self.recourse_coordinator = RecourseCoordinator(
            lifecycle=self.request_lifecycle
        )
        self._last_offer_realizations = {}
        self._pending_recourse_actions = {}
        self._same_epoch_blocked_request_ids = set()
        self.decision_mode = "integrated"  # "integrated" or "sequential"
        self.decision_mode_set = {"integrated", "aev_first","ev_first"}
        # Initialize charging station manager
        self.charging_manager = ChargingStationManager()
        self._setup_charging_stations()
        self.unserve_penalty = -0.5  # Penalty for unserved requests
        self.movingpenalty = -5e-3
        self.evaluatemode = True
        self.rebalance_battery_threshold = 0.3
        self.heuristic_battery_threshold = heuristic_battery_threshold
        self.battery_first = battery_first
        self.vehicles = {}
        self.storeactions = {}
        self.storeactions_ev = {}
        self.whole_req = 0
        # Total generated requests counter (used in _update_environment/get_episode_stats)
        self.whole_req_num = 0
        self.iftest_aev = bool(iftest_aev)
        self.aev_capacity_history = []
        self.aev_capacity_violations = 0
        self.max_required_idle_aev = 0
        self.max_idle_aev_available = 0
        self.max_observed_active_requests = 0
        self.max_pending_active_requests = 0
        self.aev_test_request_generation_rate = None
        self.aev_capacity_trimmed_requests = 0
        self.aev_test_service_time = float(aev_test_service_time) if aev_test_service_time is not None else None
        self.aev_test_request_rate_scale = float(aev_test_request_rate_scale)
        self.aev_test_request_generation_rate_override = (
            float(aev_test_request_generation_rate_override)
            if aev_test_request_generation_rate_override is not None
            else None
        )
        self.completed_service_times_ev = []
        self.completed_service_times_aev = []
        self.ev_rejected_request_ids = set()
        self.ev_rejection_times = {}
        self.ev_rejected_recovered_same_epoch_ids = set()
        self.ev_rejected_rescued_by_aev_ids = set()
        self.ev_rejected_completed_by_ev_ids = set()
        self.expired_request_ids = set()
        self.hotspot_locations = []
        self.initialise_environment()
        print(self.hotspot_locations)
        self.ev_requests = []
        # Environment state
        self.current_time = 0
        self.simulation_period = max(1, int(simulation_period))
        self.days_per_week = max(1, int(days_per_week))
        self.episode_days = int(episode_days) if episode_days is not None else max(1, int(round(200 / self.simulation_period)))
        self.episode_length = self.simulation_period * self.episode_days
        self.episode_start_day = 0
        self.randomize_episode_start_day = True
        self.heuevfirst = False
        self.idle_charging_num = {station_id: 0 for station_id in range(self.num_stations)}
        self.current_online = 0
        self.daily_online_history = []
        self.period_dropout_counts = []
        self.current_period_dropout_count = 0
        self.usemcmf = usemcmf
        self.useauction = bool(useauction)
        self.mcmf_solver = "auction" if self.useauction else mcmf_solver
        self.mcmf_backend = str(mcmf_backend)
        self.mcmf_strict = bool(mcmf_strict)
        self.mcmf_cost_scale = int(mcmf_cost_scale)
        self.mcmf_graph_reduction = bool(mcmf_graph_reduction)
        self.mcmf_verify = bool(mcmf_verify)
        self.mcmf_fallback_value = mcmf_fallback_value
        self.auction_use_gpu = bool(auction_use_gpu)
        self.auction_epsilon = float(auction_epsilon)
        self.auction_max_rounds = auction_max_rounds
        self.auction_top_k = auction_top_k
        self.usenetworkx = False
        self.gurobi_network = gurobi_network
        self.knownreject = knownreject
        self.gurobi_network_lp = gurobi_network_lp
        self.hotspot_locations_num = self.num_zones
        self.record_time = record_time
        self.assignmentnear = 10
        # Keep the feasible pickup radius proportional when the synthetic map
        # is enlarged.  A fixed radius of five covers 25% of a 20x20 axis but
        # only 17% on 30x30, which unfairly starves the global optimizer of
        # request edges while the nearest-request heuristic remains effective.
        self.assignmentrange = max(5, self.grid_size // 2)
        self.asign_allrequest = asign_allrequest
        self.use_range_requests = True  # True: use range-based, False: use nearest-K
        # Synthetic AEVs may compare every configured station.  With six
        # one-slot stations this creates more queue-sensitive alternatives
        # without increasing total physical charging capacity.
        self.chargeassignnum = self.num_stations
        self.ev_basesalary = 15
        self.dropoff_probability_rate = 0.4
        self.ev_dropoff_threshold = 0.35
        self.ev_dropoff_beta_0 = -1.0
        self.ev_dropoff_beta_idle = 0.12
        self.ev_dropoff_beta_satisfaction = -1.0
        self.ev_rejoin_gamma_0 = -1.6
        self.ev_rejoin_gamma_satisfaction = 0.8
        self.daily_drop_off = daily_drop_off
        # Time recording for performance analysis
        self.time_stats = {
            'qvalue_with_network': [],      # Q-value计算时间(使用神经网络)
            'qvalue_without_network': [],   # Q-value计算时间(不使用神经网络)
            'gurobi_solve': [],             # Gurobi求解时间
            'gurobi_variables': [],         # Gurobi变量数
            'gurobi_constraints': [],       # Gurobi约束数
            'qvalue_scale_1': [],          # Q-value矩阵规模1   
            'qvalue_scale_2': []           # Q-value矩阵规模2
        }
        
        self.active_requests = {}  # Active passenger requests
        self.completed_requests = []  # Completed requests for analysis
        self.completed_requests_ev = []  # Completed EV requests for analysis
        self.request_value_sum = 0.0
        self.request_value_sum_ev = 0.0
        self.request_value_sum_aev = 0.0
        self.rejected_requests = []  # Rejected requests for analysis
        self.request_counter = 0
        self.current_demand_phase = "legacy"
        self.generated_requests_by_phase = {}
        self.generated_requests_last_step = 0
        self.request_generation_rate = 0.8  # Increased to 60% for more active environment
        self.use_intense_requests = use_intense_requests  # Whether to use concentrated request generation
        self.battery_consum = (
            0.010 if self.synthetic_demand_profile == "predictive" else 0.015
        )
        # Assignment tracking for rebalancing analysis
        self.rebalancing_assignments_per_step = []  # Store assignments count per step
        self.rebalancing_whole = []
        self.total_rebalancing_calls = 0
        # Tracking for visualization
        self.request_generation_history = []  # Track where requests are generated
        self.vehicle_position_history = {}  # Track vehicle movement patterns
        
        # Charging station usage tracking for episode-wide statistics
        self.charging_usage_history = []  # Track charging station usage over time

        # =============================
        # Zone system (rsimulation_detail)
        # =============================
        # Node (location_id) -> zone_id mapping and zone definitions.
        # Default: partition grid into sqrt(Z) x sqrt(Z) blocks.
        self.loc_to_zone = {}
        self.zone_to_locs = {}
        self.zoneinfo = {"1": "Surge", "2": "HighDemand", "3": "CityCenter", "4": "Normal"}
        self.surge_zone_locs = set()
        self.high_demand_zone_locs = set()
        self.city_center_zone_locs = set()
        self._init_zones()

        # =============================
        # EV behavior (rsimulation_detail-inspired)
        # =============================
        # Idle time tracking: last completion -> next acceptance.
        self.ev_last_completed_time = {}
        self.ev_last_accepted_time = {}
        self.ev_current_idle_start_time = {}
        self.ev_idle_durations = []

        # Consecutive rejection penalty: 2 consecutive rejections triggers cooldown.
        self.ev_consecutive_rejections = {}
        self.ev_penalty_until_time = {}
        self.ev_penalty_duration = 2

        # Probabilistic decision model parameters (kept simple/robust).
        self.ev_charge_soc_threshold = 0.25
        self.ev_charge_soc_slope = 12.0
        self.ev_station_choice_beta = 1.0
        # Synthetic EV behaviour follows the same model families used in the
        # empirical papers.  One synthetic epoch is treated as one minute and
        # one grid edge as the pickup-time proxy.
        self.ev_acceptance_asc = 1.810
        self.ev_acceptance_beta_idle = -0.017
        self.ev_acceptance_beta_pickup = -0.050
        self.ev_acceptance_beta_surge = 0.101
        # Plain MCMF does not use acceptance information.  MCMF-K alone uses
        # the state-dependent probability defined by these coefficients.
        self.relocation_beta_match = 0.08
        self.relocation_beta_cost = 0.1
        self.relocation_cost_u = 0.6 * 0.92
        self.relocation_beta = self.relocation_beta_match

        # =============================
        # rsimulation_detail models (RelocationManager removed – using move.md MNL)
        # =============================
        
        # Initialize ValueFunction for Q-value calculation (will be set externally)
        self.value_function = None
        self.value_function_ev = None
        if self.iftest_aev:
            self.set_aev_larger_env()

        # Cross-attention: prior features cache for posterior decision phase
        # Set after Phase 1 in evfirst/aevfirst; cleared after Phase 2.
        # List of [location_norm, battery, idle_time, target_norm, action_type_float] per prior agent, or None.
        self._prior_features_for_posterior = None
        self._prior_features_for_follower = None
        self._prior_zone_dist_target_for_follower = None
        self._prev_follower_prior_features_for_leader = None
        self._prev_follower_zone_dist_target_for_leader = None
        self._bayes_external_prior = None
        self._bayes_external_posterior = None
        self._bayes_state_posterior = None
        self._prev_follower_external_prior_for_leader = None
        self._prev_follower_external_posterior_for_leader = None
        self._bayes_context_role = None
        self._leader_is_ev = None
        self._skip_bayes_distribution_training = False
        self.zone_vehicle_count_history = []
        self.zone_ev_count_history = []
        self.zone_aev_count_history = []

        print(f"✓ Initialized integrated environment: {num_vehicles} vehicles, {num_stations} charging stations")

    # =============================
    # EV metrics & penalty helpers
    # =============================
    def _is_ev(self, vehicle_id: int) -> bool:
        v = self.vehicles.get(vehicle_id)
        return bool(v is not None and v.get('type', 0) == 1)

    def _build_prior_features(self, leader_vehicle_ids, actions_dict):
        """Build prior features list from leader phase assignments for cross-attention.
        
        Each entry: [location_norm, battery, idle_time, target_location_norm, action_type_float]
        """
        num_locs = max(self.grid_size * self.grid_size, 1)
        features = []
        zone_counts = [0] * self.num_zones  # count per zone
        for vid in leader_vehicle_ids:
            v = self.vehicles.get(vid, {})
            loc = float(v.get('location', 0)) / num_locs
            batt = float(v.get('battery', 1.0))
            idle_t = float(v.get('idle_timer', 0)) / 100.0  # rough normalisation
            # Determine target and action type from action
            action = actions_dict.get(vid)
            if action is None:
                tgt = loc
                act_f = 1.0  # idle
                tgt_loc_raw = v.get('location', 0)
            else:
                from src.Action import ServiceAction, ChargingAction, IdleAction
                tgt_loc = getattr(action, 'target_location', None)
                if tgt_loc is None:
                    tgt_loc = v.get('location', 0)
                if isinstance(tgt_loc, tuple):
                    tgt_loc = tgt_loc[1] * self.grid_size + tgt_loc[0]
                tgt_loc_raw = tgt_loc
                tgt = float(tgt_loc) / num_locs
                if isinstance(action, ServiceAction):
                    act_f = 2.0
                elif isinstance(action, ChargingAction):
                    act_f = 3.0
                else:
                    act_f = 1.0
            features.append([loc, batt, idle_t, tgt, act_f])
            zone_idx = self.get_distribution_zone_index(int(tgt_loc_raw))
            if zone_idx is not None and 0 <= zone_idx < self.num_zones:
                zone_counts[zone_idx] += 1
        # Build zone distribution target (normalized counts)
        total = sum(zone_counts)
        if total > 0:
            self._prior_zone_dist_target = [c / total for c in zone_counts]
        else:
            self._prior_zone_dist_target = [1.0 / self.num_zones] * self.num_zones
        return features

    def _normalize_zone_distribution(self, values):
        if values is None or self.num_zones <= 0:
            return None
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size != self.num_zones:
            return None
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= 0.0:
            return [1.0 / self.num_zones] * self.num_zones
        return (arr / total).tolist()

    def _discretize_zone_metric(self, values, boundaries):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        bins = np.asarray(boundaries, dtype=np.float32)
        return np.digitize(arr, bins, right=False).astype(np.float32)

    def build_bayes_state_distribution(self):
        """Return p(z|s_t) from current zone surge and charging utilization bins."""
        if self.num_zones <= 0:
            return None
        try:
            surge_values = self.return_surgingpricing()
        except Exception:
            surge_values = [1.0] * self.num_zones
        try:
            _, _, charging_ratio = self.return_zone_chargingusernum()
        except Exception:
            charging_ratio = [0.0] * self.num_zones

        if len(surge_values) != self.num_zones:
            surge_values = [1.0] * self.num_zones
        if len(charging_ratio) != self.num_zones:
            charging_ratio = [0.0] * self.num_zones

        surge_bins = self._discretize_zone_metric(surge_values, [1.25, 2.0, 3.0, 4.0])
        charging_bins = self._discretize_zone_metric(charging_ratio, [0.25, 0.50, 0.75])
        scores = (surge_bins + 1.0) * (charging_bins + 1.0)
        scores = scores + 1e-3
        return self._normalize_zone_distribution(scores)

    def refresh_bayes_state_distribution(self):
        self._bayes_state_posterior = self.build_bayes_state_distribution()
        return self._bayes_state_posterior

    def _compute_bayes_beliefs_for_context(self, value_function, context_features, current_time,
                                           external_prior=None, external_posterior=None,
                                           bayes_role=None, bayes_state_dist=None):
        if value_function is None or not hasattr(value_function, 'predict_bayes_context'):
            return None, None
        if context_features is None or len(context_features) == 0:
            return None, None
        try:
            if bayes_state_dist is None:
                bayes_state_dist = self.refresh_bayes_state_distribution()
            return value_function.predict_bayes_context(
                current_time=current_time,
                prior_features=context_features,
                external_prior_dist=external_prior,
                external_posterior_dist=external_posterior,
                bayes_state_dist=bayes_state_dist,
                bayes_role=bayes_role,
            )
        except Exception as exc:
            print(f"⚠ Failed to compute Bayes beliefs for context: {exc}")
            return None, None





        

    def _compute_zone_vehicle_type_counts(self):
        """Return per-zone total/EV/AEV vehicle counts for the current environment state."""
        total_counts = [0] * self.num_zones
        ev_counts = [0] * self.num_zones
        aev_counts = [0] * self.num_zones

        for vehicle in self.vehicles.values():
            if not vehicle.get('is_online', True):
                continue
            zone_id = vehicle.get('zone_id', None)
            if zone_id is None and hasattr(self, 'get_zone_id'):
                zone_id = self.get_zone_id(vehicle.get('location', 0))
            if zone_id is None:
                continue
            zone_id = int(zone_id)
            if not (0 <= zone_id < self.num_zones):
                continue

            total_counts[zone_id] += 1
            vehicle_type = vehicle.get('type', 0)
            if vehicle_type == 1:
                ev_counts[zone_id] += 1
            elif vehicle_type == 2:
                aev_counts[zone_id] += 1

        return total_counts, ev_counts, aev_counts

    def _record_zone_vehicle_snapshot(self):
        """Record current per-zone vehicle counts for later episode-level analysis."""
        if self.num_zones <= 0:
            return
        total_counts, ev_counts, aev_counts = self._compute_zone_vehicle_type_counts()
        self.zone_vehicle_count_history.append(total_counts)
        self.zone_ev_count_history.append(ev_counts)
        self.zone_aev_count_history.append(aev_counts)

    def _in_ev_penalty(self, vehicle_id: int) -> bool:
        if not self._is_ev(vehicle_id):
            return False
        until_t = float(self.ev_penalty_until_time.get(vehicle_id, -1.0))
        return float(self.current_time) < until_t

    def _record_ev_acceptance(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_accepted_time[vehicle_id] = now
        idle_start = self.ev_current_idle_start_time.get(vehicle_id)
        if idle_start is not None:
            self.ev_idle_durations.append(max(0.0, now - float(idle_start)))
        self.ev_current_idle_start_time[vehicle_id] = None
        self.ev_consecutive_rejections[vehicle_id] = 0

    def _record_ev_completion(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_completed_time[vehicle_id] = now
        self.ev_current_idle_start_time[vehicle_id] = now

    def _get_request_final_value(self, request_id, fallback: float = 0.0) -> float:
        request = self.active_requests.get(request_id) if hasattr(self, 'active_requests') else None
        if request is not None:
            return getattr(request, 'final_value', getattr(request, 'value', fallback))

        for request in reversed(getattr(self, 'completed_requests', [])):
            if getattr(request, 'request_id', None) == request_id:
                return getattr(request, 'final_value', getattr(request, 'value', fallback))

        return fallback

    def _record_ev_rejection(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        cnt = int(self.ev_consecutive_rejections.get(vehicle_id, 0)) + 1
        self.ev_consecutive_rejections[vehicle_id] = cnt
        if cnt >= 2:
            self.ev_penalty_until_time[vehicle_id] = float(self.current_time) + float(self.ev_penalty_duration)
            self.ev_consecutive_rejections[vehicle_id] = 0

    # =============================
    # EV probabilistic decisions
    # =============================
    def _sigmoid(self, x: float) -> float:
        # numerically safe sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def compute_ev_charge_probability(self, vehicle_id: int) -> Tuple[float, Dict[int, float]]:
        """Return (p_charge, station_probs) for an EV using Zhang et al. (2023) model.

        Binary Logit: V_swap = 3.5 - 9.5*soc + 0.69*d_deadhead + η
        Station MNL:  V_i = -0.325*d_detour + 0.0529*n_battery - 0.111*l_queue
                            - 1.020*cost + 0.0927*pop + ξ_i
        Returns: (p_charge, {station_id: probability})
        """
        if not self._is_ev(vehicle_id) or not hasattr(self, 'charging_manager'):
            return 0.0, {}

        vehicle = self.vehicles[vehicle_id]

        # No-charge cooldown: if EV declined charging recently, skip (unless emergency)
        if vehicle.get('no_charge_cooldown_until', 0) > self.current_time and vehicle.get('battery', 1.0) > 0.2:
            return 0.0, {}

        # --- Binary Logit: whether to charge ---
        soc = float(vehicle.get('battery', 1.0))
        # d_deadhead: 1 if cumulative distance exceeds threshold (100 grid units ≈ heavy usage day)
        d_deadhead = 1.0 if vehicle.get('total_distance', 0) > 100 else 0.0
        # η ~ N(0, 1.760²): state-dependence noise drawn once per decision window
        eta = np.random.normal(0, 1.760)

        V_swap = 3.5 - 9.5 * soc + 0.69 * d_deadhead + eta
        p_charge = 1.0 / (1.0 + np.exp(-V_swap))

        # --- Multinomial Logit: station choice ---
        if not self.charging_manager.stations:
            return float(p_charge), {}

        vehicle_loc = int(vehicle.get('location', 0))
        station_utilities: Dict[int, float] = {}

        # Pre-compute zone-level demand popularity (ordinal 1–5)
        zone_request_num = getattr(self, 'zone_request_num', [])

        for sid, station in self.charging_manager.stations.items():
            s_loc = int(station.location)
            # d_detour: Manhattan distance (grid units, acts as km proxy)
            d_detour = float(self._manhattan_distance_loc(vehicle_loc, s_loc))
            # n_battery: available charging slots
            n_battery = float(station.available_slots)
            # l_queue: queue length at this station
            l_queue = float(len(getattr(station, 'charging_queue', [])))
            # cost: fixed unit cost (same for all stations in simulation)
            cost = 1.0
            # pop: local demand popularity based on zone type (ordinal 1–5)
            if s_loc in self.surge_zone_locs:
                pop = 5.0
            elif s_loc in self.high_demand_zone_locs:
                pop = 4.0
            elif s_loc in self.city_center_zone_locs:
                pop = 3.0
            else:
                pop = 2.0
            # ξ_i ~ N(0, 1.900²): station-specific noise
            xi = np.random.normal(0, 1.900)

            V_i = (-0.325 * d_detour + 0.0529 * n_battery
                   - 0.111 * l_queue - 1.020 * cost
                   + 0.0927 * pop + xi)
            station_utilities[int(sid)] = V_i

        # Softmax over station utilities
        if station_utilities:
            sids = list(station_utilities.keys())
            utils = np.array([station_utilities[s] for s in sids])
            exp_utils = np.exp(utils - np.max(utils))
            probs = exp_utils / np.sum(exp_utils)
            station_probs = {sids[i]: float(probs[i]) for i in range(len(sids))}
        else:
            station_probs = {}

        return float(p_charge), station_probs

    def _should_consider_ev_charging(self, vehicle_id: int) -> bool:
        if not self._is_ev(vehicle_id):
            return False

        vehicle = self.vehicles[vehicle_id]
        if not vehicle.get('is_online', True):
            return False
        if float(vehicle.get('battery', 1.0)) <= 0.2:
            return True

        return bool(
            vehicle.get('whether_finishrequest', False)
            or vehicle.get('whether_finishrelocate', False)
        )

    def _clear_ev_charge_trigger(self, vehicle_id: int) -> None:
        if not self._is_ev(vehicle_id):
            return

        vehicle = self.vehicles[vehicle_id]
        vehicle['whether_finishrequest'] = False
        vehicle['whether_finishrelocate'] = False

    def compute_ev_relocation_distribution(self, vehicle_id: int) -> Dict[int, float]:
        """MNL relocation model (move.md): idle driver chooses among neighboring grids.

        p_ij = exp(β1*s_j + β2*RC_ij) / Σ_n exp(β1*s_n + β2*RC_in)
        s_j = min(O_j / A_j, 1)  – match probability in grid j
        RC_ij = u (fixed cost) if i≠j, 0 if i==j

        Returns: ``{location_id: probability}``.
        """
        vehicle = self.vehicles.get(vehicle_id, {})
        cur_loc = int(vehicle.get('location', 0))
        cur_x = cur_loc % self.grid_size
        cur_y = cur_loc // self.grid_size

        # Xie et al. (2023), Eq. (20).  Keep these configurable so synthetic
        # sensitivity experiments can change magnitudes without changing the
        # behavioural model.
        beta1 = float(getattr(self, 'relocation_beta_match', 0.08))
        beta2 = float(getattr(self, 'relocation_beta_cost', 0.1))
        reloc_cost_u = float(getattr(self, 'relocation_cost_u', 0.6 * 0.92))

        # Build neighbor set B_i (including self)
        neighbors = [cur_loc]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur_x + dx, cur_y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append(ny * self.grid_size + nx)

        # Pre-compute supply A_j: count of idle vehicles per grid location
        idle_counts: Dict[int, int] = {}
        for vid, v in self.vehicles.items():
            if (v['assigned_request'] is None and v['passenger_onboard'] is None
                    and v['charging_station'] is None):
                loc = int(v.get('location', 0))
                idle_counts[loc] = idle_counts.get(loc, 0) + 1

        # Demand O_j: count of active requests originating from grid j
        request_counts: Dict[int, int] = {}
        if hasattr(self, 'active_requests'):
            for req in self.active_requests.values():
                pickup = int(getattr(req, 'pickup', getattr(req, 'source', 0)))
                request_counts[pickup] = request_counts.get(pickup, 0) + 1

        # Compute utilities
        utilities = {}
        for loc_j in neighbors:
            O_j = request_counts.get(loc_j, 0)
            A_j = max(idle_counts.get(loc_j, 0), 1)  # avoid division by zero
            s_j = min(O_j / A_j, 1.0)
            RC_ij = 0.0 if loc_j == cur_loc else reloc_cost_u
            utilities[loc_j] = beta1 * s_j + beta2 * RC_ij

        # Softmax
        locs = list(utilities.keys())
        u_arr = np.array([utilities[l] for l in locs])
        exp_u = np.exp(u_arr - np.max(u_arr))
        probs = exp_u / np.sum(exp_u)
        return {locs[i]: float(probs[i]) for i in range(len(locs))}

    def compute_ev_relocation_probability(self, vehicle_id: int) -> Tuple[int, Dict[int, float]]:
        """Sample one destination from the synthetic EV MNL distribution."""
        prob_dict = self.compute_ev_relocation_distribution(vehicle_id)
        locs = list(prob_dict)
        probs = np.asarray([prob_dict[loc] for loc in locs], dtype=np.float64)
        chosen = int(np.random.choice(locs, p=probs))
        return chosen, prob_dict

    def _sample_ev_default_relocation_target(self, vehicle_id: int) -> int:
        """Return the sample-once destination for the EV final matrix column."""
        step = float(getattr(self, 'current_time', 0.0) or 0.0)
        if getattr(self, '_ev_default_relocation_cache_step', None) != step:
            self._ev_default_relocation_cache_step = step
            self._ev_default_relocation_targets = {}
            self._ev_default_relocation_probabilities = {}

        vehicle_id = int(vehicle_id)
        cached = getattr(self, '_ev_default_relocation_targets', {})
        if vehicle_id in cached:
            return int(cached[vehicle_id])

        vehicle = self.vehicles[vehicle_id]
        current_location = int(vehicle.get('location', 0))
        battery_blocked = float(vehicle.get('battery', 1.0)) <= (
            float(self.min_battery_level) + 2.0 * float(self.battery_consum)
        )
        if battery_blocked:
            target = current_location
            probabilities = {current_location: 1.0}
        else:
            target, probabilities = self.compute_ev_relocation_probability(vehicle_id)

        self._ev_default_relocation_targets[vehicle_id] = int(target)
        self._ev_default_relocation_probabilities[vehicle_id] = {
            int(location): float(probability)
            for location, probability in probabilities.items()
        }
        return int(target)

    def _init_zones(self):
        """Initialize default zone partition and node<->zone membership."""
        self.loc_to_zone = {}
        self.zone_to_locs = {}

        grid_size = int(getattr(self, 'grid_size', 1))
        num_zones = int(getattr(self, 'num_zones', 4))
        if grid_size <= 0:
            return

        # Use near-square zone layout.
        zones_per_side = max(1, int(round(math.sqrt(num_zones))))
        block_w = max(1, int(math.ceil(grid_size / zones_per_side)))
        block_h = block_w

        def _zone_id_for_xy(x: int, y: int) -> int:
            zx = min(zones_per_side - 1, x // block_w)
            zy = min(zones_per_side - 1, y // block_h)
            return zy * zones_per_side + zx

        for loc in range(grid_size * grid_size):
            x = loc % grid_size
            y = loc // grid_size
            zid = _zone_id_for_xy(x, y)
            self.loc_to_zone[loc] = zid
            self.zone_to_locs.setdefault(zid, []).append(loc)
        
        # 根据 zoneinfo 定义填充各区域的位置集合
        # zoneinfo: {"1": "Surge", "2": "HighDemand", "3": "CityCenter", "4": "Normal"}
        for zid, zone_type in self.zoneinfo.items():
            zone_id_int = int(zid) - 1  # zoneinfo keys are 1-indexed, zone_to_locs keys are 0-indexed
            if zone_id_int in self.zone_to_locs:
                if zone_type == "Surge":
                    self.surge_zone_locs.update(self.zone_to_locs[zone_id_int])
                elif zone_type == "HighDemand":
                    self.high_demand_zone_locs.update(self.zone_to_locs[zone_id_int])
                elif zone_type == "CityCenter":
                    self.city_center_zone_locs.update(self.zone_to_locs[zone_id_int])


    
    
    def get_zone_id(self, location_id: int) -> int:
        """Return zone id for a node/location."""
        if not hasattr(self, 'loc_to_zone') or not self.loc_to_zone:
            return 0
        return int(self.loc_to_zone.get(int(location_id), 0))

    def get_distribution_zone_index(self, location_id: int) -> int | None:
        """Return the 0-based zone index used by zone-distribution vectors."""
        zone_id = self.get_zone_id(location_id)
        zone_id = int(zone_id)
        return zone_id if 0 <= zone_id < self.num_zones else None

    def get_zone_embedding_id(self, location_id: int) -> int:
        """Return the 1-based zone id consumed by the neural zone embedding."""
        zone_idx = self.get_distribution_zone_index(location_id)
        return int(zone_idx) + 1 if zone_idx is not None else 0

    def get_hour_of_day(self, current_time: float = None) -> float:
        """Return hour-of-day (0-24) for a given simulation step.

        For the grid environment there is no real-world clock, so we linearly
        map current_time / episode_length to a 24-hour cycle.
        """
        if current_time is None:
            current_time = self.current_time
        episode_len = getattr(self, 'episode_length', 300)
        return (current_time / max(episode_len, 1)) * 24.0

    def get_zone_locations(self, zone_id: int):
        """Return all node ids belonging to a zone."""
        if not hasattr(self, 'zone_to_locs') or not self.zone_to_locs:
            return []
        return list(self.zone_to_locs.get(int(zone_id), []))
    
    def set_random_seed(self, seed):
        """
        设置环境内部的随机数种子，确保可重复的实验结果
        
        Args:
            seed (int): 随机数种子
        """
        random.seed(seed)
        np.random.seed(seed)
        print(f"✓ Environment random seed set to {seed}", flush=True)
    
    def set_request_generation_seed(self, seed):
        """
        专门设置请求生成的随机数种子，用于控制每个episode的请求序列
        
        Args:
            seed (int): 请求生成专用的随机数种子
        """
        self.request_generation_seed = seed
        # 临时保存当前随机状态
        current_random_state = random.getstate()
        current_numpy_state = np.random.get_state()
        
        # 设置请求生成专用种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 重新保存请求生成的随机状态
        self._request_random_state = random.getstate()
        self._request_numpy_state = np.random.get_state()
        
        # 恢复原来的随机状态
        random.setstate(current_random_state)
        np.random.set_state(current_numpy_state)
        
        print(f"✓ Request generation seed set to {seed}", flush=True)
    
    def _set_request_generation_random_state(self):
        """在生成请求前设置请求生成专用的随机状态"""
        if hasattr(self, '_request_random_state') and hasattr(self, '_request_numpy_state'):
            random.setstate(self._request_random_state)
            np.random.set_state(self._request_numpy_state)
    
    def _save_request_generation_random_state(self):
        """在生成请求后保存请求生成的随机状态"""
        if hasattr(self, '_request_random_state') and hasattr(self, '_request_numpy_state'):
            self._request_random_state = random.getstate()
            self._request_numpy_state = np.random.get_state()
    
    def set_value_function(self, value_function):
        """Set the value function for Q-value calculation"""
        self.value_function = value_function
        print(f"✓ Value function set: {type(value_function).__name__}")
    def set_value_function_ev(self, value_function):
        """Set the value function for Q-value calculation"""
        self.value_function_ev = value_function
        print(f"✓ Value function ev  set: {type(value_function).__name__}")
        
        
        
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int) -> float:
        """Get Q-value for vehicle assignment using ValueFunction if available"""
        if self.value_function and hasattr(self.value_function, 'get_assignment_q_value'):
            # Provide additional context for neural network including battery level and request value
            vehicle = self.vehicles.get(vehicle_id, {})
            battery_level = vehicle.get('battery', 1.0)  # 获取车辆电池电量
            other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
            num_requests = len(self.active_requests)
            
            # 获取请求的价值信息 - 使用final_value确保与奖励一致
            request_value = 0.0
            if target_id in self.active_requests:
                # 使用value而不是final_value，确保与实际奖励计算一致
                request_value = self.active_requests[target_id].final_value
            
            return self.value_function.get_assignment_q_value(
                vehicle_id, target_id, vehicle_location, target_location, 
                self.current_time, other_vehicles, num_requests, battery_level, request_value)  # 添加request_value参数
        else:
            # Fallback to parent class method
            return super().get_assignment_q_value(vehicle_id, target_id, vehicle_location, target_location)
    
    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int) -> float:
        """Get Q-value for vehicle charging decision"""
        if self.value_function and hasattr(self.value_function, 'get_charging_q_value'):
            # Provide additional context for neural network including battery level
            vehicle = self.vehicles.get(vehicle_id, {})
            battery_level = vehicle.get('battery', 1.0)  # 获取车辆电池电量
            # 对齐训练时的计数口径：使用空闲车辆数量（不含当前车）而非“正在充电的车辆数”
            idle_count = len([v for vid, v in self.vehicles.items()
                              if v.get('assigned_request') is None and
                                 v.get('passenger_onboard') is None and
                                 v.get('charging_station') is None])
            other_vehicles = max(0, idle_count - 1)
            num_requests = len(self.active_requests)
            
            return self.value_function.get_charging_q_value(
                vehicle_id, station_id, vehicle_location, station_location, 
                self.current_time, other_vehicles, num_requests, battery_level)  # 添加battery_level参数
        else:
            # Fallback calculation
            distance = abs(vehicle_location - station_location)
            return 5.0 - distance * 0.1  # Simple heuristic

    # --- Option-value evaluators for optimizer alignment ---
    def _loc_to_xy(self, loc: int) -> tuple:
        return (loc % self.grid_size, loc // self.grid_size)

    def _manhattan_distance_loc(self, a_loc: int, b_loc: int) -> int:
        ax, ay = self._loc_to_xy(a_loc)
        bx, by = self._loc_to_xy(b_loc)
        return abs(ax - bx) + abs(ay - by)
    def _manhattan_distance_loc_time(self, a_loc: int, b_loc: int) -> int:
        ax, ay = self._loc_to_xy(a_loc)
        bx, by = self._loc_to_xy(b_loc)
        return max(abs(ax - bx), abs(ay - by))
    def _estimate_future_state_value(self, vehicle_id: int, future_loc: int, future_battery: float, future_time: float,actiontype) -> float:
        """Minimally estimate V(s_after) using current NN: take max of idle/waiting value at future state.
        Falls back to 0 if no value_function.
        """
        if not self.value_function:
            return 0.0
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        num_reqs = len(self.active_requests)
        if actiontype == 'charge':
            assign_v = self.value_function.get_charging_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=future_loc,
                current_time=future_time,
                other_vehicles=max(0, other_idle,self.num_vehicles),
                num_requests=num_reqs,
                battery_level=future_battery,
            )
            return assign_v
        elif actiontype == 'serve':
            assign_v = self.value_function.get_assignment_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=future_loc,
                current_time=future_time,
                other_vehicles=max(0, other_idle,self.num_vehicles),
                num_requests=num_reqs,
                battery_level=future_battery,
            )
            return assign_v
        elif actiontype == 'idle':
            idle_v = self.value_function.get_idle_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=future_loc,
                target_location=future_loc,
                battery_level=future_battery,
                current_time=future_time,
                other_vehicles=max(0, other_idle,self.num_vehicles),
                num_requests=num_reqs
            )
            return idle_v

    def evaluate_service_option(self, vehicle_id: int, request, ifEVQvalue=False) -> float:

        # Resolve request object from id
        if isinstance(request, (int, str)) and request in self.active_requests:
            request = self.active_requests[request]
        if request is None:
            return 0.0
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0
        
        # 根据ifEVQvalue选择使用哪个value function
        value_func = self.value_function_ev if ifEVQvalue else self.value_function
        
        # 检查对应value_function的经验缓冲区中是否有assign类型的动作经验
        if not self.evaluatemode:
            if value_func and hasattr(value_func, 'experience_buffer'):
                has_assign_experience = any(
                    exp.get('action_type', '').startswith('assign') 
                    for exp in value_func.experience_buffer
                )
                if not has_assign_experience:
                    return 0.0
        
        cur_loc = veh['location']
        cur_bat = veh['battery']
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        v_after = value_func.get_assignment_q_value(
                vehicle_id=vehicle_id,
                target_id=request.request_id,
                vehicle_location=cur_loc,
                target_reject=request.pickup,
                target_location=request.dropoff,
                current_time=self.current_time,
                other_vehicles=max(0, other_idle,self.num_vehicles),
                num_requests=num_reqs,
                battery_level=cur_bat,
            )
        
        return  v_after

    def batch_evaluate_service_options(self, vehicle_request_pairs, ifEVQvalue = False):
        """
        批量计算多个vehicle-request对的拒绝感知调整Q值，提高计算效率
        现在集成神经网络预测的拒绝率: Q_value - immediate_reward * rejection_probability
        
        每100步保存Q值与request关系的详细数据到本地
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            
        Returns:
            List of rejection-aware adjusted Q-values corresponding to each vehicle-request pair
        """
        if not vehicle_request_pairs:
            return []
        
        # 初始化计数器和数据缓冲区
        if not hasattr(self, '_batch_qvalue_save_counter'):
            self._batch_qvalue_save_counter = 0
            self._batch_qvalue_data_buffer = []
        
        self._batch_qvalue_save_counter += 1
        
        if not self.evaluatemode:
            value_func = self.value_function_ev if ifEVQvalue else self.value_function
            if value_func and hasattr(value_func, 'experience_buffer'):
                has_assign_experience = any(
                    exp.get('action_type', '').startswith('assign') 
                    for exp in value_func.experience_buffer
                )
                if not has_assign_experience:
                    # 没有assign经验，返回0值列表
                    return [0.0] * len(vehicle_request_pairs)
        
        # 准备批量输入数据
        batch_inputs = []
        valid_pairs = []
        data_records = []  # 用于保存详细信息
        
        for vehicle_id, request in vehicle_request_pairs:
            # 检查vehicle和request的有效性
            veh = self.vehicles.get(vehicle_id)
            if veh is None or request is None:
                continue
                
            # 如果request是ID，解析为对象
            if isinstance(request, (int, str)) and request in self.active_requests:
                request = self.active_requests[request]
            if request is None:
                continue
            
            # 收集request详细信息
            final_value = getattr(request, 'final_value', 0.0)
            pickup_dist = self._manhattan_distance_loc(veh['location'], request.pickup)
            order_dist = self._manhattan_distance_loc(request.pickup, request.dropoff)
                
            cur_loc = veh['location']
            cur_bat = veh['battery']
            num_reqs = len(self.active_requests)
            other_idle = len([v for vid, v in self.vehicles.items() 
                             if vid != vehicle_id and v['assigned_request'] is None 
                             and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            vehicle_idle_time = veh.get('idle_timer', 0)
            input_data = {
                'vehicle_id': vehicle_id,
                'target_id': request.request_id,
                'vehicle_location': cur_loc,
                'target_reject': request.pickup,
                'target_location': request.dropoff,
                'current_time': self.current_time,
                'other_vehicles': max(0, other_idle,self.num_vehicles),
                'num_requests': num_reqs,
                'pickup_dist': pickup_dist,
                'pick_zone': self.get_zone_id(request.pickup),
                'battery_level': cur_bat,
                'request_value': final_value,  # 添加订单价值
                'vehicle_idle_time': vehicle_idle_time,
            }
            
            batch_inputs.append(input_data)
            valid_pairs.append((vehicle_id, request))
            
            # 暂存记录（稍后补充Q值）
            data_records.append({
                'step': self._batch_qvalue_save_counter,
                'time': self.current_time,
                'vehicle_id': vehicle_id,
                'vehicle_type': veh.get('type', 0),
                'request_id': request.request_id,
                'request_value': final_value,
                'pickup_dist': pickup_dist,
                'order_dist': order_dist,
                'battery_level': cur_bat,
                'num_requests': num_reqs,
                'other_idle': other_idle,
            })
        
        if not batch_inputs:
            return []
        
        # 批量计算基础Q值
        try:
            base_q_values = []
            if hasattr(self.value_function, 'batch_get_assignment_q_value'):
                # 如果value function支持批量计算（支持多GPU）
                if ifEVQvalue:
                    base_q_values = self.value_function_ev.batch_get_assignment_q_value(batch_inputs, multi_gpu_devices=self.multi_gpu_devices)
                else:
                    base_q_values = self.value_function.batch_get_assignment_q_value(batch_inputs, multi_gpu_devices=self.multi_gpu_devices)
            else:
                # 否则使用优化的单独计算
                for input_data in batch_inputs:
                    q_value = self.value_function.get_assignment_q_value(**input_data)
                    base_q_values.append(q_value)
            
            # 批量计算拒绝感知调整值
            adjusted_q_values = []
            for i, (vehicle_id, request) in enumerate(valid_pairs):
                base_q = base_q_values[i] if i < len(base_q_values) else 0.0
                
                # 计算拒绝感知调整: Q_value - immediate_reward * rejection_probability
                adjusted_q = self._calculate_rejection_aware_adjustment(
                    vehicle_id, request, base_q
                )
                adjusted_q_values.append(adjusted_q)
                
                # 补充Q值到数据记录
                if i < len(data_records):
                    data_records[i]['base_qvalue'] = base_q
                    data_records[i]['adjusted_qvalue'] = adjusted_q
                    data_records[i]['q_minus_value'] = base_q - data_records[i]['request_value']
            
            # 添加到缓冲区
            self._batch_qvalue_data_buffer.extend(data_records)
            
            # 每100步保存一次
            # if self._batch_qvalue_save_counter % 100 == 0:
            #     self._save_qvalue_request_analysis()
            
            return base_q_values
            
        except Exception as e:
            print(f"Batch Q-value calculation failed: {e}")
            # 回退到逐个计算
            return [self.evaluate_service_option(vehicle_id, request) 
                    for vehicle_id, request in valid_pairs]
    
    def _save_qvalue_request_analysis(self):
        """保存Q值与request关系的分析数据"""
        import os
        from datetime import datetime
        import pandas as pd
        
        if not self._batch_qvalue_data_buffer:
            return
        
        # 创建保存目录
        save_dir = 'results/qvalue_request_analysis'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{save_dir}/qvalue_request_step{self._batch_qvalue_save_counter}_{timestamp}.csv'
        
        # 转换为DataFrame
        df = pd.DataFrame(self._batch_qvalue_data_buffer)
        # df.to_csv(filename, index=False)
        
        # 统计分析
        print(f"\n{'='*70}")
        print(f"Q-VALUE vs REQUEST VALUE ANALYSIS (Step {self._batch_qvalue_save_counter})")
        print(f"{'='*70}")
        print(f"📊 Saved {len(df)} records to: {filename}")
        print(f"\n📈 Request Value Statistics:")
        print(f"   Mean: {df['request_value'].mean():.2f}")
        print(f"   Std:  {df['request_value'].std():.2f}")
        print(f"   Min:  {df['request_value'].min():.2f}")
        print(f"   Max:  {df['request_value'].max():.2f}")
        
        print(f"\n🎯 Q-Value Statistics:")
        print(f"   Base Q-value Mean:     {df['base_qvalue'].mean():.2f}")
        print(f"   Adjusted Q-value Mean: {df['adjusted_qvalue'].mean():.2f}")
        
        print(f"\n🔍 Correlation Analysis:")
        correlation = df[['request_value', 'base_qvalue', 'adjusted_qvalue']].corr()
        print(f"   Request Value <-> Base Q-value:     {correlation.loc['request_value', 'base_qvalue']:.4f}")
        print(f"   Request Value <-> Adjusted Q-value: {correlation.loc['request_value', 'adjusted_qvalue']:.4f}")
        
        # 分组分析：高价值 vs 低价值订单
        median_value = df['request_value'].median()
        high_value = df[df['request_value'] >= median_value]
        low_value = df[df['request_value'] < median_value]
        
        print(f"\n📊 High vs Low Value Orders:")
        print(f"   High Value Orders (>={median_value:.1f}):")
        print(f"      Count: {len(high_value)}, Avg Q-value: {high_value['base_qvalue'].mean():.2f}")
        print(f"   Low Value Orders (<{median_value:.1f}):")
        print(f"      Count: {len(low_value)}, Avg Q-value: {low_value['base_qvalue'].mean():.2f}")
        
        if len(high_value) > 0 and len(low_value) > 0:
            q_diff = high_value['base_qvalue'].mean() - low_value['base_qvalue'].mean()
            print(f"   Q-value Difference: {q_diff:.2f} (High - Low)")
            if q_diff > 0:
                print(f"   ✓ Network prefers HIGH value orders (+{q_diff:.2f})")
            else:
                print(f"   ✗ Network prefers LOW value orders ({q_diff:.2f})")
        
        print(f"{'='*70}\n")
        
        # 清空缓冲区
        self._batch_qvalue_data_buffer = []
    


    def batch_evaluate_service_options_meanfield(self, vehicle_request_pairs, ifEVQvalue = False):
        """
        批量计算多个vehicle-request对的Mean Field Q值
        使用周围智能体的历史决策分布作为条件变量
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            ifEVQvalue: 是否使用EV的value function
            
        Returns:
            List of Mean Field Q-values: Q(s, a, μ) where μ is mean action distribution
        """
        if not vehicle_request_pairs:
            return []
        
        # 选择合适的 value function
        value_func = self.value_function_ev if ifEVQvalue else self.value_function
        
        # 检查 value function 是否支持 mean field
        if not hasattr(value_func, 'compute_mean_field') or not hasattr(value_func, 'batch_get_q_value_with_mean_field'):
            # 如果不支持 mean field，回退到普通方法
            return self.batch_evaluate_service_options(vehicle_request_pairs, ifEVQvalue)
        
        # 准备批量输入数据
        batch_inputs = []
        valid_pairs = []
        mean_fields = []
        
        # 收集所有车辆位置信息用于计算邻居
        agent_locations = {}
        for vid, vehicle in self.vehicles.items():
            loc = vehicle.get('location', 0)
            x = loc % self.grid_size
            y = loc // self.grid_size
            agent_locations[vid] = (x, y)
        
        for vehicle_id, request in vehicle_request_pairs:
            # 检查 vehicle 和 request 的有效性
            veh = self.vehicles.get(vehicle_id)
            if veh is None or request is None:
                continue
                
            # 如果 request 是 ID，解析为对象
            if isinstance(request, (int, str)) and request in self.active_requests:
                request = self.active_requests[request]
            if request is None:
                continue
            
            # 计算该车辆的邻居智能体的平均动作分布（mean field）
            mean_field = value_func.compute_mean_field(
                environment=self,
                agent_id=vehicle_id,
                agent_locations=agent_locations
            )
            mean_fields.append(mean_field)
            
            # 准备状态特征
            cur_loc = veh['location']
            cur_bat = veh['battery']
            num_reqs = len(self.active_requests)
            other_idle = len([v for vid, v in self.vehicles.items() 
                             if vid != vehicle_id and v['assigned_request'] is None 
                             and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            # 准备输入数据
            input_data = {
                'vehicle_id': vehicle_id,
                'target_id': request.request_id,
                'vehicle_location': cur_loc,
                'target_reject': request.pickup,
                'target_location': request.dropoff,
                'current_time': self.current_time,
                'other_vehicles': max(0, other_idle,self.num_vehicles),
                'num_requests': num_reqs,
                'battery_level': cur_bat,
                'request_value': getattr(request, 'final_value', getattr(request, 'value', 0.0))
            }
            
            batch_inputs.append(input_data)
            valid_pairs.append((vehicle_id, request))
        
        if not batch_inputs:
            return []
        
        # 批量计算 Mean Field Q值
        try:
            # 使用 value function 的批量 mean field Q值计算方法
            mean_field_q_values = value_func.batch_get_q_value_with_mean_field(
                batch_inputs, 
                mean_fields
            )
            
            # 应用拒绝感知调整（如果需要）
            adjusted_q_values = []
            for i, (vehicle_id, request) in enumerate(valid_pairs):
                base_q = mean_field_q_values[i] if i < len(mean_field_q_values) else 0.0
                
                # 计算拒绝感知调整
                adjusted_q = self._calculate_rejection_aware_adjustment(
                    vehicle_id, request, base_q
                )
                adjusted_q_values.append(adjusted_q)
            
            return adjusted_q_values
            
        except Exception as e:
            print(f"Mean Field batch Q-value calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # 回退到普通批量方法
            return self.batch_evaluate_service_options(vehicle_request_pairs, ifEVQvalue)


    
  
    
    
    def heuristic_find_nearest_v(self,reassignvehicles):
        """启发式寻找最近的车辆"""
        hotspot_locations = self.hotspot_locations[:self.hotspot_locations_num]
        nearest_vehicle = None
        for loc in hotspot_locations:
            available_vehicles = {}
            for vehicle_id, vehicle in reassignvehicles.items():
                loc = vehicle['location']
                battery_level = vehicle['battery']
                batt_loss = self._manhattan_distance_loc_time(loc, loc) * self.battery_consum
                if battery_level - batt_loss >= self.rebalance_battery_threshold:
                    available_vehicles[vehicle_id] = vehicle
            min_distance = float('inf')
            for vehicle_id, vehicle in available_vehicles.items():
                vehicle_loc = vehicle['location']
                distance = self._manhattan_distance_loc(vehicle_loc, loc)
                if distance < min_distance and distance !=0:
                    min_distance = distance
                    nearest_vehicle = vehicle_id
        if nearest_vehicle is not None:
            return nearest_vehicle
        else:
            loc = hotspot_locations[randint(0, len(hotspot_locations)-1)]
            available_vehicles = {}
            for vehicle_id, vehicle in reassignvehicles.items():
                loc = vehicle['location']
                battery_level = vehicle['battery']
                batt_loss = self._manhattan_distance_loc_time(loc, loc) * self.battery_consum
                if battery_level - batt_loss >= self.rebalance_battery_threshold:
                    available_vehicles[vehicle_id] = vehicle
            if available_vehicles:
                return list(available_vehicles.keys())[0]

    def _calculate_rejection_aware_adjustment(self, vehicle_id, request, base_q_value):
        """
        计算拒绝感知的Q值调整: Q_value - immediate_reward * rejection_probability
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            base_q_value: 基础Q值
            
        Returns:
            float: 调整后的Q值
        """
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return base_q_value
        
        # 计算立即收益
        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
        
        # 计算移动成本
        cur_loc = vehicle['location']
        pickup_x = request.pickup % self.grid_size
        pickup_y = request.pickup // self.grid_size
        dropoff_x = request.dropoff % self.grid_size
        dropoff_y = request.dropoff // self.grid_size
        vehicle_x = cur_loc % self.grid_size
        vehicle_y = cur_loc // self.grid_size
        
        d1 = abs(vehicle_x - pickup_x) + abs(vehicle_y - pickup_y)
        d2 = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
        moving_cost = getattr(self, 'movingpenalty', -0.1) * (d1 + d2)
        
        # 净立即收益
        net_immediate_reward = immediate_reward + moving_cost
        
        # 获取拒绝概率
        rejection_prob = self._calculate_rejection_probability(vehicle_id, request)
        
        # 计算调整后的Q值: Q值 - 立即收益 * 拒绝概率
        # 如果拒绝概率高，从Q值中减去更多的立即收益
        adjusted_q = base_q_value - (net_immediate_reward * rejection_prob)
        
        return adjusted_q

    def evaluate_charging_option(self, vehicle_id: int, station) -> float:
        """Estimate completion Q for going to charge at a station (option value)."""
        if not self.evaluatemode:
            if self.value_function and hasattr(self.value_function, 'experience_buffer'):
                has_charge_experience = any(
                    exp.get('action_type', '').startswith('charge') 
                    for exp in self.value_function.experience_buffer
                )
                if not has_charge_experience:
                    return 0.0
        
        # Resolve station object/id
        
        station_obj = None
        station_id = None
        if hasattr(station, 'id'):
            station_id = station.id
            station_obj = station
        else:
            station_id = station
            station_obj = self.charging_manager.stations.get(station_id) if hasattr(self, 'charging_manager') else None
        if station_obj is None:
            return 0.0

        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']
        station_loc = station_obj.location


        current_time = self.current_time
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        v_after = self.value_function.get_charging_q_value(vehicle_id=vehicle_id, station_id=station_id,
                                             vehicle_location=cur_loc, station_location=station_loc,
                                             current_time=current_time, other_vehicles=max(0, other_idle,self.num_vehicles),
                                             num_requests=num_reqs, battery_level=cur_bat)
        return  v_after


    def evaluate_idle_option(self, vehicle_id: int,target_loc) -> float:
        # 检查value_function的经验缓冲区中是否有idle类型的动作经验
        if not self.evaluatemode:
            if self.value_function and hasattr(self.value_function, 'experience_buffer'):
                has_idle_experience = any(
                    exp.get('action_type', '') == 'idle' 
                    for exp in self.value_function.experience_buffer
                )
                if not has_idle_experience:
                    return 0.0
        """Estimate completion Q for idling/waiting (option value)."""
        veh = self.vehicles.get(vehicle_id)
        
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']

        current_time = self.current_time
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        
        # 检查value_function是否可用
        if self.value_function is not None:
            v_after = self.value_function.get_idle_q_value(vehicle_id=vehicle_id, vehicle_location=cur_loc, 
                            target_location=target_loc,
                            battery_level=cur_bat, current_time=current_time, 
                            other_vehicles=max(0, other_idle,self.num_vehicles), num_requests=num_reqs)
        else:
            # 如果没有value_function，使用简单的启发式计算
            v_after = getattr(self, 'idle_vehicle_reward', -0.1)
        return v_after

    def evaluate_waiting_option(self, vehicle_id: int) -> float:
        if not self.evaluatemode:
            if self.value_function and hasattr(self.value_function, 'experience_buffer'):
                has_idle_experience = any(
                    exp.get('action_type', '') == 'idle' 
                    for exp in self.value_function.experience_buffer
                )
                if not has_idle_experience:
                    return 0.0
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']
        num_reqs = len(self.active_requests)
        # minimal execution reward: small movement penalty for staying put
        r_exec = self.movingpenalty * 1.0  # small penalty for idling
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        
        # 检查value_function是否可用
        if self.value_function is not None:
            v_after = self.value_function.get_waiting_q_value(vehicle_id=vehicle_id, vehicle_location=cur_loc, 
                            
                            battery_level=cur_bat, current_time=self.current_time, 
                            other_vehicles=max(0, other_idle,self.num_vehicles), num_requests=num_reqs)
        else:
            # 如果没有value_function，使用简单的启发式计算
            v_after = getattr(self, 'waiting_vehicle_reward', -0.1)
        return v_after


    def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = None, 
                        other_vehicles: int = None, num_requests: int = None,
                        target_location: int = None) -> float:
        """
        Get Q-value for vehicle idle action
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 电池电量 (0-1)
            current_time: 当前时间 (可选，如果不提供则使用self.current_time)
            other_vehicles: 其他车辆数量 (可选，如果不提供则自动计算)
            num_requests: 当前请求数量 (可选，如果不提供则自动计算)
            target_location: 目标位置 (可选，如果不提供则使用self.vehicles[vehicle_id]['idle_target'])
            
        Returns:
            float: idle动作的Q值
        """
        if self.value_function and hasattr(self.value_function, 'get_idle_q_value'):
            # 使用神经网络计算idle Q值，提供所有必要的上下文信息
            if other_vehicles is None:
                other_vehicles = len([v for v in self.vehicles.values() 
                                    if v['assigned_request'] is None and 
                                       v['passenger_onboard'] is None and 
                                       v['charging_station'] is None]) - 1  # 减去当前车辆
            if num_requests is None:
                num_requests = len(self.active_requests)
            
            # 如果没有提供target_location，从vehicle的idle_target获取
            if target_location is None:
                target_location = self.vehicles[vehicle_id].get('idle_target')
            
            return self.value_function.get_idle_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                target_location=target_location,
                battery_level=battery_level,
                current_time=current_time if current_time is not None else self.current_time,
                other_vehicles=max(0, other_vehicles),  # 确保非负
                num_requests=num_requests
            )
        else:
            # 后备计算：简单的基于电池电量和时间的启发式
            base_idle_value = -0.1  # idle的基础成本
            battery_bonus = battery_level * 0.5  # 高电量时idle的价值更高
            time_penalty = (current_time if current_time is not None else self.current_time) / self.episode_length * 0.2  # 后期idle惩罚更大
            
            return base_idle_value + battery_bonus - time_penalty


    def get_waiting_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = None, 
                        other_vehicles: int = None, num_requests: int = None) -> float:
        """
        Get Q-value for vehicle idle action
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 电池电量 (0-1)
            current_time: 当前时间 (可选，如果不提供则使用self.current_time)
            other_vehicles: 其他车辆数量 (可选，如果不提供则自动计算)
            num_requests: 当前请求数量 (可选，如果不提供则自动计算)
            
        Returns:
            float: idle动作的Q值
        """
        if self.value_function and hasattr(self.value_function, 'get_idle_q_value'):
            # 使用神经网络计算idle Q值，提供所有必要的上下文信息
            other_vehicles = len([v for v in self.vehicles.values() 
                                if v['assigned_request'] is None and 
                                   v['passenger_onboard'] is None and 
                                   v['charging_station'] is None]) - 1  # 减去当前车辆
            num_requests = len(self.active_requests)
            
            return self.value_function.get_waiting_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                battery_level=battery_level,
                current_time=self.current_time,
                other_vehicles=max(0, other_vehicles),  # 确保非负
                num_requests=num_requests
            )
        else:
            # 后备计算：简单的基于电池电量和时间的启发式
            base_idle_value = -0.1  # idle的基础成本
            battery_bonus = battery_level * 0.5  # 高电量时idle的价值更高
            time_penalty = self.current_time / self.episode_length * 0.2  # 后期idle惩罚更大
            
            return base_idle_value + battery_bonus - time_penalty
    
    # Note: store_q_learning_experience is now integrated into _update_q_learning
    # for better consistency between Q-table and neural network training
    
    def _setup_charging_stations(self):
        """Setup charging stations dynamically based on num_stations"""
        # Generate charging stations evenly distributed across the grid
        stations = []
        for i in range(self.num_stations):
            # Distribute stations evenly across the grid
            if self.num_stations == 1:
                x, y = self.grid_size // 2, self.grid_size // 2
            elif self.num_stations == 2:
                positions = [
                    (self.grid_size // 4, self.grid_size // 4),
                    (3 * self.grid_size // 4, 3 * self.grid_size // 4),
                ]
                x, y = positions[i]
            elif self.num_stations == 3:
                positions = [
                    (self.grid_size // 4, self.grid_size // 4),
                    (3 * self.grid_size // 4, self.grid_size // 4),
                    (self.grid_size // 2, 3 * self.grid_size // 4),
                ]
                x, y = positions[i]
            else:
                # For more stations, distribute more evenly
                cols = int(np.sqrt(self.num_stations))
                rows = (self.num_stations + cols - 1) // cols
                row = i // cols
                col = i % cols
                x = (col + 1) * self.grid_size // (cols + 1)
                y = (row + 1) * self.grid_size // (rows + 1)
            
            location_index = y * self.grid_size + x
            stations.append({
                'id': i + 1,
                'location': location_index,
                'capacity': self.station_capacity
            })
        
        for station_info in stations:
            self.charging_manager.add_station(
                station_info['id'],
                station_info['location'],
                station_info['capacity']
            )

    def _active_request_count_at_location(self, location: int) -> int:
        """Count unassigned pickups in the zone containing ``location``."""
        location = int(location)
        target_zone = self.get_zone_id(location)
        if target_zone is None:
            return 0
        busy_request_ids = {
            request_id
            for vehicle in self.vehicles.values()
            for request_id in (
                vehicle.get('assigned_request'),
                vehicle.get('passenger_onboard'),
            )
            if request_id is not None
        }
        return sum(
            1
            for request_id, request in self.active_requests.items()
            if request_id not in busy_request_ids
            if self.get_zone_id(
                int(getattr(request, 'pickup', getattr(request, 'source', -1)))
            ) == target_zone
        )

    def _initial_vehicle_battery(self, vehicle_id: int) -> float:
        """Return heterogeneous initial charge levels for predictive synthetic days."""
        if self.synthetic_demand_profile != "predictive":
            return random.uniform(0.8, 0.95)
        cohort = int(vehicle_id) % 3
        if cohort == 0:
            battery = random.uniform(0.35, 0.45)
        elif cohort == 1:
            battery = random.uniform(0.55, 0.70)
        else:
            battery = random.uniform(0.78, 0.92)
        if int(vehicle_id) >= int(
            getattr(self, 'ev_num_vehicles', self.num_vehicles)
        ):
            battery *= float(getattr(self, 'aev_initial_battery_scale', 1.0))
        return float(np.clip(battery, 0.05, 0.99))

    def _apply_predictive_request_patience(self, request) -> None:
        if self.synthetic_demand_profile == "predictive":
            request.pickup_deadline = (
                float(self.current_time) + self.synthetic_pickup_patience_steps
            )

    def _predictive_demand_settings(self):
        """Return a learnable within-day demand profile for synthetic runs."""
        day_step = int(self.current_time) % max(1, int(self.simulation_period))
        progress = day_step / max(1.0, float(self.simulation_period))

        if progress < 0.32:
            return "overnight", 0.15, (0.01, 0.02), [0.05, 0.05, 0.85, 0.05]
        if progress < 0.40:
            return "morning_build", 0.60, (0.025, 0.04), [0.05, 0.05, 0.85, 0.05]
        if progress < 0.54:
            return "morning_peak", 1.00, (0.06, 0.09), [0.02, 0.05, 0.90, 0.03]
        if progress < 0.72:
            return "midday_valley", 0.35, (0.015, 0.025), [0.05, 0.10, 0.10, 0.75]
        if progress < 0.90:
            return "evening_peak", 1.00, (0.07, 0.10), [0.03, 0.90, 0.05, 0.02]
        return "late_night", 0.10, (0.01, 0.02), [0.10, 0.10, 0.70, 0.10]

    def _predictive_dropoff_coordinates(self, phase: str):
        """Sample a noisy commuting destination leading into the next demand phase."""
        destination_by_phase = {
            "overnight": 2,
            "morning_build": 2,
            "morning_peak": 3,
            "midday_valley": 1,
            "evening_peak": 2,
            "late_night": 2,
        }
        if random.random() >= 0.75:
            return (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )

        hotspot_index = destination_by_phase.get(phase, 0)
        hotspot_index = min(hotspot_index, len(self.hotspot_locations) - 1)
        center_x, center_y = self.hotspot_locations[hotspot_index]
        radius = max(2, self.grid_size // 8)
        return (
            max(0, min(self.grid_size - 1, center_x + random.randint(-radius, radius))),
            max(0, min(self.grid_size - 1, center_y + random.randint(-radius, radius))),
        )

    def _request_generation_settings(self):
        """Resolve generation probability, batch size, and spatial weights."""
        if self.synthetic_demand_profile != "predictive":
            if self.current_time < self.episode_length * 0.5:
                ranges = ((0.5, 0.75), (0.75, 1.0), (1.5, 2.0))
            else:
                ranges = ((0.75, 1.0), (1.0, 1.5), (2.0, 3.0))
            draw = random.random()
            fraction_range = ranges[0] if draw < 0.3 else (ranges[1] if draw < 0.8 else ranges[2])
            low = max(1, int(self.num_vehicles * fraction_range[0]))
            high = max(low, int(self.num_vehicles * fraction_range[1]))
            return "legacy", self.request_generation_rate, low, high, [0.5, 0.3, 0.15, 0.05]

        phase, generation_rate, fraction_range, hotspot_weights = (
            self._predictive_demand_settings()
        )
        generation_rate = min(
            1.0,
            float(generation_rate)
            * float(getattr(self, "synthetic_demand_scale", 1.0)),
        )
        low = max(1, int(round(self.num_vehicles * fraction_range[0])))
        high = max(low, int(round(self.num_vehicles * fraction_range[1])))
        if self.lowrequest:
            generation_rate *= 0.25
        return phase, generation_rate, low, high, hotspot_weights

    def _record_generated_demand(self, phase: str, count: int) -> None:
        self.current_demand_phase = phase
        self.generated_requests_last_step = int(count)
        self.generated_requests_by_phase[phase] = (
            self.generated_requests_by_phase.get(phase, 0) + int(count)
        )
    
    def _setup_vehicles(self):
        """Setup vehicle states from the configured episode initialization seed."""
        # 临时保存当前随机状态
        current_random_state = random.getstate()
        current_numpy_state = np.random.get_state()
        
        initialization_seed = getattr(
            self, 'vehicle_initialization_seed', self.initial_random_seed
        )
        if initialization_seed is not None:
            random.seed(initialization_seed)
            np.random.seed(initialization_seed)
        
        for i in range(self.num_vehicles):
            # Convert grid coordinates to location index
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            location_index = y * self.grid_size + x
            
            if i < self.ev_num_vehicles:
                vehicle_type = 1  # EV (Electric Vehicle)
                vehicle_type_name = 'EV'
            else:
                vehicle_type = 2  # AEV (Autonomous Electric Vehicle)
                vehicle_type_name = 'AEV'
            
            self.vehicles[i] = {
                'type': vehicle_type,  # Vehicle type: 1=EV, 2=AEV (数值编码)
                'type_name': vehicle_type_name,  # Vehicle type name: 'EV' or 'AEV' (字符串名称)
                'location': location_index,  # Use location index instead of coordinates
                'coordinates': (x, y),  # Keep coordinates for visualization
                'battery': self._initial_vehicle_battery(i),
                'charging_station': None,
                'charging_time_left': 0,
                'charging_session_start_time': None,
                'completed_charging_durations_minutes': [],
                'total_distance': 0,
                'charging_count': 0,
                'assigned_request': None,  # Currently assigned passenger request
                'passenger_onboard': None,  # Passenger being transported
                'service_earnings': 0,  # Total earnings from completed requests
                'rejected_requests': 0,  # Track rejected requests for analysis
                'unserved_penalty': 0,  # Accumulated penalty for unserved requests
                'is_stationary': False,  # Whether the vehicle is in waiting/stationary state
                'stationary_duration': 0 , # Duration to remain stationary
                'target_location': None,
                'charging_target': None,  # Target location for idling
                'idle_target': None,  # Target location for idling
                'idle_timer': 0,  # Timer for idle duration
                'continual_reject': 0,
                'penalty_timer': 0, 
                'daily_salary': 0,  # Accumulated salary for the day
                "salary_ratio":0,  # Salary ratio based on performance
                'satisfaction':0,
                'is_online': True,
                'offline_until_time': None,
                'whether_finishrequest': False,  # Whether the vehicle has completed a request
                'whether_finishrelocate': False,  # Whether the vehicle has completed a relocation
                'ifhavetarget': False,
                'zone_id': self.get_zone_id(location_index),  # Zone ID based on initial location
                'needs_emergency_charging': False,  # Whether the vehicle needs emergency charging
                'no_charge_cooldown_until': 0,  # When EV declines charging, skip charging for 5 steps
            }
            if vehicle_type == 2:
                self.vehicles[i]['salary_ratio'] = 1e+4
            # Initialize storeactions for each vehicle
            self.storeactions[i] = None
            self.storeactions_ev[i] = None
        
        # 恢复原来的随机状态
        random.setstate(current_random_state)
        np.random.set_state(current_numpy_state)
        
        print(
            f"✓ Vehicles initialized with seed {initialization_seed}",
            flush=True,
        )
    


    def _request_surge_bonus(self, request) -> float:
        """Return the monetary surge bonus shown to a synthetic EV driver."""
        explicit = getattr(request, 'surge_bonus', None)
        if explicit is not None:
            return max(0.0, float(explicit))
        regular_fare = float(getattr(request, 'value', 0.0) or 0.0)
        final_fare = float(
            getattr(request, 'final_value', regular_fare) or regular_fare
        )
        return max(0.0, final_fare - regular_fare)

    def configure_recourse_experiment(
        self,
        variant: str = "legacy",
        *,
        rejection_logit_shift: float = 0.0,
        common_random_numbers: bool = False,
    ) -> None:
        """Configure the shared R0--R4 contract for synthetic experiments."""
        variant = str(variant or "legacy").strip().lower()
        if variant not in {"legacy", "r0", "r1", "r2", "r3", "r4"}:
            raise ValueError(
                "recourse variant must be legacy or one of r0, r1, r2, r3, r4"
            )
        shift = float(rejection_logit_shift)
        if not math.isfinite(shift):
            raise ValueError("rejection_logit_shift must be finite")
        self.recourse_variant = variant
        self.rejection_logit_shift = shift
        self.common_random_numbers = bool(common_random_numbers)
        if not hasattr(self, "recourse_run_id"):
            self.recourse_run_id = (
                f"synthetic-seed-{int(getattr(self, 'initial_random_seed', 0) or 0)}"
            )

    def _ensure_recourse_runtime(self) -> None:
        if not hasattr(self, "request_lifecycle"):
            self.request_lifecycle = RequestLifecycleTracker()
        if not hasattr(self, "recourse_coordinator"):
            self.recourse_coordinator = RecourseCoordinator(
                lifecycle=self.request_lifecycle
            )
        if not hasattr(self, "_last_offer_realizations"):
            self._last_offer_realizations = {}
        if not hasattr(self, "_pending_recourse_actions"):
            self._pending_recourse_actions = {}
        if not hasattr(self, "_current_ev_stage_request_ids"):
            self._current_ev_stage_request_ids = set()
        if not hasattr(self, "_current_ev_offered_request_ids"):
            self._current_ev_offered_request_ids = set()

    def _epoch_id(self) -> int:
        return StateSnapshotBuilder.epoch_id(self)

    def _acceptance_uniform(self, vehicle_id: int, request) -> float:
        if not bool(getattr(self, "common_random_numbers", False)):
            return random.random()
        self._ensure_recourse_runtime()
        request_id = int(getattr(request, "request_id"))
        epoch_id = self._epoch_id()
        attempt_index = self.request_lifecycle.next_attempt_index(
            epoch_id, int(vehicle_id), request_id
        )
        key = (
            f"{int(getattr(self, 'initial_random_seed', 0) or 0)}|"
            f"{getattr(self, 'recourse_run_id', '')}|"
            f"{int(getattr(self, 'cumulative_episode_index', 0) or 0)}|"
            f"{int(getattr(self, 'episode_start_day', 0) or 0)}|"
            f"{int(getattr(self, 'request_generation_seed', 0) or 0)}|"
            f"{int(getattr(self, 'vehicle_initialization_seed', 0) or 0)}|"
            f"{epoch_id}|"
            f"{int(vehicle_id)}|{request_id}|{attempt_index}"
        ).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="big", signed=False)
        return (integer + 0.5) / float(2**64)

    def _begin_joint_collection(self, mode: str):
        self._ensure_recourse_runtime()
        self._pending_recourse_actions = {}
        if getattr(self, "evaluatemode", False):
            return None
        if self.recourse_coordinator.pending is not None:
            return self.recourse_coordinator.pending
        solver_backend = (
            "auction"
            if getattr(self, "useauction", False)
            else (
                f"mcmf:{getattr(self, 'mcmf_backend', 'unknown')}"
                if getattr(self, "usemcmf", False)
                else ("gurobi" if self.assignmentgurobi else "heuristic")
            )
        )
        return self.recourse_coordinator.begin(
            self,
            mode=mode,
            recourse_variant=getattr(self, "recourse_variant", "legacy"),
            state_variant=getattr(
                self, "state_variant", "joint_state_shared_critic"
            ),
            learner_variant=getattr(
                self, "learner_variant", "optimization_anchored_residual"
            ),
            solver_backend=solver_backend,
        )

    def _annotate_recourse_actions(self, actions) -> None:
        pending = getattr(self.recourse_coordinator, "pending", None)
        if pending is None:
            return
        for vehicle_id, action in actions.items():
            if action is None:
                continue
            metadata = getattr(action, "metadata", None)
            if metadata is None:
                continue
            vehicle_type = int(self.vehicles.get(vehicle_id, {}).get("type", 1))
            aev_first = pending.mode in {"aev_first", "aevfirst"}
            is_leader = (
                vehicle_type == 2 if aev_first else vehicle_type == 1
            )
            metadata.transition_id = pending.transition_id
            metadata.stage_id = 1 if is_leader else 2
            metadata.state_snapshot = pending.pre_state if is_leader else pending.residual_state
            metadata.feasible_graph_snapshot = (
                pending.aev_stage_graph
                if vehicle_type == 2
                else pending.ev_stage_graph
            )
            metadata.joint_action_snapshot = (
                pending.aev_joint_action
                if vehicle_type == 2
                else pending.ev_joint_action
            )
            if isinstance(action, ServiceAction):
                request_id = int(action.request_id)
                if metadata.request_snapshot is None:
                    request = self.active_requests.get(request_id)
                    if request is not None:
                        metadata.request_snapshot = RequestSnapshot.from_request(request)
                metadata.acceptance_outcome = (
                    "rejected"
                    if bool(getattr(action, "was_rejected", False))
                    else "accepted"
                )
                if pending.residual_state is not None:
                    metadata.residual_category = pending.residual_state.request_label(
                        request_id
                    )
            self._pending_recourse_actions[int(vehicle_id)] = action

    def _finalize_joint_collection(self, rewards: dict[int, float], done: bool):
        if getattr(self, "evaluatemode", False):
            return None
        transition = self.recourse_coordinator.finalize(
            self, rewards=rewards, done=done
        )
        if transition is None:
            return None
        if self.value_function is not None and self.value_function_ev is not None:
            for value_function in {
                id(self.value_function): self.value_function,
                id(self.value_function_ev): self.value_function_ev,
            }.values():
                router = getattr(value_function, "set_joint_critic_router", None)
                if callable(router):
                    router(
                        ev_value_function=self.value_function_ev,
                        aev_value_function=self.value_function,
                    )
            follower_setter = getattr(
                self.value_function_ev, "set_follower_target_provider", None
            )
            follower_provider = getattr(
                self.value_function, "target_components_for_graph", None
            )
            if callable(follower_setter) and callable(follower_provider):
                follower_setter(follower_provider)
        for action in self._pending_recourse_actions.values():
            action.metadata.next_state_snapshot = transition.next_state
        seen = set()
        for value_function in (self.value_function, self.value_function_ev):
            if value_function is None or id(value_function) in seen:
                continue
            seen.add(id(value_function))
            store = getattr(value_function, "store_recourse_transition", None)
            if callable(store):
                store(transition)
        self._pending_recourse_actions = {}
        return transition

    def _ev_acceptance_probability(self, vehicle_id, request=None, distance=None):
        """Empirical binary-logit acceptance probability for a synthetic EV.

        Ashkrof et al. use idle time, pickup time and the monetary surge bonus
        in the acceptance utility.  The logit probability already integrates
        the random utility error, so drawing another normal error here would
        double-count stochasticity; the Bernoulli draw is performed only in
        ``_should_reject_request``.
        """
        vehicle = self.vehicles[vehicle_id]
        if vehicle['type'] == 2:
            return 1.0
        if not self.ifreject:
            return 1.0

        if distance is None:
            if request is None:
                distance = 0.0
            else:
                distance = self._manhattan_distance_loc(
                    vehicle['location'], request.pickup
                )
        surge_bonus = (
            self._request_surge_bonus(request) if request is not None else 0.0
        )
        utility = (
            float(getattr(self, 'ev_acceptance_asc', 1.810))
            + float(getattr(self, 'ev_acceptance_beta_idle', -0.017))
            * float(vehicle.get('idle_timer', 0.0) or 0.0)
            + float(getattr(self, 'ev_acceptance_beta_pickup', -0.050))
            * float(distance)
            + float(getattr(self, 'ev_acceptance_beta_surge', 0.101))
            * float(surge_bonus)
            + float(getattr(self, 'rejection_logit_shift', 0.0))
        )
        return min(1.0, max(0.0, self._sigmoid(utility)))

    def _calculate_rejection_probability_disttest(self, vehicle_id, distance):
        return 1.0 - self._ev_acceptance_probability(
            vehicle_id, distance=distance
        )

    def _calculate_rejection_probability(self, vehicle_id, request):
        return 1.0 - self._ev_acceptance_probability(vehicle_id, request=request)

    def _calculate_rejection_probabilityreal(self, vehicle_id, request):
        return self._calculate_rejection_probability(vehicle_id, request)

    def _calculate_known_rejection_probability(
        self, vehicle_id, request, *, sample_noise=False
    ):
        # Compatibility with the shared optimizer interface.  ``sample_noise``
        # is intentionally ignored because the logit error is already
        # represented by the Bernoulli outcome at execution time.
        return self._calculate_rejection_probability(vehicle_id, request)


    
    def _should_reject_request(self, vehicle_id, request):
        """Determine if a vehicle should reject a request"""
        variant_policy = RecourseTargetBuilder.variant_policy(
            getattr(self, "recourse_variant", "legacy")
        )
        if not self.ifreject or not variant_policy.rejection_enabled:
            rejected = False
            acceptance_probability = 1.0
            acceptance_uniform = 1.0
            self._last_offer_realizations[
                (self._epoch_id(), int(vehicle_id), int(request.request_id))
            ] = {
                "acceptance_probability": acceptance_probability,
                "uniform": acceptance_uniform,
                "rejected": rejected,
            }
            return rejected
        rejection_prob = self._calculate_rejection_probabilityreal(vehicle_id, request)
        acceptance_uniform = self._acceptance_uniform(vehicle_id, request)
        rejected = acceptance_uniform < rejection_prob
        self._last_offer_realizations[
            (self._epoch_id(), int(vehicle_id), int(request.request_id))
        ] = {
            "acceptance_probability": 1.0 - float(rejection_prob),
            "uniform": float(acceptance_uniform),
            "rejected": bool(rejected),
        }
        return rejected
    

    def _get_maximum_service_time(self):
        if self.aev_test_service_time is not None:
            return max(1, int(math.ceil(self.aev_test_service_time)))
        return max(1, int(self.grid_size * 2))


    def _count_idle_aev_vehicles(self):
        return sum(
            1
            for vehicle in self.vehicles.values()
            if vehicle.get('type') == 2
            and vehicle.get('is_online', True)
            and vehicle.get('assigned_request') is None
            and vehicle.get('passenger_onboard') is None
            and vehicle.get('charging_station') is None
        )


    def _get_pending_active_request_ids(self):
        busy_request_ids = {
            request_id
            for vehicle in self.vehicles.values()
            for request_id in (vehicle.get('assigned_request'), vehicle.get('passenger_onboard'))
            if request_id is not None
        }
        return [request_id for request_id in self.active_requests if request_id not in busy_request_ids]


    def _get_aev_capacity_snapshot(self):
        active_requests = len(self.active_requests)
        pending_active_requests = len(self._get_pending_active_request_ids())
        maximum_servicetime = self._get_maximum_service_time()
        required_idle_aev = int(math.ceil(pending_active_requests / maximum_servicetime)) if pending_active_requests > 0 else 0
        idle_aev = self._count_idle_aev_vehicles()
        total_aev = sum(
            1
            for vehicle in self.vehicles.values()
            if vehicle.get('type') == 2 and vehicle.get('is_online', True)
        )
        stats = {
            'time': self.current_time,
            'active_requests': active_requests,
            'pending_active_requests': pending_active_requests,
            'maximum_service_time': maximum_servicetime,
            'required_idle_aev': required_idle_aev,
            'idle_aev': idle_aev,
            'total_aev': total_aev,
            'feasible': idle_aev >= required_idle_aev,
        }
        return stats


    def _enforce_aev_test_capacity_limit(self):
        pending_request_ids = self._get_pending_active_request_ids()
        max_pending_requests = self._count_idle_aev_vehicles() * self._get_maximum_service_time()
        excess_pending = len(pending_request_ids) - max_pending_requests
        if excess_pending <= 0:
            return 0

        trimmed = 0
        for request_id in sorted(pending_request_ids, reverse=True):
            if trimmed >= excess_pending:
                break
            if request_id in self.active_requests:
                del self.active_requests[request_id]
                trimmed += 1

        if trimmed > 0:
            self.aev_capacity_trimmed_requests += trimmed
            self.whole_req_num = max(0, self.whole_req_num - trimmed)

        return trimmed


    def _record_aev_capacity_snapshot(self):
        snapshot = self._get_aev_capacity_snapshot()
        self.aev_capacity_history.append(snapshot)
        self.max_required_idle_aev = max(self.max_required_idle_aev, snapshot['required_idle_aev'])
        self.max_idle_aev_available = max(self.max_idle_aev_available, snapshot['idle_aev'])
        self.max_observed_active_requests = max(self.max_observed_active_requests, snapshot['active_requests'])
        self.max_pending_active_requests = max(self.max_pending_active_requests, snapshot['pending_active_requests'])
        if not snapshot['feasible']:
            self.aev_capacity_violations += 1
        return snapshot


    def set_aev_larger_env(self):
        if self.aev_test_request_generation_rate_override is not None:
            self.request_generation_rate = min(1.0, max(0.0, self.aev_test_request_generation_rate_override))
            self.aev_test_request_generation_rate = self.request_generation_rate
            return self.request_generation_rate
        maximum_servicetime = self._get_maximum_service_time()
        num_aev = max(1, self.num_vehicles - self.ev_num_vehicles)
        request_frequency = (num_aev / maximum_servicetime) * max(0.0, self.aev_test_request_rate_scale)
        self.request_generation_rate = min(1.0, float(request_frequency))
        self.aev_test_request_generation_rate = self.request_generation_rate
        return self.request_generation_rate




    def _generate_random_requests(self):
        """Generate new passenger requests in batches"""
        generated_requests = []
        
        # 设置请求生成专用的随机状态
        self._set_request_generation_random_state()
        
        phase, generation_rate, min_requests, max_requests, _ = (
            self._request_generation_settings()
        )
        if random.random() < generation_rate:
            num_requests = random.randint(min_requests, max_requests)

            
            for request_idx in range(num_requests):
                self.request_counter += 1
                
                # 为每个请求设置不同的随机种子，确保位置分布的随机性
                # 使用当前时间、请求计数器和循环索引组合作为种子
                request_seed = (self.request_generation_seed if hasattr(self, 'request_generation_seed') and self.request_generation_seed is not None else 12345) + \
                              self.current_time * 1000 + self.request_counter * 10 + request_idx
                random.seed(request_seed)
                np.random.seed(request_seed)
                
                # Random pickup and dropoff locations
                pickup_x = random.randint(0, self.grid_size - 1)
                pickup_y = random.randint(0, self.grid_size - 1)
                pickup_location = pickup_y * self.grid_size + pickup_x
                
                dropoff_x = random.randint(0, self.grid_size - 1)
                dropoff_y = random.randint(0, self.grid_size - 1)
                dropoff_location = dropoff_y * self.grid_size + dropoff_x
                
                # Ensure pickup and dropoff are different
                attempts = 0
                while dropoff_location == pickup_location and attempts < 5:
                    random.seed(request_seed+attempts)
                    np.random.seed(request_seed+attempts)
                    dropoff_x = random.randint(0, self.grid_size - 1)
                    dropoff_y = random.randint(0, self.grid_size - 1)
                    dropoff_location = dropoff_y * self.grid_size + dropoff_x
                    attempts += 1
                
                # Calculate travel time (Manhattan distance)
                travel_time = max(abs(pickup_x - dropoff_x), abs(pickup_y - dropoff_y))
                
                # Create request with demand/supply-based dynamic pricing.
                base_value = 25
                distance_value = travel_time * (2 + np.random.rand()*0.1)
                surge_factor = self.return_surgingpricing()
                surge_factor = surge_factor[self.get_zone_id(pickup_location)] if hasattr(self, 'get_zone_id') else 1.0
                point_loc = pickup_y * self.grid_size + pickup_x
                zone_loc = self.loc_to_zone.get(point_loc, None)
                if zone_loc==1:
                    distance_value += 5  # Downtown has higher surge
                elif zone_loc==2:
                    distance_value -= 2  # Suburban has moderate surge
                elif zone_loc==3:
                    distance_value -= 2  # Outskirts have lower surge
                regular_value = (base_value + distance_value) * 0.1
                final_value = regular_value * surge_factor
                
                request = Request(
                    request_id=self.request_counter,
                    source=pickup_location,
                    destination=dropoff_location,
                    current_time=self.current_time,
                    travel_time=travel_time,
                    value=regular_value,
                    final_value=final_value
                )
                request.surge_multiplier = float(surge_factor)
                request.surge_bonus = max(0.0, final_value - regular_value)
                self._apply_predictive_request_patience(request)
                
                self.active_requests[self.request_counter] = request
                generated_requests.append(request)
                
                # Track request generation for visualization
                if not hasattr(self, 'request_generation_history'):
                    self.request_generation_history = []
                self.request_generation_history.append({
                    'pickup_coords': (pickup_x, pickup_y),
                    'dropoff_coords': (dropoff_x, dropoff_y),
                    'hotspot_idx': None,  # No hotspot for random requests
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            # 保存请求生成的随机状态
            self._save_request_generation_random_state()
            
            self._record_generated_demand(phase, len(generated_requests))
            return generated_requests
        
        # 保存请求生成的随机状态
        self._save_request_generation_random_state()
        
        self._record_generated_demand(phase, 0)
        return []
    
    def _generate_intense_requests(self):
        """Generate multiple requests concentrated in 3 hotspots with probability weights"""
        generated_requests = []
        
        # 设置请求生成专用的随机状态
        self._set_request_generation_random_state()
        

        phase, generation_rate, min_requests, max_requests, probability_weights = (
            self._request_generation_settings()
        )

        # Determine how many requests to generate this step
        if random.random() < generation_rate:
            num_requests = random.randint(min_requests, max_requests)

            # Define 3 hotspot centers in the grid
            hotspots = self.hotspot_locations

            # Probability weights for each hotspot (should sum to 1.0)
            selected_hotspot_idx_reward = [30, 30, 30, 30]  # Reward weights for each hotspot
            for request_idx in range(num_requests):
                self.request_counter += 1
                
                # 为每个请求设置不同的随机种子，确保位置分布的随机性
                # 使用当前时间、请求计数器和循环索引组合作为种子
                request_seed = (self.request_generation_seed if hasattr(self, 'request_generation_seed') and self.request_generation_seed is not None else 12345) + \
                              self.current_time * 1000 + self.request_counter * 10 + request_idx
                random.seed(request_seed)
                np.random.seed(request_seed)
                # Select hotspot based on probability weights
                rand_val = random.random()
                cumulative_prob = 0
                selected_hotspot_idx = 0
                for i, weight in enumerate(probability_weights):
                    cumulative_prob += weight
                    if rand_val <= cumulative_prob:
                        selected_hotspot_idx = i
                        break
                
                hotspot_center = hotspots[selected_hotspot_idx]
                
                # Generate pickup location near selected hotspot (with some randomness)
                hotspot_radius = max(2, self.grid_size // 8)  # Radius around hotspot
                pickup_x = max(0, min(self.grid_size - 1, 
                                    hotspot_center[0] + random.randint(-hotspot_radius, hotspot_radius)))
                pickup_y = max(0, min(self.grid_size - 1, 
                                    hotspot_center[1] + random.randint(-hotspot_radius, hotspot_radius)))
                pickup_location = pickup_y * self.grid_size + pickup_x

                available_hotspot_indices = [i for i in range(len(hotspots)) if i != selected_hotspot_idx]

                if self.synthetic_demand_profile == "predictive":
                    random_dropoffx, random_dropoffy = (
                        self._predictive_dropoff_coordinates(phase)
                    )
                else:
                    random_dropoffx = random.randint(0, self.grid_size - 1)
                    random_dropoffy = random.randint(0, self.grid_size - 1)

                dropoff_location = random_dropoffy * self.grid_size + random_dropoffx
                
                # Ensure pickup and dropoff are different
                attempts = 0
                while dropoff_location == pickup_location and attempts < 5:
                    random.seed(request_seed + attempts)
                    np.random.seed(request_seed + attempts)
                    random_dropoffx = random.randint(0, self.grid_size - 1)
                    random_dropoffy = random.randint(0, self.grid_size - 1)
                    dropoff_location = random_dropoffy * self.grid_size + random_dropoffx
                    attempts += 1
                dropoff_x = random_dropoffx
                dropoff_y = random_dropoffy
                # Calculate travel time (Manhattan distance)
                travel_time = max(abs(pickup_x - dropoff_x), abs(pickup_y - dropoff_y))
                
                # Create request with dynamic pricing based on demand
                base_value = 10
                distance_value = travel_time * (1 + np.random.rand()*0.1)
                
                point_loc = pickup_y * self.grid_size + pickup_x
                zone_loc = self.loc_to_zone.get(point_loc, None)
                surge_factor = self.return_surgingpricing()
                surge_factor = surge_factor[zone_loc] if 0 <= zone_loc < len(surge_factor) else 1.0
                if zone_loc==1:
                    distance_value += 5  # Downtown has higher surge
                elif zone_loc==2:
                    distance_value -= 2  # Suburban has moderate surge
                elif zone_loc==3:
                    distance_value -= 2  # Outskirts have lower surge
                regular_value = (
                    base_value
                    + distance_value
                    + selected_hotspot_idx_reward[selected_hotspot_idx]
                ) * 0.1
                final_value = regular_value * surge_factor
                

                
                request = Request(
                    request_id=self.request_counter,
                    source=pickup_location,
                    destination=dropoff_location,
                    current_time=self.current_time,
                    travel_time=travel_time,
                    value=regular_value,
                    final_value=final_value
                )
                request.surge_multiplier = float(surge_factor)
                request.surge_bonus = max(0.0, final_value - regular_value)
                self._apply_predictive_request_patience(request)
                
                self.active_requests[self.request_counter] = request
                generated_requests.append(request)
                
                # Track request generation for visualization
                if not hasattr(self, 'request_generation_history'):
                    self.request_generation_history = []
                self.request_generation_history.append({
                    'pickup_coords': (pickup_x, pickup_y),
                    'dropoff_coords': (random_dropoffx, random_dropoffy),
                    'hotspot_idx': selected_hotspot_idx,
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            # 保存请求生成的随机状态
            self._save_request_generation_random_state()
            
            self._record_generated_demand(phase, len(generated_requests))
            return generated_requests
        
        # 保存请求生成的随机状态
        self._save_request_generation_random_state()
        
        self._record_generated_demand(phase, 0)
        return []
    
    def _record_same_epoch_recourse_if_applicable(self, vehicle_id, request_id):
        """Record an EV-rejected request accepted by an AEV in this epoch."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or int(vehicle.get('type', 1)) != 2:
            return False
        rejected_at = self.ev_rejection_times.get(request_id)
        if rejected_at is None or int(rejected_at) != self._epoch_id():
            return False
        recourse_ids = self.ev_rejected_recovered_same_epoch_ids
        is_new = request_id not in recourse_ids
        recourse_ids.add(request_id)
        self._ensure_recourse_runtime()
        self.request_lifecycle.record_aev_assignment(
            int(request_id),
            vehicle_id=int(vehicle_id),
            epoch_id=self._epoch_id(),
            vehicle_type=int(vehicle.get('type', 2)),
        )
        return is_new

    def _assign_request_to_vehicle(self, vehicle_id, request_id):
        """Assign a request to a vehicle with rejection logic"""
        #print("vehicle_id:", vehicle_id, "request_id:", request_id,"check if active_requests:", request_id in self.active_requests)
        if request_id in self.active_requests and vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            request = self.active_requests[request_id]

            # Check if this request is already assigned to another vehicle or onboard
            for other_vid, other_veh in self.vehicles.items():
                if other_vid != vehicle_id:
                    if other_veh['assigned_request'] == request_id:
                        print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: already assigned to vehicle {other_vid}")
                        return False
                    if other_veh['passenger_onboard'] == request_id:
                        print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: already onboard vehicle {other_vid}")
                        return False

            # EV penalty period: do not allow receiving orders.
            if vehicle['penalty_timer'] > 0:
                print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: vehicle in penalty period ({vehicle['penalty_timer']} steps left)")
                return False
            if not vehicle.get('is_online', True):
                return False
            #print("assign_request_to_vehicle: Vehicle {} request {} at step {}".format(vehicle_id, request_id, self.current_time))
            # Vehicle must be completely free (both assigned_request AND passenger_onboard must be None)
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                # Check if the vehicle rejects the request
                was_rejected = self._should_reject_request(vehicle_id, request)
                if int(vehicle.get('type', 1)) == 1:
                    self._ensure_recourse_runtime()
                    self._current_ev_offered_request_ids.add(int(request_id))
                    epoch_id = self._epoch_id()
                    realization = self._last_offer_realizations.get(
                        (epoch_id, int(vehicle_id), int(request_id)),
                        {
                            "acceptance_probability": 0.0 if was_rejected else 1.0,
                            "uniform": 0.0 if was_rejected else 1.0,
                        },
                    )
                    pending = self.recourse_coordinator.pending
                    transition_id = (
                        pending.transition_id
                        if pending is not None
                        else f"synthetic-eval:{epoch_id}"
                    )
                    self.request_lifecycle.record_offer(
                        transition_id=transition_id,
                        epoch_id=epoch_id,
                        request=request,
                        ev_id=int(vehicle_id),
                        vehicle=vehicle,
                        acceptance_probability=float(
                            realization["acceptance_probability"]
                        ),
                        acceptance_uniform=float(realization["uniform"]),
                        accepted=not bool(was_rejected),
                        rejection_reason=(
                            "driver_reject" if was_rejected else None
                        ),
                    )
                if was_rejected:
                    vehicle['rejected_requests'] += 1
                    self._record_ev_rejection(vehicle_id)
                    vehicle['assigned_request'] = request_id
                    self.rejected_requests.append(request)
                    if vehicle.get('type') == 1:
                        request.ev_rejection_count = getattr(request, 'ev_rejection_count', 0) + 1
                        request.was_ev_rejected = True
                        self.ev_rejected_request_ids.add(request_id)
                        self.ev_rejection_times[request_id] = self._epoch_id()
                        self.request_lifecycle.mark_residual(
                            int(request_id),
                            epoch_id=self._epoch_id(),
                            category="rejected",
                            eligible=False,
                        )
                    
                    # 清除idle相关状态（即使拒绝，也要标记车辆不再stationary）
                    vehicle['idle_target'] = None
                    vehicle['is_stationary'] = False
                    vehicle['stationary_duration'] = 0
                    
                    # 存储EV拒绝订单的经验到value_function_ev的rejection_buffer（仅对EV）
                    if vehicle['type'] == 1 and hasattr(self, 'value_function_ev') and self.value_function_ev is not None:
                        # Calculate distance for rejection experience
                        vehicle_coords = vehicle['coordinates']
                        pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                        
                        # Store rejection experience in EV's value function
                        if hasattr(self.value_function_ev, 'store_rejection_experience'):
                            self.value_function_ev.store_rejection_experience(
                                vehicle_id=vehicle_id,
                                request_id=request_id,
                                vehicle_location=vehicle['location'],
                                pickup_location=request.pickup,
                                current_time=self.current_time,
                                distance=distance,
                                rejection_reason="distance"
                            )
                    
                    #print("assign_request_to_vehicle: Vehicle {} request {} at step {}".format(vehicle_id, request_id, self.current_time))
                    return False  # Request rejected
                # Request accepted
                vehicle['assigned_request'] = request_id
                request.assigned_time = self.current_time
                self._record_ev_acceptance(vehicle_id)
                self._record_same_epoch_recourse_if_applicable(vehicle_id, request_id)
                if int(vehicle.get('type', 1)) == 2:
                    self.request_lifecycle.record_aev_assignment(
                        int(request_id),
                        vehicle_id=int(vehicle_id),
                        epoch_id=self._epoch_id(),
                        vehicle_type=int(vehicle.get('type', 2)),
                    )
                if vehicle['type'] == 1:
                    self._clear_ev_charge_trigger(vehicle_id)
                
                # 存储EV接受订单的经验到rejection_buffer（仅对EV）
                if vehicle['type'] == 1 and hasattr(self, 'value_function_ev'):
                    vehicle_coords = vehicle['coordinates']
                    pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                    
                    # 调用value_function_ev的store_acceptance_experience方法
                    if hasattr(self.value_function_ev, 'store_acceptance_experience'):
                        self.value_function_ev.store_acceptance_experience(
                            vehicle_id=vehicle_id,
                            request_id=request_id,
                            vehicle_location=vehicle['location'],
                            pickup_location=request.pickup,
                            current_time=self.current_time,
                            distance=distance
                        )
                
                # 清除idle相关状态，因为车辆现在要去接客
                vehicle['idle_target'] = None
                vehicle['is_stationary'] = False
                vehicle['stationary_duration'] = 0
                
                if vehicle['type']==1:
                    self.ev_requests.append(request)
                #print("✅  Vehicle {} accepted request {} at step {}".format(vehicle_id, request_id, self.current_time))
                return True
            else:
                # Vehicle is already busy
                print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: vehicle already has assigned_request={vehicle['assigned_request']} or passenger_onboard={vehicle['passenger_onboard']}")
                return False
        else:
            print("wrong assign_request_to_vehicle: Vehicle {} or request {} not found at step {}".format(vehicle_id, request_id, self.current_time))
            return False

    def _move_vehicle_to_charging_station(self, vehicle_id, station_id):
        """Move a vehicle to a charging station for rebalancing"""
        if vehicle_id in self.vehicles and hasattr(self, 'charging_manager'):
            vehicle = self.vehicles[vehicle_id]
            # Freeze congestion before registering this vehicle as inbound.
            # Replay must describe the queue observed when the charge action
            # was chosen, not the station state after arrival/completion.
            decision_queue_features = self._charging_queue_feature_snapshot(
                vehicle_id,
                station_id,
                current_time=self.current_time,
            )
            vehicle['charging_decision_queue_features'] = (
                list(decision_queue_features)
                if decision_queue_features is not None else None
            )
            vehicle['charging_decision_station_id'] = int(station_id)
            vehicle['charging_decision_time'] = float(self.current_time)
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            self._clear_vehicle_charging_queues(
                vehicle_id, except_station_id=station_id
            )
            vehicle['charging_target'] = None
            vehicle['idle_target'] = None
            vehicle['charging_station'] = None
            station = self.charging_manager.stations[station_id]
            # Convert station location to coordinates
            station_x = station.location % self.grid_size
            station_y = station.location // self.grid_size
            station_coords = (station_x, station_y)
            
            # Set vehicle destination to charging station
            vehicle['target_location'] = station_coords
            vehicle['charging_target'] = station_id
            self._register_aev_notarrived_reservation(vehicle_id, station_id)
            if vehicle.get('type') == 1:
                self._clear_ev_charge_trigger(vehicle_id)
            
            return True

    def _clear_aev_notarrived_reservations(self, vehicle_id):
        if self.vehicles.get(vehicle_id, {}).get('type') == 1 or not hasattr(self, 'charging_manager'):
            return
        for station in self.charging_manager.stations.values():
            station.charging_queue_notarrived = [
                queued_vehicle for queued_vehicle in station.charging_queue_notarrived
                if queued_vehicle != vehicle_id
            ]

    def _clear_vehicle_charging_queues(self, vehicle_id, except_station_id=None):
        """Remove stale physical queue entries when a vehicle changes station."""
        vehicle_key = self._queue_vehicle_key(vehicle_id)
        for station_id, station in self.charging_manager.stations.items():
            if except_station_id is not None and int(station_id) == int(except_station_id):
                continue
            station.charging_queue = [
                queued for queued in station.charging_queue
                if str(queued) != vehicle_key
            ]
            self._charging_queue_arrivals.pop(
                (int(vehicle_id), int(station_id)), None
            )

    def _register_aev_notarrived_reservation(self, vehicle_id, station_id):
        if self.vehicles.get(vehicle_id, {}).get('type') == 1 or not hasattr(self, 'charging_manager'):
            return
        self._clear_aev_notarrived_reservations(vehicle_id)
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return
        if vehicle_id not in station.charging_queue_notarrived:
            station.charging_queue_notarrived.append(vehicle_id)

    def _clear_aev_notarrived_if_arrived(self, vehicle_id, station_id):
        if self.vehicles.get(vehicle_id, {}).get('type') == 1 or not hasattr(self, 'charging_manager'):
            return
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return
        station.charging_queue_notarrived = [
            queued_vehicle for queued_vehicle in station.charging_queue_notarrived
            if queued_vehicle != vehicle_id
        ]

    def _queue_vehicle_key(self, vehicle_id):
        return str(vehicle_id)

    def _is_vehicle_waiting_for_charger(self, vehicle_id, station_id=None):
        if not hasattr(self, 'charging_manager'):
            return False
        vehicle_key = self._queue_vehicle_key(vehicle_id)
        stations = (
            [self.charging_manager.stations.get(station_id)]
            if station_id is not None else self.charging_manager.stations.values()
        )
        for station in stations:
            if station is None:
                continue
            if vehicle_key in {str(queued) for queued in getattr(station, 'charging_queue', [])}:
                return True
        return False

    def _is_vehicle_committed_to_charging(self, vehicle_id):
        vehicle = self.vehicles.get(vehicle_id, {})
        return (
            vehicle.get('charging_station') is not None
            or vehicle.get('charging_target') is not None
            or self._is_vehicle_waiting_for_charger(vehicle_id)
        )

    def _charging_stations_accepting_arrivals(self):
        """Return stations with physical or finite waiting-room capacity."""
        if not hasattr(self, 'charging_manager'):
            return []
        queue_capacity = max(
            0,
            int(getattr(self, 'station_queue_capacity', 0) or 0),
        )
        accepting = []
        for station in self.charging_manager.stations.values():
            reserved = (
                len(getattr(station, 'current_vehicles', []) or [])
                + len(getattr(station, 'charging_queue', []) or [])
                + len(getattr(station, 'charging_queue_notarrived', []) or [])
            )
            if reserved < int(station.max_capacity) + queue_capacity:
                accepting.append(station)
        return accepting

    def _charging_queue_feature_snapshot(self, vehicle_id, station_id, current_time=None):
        value_function = getattr(self, 'value_function', None)
        if value_function is None or not hasattr(value_function, '_queue_features'):
            return None
        vehicle = self.vehicles.get(vehicle_id, {})
        station = self.charging_manager.stations.get(station_id)
        if station is None:
            return None
        try:
            return value_function._queue_features(
                station_id=station_id,
                target_location=station.location,
                vehicle_id=vehicle_id,
                vehicle_location=vehicle.get('location', station.location),
                current_time=float(self.current_time if current_time is None else current_time),
                num_requests=float(len(getattr(self, 'active_requests', {}))),
                travel_duration=0.0,
            )
        except Exception:
            return None

    def _mark_charging_queue_arrival(self, vehicle_id, station_id):
        if not hasattr(self, 'charging_manager') or station_id not in self.charging_manager.stations:
            return
        key = (int(vehicle_id), int(station_id))
        if key in self._charging_queue_arrivals:
            return
        self._charging_queue_arrivals[key] = {
            'arrival_time': float(self.current_time),
            'features': self._charging_queue_feature_snapshot(vehicle_id, station_id),
            'vehicle_location': self.vehicles.get(vehicle_id, {}).get('location'),
        }

    def _mark_charging_started(self, vehicle_id, station_id):
        key = (int(vehicle_id), int(station_id))
        arrival = self._charging_queue_arrivals.pop(key, None)
        arrival_time = float(arrival.get('arrival_time', self.current_time)) if arrival else float(self.current_time)
        observed_wait = max(0.0, float(self.current_time) - arrival_time)
        predicted_wait = None
        vehicle_type = self.vehicles.get(vehicle_id, {}).get('type')
        predictor = self.value_function_ev if vehicle_type == 1 else self.value_function
        if predictor is not None and hasattr(predictor, '_queue_wait_from_feature_snapshot'):
            try:
                predicted_wait = predictor._queue_wait_from_feature_snapshot(
                    (arrival or {}).get('features')
                )
            except Exception:
                predicted_wait = None
        self.charging_wait_observations.append({
            'vehicle_id': int(vehicle_id),
            'station_id': int(station_id),
            'arrival_time': arrival_time,
            'start_time': float(self.current_time),
            'observed_wait': observed_wait,
            'predicted_wait': predicted_wait,
        })
        for value_function in (getattr(self, 'value_function', None), getattr(self, 'value_function_ev', None)):
            if value_function is None or not hasattr(value_function, 'store_queue_experience'):
                continue
            try:
                station = self.charging_manager.stations.get(station_id)
                value_function.store_queue_experience(
                    vehicle_id=vehicle_id,
                    station_id=station_id,
                    target_location=getattr(station, 'location', None),
                    vehicle_location=(arrival or {}).get('vehicle_location', self.vehicles.get(vehicle_id, {}).get('location')),
                    current_time=arrival_time,
                    num_requests=len(getattr(self, 'active_requests', {})),
                    observed_wait=observed_wait,
                    features=(arrival or {}).get('features'),
                )
            except Exception:
                continue

    def _charging_wait_step_penalty(self, vehicle_id, station_id):
        self._mark_charging_queue_arrival(vehicle_id, station_id)
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is not None:
            vehicle['charging_target'] = station_id
            vehicle['target_location'] = None
        penalty = max(0.0, float(getattr(self, 'charging_wait_penalty_per_step', 1.0)))
        self.charging_wait_steps += 1
        self.charging_wait_penalty_total += penalty
        return -penalty

    
    def _pickup_passenger(self, vehicle_id):
        """Vehicle picks up passenger at request pickup location"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成pickup
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成pickup - 订单未完成")
            # 将未完成的订单重新放回active_requests等待其他车辆
            if vehicle['assigned_request'] is not None:
                request_id = vehicle['assigned_request']
                vehicle['assigned_request'] = None
                #print(f"   订单 {request_id} 因车辆电池耗尽被重新分配")
            return False
            
        if vehicle['assigned_request'] is not None:
            # Check if the assigned request still exists
            if vehicle['assigned_request'] not in self.active_requests:
                print(f"🚫 Vehicle {vehicle_id} assigned_request {vehicle['assigned_request']} expired/removed (not in active_requests)")
                vehicle['assigned_request'] = None
                return False
                
            request = self.active_requests[vehicle['assigned_request']]
            vehicle_coords = vehicle['coordinates']
            pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
            

            distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
            request_age = self.current_time - (request.pickup_deadline - request.MAX_PICKUP_DELAY)
            is_expired = self.current_time > request.pickup_deadline
            #print(f"🚗 Vehicle {vehicle_id} moving to pickup: at {vehicle_coords}, target {pickup_coords}, distance={distance}, request_age={request_age:.0f}, expired={is_expired}")


            # Check if vehicle is at pickup location - allow pickup even if request expired (vehicle already committed)
            if vehicle_coords == pickup_coords:
                #print(f"🚗 Vehicle {vehicle_id} arrived at pickup location {pickup_coords} for request {vehicle['assigned_request']}")
                # Double-check: make sure no other vehicle has already picked up this passenger
                request_id = vehicle['assigned_request']
                already_picked_up = False
                for other_vid, other_veh in self.vehicles.items():
                    if other_vid != vehicle_id and other_veh['passenger_onboard'] == request_id:
                        already_picked_up = True
                        print(f"⚠️  Vehicle {vehicle_id} arrived at pickup but request {request_id} already picked up by vehicle {other_vid}")
                        break
                
                if not already_picked_up:
                    vehicle['passenger_onboard'] = vehicle['assigned_request']
                    request.pickup_time = self.current_time
                    self._ensure_recourse_runtime()
                    self.request_lifecycle.record_pickup(
                        int(request_id),
                        vehicle_id=int(vehicle_id),
                        epoch_id=self._epoch_id(),
                        vehicle_type=int(vehicle.get('type', 1)),
                    )
                    vehicle['assigned_request'] = None
                    if self.current_time % 25 == 0 or self.current_time > request.pickup_deadline:
                        expired_status = "EXPIRED" if self.current_time > request.pickup_deadline else ""
                        #print(f"✅ Vehicle {vehicle_id} picked up passenger (request {vehicle['passenger_onboard']}) at {vehicle_coords} {expired_status}")
                    return True
                else:
                    # Clear the assignment since another vehicle got there first
                    vehicle['assigned_request'] = None
                    return False
        return False
    


    def findchargerange_v(self):
        return_index = {}
        for j in self.charging_manager.stations.values():
            return_index[j.id] = []
            station_x = j.location % self.grid_size
            station_y = j.location // self.grid_size
            for vehicle_id, v in self.vehicles.items():
                vehicle_x = v['coordinates'][0]
                vehicle_y = v['coordinates'][1]
                distance = abs(vehicle_x - station_x) + abs(vehicle_y - station_y)
                if distance <= 5:
                    return_index[j.id].append(vehicle_id)
        return return_index

    def findchargerange_c(self,rebalance_num = 0):
        return_index = {}
        for vehicle_id, v in self.vehicles.items():
            return_index[vehicle_id] = 0
            vehicle_x = v['coordinates'][0]
            vehicle_y = v['coordinates'][1]
            battery = v['battery']
            sumcapa = 0
            for j in self.charging_manager.stations.values():
                station_x = j.location % self.grid_size
                station_y = j.location // self.grid_size
                distance = abs(vehicle_x - station_x) + abs(vehicle_y - station_y)
                if distance*self.battery_consum <= battery-0.01:
                    reserved = (
                        len(getattr(j, 'current_vehicles', []) or [])
                        + len(getattr(j, 'charging_queue', []) or [])
                        + len(getattr(j, 'charging_queue_notarrived', []) or [])
                    )
                    admission_limit = (
                        int(j.max_capacity)
                        + int(getattr(self, 'station_queue_capacity', 0))
                    )
                    sumcapa += max(0, admission_limit - reserved)
                # ``rebalance_num`` is the size of the whole dispatch batch,
                # not already reserved charging demand.  Subtracting it made
                # this value zero in every 200-vehicle dispatch and therefore
                # allowed critically depleted AEVs to wait forever.
                return_index[vehicle_id] = max(sumcapa, 0)
        return return_index

    def _dropoff_passenger(self, vehicle_id):
        """Vehicle drops off passenger at destination"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成dropoff
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成dropoff - 乘客滞留")
            # 乘客滞留在车上，订单未完成
            if vehicle['passenger_onboard'] is not None:
                request_id = vehicle['passenger_onboard']
                vehicle['passenger_onboard'] = None
                vehicle['assigned_request'] = None
                vehicle['target_location'] = None
                print(f"   乘客 {request_id} 因车辆电池耗尽而滞留在车上")
                # 保持passenger_onboard状态，等待车辆充电后继续
            return self.unserve_penalty
            
        if vehicle['passenger_onboard'] is not None:
            # Check if the passenger request still exists
            if vehicle['passenger_onboard'] not in self.active_requests:
                # Request has expired or been removed, clear the passenger
                vehicle['passenger_onboard'] = None
                vehicle['assigned_request'] = None
                return 0
                
            request = self.active_requests[vehicle['passenger_onboard']]
            vehicle_coords = vehicle['coordinates']
            dropoff_coords = (request.dropoff % self.grid_size, request.dropoff // self.grid_size)
            
            # Check if vehicle is at dropoff location
            if vehicle_coords == dropoff_coords:
                # Complete the request
                completed_request_id = int(vehicle['passenger_onboard'])
                completed_request = self.active_requests.pop(completed_request_id)
                completed_request.completed_time = self.current_time
                completed_request.completed_by_vehicle_type = self.vehicles[vehicle_id]['type']
                vehicle['whether_finishrequest'] = True
                self.completed_requests.append(completed_request)
                service_time = completed_request.completed_time - getattr(completed_request, 'created_time', self.current_time)
                if self.vehicles[vehicle_id]['type']==1:
                    self.completed_requests_ev.append(completed_request)
                    self.request_value_sum_ev += completed_request.final_value
                    self.completed_service_times_ev.append(service_time)
                    if completed_request.request_id in self.ev_rejected_request_ids:
                        self.ev_rejected_completed_by_ev_ids.add(completed_request.request_id)
                else:
                    self.request_value_sum_aev += completed_request.final_value
                    self.completed_service_times_aev.append(service_time)
                    if completed_request.request_id in self.ev_rejected_request_ids:
                        self.ev_rejected_rescued_by_aev_ids.add(completed_request.request_id)
                self.request_value_sum += completed_request.final_value
                self._ensure_recourse_runtime()
                self.request_lifecycle.record_completion(
                    completed_request_id,
                    epoch_id=self._epoch_id(),
                    vehicle_id=int(vehicle_id),
                    vehicle_type=int(vehicle.get('type', 1)),
                )
                
                earnings = completed_request.final_value
                vehicle['service_earnings'] += earnings
                vehicle['daily_salary'] += earnings
                vehicle['salary_ratio'] = vehicle['daily_salary'] / self.ev_basesalary if self.ev_basesalary > 0 else 0
                vehicle['target_location'] = None
                vehicle['idle_target'] = None
                vehicle['assigned_request'] = None
                # if self.current_time % 50 == 0:
                #     print(f"✅🎉💰 Vehicle {vehicle_id} dropped off passenger (request {vehicle['passenger_onboard']}) at {vehicle_coords}, earned ${earnings:.2f} 🚖💸✨")
                vehicle['passenger_onboard'] = None

                # EV idle time starts after completion
                self._record_ev_completion(vehicle_id)
                if not self.daily_drop_off:
                    self._handle_vehicle_dropout_event(vehicle_id)


                return earnings
        return 0
    
    def get_initial_states(self, num_agents=None, is_training=True):
        """Get initial states - implementing abstract method"""
        if num_agents is None:
            num_agents = self.num_vehicles
        
        states = {}
        for vehicle_id in range(num_agents):
            if vehicle_id in self.vehicles:
                states[vehicle_id] = self._get_vehicle_state(vehicle_id)
            else:
                # Create default state for additional agents
                states[vehicle_id] = np.array([0.5, 0.5, 0.5, 0, 0])
        return states
    
    def initialise_environment(self):
        """Initialize environment - implementing abstract method"""
        self.current_time = 0
        self._setup_vehicles()
        self.zone_vehicle_num = [0 for _ in range(self.num_zones)]
        self.zone_request_num = [0 for _ in range(self.num_zones)]
        for i in range(self.num_zones):
            zone_centerx = (self.grid_size // (int(np.sqrt(self.num_zones))))//2 + (i % int(np.sqrt(self.num_zones))) * (self.grid_size // (int(np.sqrt(self.num_zones))))
            zone_centery = (self.grid_size // (int(np.sqrt(self.num_zones))))//2 + (i // int(np.sqrt(self.num_zones))) * (self.grid_size // (int(np.sqrt(self.num_zones))))
            self.hotspot_locations.append((zone_centerx, zone_centery))

        for vehicle_id, vehicle in self.vehicles.items():
            vehicle['zone_id'] = self.get_zone_id(vehicle['location']) if hasattr(self, 'get_zone_id') else None
            if vehicle['zone_id'] is not None:
                self.zone_vehicle_num[vehicle['zone_id']] += 1
    
        if hasattr(self, 'active_requests'):
            for request_id, request in self.active_requests.items():
                request_zone_id = self.get_zone_id(request.source) if hasattr(self, 'get_zone_id') else None
                if request_zone_id is not None:
                    self.zone_request_num[request_zone_id] += 1





    def get_request_batch(self):
        """Get request batch - implementing abstract method"""
        # Return both passenger requests and charging needs
        requests = []
        
        # Add passenger requests
        for request_id, request in self.active_requests.items():
            requests.append(request)
        
        # Add charging requests for low battery vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery'] < 0.005:  # Low battery vehicles need charging
                charging_request = Request(
                    request_id=f"charge_{vehicle_id}",
                    source=vehicle['location'],
                    destination=vehicle['location'],  # Stay at same location for charging
                    current_time=self.current_time,
                    travel_time=0,
                    value=0.5  # Small value for charging necessity
                )
                requests.append(charging_request)
        
        return requests

    def generate_requests(self):
        """Public wrapper for request generation.

        Some workflows expect `env.generate_requests()`; the internal implementation
        uses `_generate_intense_requests()` / `_generate_random_requests()`.
        """
        if getattr(self, 'use_intense_requests', False):
            return self._generate_intense_requests()
        return self._generate_random_requests()

    def generate_fixed_requests(self, fixed_num_requests):
        """Generate a fixed number of requests, independent of vehicle count.

        Args:
            fixed_num_requests: exact number of requests to generate this call.

        Returns:
            list[Request]: generated requests.
        """
        generated_requests = []
        self._set_request_generation_random_state()

        hotspots = self.hotspot_locations if getattr(self, 'use_intense_requests', False) else None
        probability_weights = [0.25, 0.25, 0.25, 0.25]
        selected_hotspot_idx_reward = [30, 15, 50, 15]

        for request_idx in range(fixed_num_requests):
            self.request_counter += 1

            request_seed = (self.request_generation_seed if hasattr(self, 'request_generation_seed')
                            and self.request_generation_seed is not None else 12345) + \
                           self.current_time * 1000 + self.request_counter * 10 + request_idx
            random.seed(request_seed)
            np.random.seed(int(request_seed) % (2**31))

            if hotspots is not None:
                # intense mode: pickup near a hotspot
                rand_val = random.random()
                cumulative_prob = 0
                selected_hotspot_idx = 0
                for i, weight in enumerate(probability_weights):
                    cumulative_prob += weight
                    if rand_val <= cumulative_prob:
                        selected_hotspot_idx = i
                        break
                hotspot_center = hotspots[selected_hotspot_idx]
                hotspot_radius = max(2, self.grid_size // 8)
                pickup_x = max(0, min(self.grid_size - 1,
                                      hotspot_center[0] + random.randint(-hotspot_radius, hotspot_radius)))
                pickup_y = max(0, min(self.grid_size - 1,
                                      hotspot_center[1] + random.randint(-hotspot_radius, hotspot_radius)))
            else:
                pickup_x = random.randint(0, self.grid_size - 1)
                pickup_y = random.randint(0, self.grid_size - 1)

            pickup_location = pickup_y * self.grid_size + pickup_x

            dropoff_x = random.randint(0, self.grid_size - 1)
            dropoff_y = random.randint(0, self.grid_size - 1)
            dropoff_location = dropoff_y * self.grid_size + dropoff_x

            attempts = 0
            while dropoff_location == pickup_location and attempts < 5:
                random.seed(request_seed + attempts)
                np.random.seed(int(request_seed + attempts) % (2**31))
                dropoff_x = random.randint(0, self.grid_size - 1)
                dropoff_y = random.randint(0, self.grid_size - 1)
                dropoff_location = dropoff_y * self.grid_size + dropoff_x
                attempts += 1

            travel_time = max(abs(pickup_x - dropoff_x), abs(pickup_y - dropoff_y))

            base_value = 10 if hotspots is not None else 25
            distance_value = travel_time * (1 + np.random.rand() * 0.1)
            surge_factor = 1.0 + (fixed_num_requests - 1) * 0.0001
            point_loc = pickup_y * self.grid_size + pickup_x
            zone_loc = self.loc_to_zone.get(point_loc, None)
            if zone_loc == 1:
                distance_value += 5
            elif zone_loc == 2:
                distance_value -= 2
            elif zone_loc == 3:
                distance_value -= 2
            if hotspots is not None:
                final_value = base_value * surge_factor + distance_value + selected_hotspot_idx_reward[selected_hotspot_idx]
            else:
                final_value = base_value * surge_factor + distance_value

            request = Request(
                request_id=self.request_counter,
                source=pickup_location,
                destination=dropoff_location,
                current_time=self.current_time,
                travel_time=travel_time,
                value=base_value,
                final_value=final_value
            )

            self.active_requests[self.request_counter] = request
            generated_requests.append(request)

        self._save_request_generation_random_state()
        return generated_requests

    def get_travel_time(self, source, destination):
        """Get travel time between locations - implementing abstract method"""
        if isinstance(source, tuple) and isinstance(destination, tuple):
            # Manhattan distance as travel time
            return abs(source[0] - destination[0]) + abs(source[1] - destination[1])
        else:
            # Default travel time
            return 1.0
    
    def get_next_location(self, source, destination):
        """Get next location towards destination - implementing abstract method"""
        if isinstance(source, tuple) and isinstance(destination, tuple):
            x_diff = destination[0] - source[0]
            y_diff = destination[1] - source[1]
            
            # Move one step towards destination
            next_x = source[0]
            next_y = source[1]
            
            if x_diff > 0:
                next_x += 1
            elif x_diff < 0:
                next_x -= 1
            elif y_diff > 0:
                next_y += 1
            elif y_diff < 0:
                next_y -= 1
                
            return (next_x, next_y)
        else:
            return source
    
    def _get_vehicle_state(self, vehicle_id):
        """Get vehicle state vector"""
        vehicle = self.vehicles[vehicle_id]
        coords = vehicle['coordinates']
        state = [
            coords[0] / self.grid_size,  # Normalized x coordinate
            coords[1] / self.grid_size,  # Normalized y coordinate
            vehicle['battery'],                        # Battery level
            float(vehicle.get('is_online', True) and vehicle['charging_station'] is not None),  # Whether charging
            self.current_time / self.episode_length    # Time progress
        ]
        return np.array(state)
    
    def step(self, actions, storeactions,storeactions_ev = None):
        """执行一步环境交互"""
        rewards = {}
        dur_rewards = {}
        next_states = {}
        charging_events = []
        
        # Initialize step counters
        self.step_assignments = 0
        self.step_rejections = 0


        # 处理每个车辆的动作
        for vehicle_id, action in actions.items():
            reward,dur_reward = self._execute_action(vehicle_id, action)
            rewards[vehicle_id] = reward
            dur_rewards[vehicle_id] = dur_reward
            next_states[vehicle_id] = self._get_vehicle_state(vehicle_id)
            
            # Record charging events
            if isinstance(action, ChargingAction):
                charging_events.append({
                    'vehicle_id': vehicle_id,
                    'station_id': action.charging_station_id,
                    'duration': action.charging_duration,
                    'time': self.current_time
                })

        # 更新环境状态
        self._update_environment()
        batterypenaltyv = self._check_dead_battery_vehicles()
        done = self.current_time >= self.episode_length
        self.done = done
        self._finalize_joint_collection(rewards, done)


        saved_prior = self._prior_features_for_posterior
        saved_zone_target = getattr(self, '_prior_zone_dist_target', None)
        saved_external_prior = self._bayes_external_prior
        saved_external_posterior = self._bayes_external_posterior
        saved_bayes_role = self._bayes_context_role
        saved_skip_bayes_distribution_training = getattr(self, '_skip_bayes_distribution_training', False)
        leader_is_ev = getattr(self, '_leader_is_ev', None)
        if leader_is_ev is not None:
            # AEV update (storeactions, ifev=False): follower in evfirst, leader in aevfirst.
            if leader_is_ev:
                self._prior_features_for_posterior = self._prior_features_for_follower
                self._prior_zone_dist_target = self._prior_zone_dist_target_for_follower
                self._bayes_external_prior = saved_external_prior
                self._bayes_external_posterior = saved_external_posterior
                self._bayes_context_role = 'follower'
                self._skip_bayes_distribution_training = False
            else:
                self._prior_features_for_posterior = self._prev_follower_prior_features_for_leader
                self._prior_zone_dist_target = self._prev_follower_zone_dist_target_for_leader
                self._bayes_external_prior = self._prev_follower_external_prior_for_leader
                self._bayes_external_posterior = self._prev_follower_external_posterior_for_leader
                self._bayes_context_role = 'leader'
                self._skip_bayes_distribution_training = False
            self._update_q_learning(storeactions, False)
            # EV update (storeactions_ev, ifev=True): leader in evfirst, follower in aevfirst.
            if leader_is_ev:
                self._prior_features_for_posterior = self._prev_follower_prior_features_for_leader
                self._prior_zone_dist_target = self._prev_follower_zone_dist_target_for_leader
                self._bayes_external_prior = self._prev_follower_external_prior_for_leader
                self._bayes_external_posterior = self._prev_follower_external_posterior_for_leader
                self._bayes_context_role = 'leader'
                self._skip_bayes_distribution_training = False
            else:
                self._prior_features_for_posterior = self._prior_features_for_follower
                self._prior_zone_dist_target = self._prior_zone_dist_target_for_follower
                self._bayes_external_prior = saved_external_prior
                self._bayes_external_posterior = saved_external_posterior
                self._bayes_context_role = 'follower'
                self._skip_bayes_distribution_training = True
            self._update_q_learning(storeactions_ev, True)
            self._prior_features_for_posterior = saved_prior
            self._prior_zone_dist_target = saved_zone_target
            self._bayes_external_prior = saved_external_prior
            self._bayes_external_posterior = saved_external_posterior
            self._bayes_context_role = saved_bayes_role
            self._skip_bayes_distribution_training = saved_skip_bayes_distribution_training
        else:
            # integrated mode or unknown: no sequential encoding, keep original behavior
            self._bayes_context_role = None
            self._bayes_external_prior = None
            self._bayes_external_posterior = None
            self._skip_bayes_distribution_training = False
            self._update_q_learning(storeactions, False)
            self._update_q_learning(storeactions_ev, True)
        # Record charging station usage for this time step
        self._record_charging_usage()

        # 检查是否结束
        ev_idle_mean = float(np.mean(self.ev_idle_durations)) if getattr(self, 'ev_idle_durations', None) else 0.0
        ev_idle_count = int(len(self.ev_idle_durations)) if getattr(self, 'ev_idle_durations', None) else 0
        ev_in_penalty = int(sum(1 for vid in self.vehicles.keys() if self._in_ev_penalty(vid)))
        return next_states, rewards, dur_rewards, done, {
            'charging_events': charging_events,
            'ev_idle_mean': ev_idle_mean,
            'ev_idle_count': ev_idle_count,
            'ev_in_penalty': ev_in_penalty,
        }
    
    def _record_charging_usage(self):
        """Record charging station usage for current time step"""
        if hasattr(self, 'charging_manager') and self.charging_manager.stations:
            total_occupied = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            total_stations = len(self.charging_manager.stations)
            pressure_details = {
                station_id: (
                    len(station.current_vehicles)
                    + len(getattr(station, 'charging_queue', []) or [])
                    + len(getattr(station, 'charging_queue_notarrived', []) or [])
                )
                for station_id, station in self.charging_manager.stations.items()
            }
            pressure_values = list(pressure_details.values())
            
            usage_stats = {
                'time': self.current_time,
                'total_occupied': total_occupied,
                'total_stations': total_stations,
                'vehicles_per_station': total_occupied / max(1, total_stations),
                'mean_station_pressure': float(np.mean(pressure_values)) if pressure_values else 0.0,
                'max_station_pressure': float(max(pressure_values)) if pressure_values else 0.0,
                'station_pressure_details': pressure_details,
                'station_details': {
                    station_id: len(station.current_vehicles) 
                    for station_id, station in self.charging_manager.stations.items()
                }
            }
            self.charging_usage_history.append(usage_stats)

    def _check_dead_battery_vehicles(self):
        """检查电池耗尽的车辆"""
        dead_battery_vehicles = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery'] <= 0.0:
                vehicle['target_location'] = None
                vehicle['idle_target'] = None
                vehicle['assigned_request'] = None
                vehicle['passenger_onboard'] = None
                vehicle['charging_target'] = None
                if vehicle['charging_station'] is None:
                    dead_battery_vehicles.append(vehicle_id)
        return dead_battery_vehicles

    def generate_vehicle_requests(self,vehicle_ids):
        
        assigned_requests = []
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id]['assigned_request'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
            if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
        available_requests = list(self.active_requests.values())
        blocked_requests = set(
            getattr(self, '_same_epoch_blocked_request_ids', set())
        )
        available_requests = [
            req
            for req in available_requests
            if req.request_id not in assigned_requests
            and req.request_id not in blocked_requests
        ]
        # Persist the exact request-column order for every downstream solver.
        # The heuristic path receives the same matrix as Gurobi and must be
        # able to map its columns back to request/station identifiers.
        self._last_matrix_request_ids = [
            int(request.request_id) for request in available_requests
        ]
        
        
        request_num = len(available_requests)
        vehicle_request = np.zeros((len(vehicle_ids),request_num))
        
        use_range = getattr(self, 'use_range_requests', False)
        asign_allrequest = getattr(self, 'asign_allrequest', False)
        if asign_allrequest:
            for i,vid in enumerate(vehicle_ids):
                if self.vehicles[vid].get('penalty_timer', 0) > 0:
                    continue
                veh_loc = self.vehicles[vid]['location']
                veh_battery = self.vehicles[vid]['battery']
                for j, request in enumerate(available_requests):
                    pickuploc = request.pickup
                    dropoffloc = request.dropoff
                    pickup_dist = self._manhattan_distance_loc(veh_loc, pickuploc)
                    total_distance = pickup_dist + self._manhattan_distance_loc(pickuploc, dropoffloc)
                    feasible = veh_battery - total_distance * self.battery_consum >= self.min_battery_level
                    if feasible:
                        vehicle_request[i][j] = 1
        else:
            if use_range:
                # 基于范围的请求筛选 (find_range_requests 方式)
                range_radius = getattr(self, 'assignmentrange', 5)
                for i, vid in enumerate(vehicle_ids):
                    if self.vehicles[vid].get('penalty_timer', 0) > 0:
                        continue
                    veh_loc = self.vehicles[vid]['location']
                    veh_battery = self.vehicles[vid]['battery']
                    for j, request in enumerate(available_requests):
                        pickuploc = request.pickup
                        dropoffloc = request.dropoff
                        pickup_dist = self._manhattan_distance_loc(veh_loc, pickuploc)
                        if pickup_dist <= range_radius:
                            total_distance = pickup_dist + self._manhattan_distance_loc(pickuploc, dropoffloc)
                            feasible = veh_battery - total_distance * self.battery_consum >= self.min_battery_level
                            if feasible:
                                vehicle_request[i][j] = 1
            else:
                # 原始方式：按距离排序，只保留最近的K个订单
                near_k = getattr(self, 'assignmentnear', request_num)
                for i,vid in enumerate(vehicle_ids):
                    # penalty 期间的车辆不允许接订单，整行保持0
                    if self.vehicles[vid].get('penalty_timer', 0) > 0:
                        continue
                    # 使用 location (整数) 而不是 coordinates (元组)
                    veh_loc = self.vehicles[vid]['location']
                    veh_battery = self.vehicles[vid]['battery']
                    # 计算每个请求的接客距离用于排序
                    pickup_dists = []
                    for j,request in enumerate(available_requests):
                        pickuploc = request.pickup
                        dropoffloc = request.dropoff
                        pickup_dist = self._manhattan_distance_loc(veh_loc, pickuploc)
                        total_distance = pickup_dist + self._manhattan_distance_loc(pickuploc, dropoffloc)
                        feasible = 1 if veh_battery - total_distance * self.battery_consum >= self.min_battery_level else 0
                        pickup_dists.append((j, pickup_dist, feasible))
                    # 按接客距离排序，只保留最近的near_k个可行请求
                    pickup_dists.sort(key=lambda x: x[1])
                    kept = 0
                    for j_idx, _, feasible in pickup_dists:
                        if feasible and kept < near_k:
                            vehicle_request[i][j_idx] = 1
                            kept += 1
                        else:
                            vehicle_request[i][j_idx] = 0
        return vehicle_request
        
    def generate_vehicle_zone(self, vehicle_ids, distance_threshold=None):
        """
        计算车辆到各个 hotspot(zone) 的可达性矩阵
        
        Args:
            vehicle_ids: 车辆ID列表
            distance_threshold: 距离阈值，超过此距离则认为不可达。如果为None，使用grid_size的1/4作为默认值
            
        Returns:
            vehicle_zone: shape (len(vehicle_ids), num_zones) 的矩阵
                         值为1表示车辆可以到达该zone，0表示超过距离范围
        """
        # 获取 hotspot 位置（zones）
        hotspot_locations = self.hotspot_locations[:self.hotspot_locations_num]
        zone_num = len(hotspot_locations)
        
        # 如果没有指定距离阈值，使用grid_size的1/4作为默认值
        if distance_threshold is None:
            distance_threshold = self.grid_size // 4
        
        # 初始化车辆-zone矩阵
        vehicle_zone = np.zeros((len(vehicle_ids), zone_num))
        
        for i, vid in enumerate(vehicle_ids):
            veh_loc = self.vehicles[vid]['location']
            veh_battery = self.vehicles[vid]['battery']
            # 将车辆位置转换为坐标
            veh_x, veh_y = self._loc_to_xy(veh_loc)
            
            for j, zone_loc in enumerate(hotspot_locations):
                # zone_loc 是元组 (x, y)
                if isinstance(zone_loc, tuple):
                    zone_x, zone_y = zone_loc
                else:
                    # 如果是整数，转换为坐标
                    zone_x, zone_y = self._loc_to_xy(zone_loc)
                
                # 计算曼哈顿距离
                distance = abs(veh_x - zone_x) + abs(veh_y - zone_y)
                
                # 检查距离是否在阈值内，且电池足够
                batery_loss = distance * self.battery_consum
                if veh_battery - batery_loss >= self.min_battery_level:
                    vehicle_zone[i][j] = 1
                else:
                    vehicle_zone[i][j] = 0

        demand_predictor = getattr(self, 'value_function', None)
        if (
            self.synthetic_demand_profile == "predictive"
            and demand_predictor is not None
            and getattr(demand_predictor, 'post_demand_predictor_trained', False)
            and hasattr(demand_predictor, 'predict_post_action_demand')
        ):
            zone_locations = []
            for zone in hotspot_locations:
                if isinstance(zone, tuple):
                    zone_x, zone_y = zone
                    zone_locations.append(int(zone_y * self.grid_size + zone_x))
                else:
                    zone_locations.append(int(zone))
            for row, vehicle_id in enumerate(vehicle_ids):
                feasible_indices = np.flatnonzero(vehicle_zone[row] > 0)
                if feasible_indices.size <= 1:
                    continue
                vehicle_location = int(self.vehicles[vehicle_id]['location'])
                candidate_locations = [zone_locations[index] for index in feasible_indices]
                travel_durations = [
                    float(self._manhattan_distance_loc(vehicle_location, location))
                    for location in candidate_locations
                ]
                predicted_demand = np.asarray(
                    demand_predictor.predict_post_action_demand(
                        current_times=[float(self.current_time)] * len(candidate_locations),
                        post_action_durations=travel_durations,
                        post_action_locations=candidate_locations,
                        num_requests=[float(len(self.active_requests))] * len(candidate_locations),
                    ),
                    dtype=float,
                )
                if predicted_demand.size != feasible_indices.size or not np.all(
                    np.isfinite(predicted_demand)
                ):
                    continue
                demand_floor = (
                    float(np.max(predicted_demand)) - self.demand_forecast_filter_margin
                )
                removed = feasible_indices[predicted_demand < demand_floor]
                vehicle_zone[row, removed] = 0
                self.demand_forecast_filtered_actions += int(removed.size)
        
        return vehicle_zone
            
            
    def generate_vehicle_chargerange(self, vehicle_ids):
        """
        计算车辆到充电站的可达性矩阵
        
        Returns:
            vehicle_chargerange: shape (len(vehicle_ids), num_stations)
                                值为1表示车辆可以到达该充电站，0表示不可达或已满
        """
        chargenum = len(self.charging_manager.stations)
        vehicle_chargerange = np.zeros((len(vehicle_ids), chargenum))
        near_k = getattr(self, 'chargeassignnum', chargenum)  # 只保留最近的K个充电站
        
        # 建立充电站ID到矩阵列索引的映射
        station_id_to_idx = {station.id: idx for idx, station in enumerate(self.charging_manager.stations.values())}
        stations_list = list(self.charging_manager.stations.values())
        
        for i, vehicle_id in enumerate(vehicle_ids):
            veh_battery = self.vehicles[vehicle_id]['battery']
            veh_loc = self.vehicles[vehicle_id]['location']
            is_ev_type1 = (self.vehicles[vehicle_id]['type'] == 1)
            
            # 计算到每个充电站的距离和可行性
            station_dists = []
            for station in stations_list:
                col_idx = station_id_to_idx[station.id]
                total_reserved = (
                    len(getattr(station, 'current_vehicles', []) or [])
                    + len(getattr(station, 'charging_queue', []) or [])
                    + len(getattr(station, 'charging_queue_notarrived', []) or [])
                )
                admission_limit = (
                    int(station.max_capacity)
                    + int(getattr(self, 'station_queue_capacity', 0))
                )
                if (
                    is_ev_type1
                    or veh_battery > self.proactive_charging_max_battery
                    or total_reserved >= admission_limit
                ):
                    station_dists.append((col_idx, float('inf'), 0))
                else:
                    distance = self._manhattan_distance_loc(veh_loc, station.location)
                    batteryloss = distance * self.battery_consum
                    battery_add = self.chargeincrease_whole
                    if veh_battery - batteryloss + battery_add >= self.min_battery_level and veh_battery - batteryloss >= 0:
                        station_dists.append((col_idx, distance, 1))
                    else:
                        station_dists.append((col_idx, distance, 0))
            
            # 按距离排序，只保留最近的near_k个可行充电站
            station_dists.sort(key=lambda x: x[1])
            kept = 0
            for col_idx, _, feasible in station_dists:
                if feasible and kept < near_k:
                    vehicle_chargerange[i][col_idx] = 1
                    kept += 1
                else:
                    vehicle_chargerange[i][col_idx] = 0

        queue_predictor = getattr(self, 'value_function', None)
        if (
            self.synthetic_demand_profile == "predictive"
            and getattr(self, 'use_queue_forecast_action_filter', False)
            and queue_predictor is not None
            and getattr(queue_predictor, 'zone_distribution_mode', None) not in {
                'st_masac_gat_former2_queue_feature',
                'st_masac_gat_former2_queue_feature_greedy_alpha',
                'st_masac_gat_former2_queue_feature_fixed_alpha',
            }
            and getattr(queue_predictor, 'queue_predictor_trained', False)
            and hasattr(queue_predictor, 'predict_queue_waits')
        ):
            for row, vehicle_id in enumerate(vehicle_ids):
                feasible_indices = np.flatnonzero(vehicle_chargerange[row] > 0)
                if feasible_indices.size == 0:
                    continue
                reservation_pressure = np.asarray([
                    len(getattr(stations_list[index], 'current_vehicles', []) or [])
                    + len(getattr(stations_list[index], 'charging_queue', []) or [])
                    + len(getattr(stations_list[index], 'charging_queue_notarrived', []) or [])
                    for index in feasible_indices
                ], dtype=float)
                reservation_capacity = np.asarray([
                    max(
                        1,
                        int(math.ceil(
                            float(getattr(stations_list[index], 'max_capacity', 1) or 1)
                            * self.queue_forecast_aev_capacity_share
                        )),
                    )
                    for index in feasible_indices
                ], dtype=float)
                reservation_keep = reservation_pressure < reservation_capacity
                if not np.any(reservation_keep):
                    if (
                        float(self.vehicles[vehicle_id]['battery'])
                        <= self.critical_charging_battery
                    ):
                        reservation_keep[int(np.argmin(reservation_pressure))] = True
                    else:
                        vehicle_chargerange[row, feasible_indices] = 0
                        removed_count = int(feasible_indices.size)
                        self.queue_forecast_filtered_actions += removed_count
                        self.queue_forecast_reservation_filtered_actions += removed_count
                        self.queue_forecast_deferred_charges += 1
                        continue
                reservation_removed = feasible_indices[~reservation_keep]
                vehicle_chargerange[row, reservation_removed] = 0
                removed_count = int(reservation_removed.size)
                self.queue_forecast_filtered_actions += removed_count
                self.queue_forecast_reservation_filtered_actions += removed_count
                feasible_indices = feasible_indices[reservation_keep]
                vehicle_location = int(self.vehicles[vehicle_id]['location'])
                candidate_stations = [stations_list[index] for index in feasible_indices]
                travel_durations = [
                    float(self._manhattan_distance_loc(vehicle_location, station.location))
                    for station in candidate_stations
                ]
                predicted_waits = np.asarray(
                    queue_predictor.predict_queue_waits(
                        station_ids=[station.id for station in candidate_stations],
                        target_locations=[station.location for station in candidate_stations],
                        vehicle_ids=[vehicle_id] * len(candidate_stations),
                        vehicle_locations=[vehicle_location] * len(candidate_stations),
                        current_times=[float(self.current_time)] * len(candidate_stations),
                        num_requests=[float(len(self.active_requests))] * len(candidate_stations),
                        travel_durations=travel_durations,
                    ),
                    dtype=float,
                )
                if predicted_waits.size != feasible_indices.size or not np.all(
                    np.isfinite(predicted_waits)
                ):
                    continue
                min_predicted_wait = float(np.min(predicted_waits))
                vehicle_battery = float(self.vehicles[vehicle_id]['battery'])
                if (
                    vehicle_battery > self.critical_charging_battery
                    and min_predicted_wait > self.queue_forecast_optional_wait_limit
                ):
                    vehicle_chargerange[row, feasible_indices] = 0
                    self.queue_forecast_filtered_actions += int(feasible_indices.size)
                    self.queue_forecast_deferred_charges += 1
                    continue

                wait_ceiling = min_predicted_wait + self.queue_forecast_filter_margin
                keep = predicted_waits <= wait_ceiling
                if not np.any(keep):
                    keep[int(np.argmin(predicted_waits))] = True
                removed = feasible_indices[~keep]
                vehicle_chargerange[row, removed] = 0
                self.queue_forecast_filtered_actions += int(removed.size)

            for column, station in enumerate(stations_list):
                eligible_rows = np.flatnonzero(vehicle_chargerange[:, column] > 0)
                if eligible_rows.size == 0:
                    continue
                station_pressure = (
                    len(getattr(station, 'current_vehicles', []) or [])
                    + len(getattr(station, 'charging_queue', []) or [])
                    + len(getattr(station, 'charging_queue_notarrived', []) or [])
                )
                aev_slot_limit = max(
                    1,
                    int(math.ceil(
                        float(getattr(station, 'max_capacity', 1) or 1)
                        * self.queue_forecast_aev_capacity_share
                    )),
                )
                candidate_limit = max(0, aev_slot_limit - station_pressure)
                ranked_rows = sorted(
                    (int(row) for row in eligible_rows),
                    key=lambda row: (
                        float(self.vehicles[vehicle_ids[row]]['battery'])
                        > self.critical_charging_battery,
                        self._manhattan_distance_loc(
                            int(self.vehicles[vehicle_ids[row]]['location']),
                            int(station.location),
                        ),
                        float(self.vehicles[vehicle_ids[row]]['battery']),
                    ),
                )
                if candidate_limit <= 0:
                    ranked_rows = [
                        row for row in ranked_rows
                        if float(self.vehicles[vehicle_ids[row]]['battery'])
                        <= self.critical_charging_battery
                    ][:1]
                else:
                    ranked_rows = ranked_rows[:candidate_limit]
                keep_rows = set(ranked_rows)
                removed_rows = [
                    int(row) for row in eligible_rows if int(row) not in keep_rows
                ]
                if removed_rows:
                    vehicle_chargerange[removed_rows, column] = 0
                    removed_count = len(removed_rows)
                    self.queue_forecast_filtered_actions += removed_count
                    self.queue_forecast_reservation_filtered_actions += removed_count
        
        return vehicle_chargerange
                        
                        
            
    def generate_vehicle_wait(self, vehicle_ids,rebalance_num = 0):
        """
        Return the physical stationary outside action for every vehicle.

        Charging urgency belongs in the structured score, not feasibility.
        Keeping a real wait/outside edge makes the rollout flow and target
        MILP feasible under shared charging-capacity competition.
        
        Returns:
            vehicle_wait: shape (len(vehicle_ids), 1)
                         1 means the stationary/outside action is feasible.
        """
        del rebalance_num
        return np.ones((len(vehicle_ids), 1), dtype=np.float32)
            
    def generate_whole_matrix(self, vehicle_ids,rebalance_num = 0,onlyev = False):
        """
        生成完整的车辆-动作可行性矩阵
        
        矩阵结构：
        - 列 0 到 num_requests-1: 车辆到请求的可达性
        - 列 num_requests 到 num_requests+num_stations-1: 车辆到充电站的可达性
        - 列 num_requests+num_stations 到 num_requests+num_stations+num_zones-1: 车辆到区域的可达性
        - 列 -1: 车辆等待动作的可行性
        
        Returns:
            whole_matrix: shape (len(vehicle_ids), num_requests + num_stations + num_zones + 1)
                         值为1表示该动作可行，0表示不可行
        """
        if not onlyev:
            request_matrix = self.generate_vehicle_requests(vehicle_ids)
            chargerange_matrix = self.generate_vehicle_chargerange(vehicle_ids)
            zone_matrix = self.generate_vehicle_zone(vehicle_ids)
            wait_matrix = self.generate_vehicle_wait(vehicle_ids, rebalance_num=rebalance_num)

            self._last_matrix_num_requests = int(request_matrix.shape[1])
            self._last_matrix_num_stations = int(chargerange_matrix.shape[1])
            self._last_matrix_num_zones = int(zone_matrix.shape[1])
            self._last_matrix_charge_station_ids = [
                int(station_id)
                for station_id in self.charging_manager.stations.keys()
            ][:self._last_matrix_num_stations]
            self._last_matrix_zone_indices = list(
                range(self._last_matrix_num_zones)
            )
            self._last_matrix_zone_target_ids = [
                (
                    int(zone[1]) * self.grid_size + int(zone[0])
                    if isinstance(zone, (tuple, list, np.ndarray))
                    else int(zone)
                )
                for zone in self.hotspot_locations[:self._last_matrix_num_zones]
            ]
            
            # 拼接所有矩阵
            whole_matrix = np.concatenate((request_matrix, chargerange_matrix, zone_matrix, wait_matrix), axis=1)
            
            return whole_matrix, request_matrix.shape[1], chargerange_matrix.shape[1], zone_matrix.shape[1] 
        else:
            request_matrix = self.generate_vehicle_requests(vehicle_ids)
            wait_matrix = self.generate_vehicle_wait(vehicle_ids, rebalance_num=rebalance_num)

            self._last_matrix_num_requests = int(request_matrix.shape[1])
            self._last_matrix_num_stations = 0
            self._last_matrix_num_zones = 0
            self._last_matrix_charge_station_ids = []
            self._last_matrix_zone_indices = []
            self._last_matrix_zone_target_ids = []
            
            # 拼接请求矩阵和等待矩阵
            whole_matrix = np.concatenate((request_matrix, wait_matrix), axis=1)
            
            return whole_matrix, request_matrix.shape[1], 0, 0

    def _build_standard_masac_candidates(self, vehicle_id, max_candidates=48):
        """Return the feasible local action set used by discrete MASAC.

        This hook is called only by ``standard_masac_gat`` when a replay row
        does not already carry a next-action set.  Feasibility comes from the
        same matrix used by the MCMF optimizer, so the actor and temperature
        are trained over admissible actions rather than over invented labels.
        """
        if vehicle_id not in self.vehicles:
            return []

        vehicle = self.vehicles[vehicle_id]
        onlyev = int(vehicle.get('type', 2)) == 1
        action_matrix, num_requests, num_stations, num_zones = (
            self.generate_whole_matrix(
                [vehicle_id],
                rebalance_num=1,
                onlyev=onlyev,
            )
        )
        feasible = action_matrix[0]
        vehicle_location = int(vehicle.get('location', 0))
        idle_time = float(vehicle.get('idle_timer', 0.0) or 0.0)
        request_count = float(len(self.active_requests))

        assigned_requests = set()
        for other_vehicle in self.vehicles.values():
            assigned_request = other_vehicle.get('assigned_request')
            onboard_request = other_vehicle.get('passenger_onboard')
            if assigned_request is not None:
                assigned_requests.add(assigned_request)
            if onboard_request is not None:
                assigned_requests.add(onboard_request)
        available_requests = [
            request
            for request in self.active_requests.values()
            if request.request_id not in assigned_requests
        ]

        request_candidates = []
        for column, request in enumerate(available_requests[:num_requests]):
            if feasible[column] <= 0:
                continue
            pickup = int(request.pickup)
            dropoff = int(request.dropoff)
            pickup_distance = float(
                self._manhattan_distance_loc(vehicle_location, pickup)
            )
            trip_distance = float(self._manhattan_distance_loc(pickup, dropoff))
            total_distance = pickup_distance + trip_distance
            request_value = float(
                getattr(request, 'final_value', getattr(request, 'value', 0.0))
                or 0.0
            )
            candidate = {
                'action_type': f'assign_{request.request_id}',
                'target_location': pickup,
                'request_value': request_value,
                'target_distance': pickup_distance,
                'target_zoneid': int(self.get_zone_embedding_id(pickup)),
                'post_action_location': dropoff,
                'post_action_distance': total_distance,
                'post_action_duration': max(1.0, total_distance),
                'post_action_zoneid': int(self.get_zone_embedding_id(dropoff)),
                'vehicle_idle_time': idle_time,
                'num_requests': request_count,
            }
            request_candidates.append(
                (request_value - 0.15 * pickup_distance, candidate)
            )
        request_candidates.sort(key=lambda item: item[0], reverse=True)

        candidates = [candidate for _, candidate in request_candidates]
        offset = num_requests
        if not onlyev:
            stations = list(self.charging_manager.stations.values())
            for station_index, station in enumerate(stations[:num_stations]):
                if feasible[offset + station_index] <= 0:
                    continue
                station_location = int(station.location)
                distance = float(
                    self._manhattan_distance_loc(
                        vehicle_location,
                        station_location,
                    )
                )
                candidates.append({
                    'action_type': f'charge_{station.id}',
                    'target_station_id': int(station.id),
                    'target_location': station_location,
                    'request_value': 0.0,
                    'target_distance': distance,
                    'target_zoneid': int(
                        self.get_zone_embedding_id(station_location)
                    ),
                    'post_action_location': station_location,
                    'post_action_distance': distance,
                    'post_action_duration': (
                        max(0.0, distance) + float(self.charge_duration)
                    ),
                    'post_action_zoneid': int(
                        self.get_zone_embedding_id(station_location)
                    ),
                    'vehicle_idle_time': idle_time,
                    'num_requests': request_count,
                    'queue_features': self._charging_queue_feature_snapshot(
                        vehicle_id,
                        int(station.id),
                        current_time=self.current_time,
                    ),
                })
            offset += num_stations

            for zone_index, zone in enumerate(
                self.hotspot_locations[:num_zones]
            ):
                if feasible[offset + zone_index] <= 0:
                    continue
                if isinstance(zone, (tuple, list, np.ndarray)):
                    zone_location = int(zone[1]) * self.grid_size + int(zone[0])
                else:
                    zone_location = int(zone)
                distance = float(
                    self._manhattan_distance_loc(
                        vehicle_location,
                        zone_location,
                    )
                )
                candidates.append({
                    'action_type': 'reloc',
                    'target_location': zone_location,
                    'request_value': 0.0,
                    'target_distance': distance,
                    'target_zoneid': int(
                        self.get_zone_embedding_id(zone_location)
                    ),
                    'post_action_location': zone_location,
                    'post_action_distance': distance,
                    'post_action_duration': max(1.0, distance),
                    'post_action_zoneid': int(
                        self.get_zone_embedding_id(zone_location)
                    ),
                    'vehicle_idle_time': idle_time,
                    'num_requests': request_count,
                })

        if feasible[-1] > 0:
            if onlyev:
                outside_target = self._sample_ev_default_relocation_target(
                    vehicle_id
                )
                outside_action_type = 'reloc'
            else:
                outside_target = vehicle_location
                outside_action_type = 'idle'
            outside_distance = float(
                self._manhattan_distance_loc(
                    vehicle_location,
                    outside_target,
                )
            )
            candidates.append({
                'action_type': outside_action_type,
                'target_location': outside_target,
                'request_value': 0.0,
                'target_distance': outside_distance,
                'target_zoneid': int(
                    self.get_zone_embedding_id(outside_target)
                ),
                'post_action_location': outside_target,
                'post_action_distance': outside_distance,
                'post_action_duration': max(1.0, outside_distance),
                'post_action_zoneid': int(
                    self.get_zone_embedding_id(outside_target)
                ),
                'vehicle_idle_time': idle_time,
                'num_requests': request_count,
            })

        # Keep actor batches bounded exactly as in the NYC implementation.
        # High-value feasible requests are retained first; the non-request
        # alternatives remain represented by reserving their tail entries.
        if len(candidates) <= max_candidates:
            return candidates
        non_request_candidates = [
            candidate
            for candidate in candidates
            if not str(candidate.get('action_type', '')).startswith('assign')
        ]
        request_limit = max(0, int(max_candidates) - len(non_request_candidates))
        return candidates[:request_limit] + non_request_candidates[:max_candidates]
    
    
            
     
     
     
     
    def generate_vehicle_qvalue_withoutqnetwork(
        self, vehicles_to_rebalance, onlyev=False
    ):
        """
        Environment-aligned myopic scores used by MCMF without a learned
        value function.  Every score is in the same net-reward units as the
        executed action; charging and relocation are costs rather than
        invented positive rewards.
        """
        vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(
            vehicles_to_rebalance,
            rebalance_num=len(vehicles_to_rebalance),
            onlyev=onlyev,
        )
        
        # ==================== 计算请求的 Q 值 ====================
        batch_q_value_requests = np.zeros((len(vehicles_to_rebalance), num_requests))
        active_requests_count = len(self.active_requests)
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.active_requests.values() if hasattr(self, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0
        
        assigned_requests = []
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id]['assigned_request'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
            if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
        available_requests = list(self.active_requests.values())
        available_requests = [
            req
            for req in available_requests
            if req.request_id not in assigned_requests
            and req.request_id not in set(
                getattr(self, '_same_epoch_blocked_request_ids', set())
            )
        ]
        
        
        request_num = len(available_requests)
        for i, vehicle_id in enumerate(vehicles_to_rebalance):
            vehicle = self.vehicles[vehicle_id]
            vehicle_location = vehicle['location']
            
            for j, request in enumerate(available_requests):
                if vehicle_action_matrix[i, j] == 1:
                    request_value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    if request_value is None:
                        request_value = 0.0
                    pickup_distance = self._manhattan_distance_loc(
                        vehicle_location, request.pickup
                    )
                    trip_distance = self._manhattan_distance_loc(
                        request.pickup, request.dropoff
                    )
                    movement_cost = float(getattr(self, 'movingpenalty', -5e-3)) * (
                        pickup_distance + trip_distance
                    )
                    if vehicle['type'] == 1:
                        if (
                            self.knownreject
                            and getattr(self, 'recourse_variant', 'legacy') == 'legacy'
                        ):
                            acceptance_probability = 1.0 - (
                                self._calculate_rejection_probability(
                                    vehicle_id, request
                                )
                            )
                        else:
                            # Plain MCMF does not observe driver acceptance.
                            # The state-specific probability is exposed only
                            # by the known-rejection (MCMF-K) ablation.
                            acceptance_probability = 1.0
                    else:
                        acceptance_probability = 1.0
                    batch_q_value_requests[i, j] = acceptance_probability * (
                        request_value + movement_cost
                    )
                else:
                    batch_q_value_requests[i, j] = -100  # 不可达
        
        # ==================== 计算充电站的 Q 值 ====================
        batch_q_value_charging = np.zeros((len(vehicles_to_rebalance), num_stations))
        station_list = list(self.charging_manager.stations.values())[:num_stations]
        
        for i, vehicle_id in enumerate(vehicles_to_rebalance):
            vehicle = self.vehicles[vehicle_id]
            vehicle_location = vehicle['location']
            
            for k, station in enumerate(station_list):
                if vehicle_action_matrix[i, num_requests + k] == 1:
                    distance = self._manhattan_distance_loc(vehicle_location, station.location)
                    batch_q_value_charging[i, k] = (
                        float(getattr(self, 'movingpenalty', -5e-3)) * distance
                        - float(getattr(self, 'charging_penalty', 0.0))
                    )
                else:
                    batch_q_value_charging[i, k] = -100  # 不可达
        
        # ==================== 计算重定位区域的 Q 值 ====================
        batch_q_value_reloc = np.zeros((len(vehicles_to_rebalance), num_zones))
        for i, vehicle_id in enumerate(vehicles_to_rebalance):
            vehicle = self.vehicles[vehicle_id]
            vehicle_location = vehicle['location']
            
            for m in range(num_zones):
                if vehicle_action_matrix[i, num_requests + num_stations + m] == 1:
                    # 获取区域位置
                    zone_coords = self.hotspot_locations[m]
                    zone_location = zone_coords[1] * self.grid_size + zone_coords[0]
                    
                    distance = self._manhattan_distance_loc(vehicle_location, zone_location)
                    batch_q_value_reloc[i, m] = float(getattr(self, 'movingpenalty', -5e-3)) * max(
                        1.0, float(distance)
                    )
                else:
                    batch_q_value_reloc[i, m] = -100  # 不可达
        
        # ==================== 计算等待动作的 Q 值 ====================
        batch_q_value_wait = np.zeros((len(vehicles_to_rebalance), 1))
        for i, vehicle_id in enumerate(vehicles_to_rebalance):
            if vehicle_action_matrix[i, -1] == 1:
                batch_q_value_wait[i, 0] = float(getattr(self, 'movingpenalty', -5e-3))
            else:
                batch_q_value_wait[i, 0] = -100  # 不能等待
        
        # 拼接所有 Q 值
        batch_q_value = np.concatenate([
            batch_q_value_requests,
            batch_q_value_charging,
            batch_q_value_reloc,
            batch_q_value_wait
        ], axis=1)
        return self._round_assignment_qvalues(batch_q_value)

    def _should_force_mcmf_knownreject(self, *, onlyev: bool = False) -> bool:
        if not self.knownreject:
            return False
        if getattr(self, 'adp_value', 0.0) <= 0:
            return True
        vf = self.value_function_ev if onlyev else self.value_function
        return vf is None

    def _enforce_predictive_request_priority(
        self,
        q_values,
        action_matrix,
        num_requests,
        nonservice_columns,
    ):
        """Keep learned idle choices below feasible service without changing rewards."""
        if (
            self.synthetic_demand_profile != "predictive"
            or num_requests <= 0
            or q_values.size == 0
        ):
            return q_values

        for row in range(q_values.shape[0]):
            feasible_requests = action_matrix[row, :num_requests] > 0
            if not np.any(feasible_requests):
                continue

            best_request = float(np.max(q_values[row, :num_requests][feasible_requests]))
            idle_ceiling = best_request - self.request_priority_margin
            for column in nonservice_columns:
                if action_matrix[row, column] > 0:
                    q_values[row, column] = min(q_values[row, column], idle_ceiling)
        return q_values

    def _enforce_predictive_proactive_charge_priority(
        self,
        q_values,
        action_matrix,
        vehicle_ids,
        num_requests,
        num_stations,
    ):
        """Do not let optional early charging displace a feasible request."""
        if self.synthetic_demand_profile != "predictive" or num_requests <= 0:
            return q_values
        for row, vehicle_id in enumerate(vehicle_ids):
            if self.vehicles[vehicle_id]['battery'] <= self.critical_charging_battery:
                continue
            feasible_requests = action_matrix[row, :num_requests] > 0
            if not np.any(feasible_requests):
                continue
            best_request = float(np.max(q_values[row, :num_requests][feasible_requests]))
            charge_ceiling = best_request - self.request_priority_margin
            for column in range(num_requests, num_requests + num_stations):
                if action_matrix[row, column] > 0:
                    q_values[row, column] = min(q_values[row, column], charge_ceiling)
        return q_values

     

    def generate_vehicle_qvalue(self, vehicles_to_rebalance, onlyev=False, prior_features=None):
        """prior_features: list of dicts [{location, battery, idle_time, target_location, action_type}, ...] from leader phase."""
        
        vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance), onlyev=onlyev)
        
        # 准备全局上下文信息
        current_time = float(self.current_time) if hasattr(self, 'current_time') else 0.0
        other_vehicles = len([v for v in self.vehicles.values() 
                            if v['assigned_request'] is None 
                            and v['passenger_onboard'] is None 
                            and v['charging_station'] is None])
        num_reqs = len(self.active_requests)

        def _mixed_masac_values(value_function, rows, action_kind):
            """Adapt synthetic action rows to the NYC unified MASAC API."""
            if not rows or not hasattr(value_function, 'batch_get_mixed_q_values'):
                return None
            vehicle_ids = []
            vehicle_locations = []
            target_locations = []
            request_values = []
            target_distances = []
            post_action_locations = []
            post_action_distances = []
            post_action_durations = []
            station_ids = []
            action_ids = []
            for row in rows:
                vehicle_id = int(row['vehicle_id'])
                vehicle_location = int(row['vehicle_location'])
                request_value = float(row.get('request_value', 0.0) or 0.0)
                station_id = -1
                if action_kind == 'request':
                    target_location = int(row['target_location'])
                    target_distance = float(
                        row.get(
                            'pickup_dist',
                            self._manhattan_distance_loc(
                                vehicle_location,
                                target_location,
                            ),
                        )
                        or 0.0
                    )
                    request = self.active_requests.get(row.get('target_id'))
                    post_location = int(
                        getattr(request, 'dropoff', target_location)
                    )
                    trip_distance = float(
                        self._manhattan_distance_loc(
                            target_location,
                            post_location,
                        )
                    )
                    post_distance = target_distance + trip_distance
                    post_duration = max(1.0, post_distance)
                    action_id = 2
                elif action_kind == 'charge':
                    target_location = int(row['station_location'])
                    target_distance = float(
                        self._manhattan_distance_loc(
                            vehicle_location,
                            target_location,
                        )
                    )
                    post_location = target_location
                    post_distance = target_distance
                    post_duration = (
                        max(0.0, target_distance)
                        + float(self.charge_duration)
                    )
                    station_id = int(row.get('station_id', -1))
                    request_value = 0.0
                    action_id = 3
                elif action_kind == 'reloc':
                    target_location = int(row['target_location'])
                    target_distance = float(
                        self._manhattan_distance_loc(
                            vehicle_location,
                            target_location,
                        )
                    )
                    post_location = target_location
                    post_distance = target_distance
                    post_duration = max(1.0, target_distance)
                    request_value = 0.0
                    action_id = 1
                else:
                    target_location = vehicle_location
                    target_distance = 0.0
                    post_location = vehicle_location
                    post_distance = 0.0
                    post_duration = 1.0
                    request_value = 0.0
                    action_id = 1

                vehicle_ids.append(vehicle_id)
                vehicle_locations.append(vehicle_location)
                target_locations.append(target_location)
                request_values.append(request_value)
                target_distances.append(target_distance)
                post_action_locations.append(post_location)
                post_action_distances.append(post_distance)
                post_action_durations.append(post_duration)
                station_ids.append(station_id)
                action_ids.append(action_id)

            return value_function.batch_get_mixed_q_values(
                vehicle_ids=vehicle_ids,
                vehicle_locations=vehicle_locations,
                target_locations=target_locations,
                current_times=[float(row.get('current_time', current_time)) for row in rows],
                other_vehicles=[float(row.get('other_vehicles', other_vehicles)) for row in rows],
                num_requests=[float(row.get('num_requests', num_reqs)) for row in rows],
                battery_levels=[float(row.get('battery_level', 1.0)) for row in rows],
                request_values=request_values,
                target_distances=target_distances,
                target_zoneids=[0] * len(rows),
                vehicle_idle_times=[float(row.get('vehicle_idle_time', 0.0)) for row in rows],
                action_type_ids=action_ids,
                **({"request_ids": [row.get('target_id', -1) for row in rows]}
                   if getattr(value_function, 'supports_ev_acceptance_feature', False) else {}),
                post_action_distances=post_action_distances,
                post_action_durations=post_action_durations,
                post_action_zoneids=[0] * len(rows),
                post_action_locations=post_action_locations,
                target_station_ids=station_ids,
            )
        
        
        
        # ==================== 批量计算请求的 Q 值 ====================
        batch_q_value_requests = np.zeros((len(vehicles_to_rebalance), num_requests))
        assigned_requests = []
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id]['assigned_request'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
            if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
        available_requests = list(self.active_requests.values())
        available_requests = [
            req
            for req in available_requests
            if req.request_id not in assigned_requests
            and req.request_id not in set(
                getattr(self, '_same_epoch_blocked_request_ids', set())
            )
        ]
        
        if (self.value_function is not None or self.value_function_ev is not None) and num_requests > 0:
            batch_inputs_aev = []
            batch_inputs_ev = []
            indices_aev = []
            indices_ev = []
            for i, vehicle_id in enumerate(vehicles_to_rebalance):
                vehicle = self.vehicles[vehicle_id]
                vehicle_location = vehicle['location']
                battery_level = vehicle['battery']
                vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                if self.vehicles[vehicle_id]['type']==2:
                    for j, request_id in enumerate([req.request_id for req in available_requests]):
                        if vehicle_action_matrix[i, j] == 1:
                            request = self.active_requests[request_id]
                            request_location = request.pickup
                            request_value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            
                            # 计算pickup距离
                            pickup_dist = self._manhattan_distance_loc(vehicle_location, request_location)
                            pick_zone = self.get_zone_id(request_location) if hasattr(self, 'get_zone_id') else 0
                            
                            batch_inputs_aev.append({
                                'vehicle_id': vehicle_id,
                                'target_id': request_id,
                                'vehicle_location': vehicle_location,
                                'target_location': request_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'request_value': request_value,
                                'pickup_dist': float(pickup_dist),
                                'pick_zone': pick_zone,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices_aev.append((i, j))
                else:
                    for j, request_id in enumerate([req.request_id for req in available_requests]):
                        if vehicle_action_matrix[i, j] == 1:
                            request = self.active_requests[request_id]
                            request_location = request.pickup
                            request_value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            
                            # 计算pickup距离
                            pickup_dist = self._manhattan_distance_loc(vehicle_location, request_location)
                            pick_zone = self.get_zone_id(request_location) if hasattr(self, 'get_zone_id') else 0
                            
                            batch_inputs_ev.append({
                                'vehicle_id': vehicle_id,
                                'target_id': request_id,
                                'vehicle_location': vehicle_location,
                                'target_location': request_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'request_value': request_value,
                                'pickup_dist': float(pickup_dist),
                                'pick_zone': pick_zone,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices_ev.append((i, j))
            
            # 批量计算 Q 值（支持多GPU）
            if batch_inputs_ev:
                if self.synthetic_ev_myopic_request_q:
                    q_values = [row['request_value'] for row in batch_inputs_ev]
                    ev_q_source = "myopic_request_value"
                elif hasattr(self.value_function_ev, 'batch_get_assignment_q_value'):
                    q_values = self.value_function_ev.batch_get_assignment_q_value(
                        batch_inputs_ev,
                        multi_gpu_devices=self.multi_gpu_devices,
                    )
                    ev_q_source = "learned_batch_assignment_q"
                elif hasattr(self.value_function_ev, 'batch_get_mixed_q_values'):
                    q_values = _mixed_masac_values(
                        self.value_function_ev,
                        batch_inputs_ev,
                        'request',
                    )
                    ev_q_source = "learned_masac_request_q"
                else:
                    q_values = [row['request_value'] for row in batch_inputs_ev]
                    ev_q_source = "myopic_fallback_no_ev_q_api"
                if not self._ev_request_q_source_reported:
                    print(
                        "EV request assignment scoring: "
                        f"{ev_q_source} (edges={len(batch_inputs_ev)})",
                        flush=True,
                    )
                    self._ev_request_q_source_reported = True
                for idx, q_value in enumerate(q_values):
                    i, j = indices_ev[idx]
                    if (
                        self.knownreject
                        and getattr(self, 'recourse_variant', 'legacy') == 'legacy'
                    ):
                        request = available_requests[j]
                        rejection_pro = self._calculate_rejection_probability(
                            vehicles_to_rebalance[i], request
                        )
                        q_value = q_value * (1 - rejection_pro)
                    batch_q_value_requests[i, j] = q_value
            if batch_inputs_aev:
                if hasattr(self.value_function, 'batch_get_assignment_q_value'):
                    q_values = self.value_function.batch_get_assignment_q_value(batch_inputs_aev, multi_gpu_devices=self.multi_gpu_devices)
                elif hasattr(self.value_function, 'batch_get_mixed_q_values'):
                    q_values = _mixed_masac_values(
                        self.value_function,
                        batch_inputs_aev,
                        'request',
                    )
                else:
                    q_values = [row['request_value'] for row in batch_inputs_aev]
                for idx, q_value in enumerate(q_values):
                    i, j = indices_aev[idx]
                    batch_q_value_requests[i, j] = q_value
            # 设置不可达请求的值
            batch_q_value_requests[vehicle_action_matrix[:, :num_requests] == 0] = -1000
        
        
        if not onlyev:
            # ==================== 批量计算充电站的 Q 值 ====================
            batch_q_value_charging = np.zeros((len(vehicles_to_rebalance), num_stations))
            if self.value_function and num_stations > 0:
                station_list = list(self.charging_manager.stations.values())
                
                # 批量收集所有可达的 (vehicle, station) 对
                batch_inputs = []
                indices = []
                
                for i, vehicle_id in enumerate(vehicles_to_rebalance):
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    battery_level = vehicle['battery']
                    vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                    
                    for k, station in enumerate(station_list):
                        if vehicle_action_matrix[i, num_requests + k] == 1:
                            station_location = station.location
                            
                            batch_inputs.append({
                                'vehicle_id': vehicle_id,
                                'station_id': station.id,
                                'vehicle_location': vehicle_location,
                                'station_location': station_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices.append((i, k))
                
                # 批量计算 Q 值
                if batch_inputs:
                    if hasattr(self.value_function, 'batch_get_charging_q_value'):
                        q_values = self.value_function.batch_get_charging_q_value(batch_inputs)
                    else:
                        q_values = _mixed_masac_values(
                            self.value_function,
                            batch_inputs,
                            'charge',
                        )
                    if q_values is None:
                        q_values = [0.0] * len(batch_inputs)
                    for idx, q_value in enumerate(q_values):
                        i, k = indices[idx]
                        batch_q_value_charging[i, k] = q_value
                
                # 设置不可达充电站的值
                batch_q_value_charging[vehicle_action_matrix[:, num_requests:num_requests+num_stations] == 0] = -1000
            
            # ==================== 批量计算重定位区域的 Q 值 ====================
            batch_q_value_reloc = np.zeros((len(vehicles_to_rebalance), num_zones))
            if self.value_function and num_zones > 0:
                # 批量收集所有可达的 (vehicle, zone) 对
                batch_inputs = []
                indices = []
                
                for i, vehicle_id in enumerate(vehicles_to_rebalance):
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    battery_level = vehicle['battery']
                    vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                    
                    for m in range(num_zones):
                        if vehicle_action_matrix[i, num_requests + num_stations + m] == 1:
                            zone_coords = self.hotspot_locations[m]
                            zone_location = zone_coords[1] * self.grid_size + zone_coords[0]
                            
                            batch_inputs.append({
                                'vehicle_id': vehicle_id,
                                'vehicle_location': vehicle_location,
                                'target_location': zone_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices.append((i, m))
                
                # 批量计算 Q 值
                if batch_inputs:
                    if hasattr(self.value_function, 'batch_get_idle_q_value'):
                        q_values = self.value_function.batch_get_idle_q_value(batch_inputs)
                    else:
                        q_values = _mixed_masac_values(
                            self.value_function,
                            batch_inputs,
                            'reloc',
                        )
                    if q_values is None:
                        q_values = [0.0] * len(batch_inputs)
                    for idx, q_value in enumerate(q_values):
                        i, m = indices[idx]
                        batch_q_value_reloc[i, m] = q_value
                
                # 设置不可达区域的值
                batch_q_value_reloc[vehicle_action_matrix[:, num_requests+num_stations:num_requests+num_stations+num_zones] == 0] = -1000
            
            # ==================== 批量计算等待动作的 Q 值 ====================
            batch_q_value_wait = np.zeros((len(vehicles_to_rebalance), 1))
            def _score_outside_rows(value_function, indices, ev_relocation):
                if value_function is None or not indices:
                    return
                rows = []
                for i in indices:
                    vehicle_id = int(vehicles_to_rebalance[i])
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = int(vehicle['location'])
                    target_location = (
                        self._sample_ev_default_relocation_target(vehicle_id)
                        if ev_relocation
                        else vehicle_location
                    )
                    rows.append({
                        'vehicle_id': vehicle_id,
                        'vehicle_location': vehicle_location,
                        'target_location': target_location,
                        'current_time': current_time,
                        'other_vehicles': other_vehicles,
                        'num_requests': num_reqs,
                        'battery_level': vehicle['battery'],
                        'vehicle_idle_time': float(vehicle.get('idle_timer', 0)),
                    })

                # MASAC uses one unified critic.  Legacy value functions keep
                # their existing wait/idle adapters as fallbacks.
                if hasattr(value_function, 'batch_get_mixed_q_values'):
                    q_values = _mixed_masac_values(
                        value_function,
                        rows,
                        'reloc' if ev_relocation else 'wait',
                    )
                elif ev_relocation and hasattr(value_function, 'batch_get_idle_q_value'):
                    q_values = value_function.batch_get_idle_q_value(rows)
                elif hasattr(value_function, 'batch_get_waiting_q_value'):
                    q_values = value_function.batch_get_waiting_q_value(rows)
                else:
                    q_values = None
                if q_values is None:
                    q_values = [0.0] * len(rows)
                for row_index, q_value in zip(indices, q_values):
                    batch_q_value_wait[row_index, 0] = q_value

            feasible_wait_rows = [
                i
                for i in range(len(vehicles_to_rebalance))
                if vehicle_action_matrix[i, -1] == 1
            ]
            ev_wait_rows = [
                i for i in feasible_wait_rows
                if self._is_ev(vehicles_to_rebalance[i])
            ]
            aev_wait_rows = [
                i for i in feasible_wait_rows
                if not self._is_ev(vehicles_to_rebalance[i])
            ]
            _score_outside_rows(self.value_function, aev_wait_rows, False)
            _score_outside_rows(self.value_function_ev, ev_wait_rows, True)
            batch_q_value_wait[vehicle_action_matrix[:, -1] == 0, 0] = -1000
            
            # 拼接所有 Q 值
            batch_q_value = np.concatenate([
                batch_q_value_requests,
                batch_q_value_charging,
                batch_q_value_reloc,
                batch_q_value_wait
            ], axis=1)
            reloc_start = num_requests + num_stations
            nonservice_columns = list(range(reloc_start, reloc_start + num_zones))
            nonservice_columns.append(batch_q_value.shape[1] - 1)
            batch_q_value = self._enforce_predictive_request_priority(
                batch_q_value,
                vehicle_action_matrix,
                num_requests,
                nonservice_columns,
            )
            batch_q_value = self._enforce_predictive_proactive_charge_priority(
                batch_q_value,
                vehicle_action_matrix,
                vehicles_to_rebalance,
                num_requests,
                num_stations,
            )
            return self._round_assignment_qvalues(batch_q_value)
        else:
            batch_q_value_wait = np.zeros((len(vehicles_to_rebalance), 1))
            ev_wait_rows = [
                i
                for i, vehicle_id in enumerate(vehicles_to_rebalance)
                if vehicle_action_matrix[i, -1] == 1
            ]
            if self.value_function_ev is not None and ev_wait_rows:
                batch_inputs_ev_wait = []
                for i in ev_wait_rows:
                    vehicle_id = int(vehicles_to_rebalance[i])
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = int(vehicle['location'])
                    batch_inputs_ev_wait.append({
                        'vehicle_id': vehicle_id,
                        'vehicle_location': vehicle_location,
                        'target_location': self._sample_ev_default_relocation_target(
                            vehicle_id
                        ),
                        'current_time': current_time,
                        'other_vehicles': other_vehicles,
                        'num_requests': num_reqs,
                        'battery_level': vehicle['battery'],
                        'vehicle_idle_time': float(vehicle.get('idle_timer', 0)),
                    })
                if hasattr(self.value_function_ev, 'batch_get_mixed_q_values'):
                    q_values = _mixed_masac_values(
                        self.value_function_ev,
                        batch_inputs_ev_wait,
                        'reloc',
                    )
                elif hasattr(self.value_function_ev, 'batch_get_idle_q_value'):
                    q_values = self.value_function_ev.batch_get_idle_q_value(
                        batch_inputs_ev_wait
                    )
                else:
                    q_values = None
                if q_values is None:
                    q_values = [0.0] * len(batch_inputs_ev_wait)
                for row_index, q_value in zip(ev_wait_rows, q_values):
                    batch_q_value_wait[row_index, 0] = q_value
            batch_q_value_wait[vehicle_action_matrix[:, -1] == 0, 0] = -1000
            batch_q_value = np.concatenate([
                batch_q_value_requests,
                batch_q_value_wait
            ], axis=1)
            batch_q_value = self._enforce_predictive_request_priority(
                batch_q_value,
                vehicle_action_matrix,
                num_requests,
                [batch_q_value.shape[1] - 1],
            )
            return self._round_assignment_qvalues(batch_q_value)
    
    def generate_vehicle_qvalue_dqn(self, vehicles_to_rebalance, onlyev=False):
        """
        为DQN生成Q值矩阵，与generate_vehicle_qvalue类似但适用于DQN决策
        返回: batch_q_value matrix [num_vehicles, num_actions]
        """
        vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(
            vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance), onlyev=onlyev
        )
        
        # 准备全局上下文信息
        current_time = float(self.current_time) if hasattr(self, 'current_time') else 0.0
        other_vehicles = len([v for v in self.vehicles.values() 
                            if v['assigned_request'] is None 
                            and v['passenger_onboard'] is None 
                            and v['charging_station'] is None])
        num_reqs = len(self.active_requests)
        
        # ==================== 批量计算请求的 Q 值 ====================
        batch_q_value_requests = np.zeros((len(vehicles_to_rebalance), num_requests))
        assigned_requests = []
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id]['assigned_request'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
            if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
        available_requests = list(self.active_requests.values())
        available_requests = [
            req
            for req in available_requests
            if req.request_id not in assigned_requests
            and req.request_id not in set(
                getattr(self, '_same_epoch_blocked_request_ids', set())
            )
        ]
        
        if self.value_function and num_requests > 0:
            batch_inputs_aev = []
            batch_inputs_ev = []
            indices_aev = []
            indices_ev = []
            for i, vehicle_id in enumerate(vehicles_to_rebalance):
                vehicle = self.vehicles[vehicle_id]
                vehicle_location = vehicle['location']
                battery_level = vehicle['battery']
                vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                
                if self.vehicles[vehicle_id]['type'] == 2:  # AEV
                    for j, request_id in enumerate([req.request_id for req in available_requests]):
                        if vehicle_action_matrix[i, j] == 1:
                            request = self.active_requests[request_id]
                            request_location = request.pickup
                            request_value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            
                            pickup_dist = self._manhattan_distance_loc(vehicle_location, request_location)
                            pick_zone = self.get_zone_id(request_location) if hasattr(self, 'get_zone_id') else 0
                            
                            batch_inputs_aev.append({
                                'vehicle_id': vehicle_id,
                                'target_id': request_id,
                                'vehicle_location': vehicle_location,
                                'target_location': request_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'request_value': request_value,
                                'pickup_dist': float(pickup_dist),
                                'pick_zone': pick_zone,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices_aev.append((i, j))
                else:  # EV
                    for j, request_id in enumerate([req.request_id for req in available_requests]):
                        if vehicle_action_matrix[i, j] == 1:
                            request = self.active_requests[request_id]
                            request_location = request.pickup
                            request_value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            
                            pickup_dist = self._manhattan_distance_loc(vehicle_location, request_location)
                            pick_zone = self.get_zone_id(request_location) if hasattr(self, 'get_zone_id') else 0
                            
                            batch_inputs_ev.append({
                                'vehicle_id': vehicle_id,
                                'target_id': request_id,
                                'vehicle_location': vehicle_location,
                                'target_location': request_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'request_value': request_value,
                                'pickup_dist': float(pickup_dist),
                                'pick_zone': pick_zone,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices_ev.append((i, j))
            
            # 批量计算 Q 值
            if batch_inputs_ev and hasattr(self.value_function_ev, 'batch_get_assignment_q_value'):
                q_values = self.value_function_ev.batch_get_assignment_q_value(batch_inputs_ev)
                for idx, q_value in enumerate(q_values):
                    i, j = indices_ev[idx]
                    batch_q_value_requests[i, j] = q_value
            if batch_inputs_aev and hasattr(self.value_function, 'batch_get_assignment_q_value'):
                q_values = self.value_function.batch_get_assignment_q_value(batch_inputs_aev)
                for idx, q_value in enumerate(q_values):
                    i, j = indices_aev[idx]
                    batch_q_value_requests[i, j] = q_value
            
            # 设置不可达请求的值
            batch_q_value_requests[vehicle_action_matrix[:, :num_requests] == 0] = -1000
        
        if not onlyev:
            # ==================== 批量计算充电站的 Q 值 ====================
            batch_q_value_charging = np.zeros((len(vehicles_to_rebalance), num_stations))
            if self.value_function and num_stations > 0:
                station_list = list(self.charging_manager.stations.values())
                batch_inputs = []
                indices = []
                
                for i, vehicle_id in enumerate(vehicles_to_rebalance):
                    vehicle = self.vehicles[vehicle_id]
                    # EV(type=1)不能被分配充电动作，只有AEV(type=2)可以
                    if vehicle['type'] == 1:
                        # EV车辆：所有充电站Q值设为-1000
                        batch_q_value_charging[i, :] = -1000
                        continue
                    
                    vehicle_location = vehicle['location']
                    battery_level = vehicle['battery']
                    vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                    
                    for k, station in enumerate(station_list):
                        if vehicle_action_matrix[i, num_requests + k] == 1:
                            station_location = station.location
                            
                            batch_inputs.append({
                                'vehicle_id': vehicle_id,
                                'station_id': station.id,
                                'vehicle_location': vehicle_location,
                                'station_location': station_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices.append((i, k))
                
                # 批量计算 Q 值
                if batch_inputs and hasattr(self.value_function, 'batch_get_charging_q_value'):
                    q_values = self.value_function.batch_get_charging_q_value(batch_inputs)
                    for idx, q_value in enumerate(q_values):
                        i, k = indices[idx]
                        batch_q_value_charging[i, k] = q_value
                
                # 设置不可达充电站的值
                batch_q_value_charging[vehicle_action_matrix[:, num_requests:num_requests+num_stations] == 0] = -1000
            
            # ==================== 批量计算重定位区域的 Q 值 ====================
            batch_q_value_reloc = np.zeros((len(vehicles_to_rebalance), num_zones))
            if self.value_function and num_zones > 0:
                batch_inputs = []
                indices = []
                
                for i, vehicle_id in enumerate(vehicles_to_rebalance):
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    battery_level = vehicle['battery']
                    vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                    
                    for m in range(num_zones):
                        if vehicle_action_matrix[i, num_requests + num_stations + m] == 1:
                            zone_coords = self.hotspot_locations[m]
                            zone_location = zone_coords[1] * self.grid_size + zone_coords[0]
                            
                            batch_inputs.append({
                                'vehicle_id': vehicle_id,
                                'vehicle_location': vehicle_location,
                                'target_location': zone_location,
                                'current_time': current_time,
                                'other_vehicles': other_vehicles,
                                'num_requests': num_reqs,
                                'battery_level': battery_level,
                                'vehicle_idle_time': vehicle_idle_time
                            })
                            indices.append((i, m))
                
                # 批量计算 Q 值
                if batch_inputs and hasattr(self.value_function, 'batch_get_idle_q_value'):
                    q_values = self.value_function.batch_get_idle_q_value(batch_inputs)
                    for idx, q_value in enumerate(q_values):
                        i, m = indices[idx]
                        batch_q_value_reloc[i, m] = q_value
                
                # 设置不可达区域的值
                batch_q_value_reloc[vehicle_action_matrix[:, num_requests+num_stations:num_requests+num_stations+num_zones] == 0] = -1000
            
            # ==================== 批量计算等待动作的 Q 值 ====================
            batch_q_value_wait = np.zeros((len(vehicles_to_rebalance), 1))
            if self.value_function:
                batch_inputs = []
                indices = []
                
                for i, vehicle_id in enumerate(vehicles_to_rebalance):
                    if vehicle_action_matrix[i, -1] == 1:
                        vehicle = self.vehicles[vehicle_id]
                        vehicle_location = vehicle['location']
                        battery_level = vehicle['battery']
                        vehicle_idle_time = float(vehicle.get('idle_timer', 0))
                        
                        batch_inputs.append({
                            'vehicle_id': vehicle_id,
                            'vehicle_location': vehicle_location,
                            'current_time': current_time,
                            'other_vehicles': other_vehicles,
                            'num_requests': num_reqs,
                            'battery_level': battery_level,
                            'vehicle_idle_time': vehicle_idle_time
                        })
                        indices.append(i)
                
                # 批量计算 Q 值
                if batch_inputs and hasattr(self.value_function, 'batch_get_waiting_q_value'):
                    q_values = self.value_function.batch_get_waiting_q_value(batch_inputs)
                    for idx, q_value in enumerate(q_values):
                        i = indices[idx]
                        if self.vehicles[vehicles_to_rebalance[i]]['type'] == 1:
                            batch_q_value_wait[i, 0] = 0
                        else:
                            batch_q_value_wait[i, 0] = q_value
                
                # 设置不能等待的值
                batch_q_value_wait[vehicle_action_matrix[:, -1] == 0, 0] = -1000
            
            # 拼接所有 Q 值
            batch_q_value = np.concatenate([
                batch_q_value_requests,
                batch_q_value_charging,
                batch_q_value_reloc,
                batch_q_value_wait
            ], axis=1)
            return (
                self._round_assignment_qvalues(batch_q_value),
                vehicle_action_matrix,
                (num_requests, num_stations, num_zones),
            )
        else:
            batch_q_value_wait = np.zeros((len(vehicles_to_rebalance), 1))
            batch_q_value = np.concatenate([
                batch_q_value_requests,
                batch_q_value_wait
            ], axis=1)
            return (
                self._round_assignment_qvalues(batch_q_value),
                vehicle_action_matrix,
                (num_requests, num_stations, num_zones),
            )
      
     
     

    def find_range_requests(self, vehicle_id, radius=5):
        """Find requests within a certain radius of the vehicle's location"""
        vehicle_location = self.vehicles[vehicle_id]['location']
        nearby_requests = []
        for request in self.active_requests.values():
            pickup_location = request.pickup
            distance = self._manhattan_distance_loc(vehicle_location, pickup_location)
            if distance <= radius:
                nearby_requests.append(request)
        return nearby_requests





     
    def zone_vehicles(self, vehicle_id):
        veh_loc = self.vehicles[vehicle_id]['location']
        zone_id = self.get_zone_id(veh_loc) if hasattr(self, 'get_zone_id') else None
        return zone_id
        

    def simulate_motion(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []

        # Integrated mode has no leader/follower split.
        self.decision_mode = "integrated"
        self._leader_is_ev = None
        RecourseTargetBuilder.validate_variant(
            getattr(self, "recourse_variant", "legacy"), "integrated"
        )
        pending_transition = self._begin_joint_collection("integrated")

        # Initialize actions dictionary for all vehicles
        actions = {}
        
        # Initialize storeactions for all vehicles to prevent KeyError
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        # For ChargingIntegratedEnvironment, we handle rebalancing differently
        # Convert our vehicle states to a format compatible with Gurobi optimization
        from src.Action import ChargingAction, ServiceAction, IdleAction

        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None and self._should_consider_ev_charging(vehicle_id):
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and ((random.random() < p_charge) or vehicle['battery'] <= 0.2):
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                else:
                    # EV declined charging: set no-charge cooldown for 5 time steps
                    vehicle['no_charge_cooldown_until'] = self.current_time + 5
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v.get('is_online', True) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_wait = [vehicle_id for vehicle_id, v in self.vehicles.items() if v['is_stationary']==True and v not in idle_vehicles_1 and v not in idle_vehicles_2 and  v['penalty_timer']==0]
            idle_vehicles_v = [vehicle_id for vehicle_id, v in self.vehicles.items() if self._is_ev(vehicle_id) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['idle_target'] is not None and v not in idle_vehicles_2 and v not in idle_vehicles_1 and v not in idle_vehicles_wait and  v['penalty_timer']==0]
            # idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_wait+idle_vehicles_v
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            vehicles_to_rebalance = [
                vehicle_id
                for vehicle_id in vehicles_to_rebalance
                if self.vehicles[vehicle_id]['assigned_request'] is None
                and self.vehicles[vehicle_id]['passenger_onboard'] is None
                and self.vehicles[vehicle_id]['charging_station'] is None
                and (
                    self.vehicles[vehicle_id]['target_location'] is None
                    or self._is_ev(vehicle_id)
                )
            ]
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                
                
                assigned_requests = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = list(self.active_requests.values())
                available_requests = [
                    req
                    for req in available_requests
                    if req.request_id not in assigned_requests
                    and req.request_id not in set(
                        getattr(self, '_same_epoch_blocked_request_ids', set())
                    )
                ]
                if self.gurobi_network or self.usemcmf:
                    vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance))
                    if self.adp_value>0 and self.value_function is not None:
                        t_qvalue_start = time.time()
                        batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance)
                        t_qvalue_end = time.time()
                        if self.record_time:
                            self.time_stats['qvalue_with_network'].append(t_qvalue_end - t_qvalue_start)
                            if hasattr(batch_q_value, 'shape'):
                                self.time_stats['qvalue_scale_1'].append(batch_q_value.shape[0])
                                self.time_stats['qvalue_scale_2'].append(batch_q_value.shape[1])
                    else:
                        t_qvalue_start = time.time()
                        batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance)
          
                force_mcmf_knownreject = self._should_force_mcmf_knownreject(onlyev=False)
                if self.usemcmf or force_mcmf_knownreject:
                    rebalancing_assignments = self.gurobi_optimizer._np_vehicle_rebalancing_network(vehicles_to_rebalance, available_requests, vehicle_action_matrix, batch_q_value, iflp=True)
                else:
                    if self.assignmentgurobi:
                        rebalancing_assignments = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network(vehicles_to_rebalance,available_requests,  vehicle_action_matrix, batch_q_value,self.gurobi_network_lp)
                    else:
                        charging_stations = self._charging_stations_accepting_arrivals()
                        if self.adp_value > 0 and self.value_function is not None:

                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance))
                            batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance)
                            #rebalancing_assignments = self.gurobi_optimizer._heuristic_assignment_fast(vehicles_to_rebalance, charging_stations)
                            rebalancing_assignments = self.gurobi_optimizer._heuristic_assignment_fastqvalue(
                                vehicles_to_rebalance,
                                charging_stations,
                                vehicle_action_matrix,
                                batch_q_value,
                            )
                        else:
                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance))

                            available_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') and self.active_requests else []
                            rebalancing_assignments = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance, available_requests, charging_stations,vehicle_action_matrix)
                               
                
                
                integrated_graph = None
                if pending_transition is not None:
                    structured_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance
                    )
                    # Matrix/solver preparation removes stale charging queue
                    # entries. Freeze that actual decision state, before any
                    # selected offer is answered, not the earlier dirty queue.
                    pending_transition.pre_state = StateSnapshotBuilder.build(self)
                    integrated_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                        self,
                        vehicles_to_rebalance,
                        vehicle_action_matrix,
                        batch_q_value,
                        structured_q_value,
                        num_requests=num_requests,
                        num_stations=num_stations,
                        num_zones=num_zones,
                        stage_id=0,
                        solver_backend=pending_transition.solver_backend,
                        state=pending_transition.pre_state,
                    )
                    selected_edges = StateSnapshotBuilder.selected_edge_ids(
                        integrated_graph, rebalancing_assignments
                    )
                    RecourseTargetBuilder.verify_feasible(
                        integrated_graph, selected_edges
                    )
                    integrated_graph = integrated_graph.with_selected(
                        selected_edges, status="selected"
                    )
                    pending_transition.ev_stage_graph = integrated_graph
                    pending_transition.ev_joint_action = JointActionSnapshot.from_graph(
                        integrated_graph
                    )

                new_assignments = 0
                re_assignments_len = len(rebalancing_assignments)
                charging_assignments = 0
                self.total_rebalancing_calls += 1
                
                # 计时：处理分配结果
                if self.usemcmf and self.usenetworkx:
                    t_process_start = time.time()
                
                if len(rebalancing_assignments) != len(vehicles_to_rebalance):
                    print(f"⚠️  Warning: Mismatch in assignments - vehicles: {len(vehicles_to_rebalance)}, assignments: {len(rebalancing_assignments)} at step {self.current_time}")
                quest_num_now = len(self.active_requests)
                #print("current_time:", self.current_time, "rebalance assignments:", rebalancing_assignments)

                
                
                for vehicle_id, target_request in rebalancing_assignments.items():

                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            
                            self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].vehicle_loc = vehicle_location
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']

                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to request {target_request.request_id} at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self.vehicles[vehicle_id]['is_stationary'] = False  
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                
                                self.vehicles[vehicle_id]['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                self.vehicles[vehicle_id]['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                self.vehicles[vehicle_id]['idle_target'] = None  # Clear idle target on new assignment
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([target_request], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                request_final_value = self.active_requests[target_request.request_id].final_value
                                actions[vehicle_id].request_value = request_final_value
                                if self.vehicles[vehicle_id]['type'] == 1:
                                    target_coords = self.active_requests[target_request.request_id].pickup
                                    if  storeactions_ev[vehicle_id] is None:
                                        storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].target_location = target_coords
                                        storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions_ev[vehicle_id].target_location = target_coords
                                    else:
                                        # 替换action - 保存旧信息
                                        storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions_ev[vehicle_id].next_target_location = target_coords
                                        storeactions_ev[vehicle_id].next_action.next_value = request_final_value
                                        storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions_ev[vehicle_id] = None
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = target_coords
                                    self.vehicles[vehicle_id]['idle_timer'] = 0  # Reset idle timer on new assignment
                                else:
                                    if storeactions[vehicle_id] is None:
                                        storeactions[vehicle_id] = actions[vehicle_id]
                                        storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                    else:
                                        storeactions[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions[vehicle_id].next_action.next_value = self.active_requests[target_request.request_id].final_value
                                        storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                        storeactions[vehicle_id].next_target_location = self.active_requests[target_request.request_id].pickup
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions[vehicle_id] = None
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                    self.vehicles[vehicle_id]['idle_timer'] = 0  # Reset idle timer on new assignment
                            else:
                                self.vehicles[vehicle_id]['continual_reject'] += 1
                                self.vehicles[vehicle_id]['assigned_request'] = None  # Clear the rejected request assignment
                                # 使用location字段而不是coordinates，因为_manhattan_distance_loc期望int类型
                                
                                if self.vehicles[vehicle_id]['continual_reject'] >= self.penalty_reject_requestnum:
                                    self.vehicles[vehicle_id]['penalty_timer'] = self.ev_penalty_duration
                                    self.vehicles[vehicle_id]['continual_reject'] = 0
                                # EV拒单后的relocation决策
                                if self._is_ev(vehicle_id):
                                    target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                    
                                    rejected_request_id = target_request.request_id
                                    from src.Action import IdleAction
                                    vehicle = self.vehicles[vehicle_id]
                                    if vehicle['battery'] <= self.min_battery_level + 2*self.battery_consum:
                                        target_coords = vehicle['coordinates']  # Stay put if battery is too low
                                    vehicle['idle_target'] = target_coords
                                    current_coords = vehicle['coordinates']
                                    actions[vehicle_id] = ServiceAction([target_request], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                    actions[vehicle_id].was_rejected = True
                                    actions[vehicle_id].rejection_reason = 'driver_reject'
                                    rej_distance = self._manhattan_distance_loc(vehicle_location, self.active_requests[rejected_request_id].pickup)
                                    #print(f"❌  EV Vehicle {vehicle_id} rejected request {rejected_request_id} at step {self.current_time},distance:{rej_distance}, relocating to {target_coords} as idle target.")
                                    if  storeactions_ev[vehicle_id] is None:
                                        # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                                        storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                        rejec_pickupdist = self._manhattan_distance_loc(self.vehicles[vehicle_id]['location'], self.active_requests[rejected_request_id].pickup)
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                    else:
                                        # 替换action - 保存旧信息
                                        storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions_ev[vehicle_id].next_action.next_value = 0
                                        storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                        storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions_ev[vehicle_id].next_target_location = self.active_requests[rejected_request_id].pickup
                                        storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                        old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions_ev[vehicle_id] = None
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup

                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].target_location = current_coords
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].next_target_location = current_coords
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = None
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            vehicle['target_location'] = idle_target
                    
                            #print("successfully assign vehicle {} to idle at location {}".format(vehicle_id, idle_target))
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].next_target_location = idle_target
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                        elif isinstance(target_request, str) and target_request.startswith("reloc"):
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            if self._is_ev(vehicle_id):
                                target_location = self._sample_ev_default_relocation_target(
                                    vehicle_id
                                )
                                target_coords = (
                                    target_location % self.grid_size,
                                    target_location // self.grid_size,
                                )
                                vehicle = self.vehicles[vehicle_id]
                                from src.Action import IdleAction
                                vehicle['idle_target'] = target_coords
                                vehicle['target_location'] = target_coords
                                
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                actions[vehicle_id].learning_action_type = 'reloc'
                                actions[vehicle_id].target_location = target_location
                                actions[vehicle_id].post_action_location = target_location
                                actions[vehicle_id].post_action_distance = float(
                                    self._manhattan_distance_loc(
                                        vehicle_location,
                                        target_location,
                                    )
                                )
                                actions[vehicle_id].post_action_duration = max(
                                    1.0,
                                    actions[vehicle_id].post_action_distance,
                                )
                                actions[vehicle_id].post_action_zoneid = int(
                                    self.get_zone_embedding_id(target_location)
                                )
                                if  storeactions_ev[vehicle_id] is None:
                                    # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                                    storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    storeactions_ev[vehicle_id].target_location = target_coords
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                                else:
                                    # 替换action - 保存旧信息
                                    storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].next_action.next_value = 0
                                    storeactions_ev[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    storeactions_ev[vehicle_id].next_target_location = target_coords
                                    storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                    storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                    old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                    self.storeactions_ev[vehicle_id] = None
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                            
                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            #self._assign_idle_vehicle(vehicle_id)
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            
                            if  storeactions_ev[vehicle_id] is None:
                                storeactions_ev[vehicle_id] = actions[vehicle_id]
                                storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer'] 
                                storeactions_ev[vehicle_id].target_location = target_coords
                                self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                self.storeactions_ev[vehicle_id].dur_reward = 0
                                self.storeactions_ev[vehicle_id].current_time = self.current_time
                                self.storeactions_ev[vehicle_id].target_location = target_coords
                            else:
                                # 替换action - 保存旧信息
                                storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                storeactions_ev[vehicle_id].next_action.next_value = 0
                                storeactions_ev[vehicle_id].next_idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions_ev[vehicle_id].next_target_location = target_coords
                                storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                self.storeactions_ev[vehicle_id] = None
                                self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                self.storeactions_ev[vehicle_id].dur_reward = 0
                                self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions_ev[vehicle_id].current_time = self.current_time
                                self.storeactions_ev[vehicle_id].target_location = target_coords
                           
                    else:
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        from src.Action import IdleAction
                        #self._assign_idle_vehicle(vehicle_id)
                        idle_target = vehicle['coordinates']
                        vehicle['target_location'] = idle_target
                        vehicle['idle_target'] = idle_target
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery)
                    #print("PreTest: vehicle 41 action:", self.vehicles[41]['idle_target'])
                
        
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        vehicles_to_rebalance_aev = locals().get('vehicles_to_rebalance_aev', [])
        if vehicles_to_rebalance_aev:
            self._prev_follower_prior_features_for_leader = self._build_prior_features(vehicles_to_rebalance_aev, actions)
            self._prev_follower_zone_dist_target_for_leader = self._prior_zone_dist_target
            follower_external_prior, follower_external_posterior = self._compute_bayes_beliefs_for_context(
                self.value_function,
                self._prev_follower_prior_features_for_leader,
                self.current_time,
                external_prior=self._bayes_external_prior,
                external_posterior=self._bayes_external_posterior,
                bayes_state_dist=getattr(self, '_bayes_state_posterior', None),
                bayes_role='follower',
            )
            self._prev_follower_external_prior_for_leader = follower_external_prior
            self._prev_follower_external_posterior_for_leader = follower_external_posterior

        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        
        
        self._annotate_recourse_actions(actions)
        return actions, storeactions, storeactions_ev

    def simulate_motion_evfirst(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []
        self.decision_mode = "ev_first"
        recourse_variant = RecourseTargetBuilder.validate_variant(
            getattr(self, "recourse_variant", "legacy"), "evfirst"
        )
        recourse_policy = RecourseTargetBuilder.variant_policy(recourse_variant)
        pending_transition = self._begin_joint_collection("ev_first")
        self._same_epoch_blocked_request_ids = set()
        actions = {}
        follower_t_prediction = self.refresh_bayes_state_distribution()
        self._prior_features_for_posterior = None
        self._prior_zone_dist_target = follower_t_prediction
        self._bayes_external_prior = follower_t_prediction
        self._bayes_external_posterior = follower_t_prediction
        self._bayes_context_role = 'leader'
        vehicles_to_rebalance_aev = []
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        from src.Action import ChargingAction, ServiceAction, IdleAction

        charging_ev = []
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None and self._should_consider_ev_charging(vehicle_id):
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and ((random.random() < p_charge) or vehicle['battery'] <= 0.2):
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                else:
                    # EV declined charging: set no-charge cooldown for 5 time steps
                    vehicle['no_charge_cooldown_until'] = self.current_time + 5
                    self._clear_ev_charge_trigger(vehicle_id)
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
                        # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v.get('is_online', True) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_wait = [vehicle_id for vehicle_id, v in self.vehicles.items() if v['is_stationary']==True and v not in idle_vehicles_1 and v not in idle_vehicles_2 and  v['penalty_timer']==0]
            idle_vehicles_v = [vehicle_id for vehicle_id, v in self.vehicles.items() if self._is_ev(vehicle_id) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['idle_target'] is not None and v not in idle_vehicles_2 and v not in idle_vehicles_1 and v not in idle_vehicles_wait
                               and  v['penalty_timer']==0]
            # idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_wait+idle_vehicles_v
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            vehicles_to_rebalance = [
                vehicle_id
                for vehicle_id in vehicles_to_rebalance
                if self.vehicles[vehicle_id]['assigned_request'] is None
                and self.vehicles[vehicle_id]['passenger_onboard'] is None
                and self.vehicles[vehicle_id]['charging_station'] is None
                and (
                    self.vehicles[vehicle_id]['target_location'] is None
                    or self._is_ev(vehicle_id)
                )
            ]
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Total vehicles to rebalance: {len(vehicles_to_rebalance)}, Strict idle vehicles: {len(idle_vehicles_1)}, Available requests: {available_requests_count}")
                
                # Initialize counters for tracking assignments
                new_assignments = 0
                charging_assignments = 0
                quest_num_now = len(self.active_requests)
                re_assignments_len = len(vehicles_to_rebalance)
                self.total_rebalancing_calls += 1
                #print("videcles to rebalance:", vehicles_to_rebalance)
                vehicles_to_rebalance_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]
                requests_for_rebalance = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                debug_rebalance = self.current_time == 0
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] EV phase start: vehicles={len(vehicles_to_rebalance_ev)}, requests={len(available_requests)}",
                        flush=True,
                    )

                ev_matrix_start = time.time() if debug_rebalance else None
                vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_ev, rebalance_num=len(vehicles_to_rebalance_ev),onlyev=True)
                ev_stage_request_columns = list(
                    getattr(self, "_last_matrix_request_ids", ())
                )[:num_requests]
                self._current_ev_stage_request_ids = {
                    int(request_id)
                    for column, request_id in enumerate(ev_stage_request_columns)
                    if vehicle_action_matrix.shape[0] > 0
                    and bool(np.any(vehicle_action_matrix[:, column] > 0))
                }
                self._current_ev_offered_request_ids = set()
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] EV matrix ready in {time.time() - ev_matrix_start:.2f}s shape={vehicle_action_matrix.shape}",
                        flush=True,
                    )
                ev_q_start = time.time() if debug_rebalance else None
                if self.adp_value>0 and self.value_function is not None:
                    batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance_ev,onlyev=True)
                else:
                    batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_ev, onlyev=True
                    )
                ev_stage_graph = None
                if pending_transition is not None:
                    structured_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_ev, onlyev=True
                    )
                    ev_stage_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                        self,
                        vehicles_to_rebalance_ev,
                        vehicle_action_matrix,
                        batch_q_value,
                        structured_q_value,
                        num_requests=num_requests,
                        num_stations=num_stations,
                        num_zones=num_zones,
                        stage_id=1,
                        solver_backend=pending_transition.solver_backend,
                        state=pending_transition.pre_state,
                    )
                    pending_transition.ev_stage_graph = ev_stage_graph
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] EV q-value ready in {time.time() - ev_q_start:.2f}s shape={batch_q_value.shape}",
                        flush=True,
                    )
                ev_opt_start = time.time() if debug_rebalance else None
                    
                force_mcmf_knownreject = self._should_force_mcmf_knownreject(onlyev=True)
                if self.usemcmf or force_mcmf_knownreject:
                    rebalancing_assignments_ev = self.gurobi_optimizer._np_vehicle_rebalancing_network_ev(vehicles_to_rebalance_ev,available_requests, vehicle_action_matrix, batch_q_value, iflp=True)
                else:
                    if self.assignmentgurobi:
                        if self.gurobi_network:
                            rebalancing_assignments_ev = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network_ev(vehicles_to_rebalance_ev,available_requests,  vehicle_action_matrix, batch_q_value,self.gurobi_network_lp)
                        else:
                            rebalancing_assignments_ev,remaingrequests = self.gurobi_optimizer._gurobi_vehicle_rebalancing_ev(vehicles_to_rebalance_ev,available_requests)
                    else:
                        charging_stations = self._charging_stations_accepting_arrivals()
                        if self.adp_value > 0 and self.value_function_ev is not None:
                            rebalancing_assignments_ev = self.gurobi_optimizer._heuristic_assignment_fastqvalue_evfirst(
                                vehicles_to_rebalance_ev,
                                charging_stations,
                                vehicle_action_matrix,
                                batch_q_value,
                            )
                        else:
                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_ev, rebalance_num=len(vehicles_to_rebalance_ev), onlyev=True)
                            available_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') and self.active_requests else []
                            rebalancing_assignments_ev = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance_ev, available_requests, charging_stations, vehicle_action_matrix)
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] EV optimization ready in {time.time() - ev_opt_start:.2f}s assignments={len(rebalancing_assignments_ev)}",
                        flush=True,
                    )
                if pending_transition is not None and ev_stage_graph is not None:
                    selected_ev_edges = StateSnapshotBuilder.selected_edge_ids(
                        ev_stage_graph, rebalancing_assignments_ev
                    )
                    RecourseTargetBuilder.verify_feasible(
                        ev_stage_graph, selected_ev_edges
                    )
                    ev_stage_graph = ev_stage_graph.with_selected(
                        selected_ev_edges, status="selected"
                    )
                    pending_transition.ev_stage_graph = ev_stage_graph
                    pending_transition.ev_joint_action = JointActionSnapshot.from_graph(
                        ev_stage_graph
                    )
                                
                rejected_ev_requests = []
                for vehicle_id, target_request in rebalancing_assignments_ev.items():
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    if isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                            new_assignments += 1
                            vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                            vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                            vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                            vehicle['idle_target'] = None  # Clear idle target on new assignment
                            from src.Action import ServiceAction
                            actions[vehicle_id] = ServiceAction([target_request], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            request_final_value = self.active_requests[target_request.request_id].final_value
                            actions[vehicle_id].request_value = request_final_value
                            if vehicle['type'] == 1:
                                target_coords = self.active_requests[target_request.request_id].pickup
                                if  storeactions_ev[vehicle_id] is None:
                                    
                                    storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    storeactions_ev[vehicle_id].target_location = target_coords
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                                else:
                                    # 替换action - 保存旧信息
                                    storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].next_target_location = target_coords
                                    storeactions_ev[vehicle_id].next_action.next_value = request_final_value
                                    storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                    storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                    old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                    self.storeactions_ev[vehicle_id] = None
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                        else:
                            vehicle['continual_reject'] += 1
                            vehicle['assigned_request'] = None
                            if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                                vehicle['penalty_timer'] = self.ev_penalty_duration
                                vehicle['continual_reject'] = 0
                            # EV拒单后的relocation决策
                            if self._is_ev(vehicle_id):
                                target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                rejected_request_id = target_request.request_id
                                from src.Action import IdleAction
                                vehicle = self.vehicles[vehicle_id]
                                if vehicle['battery'] <= self.min_battery_level + 2*self.battery_consum:
                                    target_coords = vehicle['coordinates']
                                vehicle['idle_target'] = target_coords
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = ServiceAction([target_request], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                actions[vehicle_id].was_rejected = True
                                actions[vehicle_id].rejection_reason = 'driver_reject'
                                rej_distance = self._manhattan_distance_loc(vehicle_location, self.active_requests[rejected_request_id].pickup)
                                #print(f"❌  EV Vehicle {vehicle_id} rejected request {rejected_request_id} at step {self.current_time},distance:{rej_distance}, relocating to {target_coords} as idle target.")
                                if  storeactions_ev[vehicle_id] is None:
                                    # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                                    storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                    rejec_pickupdist = self._manhattan_distance_loc(self.vehicles[vehicle_id]['location'], self.active_requests[rejected_request_id].pickup)
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                else:
                                    # 替换action - 保存旧信息
                                    storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].next_action.next_value = 0
                                    storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                                    storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                    storeactions_ev[vehicle_id].next_target_location = self.active_requests[rejected_request_id].pickup
                                    storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                    old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                    self.storeactions_ev[vehicle_id] = None
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = self.active_requests[rejected_request_id].pickup
                    else:
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        target_location = self._sample_ev_default_relocation_target(
                            vehicle_id
                        )
                        target_coords = (
                            target_location % self.grid_size,
                            target_location // self.grid_size,
                        )
                        from src.Action import IdleAction
                        vehicle = self.vehicles[vehicle_id]
                        vehicle['idle_target'] = target_coords
                        vehicle['target_location'] = target_coords
                        
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                        actions[vehicle_id].learning_action_type = 'reloc'
                        actions[vehicle_id].target_location = target_location
                        actions[vehicle_id].post_action_location = target_location
                        actions[vehicle_id].post_action_distance = float(
                            self._manhattan_distance_loc(
                                vehicle_location,
                                target_location,
                            )
                        )
                        actions[vehicle_id].post_action_duration = max(
                            1.0,
                            actions[vehicle_id].post_action_distance,
                        )
                        actions[vehicle_id].post_action_zoneid = int(
                            self.get_zone_embedding_id(target_location)
                        )
                        if  storeactions_ev[vehicle_id] is None:
                            # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                            storeactions_ev[vehicle_id] = actions[vehicle_id]
                            storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                            storeactions_ev[vehicle_id].target_location = target_coords
                            self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                            self.storeactions_ev[vehicle_id].dur_reward = 0
                            self.storeactions_ev[vehicle_id].current_time = self.current_time
                            self.storeactions_ev[vehicle_id].target_location = target_coords
                        else:
                            # 替换action - 保存旧信息
                            storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                            storeactions_ev[vehicle_id].next_action.next_value = 0
                            storeactions_ev[vehicle_id].next_target_location = target_coords
                            storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                            storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                            old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                            self.storeactions_ev[vehicle_id] = None
                            self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                            self.storeactions_ev[vehicle_id].dur_reward = 0
                            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                            self.storeactions_ev[vehicle_id].current_time = self.current_time
                            self.storeactions_ev[vehicle_id].target_location = target_coords
                vehicles_to_rebalance_aev = [vid for vid in vehicles_to_rebalance if vid not in vehicles_to_rebalance_ev]
                
                # Build current leader beliefs for follower at time t.
                self._prior_features_for_follower = self._build_prior_features(vehicles_to_rebalance_ev, actions)
                self._prior_zone_dist_target_for_follower = self._prior_zone_dist_target
                leader_external_prior, leader_external_posterior = self._compute_bayes_beliefs_for_context(
                    self.value_function_ev,
                    self._prior_features_for_follower,
                    self.current_time,
                    external_prior=follower_t_prediction,
                    external_posterior=follower_t_prediction,
                    bayes_state_dist=follower_t_prediction,
                    bayes_role='leader',
                )
                self._prior_features_for_posterior = self._prior_features_for_follower
                self._prior_zone_dist_target = self._prior_zone_dist_target_for_follower
                self._bayes_external_prior = leader_external_prior
                self._bayes_external_posterior = leader_external_posterior
                self._bayes_context_role = 'follower'

                # 重新获取当前的active_requests（避免使用过期的requests_for_rebalance快照）
                current_active_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                
                # print("Assigned requests:", assigned_request)
                # Filter out both assigned AND expired requests
                available_requests = [req for req in current_active_requests 
                                    if req.request_id not in assigned_request ]
                epoch_id = self._epoch_id()
                rejected_residual_ids = {
                    int(request_id)
                    for (offer_epoch, _vehicle_id, request_id), realization
                    in self._last_offer_realizations.items()
                    if int(offer_epoch) == epoch_id
                    and bool(realization.get("rejected", False))
                }
                for request in available_requests:
                    request_id = int(request.request_id)
                    category = "other"
                    if request_id in rejected_residual_ids:
                        category = "rejected"
                    elif (
                        request_id in self._current_ev_stage_request_ids
                        and request_id not in self._current_ev_offered_request_ids
                    ):
                        category = "unoffered"
                    self.request_lifecycle.mark_residual(
                        request_id,
                        epoch_id=epoch_id,
                        category=category,
                        eligible=False,
                    )
                if not recourse_policy.same_epoch_repair:
                    self._same_epoch_blocked_request_ids = set(
                        rejected_residual_ids
                    )
                    available_requests = [
                        request
                        for request in available_requests
                        if int(request.request_id)
                        not in self._same_epoch_blocked_request_ids
                    ]
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] AEV phase start: vehicles={len(vehicles_to_rebalance_aev)}, requests={len(available_requests)}",
                        flush=True,
                    )

                aev_matrix_start = time.time() if debug_rebalance else None
                vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_aev, rebalance_num=len(vehicles_to_rebalance_aev))
                eligible_request_ids = {
                    int(request_id)
                    for column, request_id in enumerate(
                        list(getattr(self, "_last_matrix_request_ids", ()))[:num_requests]
                    )
                    if vehicle_action_matrix.shape[0] > 0
                    and bool(np.any(vehicle_action_matrix[:, column] > 0))
                }
                for request in current_active_requests:
                    request_id = int(request.request_id)
                    if request_id in assigned_request:
                        continue
                    category = "other"
                    if request_id in rejected_residual_ids:
                        category = "rejected"
                    elif (
                        request_id in self._current_ev_stage_request_ids
                        and request_id not in self._current_ev_offered_request_ids
                    ):
                        category = "unoffered"
                    self.request_lifecycle.mark_residual(
                        request_id,
                        epoch_id=epoch_id,
                        category=category,
                        eligible=request_id in eligible_request_ids,
                    )
                residual_labels = {
                    int(request.request_id): (
                        "rejected"
                        if int(request.request_id) in rejected_residual_ids
                        else (
                            "unoffered"
                            if int(request.request_id)
                            in self._current_ev_stage_request_ids
                            and int(request.request_id)
                            not in self._current_ev_offered_request_ids
                            else "other"
                        )
                    )
                    for request in current_active_requests
                    if int(request.request_id) not in assigned_request
                }
                if pending_transition is not None:
                    pending_transition.residual_state = StateSnapshotBuilder.build(
                        self, request_labels=residual_labels
                    )
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] AEV matrix ready in {time.time() - aev_matrix_start:.2f}s shape={vehicle_action_matrix.shape}",
                        flush=True,
                    )
                aev_q_start = time.time() if debug_rebalance else None
                if (
                    not recourse_policy.structured_only_follower
                    and self.adp_value > 0
                    and self.value_function is not None
                ):
                    batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance_aev)
                else:
                    batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance_aev)
                aev_stage_graph = None
                if pending_transition is not None:
                    structured_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_aev
                    )
                    aev_stage_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                        self,
                        vehicles_to_rebalance_aev,
                        vehicle_action_matrix,
                        batch_q_value,
                        structured_q_value,
                        num_requests=num_requests,
                        num_stations=num_stations,
                        num_zones=num_zones,
                        stage_id=2,
                        solver_backend=pending_transition.solver_backend,
                        state=pending_transition.residual_state,
                    )
                    pending_transition.aev_stage_graph = aev_stage_graph
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] AEV q-value ready in {time.time() - aev_q_start:.2f}s shape={batch_q_value.shape}",
                        flush=True,
                    )
                aev_opt_start = time.time() if debug_rebalance else None
                            
                force_mcmf_knownreject = self._should_force_mcmf_knownreject(onlyev=False)
                if self.usemcmf or force_mcmf_knownreject:
                    rebalancing_assignments_aev = self.gurobi_optimizer._np_vehicle_rebalancing_network(vehicles_to_rebalance_aev,available_requests, vehicle_action_matrix, batch_q_value, iflp=True)
                else:
                    if self.assignmentgurobi:
                        if self.gurobi_network:         
                            rebalancing_assignments_aev = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network(vehicles_to_rebalance_aev,available_requests,  vehicle_action_matrix, batch_q_value,self.gurobi_network_lp)
                        else:
                            charging_stations = self._charging_stations_accepting_arrivals()
                            rebalancing_assignments_aev= self.gurobi_optimizer._gurobi_vehicle_rebalancing_aev(vehicles_to_rebalance_aev,available_requests,charging_stations)
                    else:
                        charging_stations = self._charging_stations_accepting_arrivals()
                        if self.adp_value > 0 and self.value_function is not None:
                            rebalancing_assignments_aev = self.gurobi_optimizer._heuristic_assignment_fastqvalue(
                                vehicles_to_rebalance_aev,
                                charging_stations,
                                vehicle_action_matrix,
                                batch_q_value,
                            )
                        else:
                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_aev, rebalance_num=len(vehicles_to_rebalance_aev))
                            available_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') and self.active_requests else []
                            rebalancing_assignments_aev = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance_aev, available_requests, charging_stations, vehicle_action_matrix)
                if debug_rebalance:
                    print(
                        f"[evfirst][t=0] AEV optimization ready in {time.time() - aev_opt_start:.2f}s assignments={len(rebalancing_assignments_aev)}",
                        flush=True,
                    )
                if pending_transition is not None and aev_stage_graph is not None:
                    selected_aev_edges = StateSnapshotBuilder.selected_edge_ids(
                        aev_stage_graph, rebalancing_assignments_aev
                    )
                    RecourseTargetBuilder.verify_feasible(
                        aev_stage_graph, selected_aev_edges
                    )
                    aev_stage_graph = aev_stage_graph.with_selected(
                        selected_aev_edges, status="selected"
                    )
                    pending_transition.aev_stage_graph = aev_stage_graph
                    pending_transition.aev_joint_action = JointActionSnapshot.from_graph(
                        aev_stage_graph
                    )

                # for vehicle_id in vehicles_to_rebalance_aev:
                #     print("Vehicle to rebalance AEV:", vehicle_id, "Battery:", self.vehicles[vehicle_id]['battery'])
                quest_num_now = len(self.active_requests)
                for vehicle_id, target_request in rebalancing_assignments_aev.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].vehicle_loc = vehicle_location
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']


                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print("veh_id:", vehicle_id, "veh_loc:", vehicle_location, "veh_battery:", vehicle_battery)
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                vehicle = self.vehicles[vehicle_id]
                                vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                                vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                vehicle['idle_target'] = None  # Clear idle target on new assignment
                                # Generate service action
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([target_request], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                request_final_value = self.active_requests[target_request.request_id].final_value
                                actions[vehicle_id].request_value = request_final_value
                                if vehicle['type'] == 1:
                                    target_coords = self.active_requests[target_request.request_id].pickup
                                    if  storeactions_ev[vehicle_id] is None:
                                        
                                        storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions_ev[vehicle_id].target_location = target_coords
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = target_coords
                                    else:
                                        # 替换action - 保存旧信息
                                        storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].next_target_location = target_coords
                                        storeactions_ev[vehicle_id].next_action.next_value = request_final_value
                                        storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions_ev[vehicle_id] = None
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = target_coords
                                else:
                                    target_coords = self.active_requests[target_request.request_id].pickup
                                    if  storeactions[vehicle_id] is None:
                                        
                                        storeactions[vehicle_id] = actions[vehicle_id]
                                        storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions[vehicle_id].target_location = target_coords
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = target_coords
                                    else:
                                        # 替换action - 保存旧信息
                                        storeactions[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions[vehicle_id].next_target_location = target_coords
                                        storeactions[vehicle_id].next_action.next_value = 0
                                        storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions[vehicle_id] = None
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = target_coords
                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']    
                                storeactions[vehicle_id].target_location = current_coords
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = idle_target
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']

                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            #self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            target_coords = vehicle['coordinates']
                            vehicle['target_location'] = target_coords
                            vehicle['idle_target'] = target_coords
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = current_coords
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_target_location = target_coords
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                           
                    else:
                        # No assignment for this vehicle - generate idle action
                        from src.Action import IdleAction
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        #self._assign_idle_vehicle(vehicle_id)
                        vehicle = self.vehicles[vehicle_id]
                        vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                        vehicle['target_location'] = vehicle['coordinates']
                        vehicle['idle_target'] = vehicle['coordinates']
                        idle_target = vehicle.get('idle_target', None)
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery)

                for vehicle_id in vehicles_to_rebalance:
                    vehicle = self.vehicles[vehicle_id]
                    #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
                # Store the count of request assignments for this rebalancing call
                self.rebalancing_assignments_per_step.append(new_assignments)
                self.rebalancing_whole.append(re_assignments_len)
                #print(f"DEBUG Assignment Result: New request assignments: {new_assignments}, Charging assignments: {charging_assignments}, Idle assignments: {len(vehicles_to_rebalance) - new_assignments - charging_assignments}")
        
        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        vehicles_to_rebalance_ev = locals().get('vehicles_to_rebalance_ev', [])
        if vehicles_to_rebalance_ev:
            self._prev_follower_prior_features_for_leader = self._build_prior_features(vehicles_to_rebalance_ev, actions)
            self._prev_follower_zone_dist_target_for_leader = self._prior_zone_dist_target
            follower_external_prior, follower_external_posterior = self._compute_bayes_beliefs_for_context(
                self.value_function_ev,
                self._prev_follower_prior_features_for_leader,
                self.current_time,
                external_prior=self._bayes_external_prior,
                external_posterior=self._bayes_external_posterior,
                bayes_state_dist=follower_t_prediction,
                bayes_role='follower',
            )
            self._prev_follower_external_prior_for_leader = follower_external_prior
            self._prev_follower_external_posterior_for_leader = follower_external_posterior

        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        # Mark leader type so step() can avoid prior_features pollution
        self._leader_is_ev = True  # evfirst: EV is leader, AEV is follower
        self._annotate_recourse_actions(actions)
        self._same_epoch_blocked_request_ids = set()
        return actions, storeactions, storeactions_ev








    def simulate_motion_aevfirst(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []
        self.decision_mode = "aev_first"
        RecourseTargetBuilder.validate_variant(
            getattr(self, "recourse_variant", "legacy"), "aevfirst"
        )
        pending_transition = self._begin_joint_collection("aev_first")
        actions = {}
        follower_t_prediction = self.refresh_bayes_state_distribution()
        self._prior_features_for_posterior = None
        self._prior_zone_dist_target = follower_t_prediction
        self._bayes_external_prior = follower_t_prediction
        self._bayes_external_posterior = follower_t_prediction
        self._bayes_context_role = 'leader'
        vehicles_to_rebalance_ev = []
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        from src.Action import ChargingAction, ServiceAction, IdleAction

        charging_ev = []
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None and self._should_consider_ev_charging(vehicle_id):
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and ((random.random() < p_charge) or vehicle['battery'] <= 0.2):
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                else:
                    # EV declined charging: set no-charge cooldown for 5 time steps
                    vehicle['no_charge_cooldown_until'] = self.current_time + 5
                    self._clear_ev_charge_trigger(vehicle_id)
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v.get('is_online', True) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_wait = [vehicle_id for vehicle_id, v in self.vehicles.items() if v['is_stationary']==True and v not in idle_vehicles_1 and v not in idle_vehicles_2 and v['penalty_timer']==0]
            idle_vehicles_v = [vehicle_id for vehicle_id, v in self.vehicles.items() if self._is_ev(vehicle_id) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['idle_target'] is not None and v['penalty_timer']==0
                               and v not in idle_vehicles_2 and v not in idle_vehicles_1 and v not in idle_vehicles_wait]
            # idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_wait+idle_vehicles_v
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            vehicles_to_rebalance = [
                vehicle_id
                for vehicle_id in vehicles_to_rebalance
                if self.vehicles[vehicle_id]['assigned_request'] is None
                and self.vehicles[vehicle_id]['passenger_onboard'] is None
                and self.vehicles[vehicle_id]['charging_station'] is None
                and (
                    self.vehicles[vehicle_id]['target_location'] is None
                    or self._is_ev(vehicle_id)
                )
            ]
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Total vehicles to rebalance: {len(vehicles_to_rebalance)}, Strict idle vehicles: {len(idle_vehicles_1)}, Available requests: {available_requests_count}")
                
                # Initialize counters for tracking assignments
                new_assignments = 0
                charging_assignments = 0
                quest_num_now = len(self.active_requests)
                re_assignments_len = len(vehicles_to_rebalance)
                self.total_rebalancing_calls += 1
                #print("videcles to rebalance:", vehicles_to_rebalance)
                vehicles_to_rebalance_aev = [vid for vid in vehicles_to_rebalance if not self._is_ev(vid)]
                requests_for_rebalance = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                charging_stations = self._charging_stations_accepting_arrivals()
                
                vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_aev, rebalance_num=len(vehicles_to_rebalance_aev))
                if self.adp_value>0 and self.value_function is not None:
                    batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance_aev)
                else:
                    batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance_aev)
                aev_stage_graph = None
                if pending_transition is not None:
                    structured_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_aev
                    )
                    aev_stage_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                        self,
                        vehicles_to_rebalance_aev,
                        vehicle_action_matrix,
                        batch_q_value,
                        structured_q_value,
                        num_requests=num_requests,
                        num_stations=num_stations,
                        num_zones=num_zones,
                        stage_id=1,
                        solver_backend=pending_transition.solver_backend,
                        state=pending_transition.pre_state,
                    )
                force_mcmf_knownreject = self._should_force_mcmf_knownreject(onlyev=False)
                if self.usemcmf or force_mcmf_knownreject:
                    rebalancing_assignments_aev = self.gurobi_optimizer._np_vehicle_rebalancing_network(vehicles_to_rebalance_aev,available_requests, vehicle_action_matrix, batch_q_value, iflp=True)
                else:
                    if self.assignmentgurobi:
                        if self.gurobi_network:
                            rebalancing_assignments_aev = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network(vehicles_to_rebalance_aev,available_requests,  vehicle_action_matrix, batch_q_value,self.gurobi_network_lp)
                        else:
                            charging_stations = self._charging_stations_accepting_arrivals()
                            rebalancing_assignments_aev= self.gurobi_optimizer._gurobi_vehicle_rebalancing_aev(vehicles_to_rebalance_aev,available_requests,charging_stations)
                    else:
                        charging_stations = self._charging_stations_accepting_arrivals()
                        if self.adp_value > 0 and self.value_function is not None:
                            rebalancing_assignments_aev = self.gurobi_optimizer._heuristic_assignment_fastqvalue_aevfirst(
                                vehicles_to_rebalance_aev,
                                charging_stations,
                                vehicle_action_matrix,
                                batch_q_value,
                            )
                        else:
                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_aev, rebalance_num=len(vehicles_to_rebalance_aev))
                            available_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') and self.active_requests else []
                            rebalancing_assignments_aev = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance_aev, available_requests, charging_stations, vehicle_action_matrix)
                if pending_transition is not None and aev_stage_graph is not None:
                    selected_aev_edges = StateSnapshotBuilder.selected_edge_ids(
                        aev_stage_graph, rebalancing_assignments_aev
                    )
                    RecourseTargetBuilder.verify_feasible(
                        aev_stage_graph, selected_aev_edges
                    )
                    aev_stage_graph = aev_stage_graph.with_selected(
                        selected_aev_edges, status="selected"
                    )
                    pending_transition.aev_stage_graph = aev_stage_graph
                    pending_transition.aev_joint_action = JointActionSnapshot.from_graph(
                        aev_stage_graph
                    )
                quest_num_now = len(self.active_requests)
                for vehicle_id, target_request in rebalancing_assignments_aev.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            # Handle charging assignment
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print("veh_id:", vehicle_id, "veh_loc:", vehicle_location, "veh_battery:", vehicle_battery)
                            #print("target_request:", target_request.request_id, "pickup:", target_request.pickup, "dropoff:", target_request.dropoff)
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                vehicle = self.vehicles[vehicle_id]
                                vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                                vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                vehicle['idle_target'] = None  # Clear idle target on new assignment
                                # Generate service action
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([target_request], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                request_final_value = self.active_requests[target_request.request_id].final_value
                                actions[vehicle_id].request_value = request_final_value
                                if vehicle['type'] == 1:
                                    if storeactions_ev[vehicle_id] is None:
                                        storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions_ev[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                    else:
                                        storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions_ev[vehicle_id].next_target_location = self.active_requests[target_request.request_id].pickup
                                        storeactions_ev[vehicle_id].next_action.next_value = request_final_value
                                        storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                        # Save the old current_time before replacing the action
                                        old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions_ev[vehicle_id] = None
                                        self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                        self.storeactions_ev[vehicle_id].dur_reward = 0
                                        self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions_ev[vehicle_id].current_time = self.current_time
                                        self.storeactions_ev[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                else:
                                    if storeactions[vehicle_id] is None:
                                        storeactions[vehicle_id] = actions[vehicle_id]
                                        storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                        storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                                    else:
                                        storeactions[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions[vehicle_id].next_target_location = self.active_requests[target_request.request_id].pickup
                                        storeactions[vehicle_id].next_action.next_value = self.active_requests[target_request.request_id].final_value
                                        storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                        # Save the old current_time before replacing the action
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions[vehicle_id] = None
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].pickup
                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']    
                                storeactions[vehicle_id].target_location = current_coords
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = None
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']

                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            #self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            self.vehicles[vehicle_id]['is_stationary'] = False
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['idle_target'] = vehicle['coordinates']  # Default to current location if no target set
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].next_target_location = self.vehicles[vehicle_id]['target_location']
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                if pending_transition is not None:
                    pending_transition.residual_state = StateSnapshotBuilder.build(
                        self
                    )
                requests_for_rebalance = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                vehicles_to_rebalance_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]

                # Build current leader beliefs for follower at time t.
                self._prior_features_for_follower = self._build_prior_features(vehicles_to_rebalance_aev, actions)
                self._prior_zone_dist_target_for_follower = self._prior_zone_dist_target
                self._prior_features_for_posterior = self._prior_features_for_follower
                self._prior_zone_dist_target = self._prior_zone_dist_target_for_follower
                self._bayes_external_prior = self._prior_zone_dist_target_for_follower
                self._bayes_external_posterior = self._prior_zone_dist_target_for_follower
                self._bayes_context_role = 'follower'

                vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_ev, rebalance_num=len(vehicles_to_rebalance_ev),onlyev=True)
                if self.adp_value>0 and self.value_function is not None:
                    batch_q_value = self.generate_vehicle_qvalue(vehicles_to_rebalance_ev,onlyev=True)
                else:
                    batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_ev, onlyev=True
                    )
                ev_stage_graph = None
                if pending_transition is not None:
                    structured_q_value = self.generate_vehicle_qvalue_withoutqnetwork(
                        vehicles_to_rebalance_ev, onlyev=True
                    )
                    ev_stage_graph = StateSnapshotBuilder.feasible_graph_from_matrix(
                        self,
                        vehicles_to_rebalance_ev,
                        vehicle_action_matrix,
                        batch_q_value,
                        structured_q_value,
                        num_requests=num_requests,
                        num_stations=num_stations,
                        num_zones=num_zones,
                        stage_id=2,
                        solver_backend=pending_transition.solver_backend,
                        state=pending_transition.residual_state,
                    )
                force_mcmf_knownreject = self._should_force_mcmf_knownreject(onlyev=True)
                if self.usemcmf or force_mcmf_knownreject:
                    rebalancing_assignments_ev = self.gurobi_optimizer._np_vehicle_rebalancing_network_ev(vehicles_to_rebalance_ev,available_requests, vehicle_action_matrix, batch_q_value, iflp=True)
                else:
                    if self.assignmentgurobi:
                        if self.gurobi_network:
                            rebalancing_assignments_ev = self.gurobi_optimizer._gurobi_vehicle_rebalancing_network_ev(vehicles_to_rebalance_ev,available_requests,  vehicle_action_matrix, batch_q_value,self.gurobi_network_lp)
                        else:
                            rebalancing_assignments_ev,remaingrequests = self.gurobi_optimizer._gurobi_vehicle_rebalancing_ev(vehicles_to_rebalance_ev,available_requests)
                    else:
                        charging_stations = self._charging_stations_accepting_arrivals()
                        if self.adp_value > 0 and self.value_function_ev is not None:
                            rebalancing_assignments_ev = self.gurobi_optimizer._heuristic_assignment_fastqvalue(
                                vehicles_to_rebalance_ev,
                                charging_stations,
                                vehicle_action_matrix,
                                batch_q_value,
                            )
                        else:
                            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(vehicles_to_rebalance_ev, rebalance_num=len(vehicles_to_rebalance_ev), onlyev=True)
                            available_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') and self.active_requests else []
                            rebalancing_assignments_ev = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance_ev, available_requests, charging_stations, vehicle_action_matrix)
                if pending_transition is not None and ev_stage_graph is not None:
                    selected_ev_edges = StateSnapshotBuilder.selected_edge_ids(
                        ev_stage_graph, rebalancing_assignments_ev
                    )
                    RecourseTargetBuilder.verify_feasible(
                        ev_stage_graph, selected_ev_edges
                    )
                    ev_stage_graph = ev_stage_graph.with_selected(
                        selected_ev_edges, status="selected"
                    )
                    pending_transition.ev_stage_graph = ev_stage_graph
                    pending_transition.ev_joint_action = JointActionSnapshot.from_graph(
                        ev_stage_graph
                    )
                for vehicle_id, target_request in rebalancing_assignments_ev.items():
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    if isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                            new_assignments += 1
                            vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                            vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                            vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                            vehicle['idle_target'] = None  # Clear idle target on new assignment
                            from src.Action import ServiceAction
                            actions[vehicle_id] = ServiceAction([target_request], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            request_final_value = self.active_requests[target_request.request_id].final_value
                            actions[vehicle_id].request_value = request_final_value
                            if vehicle['type'] == 1:
                                target_coords = self.active_requests[target_request.request_id].pickup
                                if  storeactions_ev[vehicle_id] is None:
                                    
                                    # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                                    storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    storeactions_ev[vehicle_id].target_location = target_coords
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                                else:
                                    # 替换action - 保存旧信息
                                    storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].next_action.next_value = request_final_value
                                    storeactions_ev[vehicle_id].next_target_location = target_coords
                                    storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                    storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                    old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                    self.storeactions_ev[vehicle_id] = None
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                        else:
                            vehicle['continual_reject'] += 1
                            vehicle['assigned_request'] = None
                            if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                                vehicle['penalty_timer'] = self.ev_penalty_duration
                                vehicle['continual_reject'] = 0
                            # EV拒单后的relocation决策
                            if self._is_ev(vehicle_id):
                                target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                rejected_request_id = target_request.request_id
                                vehicle = self.vehicles[vehicle_id]
                                if vehicle['battery'] <= self.min_battery_level + 2*self.battery_consum:
                                    target_coords = vehicle['coordinates']
                                from src.Action import IdleAction
                                vehicle['idle_target'] = target_coords
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = ServiceAction([target_request], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                actions[vehicle_id].was_rejected = True
                                actions[vehicle_id].rejection_reason = 'driver_reject'
                                target_coords = self.active_requests[rejected_request_id].pickup
                                if  storeactions_ev[vehicle_id] is None:
                                    # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                                    storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].target_location = target_coords
                                    storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                                else:
                                    # 替换action - 保存旧信息
                                    storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                                    storeactions_ev[vehicle_id].next_action.next_value = 0
                                    storeactions_ev[vehicle_id].next_target_location = target_coords
                                    storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                                    storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                                    old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                                    self.storeactions_ev[vehicle_id] = None
                                    self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                                    self.storeactions_ev[vehicle_id].dur_reward = 0
                                    self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                                    self.storeactions_ev[vehicle_id].current_time = self.current_time
                                    self.storeactions_ev[vehicle_id].target_location = target_coords
                    else:
                        self.vehicles[vehicle_id]['is_stationary'] = False
                        target_location = self._sample_ev_default_relocation_target(
                            vehicle_id
                        )
                        target_coords = (
                            target_location % self.grid_size,
                            target_location // self.grid_size,
                        )
                        from src.Action import IdleAction
                        vehicle = self.vehicles[vehicle_id]
                        vehicle['idle_target'] = target_coords
                        vehicle['target_location'] = target_coords
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                        actions[vehicle_id].learning_action_type = 'reloc'
                        actions[vehicle_id].target_location = target_location
                        actions[vehicle_id].post_action_location = target_location
                        actions[vehicle_id].post_action_distance = float(
                            self._manhattan_distance_loc(
                                vehicle_location,
                                target_location,
                            )
                        )
                        actions[vehicle_id].post_action_duration = max(
                            1.0,
                            actions[vehicle_id].post_action_distance,
                        )
                        actions[vehicle_id].post_action_zoneid = int(
                            self.get_zone_embedding_id(target_location)
                        )
                        if  storeactions_ev[vehicle_id] is None:
                                    # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
                            storeactions_ev[vehicle_id] = actions[vehicle_id]
                            storeactions_ev[vehicle_id].target_location = target_coords
                            storeactions_ev[vehicle_id].idle_time = self.vehicles[vehicle_id]['idle_timer']
                            self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                            self.storeactions_ev[vehicle_id].dur_reward = 0
                            self.storeactions_ev[vehicle_id].current_time = self.current_time
                            self.storeactions_ev[vehicle_id].target_location = target_coords
                        else:
                            # 替换action - 保存旧信息
                            storeactions_ev[vehicle_id].next_action = actions[vehicle_id]
                            storeactions_ev[vehicle_id].next_action.next_value = 0
                            storeactions_ev[vehicle_id].next_target_location = target_coords
                            storeactions_ev[vehicle_id].vehicle_loc_post = vehicle_location
                            storeactions_ev[vehicle_id].vehicle_battery_post = vehicle_battery
                            old_current_time = getattr(storeactions_ev[vehicle_id], 'current_time', self.current_time)
                            self.storeactions_ev[vehicle_id] = None
                            self.storeactions_ev[vehicle_id] = actions[vehicle_id]
                            self.storeactions_ev[vehicle_id].dur_reward = 0
                            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
                            self.storeactions_ev[vehicle_id].current_time = self.current_time
                            self.storeactions_ev[vehicle_id].target_location = target_coords
                

        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        # Mark leader type so step() can avoid prior_features pollution
        self._leader_is_ev = False  # aevfirst: AEV is leader, EV is follower
        self._annotate_recourse_actions(actions)
        return actions, storeactions, storeactions_ev







    def _update_q_learning(self, actions, ifev = False):
        num_havenextaction = 0

        valuefunction = self.value_function
        valuefunction_ev = self.value_function_ev
        if valuefunction is None or not hasattr(valuefunction, 'experience_buffer'):
            return
        
        
        if ifev:
            offlinsedatalen = valuefunction_ev.experience_buffer.__len__()
            if self.current_time % 100 == 0:
                ev_total_seen = getattr(valuefunction_ev, 'total_experiences_seen', offlinsedatalen)
                ev_capacity = getattr(valuefunction_ev, 'replay_buffer_size', valuefunction_ev.experience_buffer.maxlen)
                print(f"🔄 Updating EV Q-learning - current offline data size: {offlinsedatalen} (total seen: {ev_total_seen}, capacity: {ev_capacity})")
                analyze_experience = getattr(
                    self.value_function_ev,
                    'analyze_experience_data',
                    None,
                )
                exp_analysis = (
                    analyze_experience() if callable(analyze_experience) else None
                )
                if exp_analysis:
                    reward_stats = exp_analysis['reward_stats']
                    action_stats = exp_analysis['action_stats']
                    print(f" Assign: {action_stats['assign_count']}, chargelength: {action_stats['charge_count']}, idlelength: {action_stats['idle_count']}")
                    print(f"      EV assign sample pool: positive(>10)={action_stats['ev_assign_positive_count']}, negative(<10)={action_stats['ev_assign_negative_count']}, positive_ratio={action_stats['ev_assign_positive_ratio']:.1%}")
                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']} | 0{reward_stats['neutral_ratio']} | -{reward_stats['negative_ratio']}")
                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']}, Assign={action_stats['assign_mean_reward']}, Charge={action_stats['charge_mean_reward']}, Idle={action_stats['idle_mean_reward']}")
                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                else:
                    print("    ⚠️ No experience data available for analysis yet")
        else:
            offlinsedatalen = valuefunction.experience_buffer.__len__()
            if self.current_time % 100 == 0:
                print(f"🔄 Updating Q-learning - current offline data size: {offlinsedatalen}")
                analyze_experience = getattr(
                    self.value_function,
                    'analyze_experience_data',
                    None,
                )
                exp_analysis = (
                    analyze_experience() if callable(analyze_experience) else None
                )
                if exp_analysis:
                    reward_stats = exp_analysis['reward_stats']
                    action_stats = exp_analysis['action_stats']
                    print(f" Assign: {action_stats['assign_count']}, chargelength: {action_stats['charge_count']}, idlelength: {action_stats['idle_count']}")
                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']} | 0{reward_stats['neutral_ratio']} | -{reward_stats['negative_ratio']}")
                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']}, Assign={action_stats['assign_mean_reward']}, Charge={action_stats['charge_mean_reward']}, Idle={action_stats['idle_mean_reward']}")
                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                else:
                    print("    ⚠️ No experience data available for analysis yet")
        
        # Save training dataset at line 2085
        if ifev:
            if self.current_time % 200 == 0 and offlinsedatalen > 100:  # Save every 200 time steps with enough data
                #self._save_training_dataset(valuefunction_ev)
                self._analyze_q_value_issues(valuefunction_ev)
            
            # 额外分析：每50步检查Q-value趋势
            if self.current_time % 50 == 0 and offlinsedatalen > 50:
                self._quick_q_value_analysis(valuefunction_ev)

        else:
            if self.current_time % 200 == 0 and offlinsedatalen > 100:  # Save every 200 time steps with enough data
                #self._save_training_dataset(valuefunction)
                self._analyze_q_value_issues(valuefunction)
            
            # 额外分析：每50步检查Q-value趋势
            if self.current_time % 50 == 0 and offlinsedatalen > 50:
                self._quick_q_value_analysis(valuefunction)


        from .Action import ServiceAction, ChargingAction, IdleAction

        def _manhattan_distance_loc(a_loc: int, b_loc: int) -> int:
            ax, ay = (a_loc % self.grid_size, a_loc // self.grid_size)
            bx, by = (b_loc % self.grid_size, b_loc // self.grid_size)
            return abs(ax - bx) + abs(ay - by)

        def _location_id(value, fallback: int) -> int:
            """Normalize synthetic ``(x, y)`` coordinates to a grid id."""
            if value is None:
                return int(fallback)
            if isinstance(value, (tuple, list, np.ndarray)):
                coordinates = np.asarray(value).reshape(-1)
                if coordinates.size >= 2:
                    return int(coordinates[1]) * self.grid_size + int(coordinates[0])
                if coordinates.size == 1:
                    return int(coordinates[0])
                return int(fallback)
            return int(value)

        def _post_demand_experience_kwargs(
            current_value_function,
            completed_action,
            post_location,
            request_num_at_start,
        ):
            if current_value_function is None or not (
                getattr(current_value_function, 'uses_post_demand_feature', False)
                or getattr(current_value_function, 'uses_post_demand_direct_q', False)
            ):
                return {}
            if post_location is None:
                return {}
            duration = max(
                0.0,
                float(
                    getattr(
                        completed_action,
                        'post_action_duration',
                        getattr(completed_action, 'dur_time', 1.0),
                    )
                    or 0.0
                ),
            )
            start_time = float(
                getattr(completed_action, 'current_time', self.current_time - duration)
            )
            post_location = int(post_location)
            return {
                'post_demand_current_time': start_time,
                'post_action_duration': duration,
                'post_action_location': post_location,
                'post_demand_num_requests_at_start': float(request_num_at_start or 0.0),
                'post_demand_current_zone_count': 0.0,
                'post_demand_snapshot_available': 0.0,
                'observed_post_demand': float(
                    self._active_request_count_at_location(post_location)
                ),
            }

        def _store_experience_with_masac_candidates(
            current_value_function,
            **experience,
        ):
            """Mirror NYC's feasible-action replay interface in synthetic runs.

            NYC places ``next_candidate_actions`` on every MASAC replay row.
            The synthetic environment previously omitted that field for the
            former2 queue-feature mode, making its actor, temperature update,
            and Bellman bootstrap identically zero after training began.
            """
            needs_candidates = bool(
                getattr(
                    current_value_function,
                    'requires_candidate_action_replay',
                    False,
                )
                or getattr(
                    current_value_function,
                    'standard_entropy_tuning',
                    False,
                )
            )
            terminal = bool(
                experience.get('is_system_done', False)
                or experience.get('is_vehicle_done', False)
            )
            if (
                needs_candidates
                and not terminal
                and not experience.get('next_candidate_actions')
            ):
                experience['next_candidate_actions'] = (
                    self._build_standard_masac_candidates(
                        int(experience.get('vehicle_id', -1))
                    )
                )

            if needs_candidates and not terminal:
                if experience.get('next_candidate_actions'):
                    current_value_function.replay_rows_with_candidates = int(
                        getattr(
                            current_value_function,
                            'replay_rows_with_candidates',
                            0,
                        )
                    ) + 1
                else:
                    current_value_function.replay_rows_without_candidates = int(
                        getattr(
                            current_value_function,
                            'replay_rows_without_candidates',
                            0,
                        )
                    ) + 1
            current_value_function.store_experience(**experience)


        whether_finish = self.current_time >= self.episode_length


        for vehicle_id in actions.keys():
            action = actions[vehicle_id]
            # Skip None actions to prevent AttributeError
            if action is None:
                continue
            selected_value_function = (
                self.value_function_ev
                if int(self.vehicles[vehicle_id].get('type', 1)) == 1
                else self.value_function
            )
            bind_context = getattr(
                selected_value_function, 'set_replay_collection_context', None
            )
            if callable(bind_context):
                bind_context(action)
            batterynow = self.vehicles[vehicle_id]['battery']
            # Use pre-action as decision state
            current_location = action.vehicle_loc
            current_battery = action.vehicle_battery
            current_request_num = action.req_num
            veh_curloc = self.vehicles[vehicle_id]['location']
            veh_type = self.vehicles[vehicle_id]['type']
            action_start_time = float(
                getattr(action, 'current_time', self.current_time)
            )
            default_action_duration = max(
                1.0,
                float(self.current_time) - action_start_time,
            )
            action_dur_time = float(
                getattr(action, 'dur_time', default_action_duration)
                or default_action_duration
            )
            if veh_type == 2:
                
                if (self.value_function and hasattr(self.value_function, 'store_experience')):
                    other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                    num_requests = len(self.active_requests)
                    store_threshold = 5
                    # Service option at assignment decision
                    next_action = getattr(action, 'next_action', None)
                    if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and next_action is not None and action.dur_reward >store_threshold:
                        idle_time  =action.idle_time if hasattr(action, 'idle_time') else 0
                        #print(f"🚗 Storing service experience for AEV vehicle {vehicle_id} at step {self.current_time} of reward {actions[vehicle_id].dur_reward} with idle time {idle_time}")
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                        next_value = getattr(next_action, 'next_value', r_exec)
                        req = action.target_location
                        next_target = action.next_target_location if hasattr(action, 'next_target_location') else req
                        next_battery = batterynow
                        target_location = req
                        next_location = req
                        if isinstance(next_action, ServiceAction) :
                            next_action_type = "assign"
                        elif isinstance(next_action, ChargingAction) :
                            next_action_type = "charge"
                        else:
                            next_action_type = "idle"
                        # 使用实际的 request.final_value 而不是 dur_reward
                        request_obj = self.active_requests.get(action.request_id) if action.request_id in self.active_requests else None
                        req_final_value = request_obj.final_value if request_obj else r_exec
                        _store_experience_with_masac_candidates(
                            self.value_function,
                            vehicle_id=vehicle_id,
                            action_type=f"assign_{action.request_id}",
                            vehicle_location=actions[vehicle_id].vehicle_loc,
                            target_location=target_location,
                            current_time=self.current_time,
                            reward=r_exec,
                            next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                            next_target_location=next_target,
                            battery_level=current_battery,
                            next_battery_level=next_battery,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=req_final_value,
                            next_action_type = next_action_type,
                            next_request_value = next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False),
                            vehicle_idle_time = action.idle_time if hasattr(action, 'idle_time') else 0,
                            next_vehicle_idle_time = self.vehicles[vehicle_id]['idle_timer'],
                            **_post_demand_experience_kwargs(
                                self.value_function,
                                action,
                                actions[vehicle_id].vehicle_loc_post,
                                current_request_num,
                            ),
                        )

                    elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                        # print(f"🔋 Storing charging experience for vehicle {vehicle_id} at step {self.current_time}")
                        st_id = action.charging_station_id
                        next_action = getattr(action, 'next_action', None)
                        if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch and next_action is not None:
                            #print(f"🔋 Storing charging experience for vehicle {vehicle_id} at step {self.current_time}")
                            next_value = getattr(next_action, 'next_value', 0)
                            st = self.charging_manager.stations[st_id]
                            station_loc = st.location
                            r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                            next_battery = batterynow
                            next_target = action.next_target_location if hasattr(action, 'next_target_location') else station_loc
                            target_location = station_loc
                            next_location = station_loc
                            if isinstance(next_action, ServiceAction) :
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction) :
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                            _store_experience_with_masac_candidates(
                                self.value_function,
                                vehicle_id=vehicle_id,
                                action_type=f"charge_{st_id}",
                                target_station_id=int(st_id),
                                queue_features=(
                                    getattr(action, 'queue_features', None)
                                    or self.vehicles[vehicle_id].get(
                                        'charging_decision_queue_features'
                                    )
                                ),
                                vehicle_location=actions[vehicle_id].vehicle_loc,
                                target_location=target_location,
                                current_time=action_start_time,
                                reward=r_exec,
                                next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                                next_target_location=next_target,
                                battery_level=current_battery,
                                next_battery_level=next_battery,
                                other_vehicles=other_vehicles,
                                num_requests=num_requests,
                                request_value=0.0,
                                next_action_type=next_action_type,
                                next_request_value=next_value,
                                dur_time=getattr(action, 'dur_time', 1.0),
                                is_system_done=getattr(self, 'done', False),
                                vehicle_idle_time = action.idle_time if hasattr(action, 'idle_time') else 0,
                                next_vehicle_idle_time = self.vehicles[vehicle_id]['idle_timer'],
                                **_post_demand_experience_kwargs(
                                    self.value_function,
                                    action,
                                    actions[vehicle_id].vehicle_loc_post,
                                    current_request_num,
                                ),
                            )

                    # Idle/wait decision treated as single-step option
                    elif isinstance(action, IdleAction):

                        # print(f"⏳ Storing idle experience for vehicle {vehicle_id} at step {self.current_time}")
                        next_action = getattr(action, 'next_action', None)
                        # 确定next_action类型和价值，即使next_action为None也要存储
                        if next_action is not None:
                            next_value = getattr(next_action, 'next_value', actions[vehicle_id].dur_reward)
                            if isinstance(next_action, ServiceAction):
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction):
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                        else:
                            # 没有next_action时，使用当前的dur_reward作为价值估计
                            next_value = actions[vehicle_id].dur_reward
                            next_action_type = "idle"
                        
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                        next_target = action.next_target_location if hasattr(action, 'next_target_location') else veh_curloc
                        _store_experience_with_masac_candidates(
                            self.value_function,
                            vehicle_id=vehicle_id,
                            action_type="idle",
                            vehicle_location=actions[vehicle_id].vehicle_loc,
                            target_location=veh_curloc,
                            current_time=self.current_time,
                            reward=r_exec,
                            next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                            next_target_location=next_target,
                            battery_level=current_battery,
                            next_battery_level=batterynow,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=0.0,
                            next_action_type=next_action_type,
                            next_request_value=next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False),
                            vehicle_idle_time = action.idle_time if hasattr(action, 'idle_time') else 0,
                            next_vehicle_idle_time = self.vehicles[vehicle_id]['idle_timer'],
                            **_post_demand_experience_kwargs(
                                self.value_function,
                                action,
                                actions[vehicle_id].vehicle_loc_post,
                                current_request_num,
                            ),
                        )
            else:
                if not (
                    self.value_function_ev
                    and hasattr(self.value_function_ev, 'store_experience')
                ):
                    continue
                if isinstance(action, IdleAction):
                    other_vehicles = len([
                        vehicle
                        for vehicle in self.vehicles.values()
                        if vehicle['assigned_request'] is not None
                    ])
                    num_requests = len(self.active_requests)
                    source_location = _location_id(
                        current_location,
                        veh_curloc,
                    )
                    target_location = _location_id(
                        getattr(
                            action,
                            'post_action_location',
                            getattr(action, 'target_location', None),
                        ),
                        source_location,
                    )
                    target_distance = float(_manhattan_distance_loc(
                        source_location,
                        target_location,
                    ))
                    post_duration = float(getattr(
                        action,
                        'post_action_duration',
                        max(1.0, target_distance),
                    ))
                    next_action = getattr(action, 'next_action', None)
                    if isinstance(next_action, ServiceAction):
                        next_action_type = 'assign'
                        next_value = float(getattr(next_action, 'request_value', 0.0))
                    elif isinstance(next_action, ChargingAction):
                        next_action_type = 'charge'
                        next_value = 0.0
                    else:
                        next_action_type = 'reloc'
                        next_value = float(getattr(action, 'dur_reward', 0.0))
                    next_target = _location_id(
                        getattr(next_action, 'target_location', None),
                        target_location,
                    )
                    ev_post_demand_kwargs = _post_demand_experience_kwargs(
                        self.value_function_ev,
                        action,
                        target_location,
                        current_request_num,
                    )
                    ev_post_demand_kwargs.pop('post_action_location', None)
                    ev_post_demand_kwargs.pop('post_action_duration', None)
                    _store_experience_with_masac_candidates(
                        self.value_function_ev,
                        vehicle_id=vehicle_id,
                        action_type='reloc',
                        vehicle_location=source_location,
                        target_location=target_location,
                        target_distance=target_distance,
                        target_zoneid=int(
                            self.get_zone_embedding_id(target_location)
                        ),
                        current_time=action_start_time,
                        reward=float(getattr(action, 'dur_reward', 0.0)),
                        next_vehicle_location=veh_curloc,
                        next_target_location=next_target,
                        battery_level=current_battery,
                        next_battery_level=batterynow,
                        other_vehicles=other_vehicles,
                        num_requests=num_requests,
                        request_value=0.0,
                        next_action_type=next_action_type,
                        next_request_value=next_value,
                        dur_time=action_dur_time,
                        post_action_location=target_location,
                        post_action_distance=target_distance,
                        post_action_duration=post_duration,
                        post_action_zoneid=int(
                            self.get_zone_embedding_id(target_location)
                        ),
                        is_system_done=getattr(self, 'done', False),
                        vehicle_idle_time=getattr(action, 'idle_time', 0),
                        next_vehicle_idle_time=self.vehicles[vehicle_id]['idle_timer'],
                        **ev_post_demand_kwargs,
                    )
                    continue
                if not (
                    isinstance(action, ServiceAction)
                    and hasattr(action, 'request_id')
                ):
                    continue

                other_vehicles = len([
                    vehicle
                    for vehicle in self.vehicles.values()
                    if vehicle['assigned_request'] is not None
                ])
                num_requests = len(self.active_requests)
                store_threshold = 5
                next_action = getattr(action, 'next_action', None)
                dropout_after_action = bool(
                    getattr(action, 'ev_dropout_after_action', False)
                )
                was_rejected = bool(getattr(action, 'was_rejected', False))

                # Rejection is an observed outcome of the selected service
                # action, so it trains both the rejection head and the critic.
                if was_rejected:
                    rejection_hook = getattr(
                        self.value_function_ev,
                        'store_rejection_experience',
                        None,
                    )
                    if callable(rejection_hook):
                        reject_target = getattr(
                            action,
                            'target_location',
                            current_location,
                        )
                        rejection_hook(
                            vehicle_id=vehicle_id,
                            request_id=action.request_id,
                            vehicle_location=current_location,
                            pickup_location=reject_target,
                            current_time=action_start_time,
                            distance=_manhattan_distance_loc(
                                int(current_location),
                                int(reject_target),
                            ),
                            rejection_reason=getattr(
                                action,
                                'rejection_reason',
                                'driver_reject',
                            ),
                            rejection_sample=getattr(
                                action,
                                'rejection_sample',
                                None,
                            ),
                        )
                    request_snapshot = getattr(
                        getattr(action, 'metadata', None),
                        'request_snapshot',
                        None,
                    )
                    request_value = float(
                        getattr(
                            request_snapshot,
                            'final_value',
                            getattr(action, 'request_value', 0.0),
                        )
                        or 0.0
                    )
                    reject_target = int(
                        getattr(action, 'target_location', current_location)
                    )
                    rejection_reward = float(
                        getattr(action, 'dur_reward', 0.0) or 0.0
                    )
                    if rejection_reward == 0.0:
                        rejection_reward = -float(
                            getattr(self, 'rejection_penalty_base', 1.0)
                        )
                    next_action = getattr(action, 'next_action', None)
                    if isinstance(next_action, ServiceAction):
                        next_action_type = 'assign'
                    elif isinstance(next_action, ChargingAction):
                        next_action_type = 'charge'
                    else:
                        next_action_type = 'idle'
                    _store_experience_with_masac_candidates(
                        self.value_function_ev,
                        vehicle_id=vehicle_id,
                        action_type=f"assign_{action.request_id}",
                        vehicle_location=int(current_location),
                        target_location=reject_target,
                        current_time=action_start_time,
                        reward=rejection_reward,
                        next_vehicle_location=int(veh_curloc),
                        next_target_location=int(veh_curloc),
                        battery_level=float(current_battery),
                        next_battery_level=float(batterynow),
                        other_vehicles=sum(
                            vehicle['assigned_request'] is not None
                            for vehicle in self.vehicles.values()
                        ),
                        num_requests=len(self.active_requests),
                        request_value=request_value,
                        next_action_type=next_action_type,
                        next_request_value=float(
                            getattr(next_action, 'next_value', 0.0) or 0.0
                        ),
                        dur_time=action_dur_time,
                        post_action_location=int(veh_curloc),
                        post_action_distance=0.0,
                        post_action_duration=0.0,
                        post_action_zoneid=int(
                            self.get_zone_embedding_id(int(veh_curloc))
                        ),
                        is_system_done=bool(getattr(self, 'done', False)),
                        vehicle_idle_time=float(
                            getattr(action, 'idle_time', 0.0) or 0.0
                        ),
                        next_vehicle_idle_time=float(
                            self.vehicles[vehicle_id].get('idle_timer', 0.0)
                            or 0.0
                        ),
                        was_rejected=True,
                    )
                    continue

                if next_action is None and not dropout_after_action:
                    continue

                r_exec = float(actions[vehicle_id].dur_reward)
                # NYC records non-rejected low-return service attempts with a
                # neutral reward instead of mixing relocation/rejection cost
                # into the request critic target.
                if r_exec <= store_threshold and not dropout_after_action:
                    r_exec = 0.0
                next_value = getattr(
                    next_action,
                    'next_value',
                    0.0 if dropout_after_action else r_exec,
                )
                if isinstance(next_action, ServiceAction):
                    next_action_type = "assign"
                elif isinstance(next_action, ChargingAction):
                    next_action_type = "charge"
                else:
                    next_action_type = "idle"

                target_location = getattr(
                    action,
                    'target_location',
                    current_location,
                )
                post_location = getattr(
                    action,
                    'vehicle_loc_post',
                    veh_curloc,
                )
                if post_location is None:
                    post_location = veh_curloc
                target_distance = float(_manhattan_distance_loc(
                    int(current_location),
                    int(target_location),
                ))
                trip_distance = float(_manhattan_distance_loc(
                    int(target_location),
                    int(post_location),
                ))
                post_distance = target_distance + trip_distance
                target_zoneid = int(
                    self.get_zone_embedding_id(int(target_location))
                )
                post_zoneid = int(
                    self.get_zone_embedding_id(int(post_location))
                )
                req_final_value = getattr(
                    action,
                    'request_value',
                    self._get_request_final_value(action.request_id, r_exec),
                )
                next_target = getattr(
                    action,
                    'next_target_location',
                    post_location,
                )
                ev_post_demand_kwargs = _post_demand_experience_kwargs(
                    self.value_function_ev,
                    action,
                    post_location,
                    current_request_num,
                )
                # These two action fields are supplied explicitly below for
                # every EV value-function type.  The post-demand helper also
                # returns them for demand-aware MASAC models, which would make
                # the function call contain duplicate keyword arguments.
                ev_post_demand_kwargs.pop('post_action_location', None)
                ev_post_demand_kwargs.pop('post_action_duration', None)

                _store_experience_with_masac_candidates(
                    self.value_function_ev,
                    vehicle_id=vehicle_id,
                    action_type=f"assign_{action.request_id}",
                    vehicle_location=current_location,
                    target_location=target_location,
                    target_distance=target_distance,
                    target_zoneid=target_zoneid,
                    current_time=action_start_time,
                    reward=r_exec,
                    next_vehicle_location=post_location,
                    next_target_location=next_target,
                    battery_level=current_battery,
                    next_battery_level=batterynow,
                    other_vehicles=other_vehicles,
                    num_requests=num_requests,
                    request_value=req_final_value,
                    next_action_type=next_action_type,
                    next_request_value=next_value,
                    dur_time=action_dur_time,
                    post_action_location=post_location,
                    post_action_distance=post_distance,
                    post_action_duration=action_dur_time,
                    post_action_zoneid=post_zoneid,
                    is_system_done=getattr(self, 'done', False),
                    is_vehicle_done=dropout_after_action,
                    ev_dropout_after_action=dropout_after_action,
                    dropout_penalty=getattr(action, 'dropout_penalty', 0.0),
                    dropout_satisfaction=getattr(
                        action,
                        'dropout_satisfaction',
                        0.0,
                    ),
                    dropout_salary_ratio=getattr(
                        action,
                        'dropout_salary_ratio',
                        0.0,
                    ),
                    dropout_probability=getattr(
                        action,
                        'dropout_probability',
                        0.0,
                    ),
                    next_dropout_satisfaction=(
                        0.0
                        if dropout_after_action
                        else self._get_ev_dropout_state_features(vehicle_id)[0]
                    ),
                    next_dropout_salary_ratio=(
                        0.0
                        if dropout_after_action
                        else self._get_ev_dropout_state_features(vehicle_id)[1]
                    ),
                    next_dropout_probability=(
                        0.0
                        if dropout_after_action
                        else self._get_ev_dropout_state_features(vehicle_id)[2]
                    ),
                    vehicle_idle_time=getattr(action, 'idle_time', 0),
                    next_vehicle_idle_time=self.vehicles[vehicle_id]['idle_timer'],
                    was_rejected=False,
                    **ev_post_demand_kwargs,
                )
                if hasattr(action, 'awaiting_new_assignment'):
                    action.awaiting_new_assignment = False




                    # Charging option at decision
                    # elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                    #     st_id = action.charging_station_id
                    #     next_action = getattr(action, 'next_action', None)
                    #     if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch and next_action is not None:
                    #         next_value = getattr(next_action, 'next_value', 0)
                    #         st = self.charging_manager.stations[st_id]
                    #         station_loc = st.location
                    #         r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                    #         next_battery = batterynow
                    #         target_location = station_loc
                    #         next_location = station_loc
                    #         if isinstance(next_action, ServiceAction) :
                    #             next_action_type = "assign"
                    #         elif isinstance(next_action, ChargingAction) :
                    #             next_action_type = "charge"
                    #         else:
                    #             next_action_type = "idle"
                    #         self.value_function_ev.store_experience(
                    #             vehicle_id=vehicle_id,
                    #             action_type=f"charge_{st_id}",
                    #             vehicle_location=actions[vehicle_id].vehicle_loc,
                    #             target_location=target_location,
                    #             current_time=self.current_time,
                    #             reward=r_exec,
                    #             next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                    #             next_target_location=next_target,
                    #             battery_level=current_battery,
                    #             next_battery_level=next_battery,
                    #             other_vehicles=other_vehicles,
                    #             num_requests=num_requests,
                    #             request_value=0.0,
                    #             next_action_type=next_action_type,
                    #             next_request_value=next_value,
                    #             dur_time=getattr(action, 'dur_time', 1.0),
                    #             is_system_done=getattr(self, 'done', False),
                    #             vehicle_idle_time = action.idle_time if hasattr(action, 'idle_time') else 0,
                    #         next_vehicle_idle_time = self.vehicles[vehicle_id]['idle_timer']
                    #         )
                    # elif isinstance(action, IdleAction):
                    #     next_action = getattr(action, 'next_action', None)
                    #     # 确定next_action类型和价值，即使next_action为None也要存储
                    #     if next_action is not None:
                    #         next_value = getattr(next_action, 'next_value', actions[vehicle_id].dur_reward)
                    #         if isinstance(next_action, ServiceAction):
                    #             next_action_type = "assign"
                    #         elif isinstance(next_action, ChargingAction):
                    #             next_action_type = "charge"
                    #         else:
                    #             next_action_type = "idle"
                    #     else:
                    #         next_value = actions[vehicle_id].dur_reward
                    #         next_action_type = "idle"
                    #     r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from actio
                    #     self.value_function_ev.store_experience(
                    #         vehicle_id=vehicle_id,
                    #         action_type="idle",
                    #         vehicle_location=actions[vehicle_id].vehicle_loc,
                    #         target_location=veh_curloc,
                    #         current_time=self.current_time,
                    #         reward=-10,
                    #         next_vehicle_location=actions[vehicle_id].vehicle_loc_post,
                    #         next_target_location=next_target,
                    #         battery_level=current_battery,
                    #         next_battery_level=batterynow,
                    #         other_vehicles=other_vehicles,
                    #         num_requests=num_requests,
                    #         request_value=0.0,
                    #         next_action_type=next_action_type,
                    #         next_request_value=next_value,
                    #         dur_time=getattr(action, 'dur_time', 1.0),
                    #         is_system_done=getattr(self, 'done', False),
                    #         vehicle_idle_time = action.idle_time if hasattr(action, 'idle_time') else 0,
                    #         next_vehicle_idle_time = self.vehicles[vehicle_id]['idle_timer']
                    #     )

    def _execute_action(self, vehicle_id, action):
        """Execute vehicle action with immediate reward aligned to Gurobi optimization objective"""
        from src.Action import ChargingAction, ServiceAction, IdleAction

        vehicle = self.vehicles[vehicle_id]
        if vehicle.get('type') == 1 and isinstance(action, ServiceAction):
            self._annotate_ev_dropout_state_features(action, vehicle_id)
        if not vehicle.get('is_online', True):
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['charging_target'] = None
            return 0.0, 0.0
        # Ensure storeactions[vehicle_id] exists
        if vehicle_id not in self.storeactions or self.storeactions[vehicle_id] is None:
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].current_time = self.current_time
        
        # Check if vehicle is in stationary state
        
        
        reward = 0
        dur_reward = 0.0
        # Get parameters for consistency with Gurobi
        charging_penalty = getattr(self, 'charging_penalty', 2.0)

        if isinstance(action, ChargingAction):
            vehicle['idle_target'] = None  # Clear idle target when servicing
            if vehicle['charging_station'] is None:
                # Get the charging station and check location
                station_id = action.charging_station_id
                if station_id in self.charging_manager.stations:
                    station = self.charging_manager.stations[station_id]
                    current_location = vehicle['location']  # Use location index, not coordinates[0]
                    station_location = station.location
                    # Check if vehicle is already at the charging station
                    if current_location == station_location:
                        vehicle['charging_target'] = None
                        self._clear_aev_notarrived_if_arrived(vehicle_id, station_id)
                        self._mark_charging_queue_arrival(vehicle_id, station_id)
                        
                        success = station.start_charging(str(vehicle_id))
                        if success:
                            self._mark_charging_started(vehicle_id, station_id)
                            vehicle['charging_station'] = station_id
                            self._set_vehicle_charging_session(vehicle_id)
                            vehicle['charging_count'] += 1
                            vehicle['target_location'] = None
                            vehicle.pop('target_charging_station', None)
                            reward = -charging_penalty - np.random.random()*0.2
                        else:
                            reward = self._charging_wait_step_penalty(vehicle_id, station_id)
                    else:
                        # Vehicle needs to move to charging station
                        vehicle['target_charging_station'] = station_id
                        #print(f"DEBUG: Vehicle {vehicle_id} moving towards charging station {station_id}")
                        reward = self._execute_movement_towards_charging_station(vehicle_id, station_id)
                else:
                    reward = -charging_penalty - np.random.random()*0.2  # Invalid station penalty
            else:
                reward = -charging_penalty - np.random.random()*0.2  # Invalid station penalty
            if vehicle['type'] == 1:
                reward += -0.1  # Extra penalty for EVs to encourage efficiency
                if self.storeactions_ev[vehicle_id] is not None:
                    self.storeactions_ev[vehicle_id].dur_reward += reward  # Store for reference
            else:
                reward += -0.01  # Smaller penalty for ICE vehicles
                if self.storeactions[vehicle_id] is not None:
                    self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
            dur_reward = action.dur_reward  # Total reward over charging duration
        elif isinstance(action, ServiceAction):
            self._clear_aev_notarrived_reservations(vehicle_id)
            #print(f"🚖 Executing ServiceAction for vehicle {vehicle_id} at time {self.current_time}")
            if vehicle['idle_target'] is not None and vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                #print(f"   ⚠️ Vehicle rejecting while idle: {vehicle_id}")
                reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target', None))
                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += reward
                        dur_reward = self.storeactions_ev[vehicle_id].dur_reward  # Total reward over charging duration
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += reward
                        dur_reward = self.storeactions[vehicle_id].dur_reward  # Total reward over charging duration
                request_reassign_penalty = -5         
                return reward, dur_reward
            elif  vehicle['target_location'] is not None and vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                vehicle['idle_target'] = None  # Clear idle target when servicing
                vehicle['target_location'] = None
                print(f"   ⚠️ Vehicle rejecting while moving to target: {vehicle_id}")
                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += -1.0
                        dur_reward = self.storeactions_ev[vehicle_id].dur_reward
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += -1.0
                        dur_reward = self.storeactions[vehicle_id].dur_reward
                return -1.0, dur_reward
            elif vehicle['assigned_request'] is not None:
                vehicle['idle_target'] = None  # Clear idle target when servicing
                if self._pickup_passenger(vehicle_id):
                    reward = 0.5 + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:
                        vehicle['target_location'] = None
                        vehicle['idle_target'] = None
                        vehicle['assigned_request'] = None
                        vehicle['passenger_onboard'] = None
                        vehicle['charging_target'] = None
                        #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法继续前往pickup位置")
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += reward  # Store for reference
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
            elif vehicle['passenger_onboard'] is not None:
                vehicle['idle_target'] = None  # Clear idle target when servicing
                earnings = self._dropoff_passenger(vehicle_id)
                if earnings > 0:
                    reward = earnings + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:
                        #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，乘客滞留无法到达dropoff位置")
                        vehicle['target_location'] = None
                        vehicle['idle_target'] = None
                        vehicle['assigned_request'] = None
                        vehicle['passenger_onboard'] = None
                        vehicle['charging_target'] = None
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        dropout_penalty = getattr(self.storeactions_ev[vehicle_id], 'dropout_penalty', 0.0)
                        adjusted_reward = reward - dropout_penalty
                        if reward>5 and self.storeactions_ev[vehicle_id].dur_reward>-200:
                            self.storeactions_ev[vehicle_id].dur_reward = adjusted_reward
                        else:
                            self.storeactions_ev[vehicle_id].dur_reward += adjusted_reward  # Store for reference
                        if earnings > 0:
                            self.storeactions_ev[vehicle_id].awaiting_new_assignment = True
                else:
                    if self.storeactions[vehicle_id] is not None:
                        if reward>5 and self.storeactions[vehicle_id].dur_reward>-200:
                            self.storeactions[vehicle_id].dur_reward = reward
                        else:
                            self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
        
        elif isinstance(action, IdleAction):
            self._clear_aev_notarrived_reservations(vehicle_id)
            if vehicle.get('is_stationary', False):
                idle_penalty = self.movingpenalty
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += idle_penalty
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += idle_penalty
                return idle_penalty, idle_penalty
            else:
                if vehicle['idle_target'] is None:
                    reward = 0
                    dur_reward = 0
                else:
                    reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target', None))
                    if vehicle['type'] == 1:
                        if self.storeactions_ev[vehicle_id] is not None:
                            self.storeactions_ev[vehicle_id].dur_reward += reward
                            dur_reward = self.storeactions_ev[vehicle_id].dur_reward  # Total reward over charging duration
                    else:
                        if self.storeactions[vehicle_id] is not None:
                            self.storeactions[vehicle_id].dur_reward += reward
                            dur_reward = self.storeactions[vehicle_id].dur_reward  # Total reward over charging duration
        # if vehicle.get('is_stationary', False):
        #     # Reduce stationary duration
        #     vehicle['stationary_duration'] = 1
            
        #     # If stationary duration is finished, remove stationary status
        #     if vehicle['stationary_duration'] <= 0:
        #         vehicle['is_stationary'] = False
        #         vehicle['stationary_duration'] = 0
            

        #     active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        #     active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        #     avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
        #     self.storeactions[vehicle_id].dur_reward += - avg_request_value*0.01
        #     return - avg_request_value*0.01, - avg_request_value*0.01
        return reward, dur_reward
    
    def _should_store_experience(self, action_type: str, reward: float, battery_level: float) -> bool:
        """
        决定是否存储experience，只存储关键决策点：
        1. 完成订单的experience（reward >= 15）
        2. 充电决策的experience（低电量情况）
        3. idle决策的experience（空闲移动决策）
        排除pickup/dropoff执行过程中的中间experience
        """
        # 1. 完成订单的experience（最高优先级）
        if reward >= 15:
            #print(f"✓ Storing COMPLETED ORDER experience: reward={reward}")
            return True
        
        # 2. 充电决策experience（电池管理的关键决策）
        if action_type.startswith('charge') and battery_level < 0.5:
            #print(f"✓ Storing CHARGING decision experience: battery={battery_level}")
            return True
        
        # 3. Idle决策experience（空闲状态的移动决策）
        if action_type == 'idle':
            return True  # 所有idle决策都存储
        
        # 4. 初始assignment决策（只存储决策时刻，不存储执行过程）
        if action_type.startswith('assign'):
            # 只存储真正的分配决策时刻或失败的分配
            if reward > 0 or reward <= -10:  # 成功分配或严重失败
                #print(f"✓ Storing assignment decision: reward={reward}")
                return True
            else:
                # 排除pickup/dropoff执行过程中的中间状态
                return False
        
        # 5. 其他情况不存储
        return False
    
    def _execute_movement_towards_target(self, vehicle_id):
        """Execute intelligent movement towards target (pickup/dropoff/charging)"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is not None:
            return -0.2  # Charging penalty for movement
            
        old_coords = vehicle['coordinates']
        target_coords = None
        movement_purpose = "idle"
        
        # 1. Priority 1: Move towards dropoff if passenger onboard
        if vehicle['passenger_onboard'] is not None:
            if vehicle['passenger_onboard'] in self.active_requests:
                request = self.active_requests[vehicle['passenger_onboard']]
                target_coords = (request.dropoff % self.grid_size, request.dropoff // self.grid_size)
                movement_purpose = "dropoff"
        
        # 2. Priority 2: Move towards pickup if request assigned
        elif vehicle['assigned_request'] is not None:
            if vehicle['assigned_request'] in self.active_requests:
                request = self.active_requests[vehicle['assigned_request']]
                target_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
                movement_purpose = "pickup"
        
        # 3. Priority 3: Move towards charging station if low battery
        elif vehicle['battery'] < self.min_battery_level and hasattr(vehicle, 'charging_target'):
            if vehicle['charging_target'] in self.charging_manager.stations:
                station = self.charging_manager.stations[vehicle['charging_target']]
                target_coords = (station.location % self.grid_size, station.location // self.grid_size)
                movement_purpose = "charging"
        
        # 4. Priority 4: Move towards charging station if target_location set
        elif 'target_location' in vehicle and vehicle['target_location'] is not None:
            target_coords = vehicle['target_location']
            movement_purpose = "rebalance_charging"
        
        # Calculate intelligent movement towards target
        if target_coords:
            current_x, current_y = old_coords
            target_x, target_y = target_coords
            
            # Move one step towards target (Manhattan distance)
            if current_x < target_x:
                new_x = current_x + 1
                new_y = current_y
            elif current_x > target_x:
                new_x = current_x - 1
                new_y = current_y
            elif current_y < target_y:
                new_x = current_x
                new_y = current_y + 1
            elif current_y > target_y:
                new_x = current_x
                new_y = current_y - 1
            else:
                # Already at target
                new_x, new_y = current_x, current_y
        else:
            # No specific target - random movement (exploration)
            new_x = max(0, min(self.grid_size-1, 
                             old_coords[0] + random.randint(-1, 1)))
            new_y = max(0, min(self.grid_size-1, 
                             old_coords[1] + random.randint(-1, 1)))
            movement_purpose = "exploration"
        
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        new_location_index = new_y * self.grid_size + new_x
        
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        vehicle['total_distance'] += distance
        
        # Track vehicle position for visualization
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []
        self.vehicle_position_history[vehicle_id].append({
            'coords': (new_x, new_y),
            'time': self.current_time,
            'action_type': movement_purpose
        })
        
        # Movement consumes battery
        vehicle['battery'] -= distance * (self.battery_consum )
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            vehicle['needs_emergency_charging'] = True
            print(f"⚠️  车辆 {vehicle_id} 在智能移动后电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return self.movingpenalty * distance
    

    def return_nearest_idle_target(self, vehicle_id):
        if self.use_intense_requests:
            return self.hotspot_locations[0]
        else:
            vehicle_loc = self.vehicles[vehicle_id]['location']
            x = vehicle_loc % self.grid_size
            y = vehicle_loc // self.grid_size
            distance_list = []
            for loc in self.hotspot_locations:
                hx, hy = loc
                dist = abs(hx - x) + abs(hy - y)
                distance_list.append((dist, loc))
            if distance_list:
                # Return the closest hotspot location
                return min(distance_list, key=lambda item: item[0])[1]
            return None

    
    
    
    
    
    def return_nearest_hotspot_index(self, vehicle_id):
        vehicle_loc = self.vehicles[vehicle_id]['location']
        x = vehicle_loc % self.grid_size
        y = vehicle_loc // self.grid_size
        distance_list = []
        for idx, loc in enumerate(self.hotspot_locations):
            hx, hy = loc
            dist = abs(hx - x) + abs(hy - y)
            distance_list.append((dist, idx))
        if distance_list:
            # Return the index of the closest hotspot location
            return min(distance_list, key=lambda item: item[0])[1]
        return None



    def _store_action(self, vehicle_id, action, storeactions_dict, vehicle_location, vehicle_battery, target_coords=None, next_value=0):
        """封装storeactions的赋值逻辑
        
        Args:
            vehicle_id: 车辆ID
            action: 要存储的action对象
            storeactions_dict: 本地storeactions字典（用于immediate更新）
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 可选的目标坐标，如果不提供则使用vehicle['target_location']
            next_value: next_action的value值，默认为0（用于ServiceAction时可传入final_value）
        """
        if target_coords is None:
            target_coords = self.vehicles[vehicle_id]['target_location']
        
        if self.storeactions[vehicle_id] is None:
            # 新action - 直接存储
            storeactions_dict[vehicle_id] = action
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].current_time = self.current_time
            self.storeactions[vehicle_id].target_location = target_coords
        else:
            # 替换action - 保存旧信息
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].next_action.next_value = next_value
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            
            self.storeactions[vehicle_id] = None
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions[vehicle_id].current_time = self.current_time
            self.storeactions[vehicle_id].target_location = target_coords
    

    def _store_rejected_ev_action(self, vehicle_id, idle_action, rejected_request_id, storeactions_ev_dict, vehicle_location, vehicle_battery, target_coords):
        """封装EV拒单后的storeactions_ev逻辑，存储ServiceAction用于拒单penalty计算
        
        Args:
            vehicle_id: 车辆ID
            idle_action: 实际执行的IdleAction
            rejected_request_id: 被拒绝的请求ID
            storeactions_ev_dict: 本地storeactions_ev字典
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 目标坐标
        """
        quest_num_now = len(self.active_requests)
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
        penalty_reward = - avg_request_value * 0.1
        
        from src.Action import ServiceAction
        
        if self.storeactions_ev[vehicle_id] is None:
            # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
            storeactions_ev_dict[vehicle_id] = idle_action
            self.storeactions_ev[vehicle_id] = ServiceAction([], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
        else:
            # 替换action - 保存旧信息
            storeactions_ev_dict[vehicle_id].next_action = ServiceAction([], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            storeactions_ev_dict[vehicle_id].next_action.next_value = 0
            storeactions_ev_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_ev_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_ev_dict[vehicle_id], 'current_time', self.current_time)
            
            self.storeactions_ev[vehicle_id] = None
            self.storeactions_ev[vehicle_id] = idle_action
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
        
        # 保存拒绝经验数据并检查predictor训练
        self._save_and_train_rejection_predictor()
    
    def _save_and_train_rejection_predictor(self):
        """保存接受/拒绝订单数据，并训练EV的rejection predictor"""
        # 初始化计数器
        if not hasattr(self, '_rejection_save_counter'):
            self._rejection_save_counter = 0
            self._rejection_data_buffer = []
        
        self._rejection_save_counter += 1
        
        # 每100步保存并训练一次
        if self._rejection_save_counter % 10 == 0:
            # 从value_function_ev获取rejection_buffer数据
            if hasattr(self, 'value_function_ev') and hasattr(self.value_function_ev, 'rejection_buffer'):
                rejection_buffer = self.value_function_ev.rejection_buffer
                
                if len(rejection_buffer) > 0:
                    # 保存数据到本地
                    self._save_rejection_acceptance_data(rejection_buffer)
                    
                    # 训练rejection predictor（仅对EV）
                    if hasattr(self.value_function_ev, 'train_rejection_predictor'):
                        print(f"\n{'='*60}")
                        print(f"Training EV Rejection Predictor at step {self._rejection_save_counter}")
                        print(f"{'='*60}")
                        
                        loss = self.value_function_ev.train_rejection_predictor(batch_size=64)
                        
                        if loss is not None:
                            print(f"✓ Training completed. Loss: {loss:.6f}")
                            print(f"  Buffer size: {len(rejection_buffer)}")
                            
                            # 统计接受/拒绝比例
                            rejected_count = sum(1 for d in rejection_buffer if d.get('was_rejected', False))
                            accepted_count = len(rejection_buffer) - rejected_count
                            print(f"  Rejected: {rejected_count} ({rejected_count/len(rejection_buffer)*100:.1f}%)")
                            print(f"  Accepted: {accepted_count} ({accepted_count/len(rejection_buffer)*100:.1f}%)")
                            
                            # 记录loss并设置pretrained标志（仅在训练成功时）
                            self.rejection_loss.append(loss)
                            self.rejection_pretrained = True
                        else:
                            print(f"⚠ Training skipped (insufficient data)")
                            self.rejection_loss.append(0.0)
                        
                        print(f"{'='*60}\n")
    def _save_rejection_acceptance_data(self, rejection_buffer):
        """保存接受和拒绝的订单数据到本地"""
        import os
        from datetime import datetime
        import pandas as pd
        
        if not rejection_buffer:
            return
        
        # 创建保存目录
        save_dir = 'results/rejection_analysis'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{save_dir}/rejection_acceptance_step{self._rejection_save_counter}_{timestamp}.csv'
        
        # 转换为DataFrame
        data_list = []
        for data in rejection_buffer:
            data_list.append({
                'distance': data.get('distance', 0),
                'battery_level': data.get('battery_level', 0),
                'current_time': data.get('current_time', 0),
                'num_requests': data.get('num_requests', 0),
                'vehicle_type': data.get('vehicle_type', 0),
                'was_rejected': data.get('was_rejected', False),
                'rejection_label': 'Rejected' if data.get('was_rejected', False) else 'Accepted'
            })
        
        df = pd.DataFrame(data_list)
        df.to_csv(filename, index=False)
        
        print(f"\n📊 Saved {len(data_list)} rejection/acceptance records to: {filename}")
        print(f"   Rejected: {df['was_rejected'].sum()} ({df['was_rejected'].sum()/len(df)*100:.1f}%)")
        print(f"   Accepted: {(~df['was_rejected']).sum()} ({(~df['was_rejected']).sum()/len(df)*100:.1f}%)")
        print(f"   Avg Distance: {df['distance'].mean():.2f}\n")
    
    def _handle_ev_rejection_relocation(self, vehicle_id):
        """EV拒单后使用move.md MNL选择相邻grid移动
        
        Returns:
            tuple: (target_coords, rel_action) 目标坐标和relocation动作类型
        """
        chosen_loc, _ = self.compute_ev_relocation_probability(vehicle_id)

        vehicle = self.vehicles[vehicle_id]
        cur_loc = int(vehicle.get('location', 0))
        rel_action = 'Wait' if chosen_loc == cur_loc else 'Move'
        target_coords = (chosen_loc % self.grid_size, chosen_loc // self.grid_size)

        return target_coords, rel_action



    def _set_vehicle_charging_session(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        vehicle['charging_time_left'] = int(getattr(self, 'charge_duration', 2))
        vehicle['charging_session_start_time'] = float(self.current_time)

    def _execute_movement_towards_charging_station(self, vehicle_id, station_id):
        """Execute movement towards charging station"""
        vehicle = self.vehicles[vehicle_id]

            
        station = self.charging_manager.stations[station_id]
        current_location = vehicle['location']  # Use location index, not coordinates[0]
        station_location = station.location
        
        # Convert locations to coordinates
        current_x = current_location % self.grid_size
        current_y = current_location // self.grid_size
        target_x = station_location % self.grid_size
        target_y = station_location // self.grid_size
        if (current_x, current_y) == (target_x, target_y):
            vehicle['charging_target'] = None
            self._clear_aev_notarrived_if_arrived(vehicle_id, station_id)
            self._mark_charging_queue_arrival(vehicle_id, station_id)
            success = station.start_charging(str(vehicle_id))
            #print(f"DEBUG: Vehicle {vehicle_id} at charging station {station_id}, trying to start: success={success}")
            if success:
                self._mark_charging_started(vehicle_id, station_id)
                vehicle['charging_station'] = station_id
                self._set_vehicle_charging_session(vehicle_id)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None  # Clear any rebalance target
                vehicle.pop('target_charging_station', None)  # Remove target
                #print(f"DEBUG: Vehicle {vehicle_id} started charging at station {station_id}")
                
                # Return charging penalty (same as Gurobi)
                charging_penalty = getattr(self, 'charging_penalty', 2.0)
                return -charging_penalty
            else:
                return self._charging_wait_step_penalty(vehicle_id, station_id)
        # Move one step towards charging station (Manhattan distance)
        old_coords = vehicle['coordinates']
        if current_x < target_x:
            new_x = current_x + 1
            new_y = current_y
        elif current_x > target_x:
            new_x = current_x - 1
            new_y = current_y
        elif current_y < target_y:
            new_x = current_x
            new_y = current_y + 1
        elif current_y > target_y:
            new_x = current_x
            new_y = current_y - 1
        else:
            # Already at target
            new_x, new_y = current_x, current_y
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        
        if (new_x, new_y) == (target_x, target_y):
            vehicle['coordinates'] = (new_x, new_y)
            vehicle['location'] = new_y * self.grid_size + new_x
            vehicle['battery'] -= distance * self.battery_consum
            vehicle['battery'] = max(0, vehicle['battery'])
            vehicle['charging_target'] = None
            self._clear_aev_notarrived_if_arrived(vehicle_id, station_id)
            self._mark_charging_queue_arrival(vehicle_id, station_id)
            success = station.start_charging(str(vehicle_id))
            #print(f"DEBUG: Vehicle {vehicle_id} at charging station {station_id}, trying to start: success={success}")
            if success:
                self._mark_charging_started(vehicle_id, station_id)
                vehicle['charging_station'] = station_id
                self._set_vehicle_charging_session(vehicle_id)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None  # Clear any rebalance target
                vehicle.pop('target_charging_station', None)  # Remove target
                #print(f"DEBUG: Vehicle {vehicle_id} started charging at station {station_id}")
                
                # Return charging penalty (same as Gurobi)
                charging_penalty = getattr(self, 'charging_penalty', 2.0)
                return -charging_penalty
            else:
                return self._charging_wait_step_penalty(vehicle_id, station_id)
        
        # Update vehicle position
        new_location_index = new_y * self.grid_size + new_x
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        
        # Calculate distance moved
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        
        # Movement consumes battery
        vehicle['battery'] -= distance * self.battery_consum
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            vehicle['battery'] = 1
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            print(f"⚠️  车辆 {vehicle_id} 前往充电站时电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return self.movingpenalty * distance
    
    def _execute_movement_towards_idle(self, vehicle_id, target_coords):
        if target_coords is None:
            self.vehicles[vehicle_id]['idle_target'] = None
            self.vehicles[vehicle_id]['target_location'] = None
            return 0
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is not None:
            return 0
        
        # No target to move towards
        if not target_coords:
            return 0

        old_coords = vehicle['coordinates']
        current_x, current_y = old_coords
        target_x, target_y = target_coords
        if current_x == target_x and current_y == target_y:
            vehicle['idle_target'] = None
            vehicle['target_location'] = None
            # 已经在目标点：清空idle目标，打印一次调试信息
            #print(f"DEBUG: Idle movement - already at target {target_coords}")
            return 0
        # Move one step towards target coordinates (Manhattan distance)
        if current_x < target_x:
            new_x = current_x + 1
            new_y = current_y
        elif current_x > target_x:
            new_x = current_x - 1
            new_y = current_y
        elif current_y < target_y:
            new_x = current_x
            new_y = current_y + 1
        elif current_y > target_y:
            new_x = current_x
            new_y = current_y - 1
        else:
            # Already at target
            new_x, new_y = current_x, current_y
            vehicle['idle_target'] = None
            #print("DEBUG: Idle movement - already at target (redundant branch)")
        # Update vehicle position
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        new_location_index = new_y * self.grid_size + new_x
        
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        vehicle['total_distance'] += distance
        # If reached target after this move, clear and log
        if (new_x, new_y) == (target_x, target_y):
            vehicle['idle_target'] = None
            vehicle['target_location'] = None
            vehicle['whether_finishrelocate'] = True
            if not self.daily_drop_off:
                self._handle_vehicle_dropout_event(vehicle_id)
        
        # Track vehicle position for visualization
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []
        self.vehicle_position_history[vehicle_id].append({
            'coords': (new_x, new_y),
            'time': self.current_time,
            'action_type': 'idle_movement'
        })
        
        # Movement consumes battery (same as other movement methods)
        vehicle['battery'] -= distance * (self.battery_consum )
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            vehicle['needs_emergency_charging'] = True
            print(f"⚠️  车辆 {vehicle_id} 在闲置移动后电池耗尽 (位置: {new_x}, {new_y})")
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
        # Small time penalty for movement (consistent with other methods)
        return self.movingpenalty * distance
    


    def return_surgingpricing(self):
        zone_surge = [1.0 for _ in range(self.num_zones)]
        demand = [0 for _ in range(self.num_zones)]
        supply = [0 for _ in range(self.num_zones)]
        busy_request_ids = {
            request_id
            for vehicle in self.vehicles.values()
            for request_id in (
                vehicle.get('assigned_request'),
                vehicle.get('passenger_onboard'),
            )
            if request_id is not None
        }
        for request_id, request in self.active_requests.items():
            if request_id in busy_request_ids:
                continue
            zone_id = self.get_zone_id(request.pickup)
            if 0 <= zone_id < self.num_zones:
                demand[zone_id] += 1
        for vehicle in self.vehicles.values():
            if (
                vehicle.get('is_online', True)
                and vehicle.get('assigned_request') is None
                and vehicle.get('passenger_onboard') is None
                and vehicle.get('charging_station') is None
                and vehicle.get('charging_target') is None
                and vehicle.get('penalty_timer', 0) <= 0
            ):
                zone_id = self.get_zone_id(vehicle.get('location', 0))
                if 0 <= zone_id < self.num_zones:
                    supply[zone_id] += 1

        # Ashkrof et al., Eq. (8): 1x below balance, linearly increasing to
        # 5x at the largest current zonal demand/supply ratio.
        S_values = [
            demand[zone_id] / max(1, supply[zone_id])
            for zone_id in range(self.num_zones)
        ]
        Smax = max(S_values) if S_values else 1
        for zone_id in range(self.num_zones):
            S = S_values[zone_id]
            if S <= 1:
                zone_surge[zone_id] = 1
            elif Smax <= 1:
                zone_surge[zone_id] = 1
            else:
                zone_surge[zone_id] = 1 + 4 * (S - 1) / (Smax - 1)
        return zone_surge


    def return_zone_chargingusernum(self):
        zone_charging_users = [0 for _ in range(self.num_zones)]
        zone_max_capacity = [0 for _ in range(self.num_zones)]
        zone_chargingratio = [0 for _ in range(self.num_zones)]

        for charging_station in self.charging_manager.stations.values():
            zone_id = self.get_zone_id(int(charging_station.location)) if hasattr(self, 'get_zone_id') else 0
            if not (0 <= zone_id < self.num_zones):
                continue
            zone_max_capacity[zone_id] += charging_station.max_capacity
            zone_charging_users[zone_id] += len(charging_station.current_vehicles)
            if zone_max_capacity[zone_id] > 0:
                zone_chargingratio[zone_id] = min(1.0, zone_charging_users[zone_id] / zone_max_capacity[zone_id])
        return zone_charging_users, zone_max_capacity, zone_chargingratio




    def update_satisfaction(self, vehicle_id):
        vehicle_base_salalry = self.ev_basesalary
        satisfaction = self.vehicles[vehicle_id].get('satisfaction', 0)
        daily_salary = self.vehicles[vehicle_id].get('daily_salary', 0)
        baseline_gap = (daily_salary - vehicle_base_salalry) / max(vehicle_base_salalry, 1e-6)
        updated_satisfaction = satisfaction * self.dropoff_probability_rate + (1 - self.dropoff_probability_rate) * baseline_gap
        self.vehicles[vehicle_id]['satisfaction'] = float(np.clip(updated_satisfaction, -1.0, 1.0))


    def _update_all_ev_satisfaction(self):
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.get('type') == 1:
                self.update_satisfaction(vehicle_id)

    def _current_day_index(self, current_time=None):
        if current_time is None:
            current_time = self.current_time
        return int(current_time // max(self.simulation_period, 1))

    def _current_weekday_index(self, current_time=None):
        return (self.episode_start_day + self._current_day_index(current_time)) % max(self.days_per_week, 1)

    def _is_vehicle_online(self, vehicle_id, current_time=None):
        vehicle = self.vehicles[vehicle_id]
        if vehicle.get('is_online', True):
            return True
        offline_until_time = vehicle.get('offline_until_time')
        if offline_until_time is None:
            return False
        if current_time is None:
            current_time = self.current_time
        return current_time >= offline_until_time

    def _set_vehicle_offline_until_next_day(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        self._mark_ev_pending_dropout_penalty(vehicle_id)
        next_day_time = (self._current_day_index() + 1) * self.simulation_period
        vehicle['is_online'] = False
        vehicle['offline_until_time'] = next_day_time
        self.current_period_dropout_count += 1
        vehicle['assigned_request'] = None
        vehicle['passenger_onboard'] = None
        vehicle['target_location'] = None
        vehicle['idle_target'] = None
        vehicle['charging_target'] = None
        vehicle['is_stationary'] = False
        vehicle['stationary_duration'] = 0
        vehicle['whether_finishrequest'] = False
        vehicle['whether_finishrelocate'] = False

    def _mark_ev_pending_dropout_penalty(self, vehicle_id):
        if vehicle_id not in self.vehicles or self.vehicles[vehicle_id].get('type') != 1:
            return 0.0
        pending_action = self.storeactions_ev.get(vehicle_id) if hasattr(self, 'storeactions_ev') else None
        if pending_action is None or not isinstance(pending_action, ServiceAction):
            return 0.0
        if getattr(pending_action, 'ev_dropout_after_action', False):
            return float(getattr(pending_action, 'dropout_penalty', 0.0))
        dropout_penalty = float(getattr(self, 'ev_basesalary', 0.0))
        pending_action.ev_dropout_after_action = True
        pending_action.dropout_penalty = dropout_penalty
        pending_action.is_vehicle_done = True
        return 0

    def _get_ev_dropout_state_features(self, vehicle_id):
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or vehicle.get('type') != 1:
            return 0.0, 0.0, 0.0
        base_salary = max(float(getattr(self, 'ev_basesalary', 0.0)), 1e-6)
        salary_ratio = float(np.clip(vehicle.get('daily_salary', 0.0) / base_salary, 0.0, 5.0))
        return 0.0, salary_ratio, 0.0

    def _annotate_ev_dropout_state_features(self, action, vehicle_id):
        if getattr(action, 'has_dropout_state_features', False):
            return
        satisfaction, salary_ratio, dropout_probability = self._get_ev_dropout_state_features(vehicle_id)
        action.dropout_satisfaction = satisfaction
        action.dropout_salary_ratio = salary_ratio
        action.dropout_probability = dropout_probability
        action.has_dropout_state_features = True

    def _handle_vehicle_dropout_event(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        if vehicle.get('type') != 1 or not vehicle.get('is_online', True):
            return False
        dropoff_probability = self.calculate_dropoff_probability(vehicle_id)
        if dropoff_probability > self.ev_dropoff_threshold:
            self._set_vehicle_offline_until_next_day(vehicle_id)
            return True
        return False

    def _can_daily_dropout(self, vehicle):
        return (
            vehicle.get('type') == 1
            and vehicle.get('is_online', True)
            and vehicle.get('assigned_request') is None
            and vehicle.get('passenger_onboard') is None
            and vehicle.get('charging_station') is None
        )

    def _refresh_daily_driver_states(self):
        previous_period_dropout_count = self.current_period_dropout_count
        self.current_period_dropout_count = 0
        daily_online = 0
        self._update_all_ev_satisfaction()
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle['idle_timer'] = 0
            was_online = vehicle.get('is_online', True)
            if self.daily_drop_off and self._can_daily_dropout(vehicle):
                self._handle_vehicle_dropout_event(vehicle_id)
            elif not was_online:
                rejoin_probability = self.calculate_rejoin_probability(vehicle_id)
                if random.random() < rejoin_probability:
                    vehicle['is_online'] = True
                    vehicle['offline_until_time'] = None
            if vehicle.get('is_online', True):
                daily_online += 1
            vehicle['daily_salary'] = 0
            if vehicle.get('type') == 1:
                vehicle['salary_ratio'] = 0
        self.current_online = daily_online
        self.daily_online_history.append(daily_online)
        if self.daily_drop_off:
            self.period_dropout_counts.append(previous_period_dropout_count + self.current_period_dropout_count)
            self.current_period_dropout_count = 0
        else:
            self.period_dropout_counts.append(previous_period_dropout_count)


    def calculate_dropoff_probability(self, vehicle_id):
        idle_time = self.vehicles[vehicle_id].get('idle_timer', 0)
        satisfaction = self.vehicles[vehicle_id].get('satisfaction', 0)
        beta_0 = self.ev_dropoff_beta_0
        beta_idle = self.ev_dropoff_beta_idle
        beta_satisfaction = self.ev_dropoff_beta_satisfaction
        logit = beta_0 + beta_idle * idle_time + beta_satisfaction * satisfaction
        if self.ifdropoff:
            dropoff_probability = 1.0 / (1.0 + np.exp(-logit))
        else:
            dropoff_probability = 0
        return dropoff_probability

    def calculate_rejoin_probability(self, vehicle_id): 
        gamma_0 = self.ev_rejoin_gamma_0
        gamme_satisfaction = self.ev_rejoin_gamma_satisfaction
        satisfaction = self.vehicles[vehicle_id].get('satisfaction', 0)
        logit = gamma_0 + gamme_satisfaction * satisfaction
        if self.ifdropoff:
            rejoin_probability = 1.0 / (1.0 + np.exp(-logit))
        else:
            rejoin_probability = 0
        return rejoin_probability







    def _update_environment(self):
        """Update environment state"""
        previous_day_index = self._current_day_index()
        self.current_time += 1
        if self._current_day_index() != previous_day_index:
            self._refresh_daily_driver_states()
        
        # Generate new requests using selected method
        if self.use_intense_requests:
            new_requests = self._generate_intense_requests()  # Now returns a list

        else:
            new_requests = self._generate_random_requests()  # Now also returns a list
        request_num = len(new_requests)
        self.whole_req_num += request_num
        # Update charging status
        
        
        for vehicle_id, vehicle in self.vehicles.items():
            veh_loc = vehicle['location']
            if not vehicle.get('is_online', True):
                vehicle['zone_id'] = None
                continue
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:  
                vehicle['idle_timer'] += 1
            if vehicle['penalty_timer'] > 0:
                vehicle['penalty_timer'] -= 1
            vehicle['zone_id'] = self.get_zone_id(veh_loc) if hasattr(self, 'get_zone_id') else None
                    
        self.zone_vehicle_num = [0 for _ in range(self.num_zones)]
        self.zone_request_num = [0 for _ in range(self.num_zones)]

        busy_request_ids = {
            request_id
            for vehicle in self.vehicles.values()
            for request_id in (
                vehicle.get('assigned_request'),
                vehicle.get('passenger_onboard'),
            )
            if request_id is not None
        }
        for request_id, request in self.active_requests.items():
            if request_id in busy_request_ids:
                continue
            pickup_zone_id = self.get_zone_id(request.pickup) if hasattr(self, 'get_zone_id') else None
            if pickup_zone_id is not None and 0 <= pickup_zone_id < self.num_zones:
                self.zone_request_num[pickup_zone_id] += 1

        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.get('is_online', True):
                continue
            zone_id = vehicle.get('zone_id', None)
            if zone_id is not None and 0 <= zone_id < self.num_zones:
                self.zone_vehicle_num[zone_id] += 1

        self.current_online = sum(1 for vehicle in self.vehicles.values() if vehicle.get('is_online', True))

        self._record_zone_vehicle_snapshot()








        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['charging_station'] is not None:
                vehicle['charging_time_left'] -= 1
                #print(f"DEBUG: Vehicle {vehicle_id} charging at station {vehicle['charging_station']}, time left: {vehicle['charging_time_left']}")
                # Charging increases battery (reduced rate for more realistic charging)
                vehicle['battery'] += self.chargeincrease_per_epoch + np.random.random()*0.001
                vehicle['battery'] = min(1.0, vehicle['battery'])
                
                # Charging complete
                if vehicle['charging_time_left'] <= 0:
                    station_id = vehicle['charging_station']
                    if station_id in self.charging_manager.stations:
                        station = self.charging_manager.stations[station_id]
                        station.stop_charging(str(vehicle_id))
                    session_start = vehicle.get('charging_session_start_time')
                    if session_start is not None:
                        duration_epochs = max(
                            0.0,
                            float(self.current_time) - float(session_start),
                        )
                        duration_minutes = (
                            duration_epochs
                            * 1440.0
                            / max(1.0, float(self.simulation_period))
                        )
                        vehicle.setdefault(
                            'completed_charging_durations_minutes', []
                        ).append(duration_minutes)
                    vehicle['charging_session_start_time'] = None
                    vehicle['charging_station'] = None
                    self.charge_finished += 1
                    self.charge_stats[station_id].append(self.current_time)
                    

        # This fixes the issue where stations auto-start vehicles from queue but don't update vehicle state
        for station_id, station in self.charging_manager.stations.items():
            for charging_vehicle_id in station.current_vehicles:
                vehicle_id = int(charging_vehicle_id)
                if vehicle_id in self.vehicles:
                    vehicle = self.vehicles[vehicle_id]
                    vehicle['charging_target'] = None
                    
                    # If vehicle is in station but doesn't know it's charging, sync the state
                    if vehicle['charging_station'] is None:
                        self._mark_charging_started(vehicle_id, station_id)
                        vehicle['charging_station'] = station_id
                        self._set_vehicle_charging_session(vehicle_id)
                        # Auto-starting the next queued vehicle is a real session start.
                        vehicle['charging_count'] = int(vehicle.get('charging_count', 0)) + 1
            self.idle_charging_num[station_id] = station.max_capacity - len(station.current_vehicles)
        self.current_online = sum(1 for vehicle in self.vehicles.values() if vehicle.get('is_online', True))

        # Remove expired requests and apply unserved penalty
        current_time = self.current_time
        expired_requests = []
        unserved_penalty_total = 0
        
        for request_id, request in self.active_requests.items():
            if current_time > request.pickup_deadline:
                expired_requests.append(request_id)
                # Apply penalty for unserved request
                unserved_penalty_total += self.unserved_penalty

        # Debug info
        if len(expired_requests) > 0 and self.current_time % 50 == 0:
            print(f"⏰ Time {current_time}: {len(expired_requests)} requests expired (pickup_deadline passed)")

        # Distribute unserved penalty among all vehicles
        expired_being_served = 0
        for request_id in expired_requests:
            # 检查订单是否正在被车辆服务（assigned或onboard）
            request_being_served = False
            for vehicle_id, vehicle in self.vehicles.items():
                if (vehicle['assigned_request'] == request_id or 
                    vehicle['passenger_onboard'] == request_id):
                    request_being_served = True
                    expired_being_served += 1
                    # if self.current_time % 50 == 0:
                    #     print(f"   ⚠️  Request {request_id} expired but still being served by vehicle {vehicle_id}")
                    break
            
            # 只删除不在服务中的过期订单
            if not request_being_served:
                request = self.active_requests[request_id]
                #self.rejected_requests.append(request)
                self._ensure_recourse_runtime()
                self.request_lifecycle.record_expiry(int(request_id))
                self.expired_request_ids.add(int(request_id))
                del self.active_requests[request_id]
        
        if expired_being_served > 0 and self.current_time % 50 == 0:
            print(f"   📊 {expired_being_served} expired requests still being served (kept in active_requests)")

        if self.iftest_aev:
            self._enforce_aev_test_capacity_limit()
            self._record_aev_capacity_snapshot()
        


            
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self._ensure_recourse_runtime()
        self.request_lifecycle.reset()
        self.recourse_coordinator.pending = None
        self._pending_recourse_actions = {}
        self._same_epoch_blocked_request_ids = set()
        self._last_offer_realizations = {}
        self._current_ev_stage_request_ids = set()
        self._current_ev_offered_request_ids = set()
        if self.randomize_episode_start_day:
            self.episode_start_day = random.randint(0, self.days_per_week - 1)
        # Reset request system
        self.request_value_sum = 0
        self.request_value_sum_ev = 0
        self.request_value_sum_aev = 0
        self.whole_req = 0
        self.ev_requests = []
        self.active_requests = {}
        self.whole_req_num = 0
        self.completed_requests = []
        self.completed_requests_ev = []
        self.rejected_requests = []
        self.request_counter = 0
        self.current_demand_phase = "legacy"
        self.generated_requests_by_phase = {}
        self.generated_requests_last_step = 0
        self.charge_finished = 0
        self.charge_stats = {station_id: [] for station_id in self.charging_manager.stations}
        self.charging_wait_penalty_total = 0.0
        self.charging_wait_steps = 0
        self.charging_wait_observations = []
        self._charging_queue_arrivals = {}
        self.queue_forecast_filtered_actions = 0
        self.queue_forecast_deferred_charges = 0
        self.queue_forecast_reservation_filtered_actions = 0
        self.demand_forecast_filtered_actions = 0
        self.charging_usage_history = []
        for station in self.charging_manager.stations.values():
            station.current_vehicles.clear()
            station.charging_queue.clear()
            station.charging_queue_notarrived.clear()
            station.available_slots = station.max_capacity
        # Reset rebalancing assignment tracking
        self.rebalancing_assignments_per_step = []
        self.rebalancing_whole = []
        self.total_rebalancing_calls = 0
        self.zone_vehicle_count_history = []
        self.zone_ev_count_history = []
        self.zone_aev_count_history = []
        self.storeactions = {}
        self.storeactions_ev = {}
        self.storeactions_next = {}
        self._ev_default_relocation_cache_step = None
        self._ev_default_relocation_targets = {}
        self._ev_default_relocation_probabilities = {}
        self._prior_features_for_posterior = None
        self._prior_features_for_follower = None
        self._prior_zone_dist_target_for_follower = None
        self._prev_follower_prior_features_for_leader = None
        self._prev_follower_zone_dist_target_for_leader = None
        self._bayes_external_prior = None
        self._bayes_external_posterior = None
        self._prev_follower_external_prior_for_leader = None
        self._prev_follower_external_posterior_for_leader = None
        self._bayes_context_role = None
        self._skip_bayes_distribution_training = False
        self.aev_capacity_history = []
        self.aev_capacity_violations = 0
        self.max_required_idle_aev = 0
        self.max_idle_aev_available = 0
        self.max_observed_active_requests = 0
        self.max_pending_active_requests = 0
        self.aev_capacity_trimmed_requests = 0
        self.completed_service_times_ev = []
        self.completed_service_times_aev = []
        self.ev_rejected_request_ids = set()
        self.ev_rejection_times = {}
        self.ev_rejected_recovered_same_epoch_ids = set()
        self.ev_rejected_rescued_by_aev_ids = set()
        self.ev_rejected_completed_by_ev_ids = set()
        self.expired_request_ids = set()
        if self.iftest_aev:
            self.set_aev_larger_env()
        self._setup_vehicles()
        self.current_online = len(self.vehicles)
        self.daily_online_history = [self.current_online]
        self.period_dropout_counts = []
        self.current_period_dropout_count = 0
        self._record_zone_vehicle_snapshot()
        return self.get_initial_states()
    
    def _charging_session_stats(self):
        simulated_days = float(self.current_time) / max(
            1.0,
            float(self.simulation_period),
        )
        return charging_session_metrics(self.vehicles, simulated_days)

    def get_episode_stats(self):
        """Get detailed statistics for current episode"""
        if not self.daily_online_history:
            self.daily_online_history = [sum(1 for vehicle in self.vehicles.values() if vehicle.get('is_online', True))]

        # Calculate average battery level
        total_battery = sum(v['battery'] for v in self.vehicles.values())
        avg_battery = total_battery / len(self.vehicles) if self.vehicles else 0
        
        # Calculate total rejected requests (unique orders that were rejected)
        total_rejected = len(self.rejected_requests)
        total_ev_request = len(self.ev_requests)
        # Calculate average charging station utilization
        if not hasattr(self, 'charging_manager') or not self.charging_manager.stations:
            #print("DEBUG: No charging manager or stations found!")
            total_capacity = 0
            total_occupied = 0
            avg_station_utilization = 0
            avg_vehicles_per_station = 0
        else:
            total_capacity = sum(station.max_capacity for station in self.charging_manager.stations.values())
            total_occupied = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            
            # Debug: Check vehicle charging status consistency
            vehicles_with_charging_station = [v for v in self.vehicles.values() if v['charging_station'] is not None]
            station_vehicle_count = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            

            if self.charging_usage_history:
                # Calculate average over entire episode
                avg_vehicles_per_station = sum(usage['vehicles_per_station'] for usage in self.charging_usage_history) / len(self.charging_usage_history)
                avg_total_occupied = sum(usage['total_occupied'] for usage in self.charging_usage_history) / len(self.charging_usage_history)
                avg_station_utilization = avg_total_occupied / max(1, total_capacity)
                
                #print(f"DEBUG: Episode charging stats - History points: {len(self.charging_usage_history)}, Avg occupied: {avg_total_occupied:.1f}, Avg per station: {avg_vehicles_per_station:.2f}")
            else:
                # Fallback to current state if no history
                avg_station_utilization = total_occupied / max(1, total_capacity)
                avg_vehicles_per_station = total_occupied / max(1, len(self.charging_manager.stations))
                #print(f"DEBUG: No charging history - using current state: {avg_vehicles_per_station:.2f}")
            

            # Debug: Show vehicles that think they're charging
            charging_vehicles = [vid for vid, v in self.vehicles.items() if v['charging_station'] is not None]
            #print(f"  Vehicles with charging_station set: {charging_vehicles}")
        
        # Count active and completed requests
        active_orders = len(self.active_requests)
        completed_orders = len(self.completed_requests)
        completed_ev_orders = len(self.completed_requests_ev)
        completed_aev_orders = max(0, completed_orders - completed_ev_orders)
        rejected_requests = len({
            request.request_id for request in self.rejected_requests
        })
        recourse_requests = len(self.ev_rejected_recovered_same_epoch_ids)
        unresolved_requests = max(0, int(self.whole_req_num) - completed_orders)
        lost_requests = len(self.expired_request_ids)
        service_ratio = completed_orders/self.whole_req_num if self.whole_req_num > 0 else 0
        avg_request_value1 = self.request_value_sum/completed_orders if completed_orders > 0 else 0
        avg_ev_request_value = self.request_value_sum_ev / completed_ev_orders if completed_ev_orders > 0 else 0
        avg_aev_request_value = self.request_value_sum_aev / completed_aev_orders if completed_aev_orders > 0 else 0
        avg_ev_completion_time = float(np.mean(self.completed_service_times_ev)) if self.completed_service_times_ev else 0.0
        avg_aev_completion_time = float(np.mean(self.completed_service_times_aev)) if self.completed_service_times_aev else 0.0
        ev_rejected_unique = len(self.ev_rejected_request_ids)
        ev_rejected_rescued_by_aev = len(self.ev_rejected_rescued_by_aev_ids)
        ev_rejected_completed_by_ev = len(self.ev_rejected_completed_by_ev_ids)
        ev_rejected_unrescued = max(0, ev_rejected_unique - ev_rejected_rescued_by_aev)
        ev_rejected_rescue_rate = ev_rejected_rescued_by_aev / ev_rejected_unique if ev_rejected_unique > 0 else 1.0
        avg_charging_wait_time = (
            float(np.mean([obs['observed_wait'] for obs in self.charging_wait_observations]))
            if self.charging_wait_observations else 0.0
        )
        wait_metrics = positive_wait_metrics(
            self.charging_wait_observations,
            active_arrivals=getattr(self, '_charging_queue_arrivals', {}),
            current_time=float(self.current_time),
        )
        queue_prediction_errors = [
            (float(obs['predicted_wait']) - float(obs['observed_wait'])) ** 2
            for obs in self.charging_wait_observations
            if obs.get('predicted_wait') is not None
        ]
        queue_predictor_holdout_mse = (
            float(np.mean(queue_prediction_errors)) if queue_prediction_errors else 0.0
        )
        if self.charging_usage_history:
            mean_station_pressure = float(np.mean([
                usage.get('mean_station_pressure', 0.0)
                for usage in self.charging_usage_history
            ]))
            mean_max_station_pressure = float(np.mean([
                usage.get('max_station_pressure', 0.0)
                for usage in self.charging_usage_history
            ]))
            max_station_pressure = float(max(
                usage.get('max_station_pressure', 0.0)
                for usage in self.charging_usage_history
            ))
        else:
            mean_station_pressure = 0.0
            mean_max_station_pressure = 0.0
            max_station_pressure = 0.0
        total_orders = active_orders + completed_orders + total_rejected
        accepted_orders = active_orders + completed_orders
        assignment_success_rate = accepted_orders / total_orders if total_orders > 0 else 0
        completion_rate = completed_orders / total_orders if total_orders > 0 else 0
        
        # Vehicle type breakdown
        ev_vehicles = [v for v in self.vehicles.values() if v['type'] == 1]  # EV
        aev_vehicles = [v for v in self.vehicles.values() if v['type'] == 2]  # AEV
        
        ev_rejected = sum(v['rejected_requests'] for v in ev_vehicles)
        aev_rejected = sum(v['rejected_requests'] for v in aev_vehicles)
        
        # Calculate rebalancing assignment statistics
        avg_rebalancing_assignments = 0
        total_rebalancing_assignments = 0
        avg_rebalance_whole = 0
        if self.rebalancing_assignments_per_step:
            total_rebalancing_assignments = sum(self.rebalancing_assignments_per_step)
            avg_rebalancing_assignments = total_rebalancing_assignments / len(self.rebalancing_assignments_per_step)
            avg_rebalance_whole = sum(self.rebalancing_whole) / len(self.rebalancing_whole) if self.rebalancing_whole else 0

        zone_vehicle_history = self.zone_vehicle_count_history.copy()
        zone_ev_history = self.zone_ev_count_history.copy()
        zone_aev_history = self.zone_aev_count_history.copy()

        if zone_vehicle_history:
            avg_zone_vehicle_counts = np.mean(np.array(zone_vehicle_history, dtype=float), axis=0).tolist()
            last_zone_vehicle_counts = zone_vehicle_history[-1]
        else:
            avg_zone_vehicle_counts = []
            last_zone_vehicle_counts = []

        if zone_ev_history:
            avg_zone_ev_counts = np.mean(np.array(zone_ev_history, dtype=float), axis=0).tolist()
            last_zone_ev_counts = zone_ev_history[-1]
        else:
            avg_zone_ev_counts = []
            last_zone_ev_counts = []

        if zone_aev_history:
            avg_zone_aev_counts = np.mean(np.array(zone_aev_history, dtype=float), axis=0).tolist()
            last_zone_aev_counts = zone_aev_history[-1]
        else:
            avg_zone_aev_counts = []
            last_zone_aev_counts = []

        mean_daily_online = float(np.mean(self.daily_online_history)) if self.daily_online_history else float(self.current_online)
        drop_off_rate = 1.0 - (mean_daily_online / max(1, len(self.vehicles)))
        mean_period_dropout_count = float(np.mean(self.period_dropout_counts)) if self.period_dropout_counts else 0.0
        latest_aev_capacity = self._get_aev_capacity_snapshot()
        feasible_steps = sum(1 for snapshot in self.aev_capacity_history if snapshot['feasible'])
        total_capacity_steps = len(self.aev_capacity_history)
        aev_capacity_feasible_rate = feasible_steps / total_capacity_steps if total_capacity_steps > 0 else 1.0

        stats = {
            'episode_time': self.current_time,
            'episode_days_covered': self._current_day_index() + 1,
            'episode_start_weekday': self.episode_start_day,
            'episode_end_weekday': self._current_weekday_index(),
            'total_orders': total_orders,
            'accepted_orders': accepted_orders,
            'active_orders': active_orders,
            'rejected_orders': total_rejected,
            'rejected_requests': rejected_requests,
            'recourse_requests': recourse_requests,
            'lost_requests': lost_requests,
            'unresolved_requests': unresolved_requests,
            'ev_accept': total_ev_request,
            'completed_orders': completed_orders,
            'completed_ev_orders': completed_ev_orders,
            'completed_aev_orders': completed_aev_orders,
            'service_ratio': service_ratio,
            'avg_request_value': avg_request_value1,
            'avg_completed_order_value': avg_request_value1,
            'avg_ev_completed_order_value': avg_ev_request_value,
            'avg_aev_completed_order_value': avg_aev_request_value,
            'avg_ev_completion_time': avg_ev_completion_time,
            'avg_aev_completion_time': avg_aev_completion_time,
            'ev_rejected_unique_requests': ev_rejected_unique,
            'ev_rejected_rescued_by_aev': ev_rejected_rescued_by_aev,
            'ev_rejected_completed_by_ev': ev_rejected_completed_by_ev,
            'ev_rejected_unrescued_by_aev': ev_rejected_unrescued,
            'ev_rejected_rescue_rate_by_aev': ev_rejected_rescue_rate,
            'assignment_success_rate': assignment_success_rate,
            'completion_rate': completion_rate,
            'avg_battery_level': avg_battery,
            'finished_charge': self.charge_finished,
            **self._charging_session_stats(),
            'charge_stats': self.charge_stats,
            'total_vehicles': len(self.vehicles),
            'online_vehicles': self.current_online,
            'offline_vehicles': max(0, len(self.vehicles) - self.current_online),
            'daily_online_history': self.daily_online_history.copy(),
            'period_dropout_counts': self.period_dropout_counts.copy(),
            'mean_period_dropout_count': mean_period_dropout_count,
            'mean_daily_online_vehicles': mean_daily_online,
            'drop_off_rate': drop_off_rate,
            'ev_count': len(ev_vehicles),
            'aev_count': len(aev_vehicles),
            'ev_rejected': ev_rejected,
            'aev_rejected': aev_rejected,
            'total_stations': len(self.charging_manager.stations),
            'vehicles_charging': len([v for v in self.vehicles.values() if v['charging_station'] is not None]),
            'charging_wait_penalty_total': float(self.charging_wait_penalty_total),
            'charging_wait_penalty_per_step': float(self.charging_wait_penalty_per_step),
            'charging_penalty_per_session': float(self.charging_penalty_per_session),
            'charging_penalty_per_step': float(self.charging_penalty),
            'proactive_charging_max_battery': float(self.proactive_charging_max_battery),
            'queue_forecast_filtered_actions': int(self.queue_forecast_filtered_actions),
            'queue_forecast_deferred_charges': int(self.queue_forecast_deferred_charges),
            'queue_forecast_reservation_filtered_actions': int(
                self.queue_forecast_reservation_filtered_actions
            ),
            'demand_forecast_filtered_actions': int(self.demand_forecast_filtered_actions),
            'charging_wait_steps': int(self.charging_wait_steps),
            'avg_charging_wait_time': avg_charging_wait_time,
            'avg_wait': float(wait_metrics['avg_wait']),
            'waiting_vehicle_count': int(wait_metrics['waiting_vehicle_count']),
            'completed_waiting_vehicle_count': int(
                wait_metrics['completed_waiting_vehicle_count']
            ),
            'ongoing_waiting_vehicle_count': int(
                wait_metrics['ongoing_waiting_vehicle_count']
            ),
            'queue_predictor_holdout_mse': queue_predictor_holdout_mse,
            'queue_predictor_holdout_count': len(queue_prediction_errors),
            'mean_station_pressure': mean_station_pressure,
            'mean_max_station_pressure': mean_max_station_pressure,
            'max_station_pressure': max_station_pressure,
            'charging_wait_observations': list(self.charging_wait_observations),
            # Rebalancing assignment statistics
            'total_rebalancing_calls': self.total_rebalancing_calls,
            'total_rebalancing_assignments': total_rebalancing_assignments,
            'avg_rebalancing_assignments_per_call': avg_rebalancing_assignments,
            'avg_rebalancing_assignments_per_whole': avg_rebalance_whole,
            'rebalancing_assignments_per_step': self.rebalancing_assignments_per_step.copy(),
            'rebalance_whole': self.rebalancing_whole.copy(),
            'avg_zone_vehicle_counts': avg_zone_vehicle_counts,
            'avg_zone_ev_counts': avg_zone_ev_counts,
            'avg_zone_aev_counts': avg_zone_aev_counts,
            'last_zone_vehicle_counts': last_zone_vehicle_counts,
            'last_zone_ev_counts': last_zone_ev_counts,
            'last_zone_aev_counts': last_zone_aev_counts,
            'zone_vehicle_count_history': zone_vehicle_history,
            'zone_ev_count_history': zone_ev_history,
            'zone_aev_count_history': zone_aev_history,
            'iftest_aev': self.iftest_aev,
            'aev_test_request_generation_rate': self.aev_test_request_generation_rate,
            'aev_test_request_rate_scale': self.aev_test_request_rate_scale,
            'aev_test_request_generation_rate_override': self.aev_test_request_generation_rate_override,
            'aev_max_service_time': latest_aev_capacity['maximum_service_time'],
            'aev_pending_active_current': latest_aev_capacity['pending_active_requests'],
            'aev_required_idle_current': latest_aev_capacity['required_idle_aev'],
            'aev_idle_current': latest_aev_capacity['idle_aev'],
            'aev_capacity_feasible_current': latest_aev_capacity['feasible'],
            'aev_capacity_violations': self.aev_capacity_violations,
            'aev_capacity_feasible_rate': aev_capacity_feasible_rate,
            'aev_capacity_checked_steps': total_capacity_steps,
            'aev_capacity_trimmed_requests': self.aev_capacity_trimmed_requests,
            'max_required_idle_aev': self.max_required_idle_aev,
            'max_idle_aev_available': self.max_idle_aev_available,
            'max_observed_active_requests': self.max_observed_active_requests,
            'max_pending_active_requests': self.max_pending_active_requests,
            'synthetic_demand_profile': self.synthetic_demand_profile,
            'current_demand_phase': self.current_demand_phase,
            'generated_requests_last_step': int(self.generated_requests_last_step),
            'generated_requests_by_phase': dict(self.generated_requests_by_phase),
            'total_generated_requests': int(self.whole_req_num),
            'expired_request_count': len(self.expired_request_ids),
            'request_lifecycle_accounted_count': (
                active_orders + completed_orders + len(self.expired_request_ids)
            ),
            'request_lifecycle_gap': int(self.whole_req_num) - (
                active_orders + completed_orders + len(self.expired_request_ids)
            ),
        }
        self._ensure_recourse_runtime()
        self.request_lifecycle.assert_reconciled()
        stats.update(self.request_lifecycle.metrics())
        return stats

    def get_stats(self):
        """Get environment statistics including request fulfillment and vehicle types"""
        total_battery = sum(v['battery'] for v in self.vehicles.values())
        avg_battery = total_battery / len(self.vehicles)
        
        total_charging = sum(v['charging_count'] for v in self.vehicles.values())
        total_rejected = sum(v['rejected_requests'] for v in self.vehicles.values())
        
        # Vehicle type statistics
        ev_vehicles = [v for v in self.vehicles.values() if v['type'] == 1]  # EV
        aev_vehicles = [v for v in self.vehicles.values() if v['type'] == 2]  # AEV
        
        ev_rejected = sum(v['rejected_requests'] for v in ev_vehicles)
        aev_rejected = sum(v['rejected_requests'] for v in aev_vehicles)
        
        vehicles_with_requests = len([v for v in self.vehicles.values() 
                                    if v['assigned_request'] is not None or v['passenger_onboard'] is not None])
        
        return {
            'average_battery': avg_battery,
            'total_charging_events': total_charging,
            **self._charging_session_stats(),
            'vehicles_charging': len([v for v in self.vehicles.values() 
                                    if v['charging_station'] is not None]),
            'online_vehicles': sum(1 for v in self.vehicles.values() if v.get('is_online', True)),
            'offline_vehicles': sum(1 for v in self.vehicles.values() if not v.get('is_online', True)),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'completed_orders_req': self.request_value_sum/len(self.completed_requests) if len(self.completed_requests) > 0 else 0,
            'vehicles_with_requests': vehicles_with_requests,
            'request_fulfillment_rate': len(self.completed_requests) / max(1, len(self.completed_requests) + len(self.active_requests)),
            'total_rejected_requests': total_rejected,
            'rejected_requests': len({
                request.request_id for request in self.rejected_requests
            }),
            'recourse_requests': len(self.ev_rejected_recovered_same_epoch_ids),
            'lost_requests': len(self.expired_request_ids),
            'unresolved_requests': max(
                0, int(self.whole_req_num) - len(self.completed_requests)
            ),
            'ev_count': len(ev_vehicles),
            'aev_count': len(aev_vehicles),
            'ev_rejected': ev_rejected,
            'aev_rejected': aev_rejected,
            'ev_rejection_rate': ev_rejected / max(1, ev_rejected + len(self.completed_requests)) if ev_vehicles else 0,
            'aev_rejection_rate': aev_rejected / max(1, aev_rejected + len(self.completed_requests)) if aev_vehicles else 0,
            'synthetic_demand_profile': self.synthetic_demand_profile,
            'current_demand_phase': self.current_demand_phase,
            'generated_requests_last_step': int(self.generated_requests_last_step),
            'generated_requests_by_phase': dict(self.generated_requests_by_phase),
            'total_generated_requests': int(self.whole_req_num),
        }
    
    def save_time_stats(self, file_path=None):
        """Save time statistics to Excel/CSV file with averages
        
        Args:
            file_path: Path to save the time statistics. If None, saves to results/time_analysis/
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        if not self.record_time:
            print("Warning: record_time is False, no time statistics recorded")
            return
        
        # Calculate statistics for each metric
        stats_summary = {}
        for key, values in self.time_stats.items():
            if values:
                stats_summary[key] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5
                }
            else:
                stats_summary[key] = {
                    'count': 0,
                    'mean': 0,
                    'min': 0,
                    'max': 0,
                    'std': 0
                }
        
        # Create DataFrame for summary statistics
        df_summary = pd.DataFrame(stats_summary).T
        df_summary.index.name = 'Metric'
        
        # Create DataFrame for detailed time series
        max_len = max(len(v) for v in self.time_stats.values()) if self.time_stats else 0
        df_details = pd.DataFrame({
            key: values + [None] * (max_len - len(values))
            for key, values in self.time_stats.items()
        })
        
        # Determine file path
        if file_path is None:
            os.makedirs('results/time_analysis', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'results/time_analysis/time_stats_{timestamp}.xlsx'
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary')
            df_details.to_excel(writer, sheet_name='Details', index=False)
        
        print(f"Time statistics saved to: {file_path}")
        
        # Print summary to console
        print("\n=== Time Statistics Summary ===")
        print(df_summary.to_string())
        
        # Print Q-value matrix dimension analysis if available
        if self.time_stats.get('qvalue_scale_1') and self.time_stats.get('qvalue_scale_2'):
            scale_1_values = self.time_stats['qvalue_scale_1']
            scale_2_values = self.time_stats['qvalue_scale_2']
            avg_dim1 = sum(scale_1_values) / len(scale_1_values)
            avg_dim2 = sum(scale_2_values) / len(scale_2_values)
            print(f"\n=== Q-value Matrix Dimensions ===")
            print(f"Dimension 1 (vehicles): avg={avg_dim1:.1f}, min={min(scale_1_values)}, max={max(scale_1_values)}")
            print(f"Dimension 2 (actions):  avg={avg_dim2:.1f}, min={min(scale_2_values)}, max={max(scale_2_values)}")
            print(f"Average matrix size: {avg_dim1:.1f} × {avg_dim2:.1f} = {avg_dim1 * avg_dim2:.0f} elements")
        
        return df_summary
        
    
        
    def simulate_motion_dqn(self, dqn_agent=None, current_requests: List[Request] = None, training=True):
        """
        DQN-based simulation for vehicle dispatch as benchmark comparison to ILP-ADP.
        存储的 transition 以“动作完成”为 done（接单完成/失败、充电完成、wait/idle 单步）。
        """
        # Import lightweight DQN utilities without changing other components
        try:
            from .ValueFunction_pytorch import DQNAgent, create_dqn_state_features
        except Exception:
            print("Warning: DQN components not available. Please ensure ValueFunction_pytorch.py defines DQNAgent and create_dqn_state_features.")
            return None
        actions = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None:
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and ((random.random() < p_charge) or vehicle['battery'] <= 0.2):
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                else:
                    # EV declined charging: set no-charge cooldown for 5 time steps
                    vehicle['no_charge_cooldown_until'] = self.current_time + 5
  
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        if leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v.get('is_online', True) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_wait = [vehicle_id for vehicle_id, v in self.vehicles.items() if v['is_stationary']==True and v not in idle_vehicles_1 and v not in idle_vehicles_2 and  v['penalty_timer']==0]
            idle_vehicles_v = [vehicle_id for vehicle_id, v in self.vehicles.items() if self._is_ev(vehicle_id) and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['idle_target'] is not None and v not in idle_vehicles_2 and v not in idle_vehicles_1 and v not in idle_vehicles_wait and  v['penalty_timer']==0]
            # idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_wait+idle_vehicles_v
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            vehicles_to_rebalance = [
                vehicle_id for vehicle_id in vehicles_to_rebalance
                if not self._is_vehicle_committed_to_charging(vehicle_id)
            ]
            vehicles_to_rebalance = [
                vehicle_id
                for vehicle_id in vehicles_to_rebalance
                if self.vehicles[vehicle_id]['assigned_request'] is None
                and self.vehicles[vehicle_id]['passenger_onboard'] is None
                and self.vehicles[vehicle_id]['charging_station'] is None
                and (
                    self.vehicles[vehicle_id]['target_location'] is None
                    or self._is_ev(vehicle_id)
                )
            ]
        
        
        
        
        
        # 生成车辆动作可行性矩阵和Q值矩阵
        if self.adp_value > 0 and self.value_function is not None:
            batch_q_value, vehicle_action_matrix, (num_requests, num_stations, num_zones) = self.generate_vehicle_qvalue_dqn(vehicles_to_rebalance)
        else:
            batch_q_value = self.generate_vehicle_qvalue_withoutqnetwork(vehicles_to_rebalance)
            vehicle_action_matrix, num_requests, num_stations, num_zones = self.generate_whole_matrix(
                vehicles_to_rebalance, rebalance_num=len(vehicles_to_rebalance)
            )
        
        # 获取可用请求和充电站列表
        assigned_requests = []
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id]['assigned_request'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['assigned_request'])
            if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                assigned_requests.append(self.vehicles[vehicle_id]['passenger_onboard'])
        available_requests = list(self.active_requests.values())
        available_requests = [
            req
            for req in available_requests
            if req.request_id not in assigned_requests
            and req.request_id not in set(
                getattr(self, '_same_epoch_blocked_request_ids', set())
            )
        ]
        station_list = list(self.charging_manager.stations.values()) if hasattr(self, 'charging_manager') else []
        

        # Instantiate a default agent if needed (benchmark use only)
        if dqn_agent is None:
            device = 'cuda' if hasattr(self, 'device') else 'cpu'
            dqn_agent = DQNAgent(state_dim=64, action_dim=32, device=device)

        results = {
            'total_reward': 0.0,
            'actions_taken': [],
            'vehicle_utilization': 0.0,
            'request_completion_rate': 0.0,
            'average_battery_level': 0.0,
            'dqn_decisions': [],
            'transitions_added': 0
        }

        # Snapshot counts for metrics
        completed_before = len(self.completed_requests)

        # Prepare per-action buffer (persist across ticks)
        if not hasattr(self, '_dqn_action_buffers'):
            self._dqn_action_buffers = {}

        # 1) Progress ongoing actions for vehicles already in buffer
        for vehicle_id, buf in list(self._dqn_action_buffers.items()):
            v = self.vehicles.get(vehicle_id)
            if not v:
                del self._dqn_action_buffers[vehicle_id]
                continue

            a_type = buf.get('action_type', 'idle')
            step_reward = 0.0
            action_done = False

            if a_type == 'assign':
                # Continue towards pickup/dropoff
                if self._pickup_passenger(vehicle_id):
                    step_reward += 0.5 + np.random.normal(0, 0.2)
                else:
                    step_reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)
                # Try dropoff if onboard
                if v.get('passenger_onboard') is not None:
                    drop_r = self._dropoff_passenger(vehicle_id)
                    if drop_r > 0:
                        step_reward += drop_r + np.random.normal(0, 0.2)
                        # Mark successful completion for this buffered assign
                        buf['dropoff_done'] = True
                    else:
                        step_reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)
                # Done when vehicle becomes free (no assigned and no onboard)
                if v.get('assigned_request') is None and v.get('passenger_onboard') is None:
                    # If we became free without a recorded dropoff, treat as assignment failure
                    if not buf.get('dropoff_done', False):
                        active_requests_count = len(self.active_requests)
                        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
                        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
                        step_reward += -avg_request_value*0.01
                    action_done = True

            elif a_type == 'charge':
                station_id = buf.get('station_id')
                if v.get('charging_station') is None:
                    # Move towards station / try start charging
                    vloc = v.get('location', 0)
                    # Ensure station_id is valid; remap to nearest if missing
                    if not hasattr(self, 'charging_manager') or not getattr(self.charging_manager, 'stations', None):
                        action_done = True  # no stations; consider action finished
                    else:
                        stations = self.charging_manager.stations
                        if station_id not in stations:
                            best_sid, best_d = None, 1e9
                            for sid, st in stations.items():
                                d = self._manhattan_distance_loc(vloc, st.location)
                                if d < best_d:
                                    best_sid, best_d = sid, d
                            station_id = best_sid if best_sid is not None else list(stations.keys())[0]
                            buf['station_id'] = station_id
                        self._move_vehicle_to_charging_station(vehicle_id, station_id)
                        step_reward += self._execute_movement_towards_charging_station(vehicle_id, station_id)
                # Detect completion: charging_duration == 0 if present; else charging_time_left <= 0; or finished charging (was charging -> now not charging)
                cd = v.get('charging_duration', None)
                ctl = v.get('charging_time_left', None)
                finished_by_time = (cd == 0) if cd is not None else (ctl is not None and ctl <= 0)
                if finished_by_time:
                    action_done = True
                else:
                    if buf.get('was_charging', False) and v.get('charging_station') is None:
                        action_done = True
                    if v.get('charging_station') is not None:
                        buf['was_charging'] = True

            # wait/idle are stored immediately at creation; buffers shouldn't exist for them

            # Accumulate and finalize if needed
            buf['acc_reward'] = buf.get('acc_reward', 0.0) + float(step_reward)
            results['total_reward'] += float(step_reward)

            if action_done and training and hasattr(dqn_agent, 'store_transition'):
                next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                # Ensure action index is within DQN agent's action dimension bounds
                action_dim = getattr(dqn_agent, 'action_dim', 32)
                safe_action_idx = min(int(buf['action_idx']), action_dim - 1)
                dqn_agent.store_transition(buf['state'], safe_action_idx, float(buf['acc_reward']), next_state, done=True)
                results['transitions_added'] += 1
                del self._dqn_action_buffers[vehicle_id]

        # 2) 对vehicles_to_rebalance中的车辆使用贪心Q值选择
        # 区分EV和AEV车辆列表（类似6562行逻辑）
        ev_vehicles_to_rebalance = [vid for vid in vehicles_to_rebalance if self.vehicles[vid]['type'] == 1]
        aev_vehicles_to_rebalance = [vid for vid in vehicles_to_rebalance if self.vehicles[vid]['type'] == 2]
        
        # 维护本轮已分配的订单集合，避免重复分配
        assigned_requests_this_step = set()
        num_requests = len(available_requests)
        for i, vehicle_id in enumerate(vehicles_to_rebalance):
            if vehicle_id in self._dqn_action_buffers:
                continue
            
            v = self.vehicles[vehicle_id]
            # Skip busy vehicles
            is_free = (v.get('assigned_request') is None and
                       v.get('passenger_onboard') is None and
                       v.get('charging_station') is None)
            if not is_free:
                continue

            # 获取该车辆的Q值向量（拷贝以便修改）
            vehicle_q_values = batch_q_value[i, :].copy()
            
            # 将已分配的订单的Q值设为负无穷，避免重复选择
            for assigned_req_idx in range(num_requests):
                req = available_requests[assigned_req_idx]
                if req.request_id in assigned_requests_this_step:
                    vehicle_q_values[assigned_req_idx] = -np.inf
            
            # 贪心选择：选择Q值最大的可行动作
            best_action_idx = np.argmax(vehicle_q_values)
            best_q_value = vehicle_q_values[best_action_idx]
            
            # 将action_idx映射到环境动作
            if best_action_idx < num_requests:
                request = available_requests[best_action_idx]
                env_action = {
                    'type': 'assign',
                    'request_id': request.request_id,
                    'pickup': request.pickup,
                    'dropoff': request.dropoff
                }
                # 标记该订单已在本轮分配，避免其他车辆重复选择
                assigned_requests_this_step.add(request.request_id)
            elif best_action_idx < num_requests + num_stations:
                # 去充电站
                station_idx = best_action_idx - num_requests
                if station_idx < len(station_list):
                    station = station_list[station_idx]
                    env_action = {
                        'type': 'charge',
                        'station_id': station.id,
                        'location': station.location
                    }
                else:
                    env_action = {'type': 'idle'}
            elif best_action_idx < num_requests + num_stations + num_zones:
                # 重定位到hotspot
                zone_idx = best_action_idx - num_requests - num_stations
                if zone_idx < len(self.hotspot_locations):
                    zone_coords = self.hotspot_locations[zone_idx]
                    zone_location = zone_coords[1] * self.grid_size + zone_coords[0]
                    env_action = {
                        'type': 'relocate',
                        'target_location': zone_location,
                        'zone_id': zone_idx
                    }
                else:
                    env_action = {'type': 'idle'}
            else:
                # 等待/idle
                env_action = {'type': 'idle'}
            
            # Build DQN state features for transition storage
            state = create_dqn_state_features(self, vehicle_id, self.current_time)
            
            # Execute action via DQN executor
            reward = self._execute_dqn_action(vehicle_id, env_action, available_requests)

            # Record
            results['total_reward'] += float(reward)
            results['actions_taken'].append({
                'vehicle_id': vehicle_id,
                'action_idx': int(best_action_idx),
                'env_action': env_action,
                'reward': float(reward),
                'q_value': float(best_q_value)
            })

            # Store transition logic based on action completeness semantics
            if training and dqn_agent and hasattr(dqn_agent, 'store_transition'):
                # Get DQN agent's action dimension to ensure index is within bounds
                action_dim = getattr(dqn_agent, 'action_dim', 32)
                safe_action_idx = min(int(best_action_idx), action_dim - 1)
                
                a_type = env_action.get('type', 'idle')
                if a_type == 'assign':
                    # If assignment failed this step (仍然空闲且没有乘客)，立即存储并结束
                    if v.get('assigned_request') is None and v.get('passenger_onboard') is None:
                        next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                        dqn_agent.store_transition(state, safe_action_idx, float(reward), next_state, done=True)
                        results['transitions_added'] += 1
                    else:
                        # 成功接单：进入缓冲，累计到 dropoff 完成
                        self._dqn_action_buffers[vehicle_id] = {
                            'state': state,
                            'action_idx': safe_action_idx,
                            'env_action': env_action,
                            'action_type': 'assign',
                            'acc_reward': float(reward)
                        }
                elif a_type == 'charge':
                    # 充电：进入缓冲，直到 charging_duration==0 或 charging_time_left<=0 完成
                    self._dqn_action_buffers[vehicle_id] = {
                        'state': state,
                        'action_idx': safe_action_idx,
                        'env_action': env_action,
                        'action_type': 'charge',
                        'station_id': env_action.get('station_id'),
                        'acc_reward': float(reward),
                        'was_charging': v.get('charging_station') is not None
                    }
                else:
                    # wait/idle 单步即完成
                    next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                    dqn_agent.store_transition(state, safe_action_idx, float(reward), next_state, done=True)
                    results['transitions_added'] += 1

        # Advance environment by one tick to keep parity with built-in step()
        self._update_environment()

        # Compute benchmark metrics from current env
        self._calculate_dqn_performance_metrics(results, current_requests)

        # Snapshot deltas
        results['orders_completed_delta'] = max(0, len(self.completed_requests) - completed_before)

        return results
    

    def _execute_dqn_action(self, vehicle_id, action, current_requests):
        """
        Execute DQN action in environment and return reward
        
        Args:
            vehicle_id: ID of the vehicle
            action: Action to execute
            current_requests: Available requests
        
        Returns:
            float: Reward for the action
        """
        vehicle = self.vehicles.get(vehicle_id)
        if not vehicle:
            return -1.0

        a_type = action.get('type', 'idle')

        # Helper to safely extract request from provided mapping or index
        def _resolve_request(req_id_or_index):
            # Prefer direct id lookup
            if isinstance(req_id_or_index, (int, str)) and req_id_or_index in self.active_requests:
                return self.active_requests[req_id_or_index]
            # Fallback: index into current_requests if valid
            if isinstance(req_id_or_index, int) and 0 <= req_id_or_index < len(current_requests or []):
                candidate = current_requests[req_id_or_index]
                rid = getattr(candidate, 'request_id', None)
                if rid in self.active_requests:
                    return self.active_requests[rid]
                return candidate
            return None

        # Assign/serve: accept one existing order
        if a_type == 'assign':
            req_id = action.get('request_id')
            # Also accept alternate keys from mapper
            if req_id is None:
                req_id = action.get('req_id')
            req = _resolve_request(req_id if req_id is not None else 0)
            if req is None:
                return -0.5  # No valid request to accept

            # Ensure the id we use exists in active_requests
            rid = getattr(req, 'request_id', None)
            if rid is None or rid not in self.active_requests:
                return -0.5

            # Attempt to assign (may reject based on EV behaviour)
            accepted = self._assign_request_to_vehicle(vehicle_id, rid)
            if not accepted:
                # Vehicle chose to reject; mark stationary penalty via stationary branch
                vehicle['is_stationary'] = True
                vehicle['stationary_duration'] = 1
                active_requests_count = len(self.active_requests)
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
                vehicle['waiting_for_requests'] = True
                return 0

            # Move/pickup/dropoff using existing helpers
            reward = 0.0
            if self._pickup_passenger(vehicle_id):
                reward += 0.5 + np.random.normal(0, 0.2)
            else:
                # Move one step toward target (pickup or dropoff)
                reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)

            # If passenger onboard after movement, attempt dropoff
            if vehicle.get('passenger_onboard') is not None:
                drop_reward = self._dropoff_passenger(vehicle_id)
                if drop_reward > 0:
                    reward += drop_reward + np.random.normal(0, 0.2)
                else:
                    reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
            return float(reward)

        # Charge: send to a station and let charging start when reached
        if a_type == 'charge' and vehicle.get('type') == 1:
            # Choose station: prefer provided id else nearest available
            station_id = action.get('station_id', None)
            if not hasattr(self, 'charging_manager') or not self.charging_manager.stations:
                return -1.0
            stations = self.charging_manager.stations
            if station_id not in stations:
                # Pick nearest station by manhattan distance
                vloc = vehicle.get('location', 0)
                best_sid, best_d = None, 1e9
                for sid, st in stations.items():
                    d = self._manhattan_distance_loc(vloc, st.location)
                    if d < best_d:
                        best_sid, best_d = sid, d
                station_id = best_sid if best_sid is not None else list(stations.keys())[0]

            # Set charging goal and move one step
            self._move_vehicle_to_charging_station(vehicle_id, station_id)
            reward = self._execute_movement_towards_charging_station(vehicle_id, station_id)
            vehicle['waiting_for_requests'] = False
            return float(reward)

        # Wait: stay in place for one step with small opportunity penalty
        if a_type == 'wait':
            vehicle['is_stationary'] = True
            vehicle['stationary_duration'] = int(action.get('duration', 1))
            active_requests_count = len(self.active_requests)
            active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
            avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 50.0
            vehicle['waiting_for_requests'] = True
            return float(-avg_request_value * 0.01)

        # Idle: move toward a simple idle target for exploration
        # If rebalance target provided, treat as idle to that target
        if a_type in ('idle', 'rebalance'):
            battery_level = vehicle.get('battery', 1.0)
            if battery_level - 2*self.battery_consum>=self.min_battery_level:
                if a_type == 'rebalance' and 'target_location' in action:
                    # Convert target index to coordinates
                    tloc = int(action['target_location'])
                    tx, ty = (tloc % self.grid_size, tloc // self.grid_size)
                    vehicle['idle_target'] = (tx, ty)
                else:
                    # Get nearest idle hotspot target
                    idle_target = self.return_nearest_idle_target(vehicle_id)
                    if idle_target is not None:
                        vehicle['idle_target'] = idle_target
                        vehicle['target_location'] = idle_target
                reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target'))
                vehicle['waiting_for_requests'] = False
            else:
                vehicle['idle_target'] = None
                vehicle['target_location'] = None
                reward = -0.2  # Penalty for idling with low battery
            return float(reward)

        # Default fallback
        vehicle['waiting_for_requests'] = False
        return -0.1
    
    def _update_storeaction(self, vehicle_id, action, storeactions_dict, is_ev=False):
        """
        封装storeaction更新逻辑，避免代码重复
        
        Args:
            vehicle_id: 车辆ID
            action: 新的action对象
            storeactions_dict: storeactions或storeactions_ev字典
            is_ev: 是否为EV车辆
        """
        vehicle = self.vehicles[vehicle_id]
        if isinstance(action, ChargingAction):
            queue_features = vehicle.get('charging_decision_queue_features')
            action.queue_features = (
                list(queue_features) if queue_features is not None else None
            )
            action.target_station_id = int(action.charging_station_id)
            action.queue_decision_time = float(
                vehicle.get('charging_decision_time', self.current_time)
            )
        vehicle_location = vehicle['location']
        vehicle_battery = vehicle['battery']
        target_coords = vehicle.get('target_location')
        
        # 获取对应的全局存储
        global_store = self.storeactions_ev if is_ev else self.storeactions

        if is_ev:
            pending_action = global_store.get(vehicle_id)
            if (
                pending_action is not None
                and isinstance(pending_action, ServiceAction)
                and getattr(pending_action, 'awaiting_new_assignment', False)
                and not isinstance(action, ServiceAction)
            ):
                storeactions_dict[vehicle_id] = pending_action
                return
        
        if storeactions_dict[vehicle_id] is None:
            # 首次创建action
            storeactions_dict[vehicle_id] = action
            global_store[vehicle_id] = action
            global_store[vehicle_id].dur_reward = 0
            global_store[vehicle_id].current_time = self.current_time
            global_store[vehicle_id].target_location = target_coords
        else:
            # 更新已有action
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].next_action.next_value = 0
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            
            # 替换为新action
            global_store[vehicle_id] = action
            global_store[vehicle_id].dur_reward = 0
            global_store[vehicle_id].dur_time = self.current_time - old_current_time
            global_store[vehicle_id].current_time = self.current_time
            global_store[vehicle_id].target_location = target_coords
    

    def _calculate_dqn_performance_metrics(self, results, current_requests):
        """
        Calculate performance metrics for DQN simulation
        
        Args:
            results: Simulation results dictionary to update
            current_requests: List of current requests
        """
        # Vehicle utilization: vehicles engaged in any activity (assigned, onboard, charging)
        total_vehicles = len(self.vehicles)
        engaged = sum(1 for v in self.vehicles.values() if (
            v.get('assigned_request') is not None or
            v.get('passenger_onboard') is not None or
            v.get('charging_station') is not None
        ))
        results['vehicle_utilization'] = engaged / max(1, total_vehicles)

        # Request completion rate based on environment accounting
        completed = len(self.completed_requests)
        total_requests = completed + len(self.active_requests) + len(self.rejected_requests)
        results['request_completion_rate'] = completed / max(1, total_requests)

        # Average EV battery level
        evs = [v for v in self.vehicles.values() if v.get('type') == 1]
        results['average_battery_level'] = (sum(v.get('battery', 1.0) for v in evs) / len(evs)) if evs else 1.0

        # Action distribution inferred from chosen DQN indices if available
        act_indices = [a.get('dqn_action') for a in results.get('actions_taken', []) if 'dqn_action' in a]
        if act_indices:
            n = len(act_indices)
            results['action_distribution'] = {
                'assign': sum(1 for a in act_indices if a is not None and a < 10) / n,
                'rebalance': sum(1 for a in act_indices if a is not None and 10 <= a < 20) / n,
                'charge': sum(1 for a in act_indices if a is not None and 20 <= a < 25) / n,
                'wait': sum(1 for a in act_indices if a is not None and 25 <= a < 28) / n,
                'idle': sum(1 for a in act_indices if a is not None and a >= 28) / n,
            }
        else:
            results['action_distribution'] = {'assign': 0, 'rebalance': 0, 'charge': 0, 'wait': 0, 'idle': 1}

        # Waiting stats
        waiting = sum(1 for v in self.vehicles.values() if v.get('is_stationary', False) or v.get('waiting_for_requests', False))
        results['vehicles_waiting'] = waiting
        results['wait_utilization'] = waiting / max(1, total_vehicles)

        return results

    def _save_training_dataset(self, value_function):
        """
        保存Q-network训练的experience数据集到本地
        """
        import json
        import pickle
        import os
        from datetime import datetime
        
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            print("⚠️ No experience buffer found or empty buffer")
            return
        
        # 创建保存目录
        save_dir = "results/training_datasets"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换经验数据为可保存格式
        experiences = list(value_function.experience_buffer)
        dataset = {
            'timestamp': timestamp,
            'current_time': self.current_time,
            'dataset_size': len(experiences),
            'experiences': experiences,
            'environment_info': {
                'grid_size': self.grid_size,
                'num_vehicles': self.NUM_AGENTS,
                'num_charging_stations': len(self.charging_stations) if hasattr(self, 'charging_stations') else 0
            }
        }
        
        # 保存为pickle文件（用于后续训练）
        pickle_file = f"{save_dir}/training_dataset_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # 保存为JSON文件（便于查看和分析）
        json_file = f"{save_dir}/training_dataset_{timestamp}.json"
        # 将numpy类型转换为Python原生类型以便JSON序列化
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            else:
                return obj
        
        json_dataset = convert_for_json(dataset)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training dataset saved:")
        print(f"   📁 Pickle file: {pickle_file}")
        print(f"   📄 JSON file: {json_file}")
        print(f"   📊 Dataset size: {len(experiences)} experiences")
        
    def _analyze_q_value_issues(self, value_function):
        """
        分析为什么接受订单的Q-value比idle还要小的问题
        """
        print("\n🔍 Analyzing Q-value issues (accept vs idle)...")
        
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            print("⚠️ No experience buffer found")
            return
        
        experiences = list(value_function.experience_buffer)
        
        # 分离不同类型的动作
        assign_experiences = [exp for exp in experiences if exp['action_type'].startswith('assign')]
        idle_experiences = [exp for exp in experiences if exp['action_type'] == 'idle']
        charge_experiences = [exp for exp in experiences if exp['action_type'].startswith('charge')]
        
        print(f"📈 Action type distribution:")
        print(f"   Assign actions: {len(assign_experiences)}")
        print(f"   Idle actions: {len(idle_experiences)}")
        print(f"   Charge actions: {len(charge_experiences)}")
        
        if len(assign_experiences) > 0 and len(idle_experiences) > 0:
            # 计算奖励统计
            assign_rewards = [exp['reward'] for exp in assign_experiences]
            idle_rewards = [exp['reward'] for exp in idle_experiences]
            
            assign_mean = sum(assign_rewards) / len(assign_rewards)
            idle_mean = sum(idle_rewards) / len(idle_rewards)
            
            assign_positive = len([r for r in assign_rewards if r > 0])
            idle_positive = len([r for r in idle_rewards if r > 0])
            
            print(f"\n🎯 Reward Analysis:")
            print(f"   Assign - Mean: {assign_mean:.3f}, Positive: {assign_positive}/{len(assign_rewards)} ({assign_positive/len(assign_rewards)*100:.1f}%)")
            print(f"   Idle   - Mean: {idle_mean:.3f}, Positive: {idle_positive}/{len(idle_rewards)} ({idle_positive/len(idle_rewards)*100:.1f}%)")
            
            # 计算当前Q值
            sample_vehicle_id = 0
            sample_location = 50  # 网格中心位置
            sample_time = self.current_time
            
            try:
                # 获取sample state下的Q值
                assign_q = value_function.get_q_value(
                    vehicle_id=sample_vehicle_id,
                    action_type="assign_0",
                    vehicle_location=sample_location,
                    target_location=sample_location + 10,
                    current_time=sample_time,
                    battery_level=0.8,
                    request_value=10.0
                )
                
                idle_q = value_function.get_q_value(
                    vehicle_id=sample_vehicle_id,
                    action_type="idle",
                    vehicle_location=sample_location,
                    target_location=sample_location,
                    current_time=sample_time,
                    battery_level=0.8,
                    request_value=0.0
                )
                
                print(f"\n🧠 Current Q-values (sample state):")
                print(f"   Assign Q-value: {assign_q:.3f}")
                print(f"   Idle Q-value:   {idle_q:.3f}")
                print(f"   Difference:     {assign_q - idle_q:.3f}")
                
                if assign_q < idle_q:
                    print(f"⚠️  ISSUE DETECTED: Assign Q-value is lower than idle!")
                    
                    # 分析可能的原因
                    print(f"\n🔍 Possible causes:")
                    print(f"   1. Assign actions getting more negative rewards: {assign_mean < idle_mean}")
                    print(f"   2. Idle actions more consistently positive: {idle_positive/len(idle_rewards) > assign_positive/len(assign_rewards) if len(assign_rewards) > 0 else False}")
                    print(f"   3. Training imbalance - more negative assign examples")
                    
                    # 分析距离对奖励的影响
                    assign_with_distance = [(exp['reward'], abs(exp['vehicle_location'] - exp['target_location'])) 
                                          for exp in assign_experiences 
                                          if 'vehicle_location' in exp and 'target_location' in exp]
                    
                    if assign_with_distance:
                        avg_distance = sum(d[1] for d in assign_with_distance) / len(assign_with_distance)
                        high_distance_rewards = [r for r, d in assign_with_distance if d > avg_distance]
                        low_distance_rewards = [r for r, d in assign_with_distance if d <= avg_distance]
                        
                        print(f"\n📏 Distance analysis for assign actions:")
                        print(f"   Average distance: {avg_distance:.1f}")
                        if high_distance_rewards:
                            print(f"   High distance rewards: {sum(high_distance_rewards)/len(high_distance_rewards):.3f}")
                        if low_distance_rewards:
                            print(f"   Low distance rewards:  {sum(low_distance_rewards)/len(low_distance_rewards):.3f}")
                
            except Exception as e:
                print(f"❌ Error calculating Q-values: {e}")
        
        else:
            print("⚠️ Not enough data for both assign and idle actions")

    def _quick_q_value_analysis(self, value_function):
        """
        快速Q-value分析 - 每50步运行一次，检查Q-value趋势和矛盾
        """
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            return
        
        experiences = list(value_function.experience_buffer)
        
        # 只分析最近100个experience
        recent_experiences = experiences[-100:] if len(experiences) > 100 else experiences
        
        # 快速统计
        assign_rewards = [exp['reward'] for exp in recent_experiences if exp['action_type'].startswith('assign')]
        idle_rewards = [exp['reward'] for exp in recent_experiences if exp['action_type'] == 'idle']
        
        if len(assign_rewards) > 0 and len(idle_rewards) > 0:
            assign_mean = sum(assign_rewards) / len(assign_rewards)
            idle_mean = sum(idle_rewards) / len(idle_rewards)
            
            print(f"🔍 Quick Q-Value Check (last {len(recent_experiences)} experiences):")
            print(f"   Assign avg reward: {assign_mean:.3f} (n={len(assign_rewards)})")
            print(f"   Idle avg reward:   {idle_mean:.3f} (n={len(idle_rewards)})")
            print(f"   Difference: {assign_mean - idle_mean:.3f}")
            
            # 检查Q值与奖励的矛盾
            if hasattr(value_function, 'get_q_value'):
                try:
                    # 快速获取当前Q值估计 (使用平均状态)
                    avg_location = (
                        (self.grid_size // 2) * self.grid_size
                        + self.grid_size // 2
                    )
                    avg_time = self.current_time
                    avg_battery = 0.7
                    other_vehicles = max(0, len([v for v in self.vehicles.values() 
                                               if v.get('assigned_request') is None and 
                                                  v.get('passenger_onboard') is None]) - 1)
                    num_requests = len(self.active_requests)
                    
                    # 获取当前Q值预测
                    assign_q = value_function.get_q_value(
                        vehicle_id=1, action_type='assign_1', 
                        vehicle_location=avg_location, target_location=avg_location,
                        current_time=avg_time, other_vehicles=other_vehicles,
                        num_requests=num_requests, battery_level=avg_battery,
                        request_value=10.0
                    )
                    
                    idle_q = value_function.get_q_value(
                        vehicle_id=1, action_type='idle',
                        vehicle_location=avg_location,
                        target_location=avg_location,
                        battery_level=avg_battery, current_time=avg_time,
                        other_vehicles=other_vehicles,
                        num_requests=num_requests,
                    )

                    charge_q = value_function.get_q_value(
                        vehicle_id=1, action_type='charge_1',
                        vehicle_location=avg_location,
                        target_location=avg_location,
                        current_time=avg_time, other_vehicles=other_vehicles,
                        num_requests=num_requests, battery_level=avg_battery,
                        request_value=0.0,
                    )
                    
                    print(f"   Current Q-predictions: Assign={assign_q:.3f}, Idle={idle_q:.3f}, Charge={charge_q:.3f}")
                    
                    # 检测矛盾
                    if assign_mean > idle_mean + 5.0 and idle_q > assign_q + 0.5:
                        print(f"🚨 CONTRADICTION DETECTED!")
                        print(f"   Assign rewards ({assign_mean:.1f}) > Idle rewards ({idle_mean:.1f})")
                        print(f"   But Idle Q-value ({idle_q:.3f}) > Assign Q-value ({assign_q:.3f})")
                        print(f"   💡 Possible causes:")
                        print(f"      1. Training hasn't converged yet (need more steps)")
                        print(f"      2. Sample imbalance in training batch")
                        print(f"      3. Network capacity insufficient")
                        print(f"      4. Learning rate too high/low")
                        
                        # 调用详细的矛盾分析
                        self._analyze_q_reward_contradiction(value_function, recent_experiences)
                        
                except Exception as e:
                    print(f"   Q-value prediction error: {e}")
            
            if assign_mean < idle_mean - 0.1:  # 阈值0.1
                print(f"⚠️  WARNING: Assign rewards significantly lower than idle!")
                
                # 导出最近的experience数据为CSV
                self._export_recent_experiences_csv(recent_experiences)
    
    def _export_recent_experiences_csv(self, experiences):
        """
        导出最近的experience数据为CSV文件
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        # 创建导出目录
        export_dir = "results/q_value_analysis"
        os.makedirs(export_dir, exist_ok=True)
        
        # 准备数据
        rows = []
        for i, exp in enumerate(experiences):
            # 计算距离 - 安全处理位置坐标
            v_loc = exp.get('vehicle_location', 0)
            t_loc = exp.get('target_location', 0)
            grid_size = getattr(self, 'grid_size', 10)
            
            # 安全转换位置为整数索引
            def _safe_location_to_int(loc):
                if isinstance(loc, tuple) and len(loc) == 2:
                    # 如果是坐标元组，转换为索引
                    x, y = loc
                    return y * grid_size + x
                elif isinstance(loc, (int, float)):
                    return int(loc)
                else:
                    return 0
            
            v_loc_int = _safe_location_to_int(v_loc)
            t_loc_int = _safe_location_to_int(t_loc)
            
            # 计算坐标
            vx, vy = v_loc_int % grid_size, v_loc_int // grid_size
            tx, ty = t_loc_int % grid_size, t_loc_int // grid_size
            distance = abs(vx - tx) + abs(vy - ty)
            
            # 简化动作类型
            action_type = exp.get('action_type', '')
            if action_type == 'idle':
                action_category = 'idle'
            elif action_type.startswith('assign'):
                action_category = 'assign'
            elif action_type.startswith('charge'):
                action_category = 'charge'
            else:
                action_category = 'other'
            
            row = {
                'exp_id': i,
                'vehicle_id': exp.get('vehicle_id', 0),
                'action_type': action_type,
                'action_category': action_category,
                'vehicle_location': v_loc,
                'target_location': t_loc,
                'distance': distance,
                'battery_level': exp.get('battery_level', 1.0),
                'current_time': exp.get('current_time', 0.0),
                'reward': exp.get('reward', 0.0),
                'next_battery_level': exp.get('next_battery_level', 1.0),
                'num_requests': exp.get('num_requests', 0),
                'request_value': exp.get('request_value', 0.0),
                'is_rejection': exp.get('is_rejection', False)
            }
            rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(export_dir, f"recent_experiences_{timestamp}.csv")
        # df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📊 Recent experiences exported to: {csv_file}")
        
        # 显示基础统计
        print(f"📈 Quick Statistics:")
        reward_by_action = df.groupby('action_category')['reward'].agg(['count', 'mean', 'std']).round(3)
        print(reward_by_action)
        return csv_file
    
    def _analyze_q_reward_contradiction(self, value_function, experiences):
        """
        深入分析Q值与奖励矛盾的详细原因
        """
        print(f"\n🔬 详细矛盾分析:")
        print("=" * 50)
        csv_file = None
        
        try:
            # 1. 分析training step和buffer状态
            training_step = getattr(value_function, 'training_step', 0)
            buffer_size = len(value_function.experience_buffer) if hasattr(value_function, 'experience_buffer') else 0
            
            print(f"📊 训练状态:")
            print(f"   训练步数: {training_step}")
            print(f"   缓冲区大小: {buffer_size}")
            print(f"   分析样本数: {len(experiences)}")
            
            # 2. 分析最近的训练批次构成
            if hasattr(value_function, '_action_balanced_sample'):
                try:
                    sample_batch = value_function._action_balanced_sample(64)
                    batch_assign = len([exp for exp in sample_batch if exp['action_type'].startswith('assign')])
                    batch_idle = len([exp for exp in sample_batch if exp['action_type'] == 'idle'])
                    batch_charge = len([exp for exp in sample_batch if exp['action_type'].startswith('charge')])
                    
                    print(f"🎲 最近训练批次构成:")
                    print(f"   Assign: {batch_assign}/64 ({batch_assign/64:.1%})")
                    print(f"   Idle: {batch_idle}/64 ({batch_idle/64:.1%})")
                    print(f"   Charge: {batch_charge}/64 ({batch_charge/64:.1%})")
                    
                    # 检查是否过度倾向于idle
                    if batch_idle > batch_assign * 2:
                        print(f"   ⚠️  Idle样本过多，可能影响学习")
                        
                except Exception as e:
                    print(f"   采样分析失败: {e}")
            
            # 3. 分析奖励分布的细节
            assign_rewards = [exp['reward'] for exp in experiences if exp['action_type'].startswith('assign')]
            idle_rewards = [exp['reward'] for exp in experiences if exp['action_type'] == 'idle']
            
            if assign_rewards and idle_rewards:
                import numpy as np
                
                # 正负奖励分布
                assign_pos = len([r for r in assign_rewards if r > 0])
                assign_neg = len([r for r in assign_rewards if r <= 0])
                idle_pos = len([r for r in idle_rewards if r > 0])
                idle_neg = len([r for r in idle_rewards if r <= 0])
                
                print(f"\n🎯 奖励分布分析:")
                print(f"   Assign: {assign_pos} 正奖励, {assign_neg} 负/零奖励")
                print(f"   Idle: {idle_pos} 正奖励, {idle_neg} 负/零奖励")
                
                # 奖励量级分析
                if assign_rewards:
                    print(f"   Assign奖励范围: [{np.min(assign_rewards):.1f}, {np.max(assign_rewards):.1f}]")
                if idle_rewards:
                    print(f"   Idle奖励范围: [{np.min(idle_rewards):.1f}, {np.max(idle_rewards):.1f}]")
                
                # 检查奖励scale问题
                assign_scale = np.std(assign_rewards) if len(assign_rewards) > 1 else 0
                idle_scale = np.std(idle_rewards) if len(idle_rewards) > 1 else 0
                print(f"   奖励变异性: Assign std={assign_scale:.2f}, Idle std={idle_scale:.2f}")
                
                if assign_scale > idle_scale * 3:
                    print(f"   ⚠️  Assign奖励变异性过大，可能影响学习稳定性")
            
            # 4. 提供具体的改进建议
            print(f"\n💡 改进建议:")
            
            if training_step < 1000:
                print(f"   🔄 训练步数较少({training_step})，建议继续训练至少2000步")
                
            if buffer_size < 5000:
                print(f"   📊 缓冲区数据较少({buffer_size})，建议积累更多经验")
                
            # 检查学习率
            if hasattr(value_function, 'optimizer'):
                current_lr = value_function.optimizer.param_groups[0]['lr']
                print(f"   📈 当前学习率: {current_lr:.6f}")
                if current_lr > 0.01:
                    print(f"      建议降低学习率到 0.001-0.005 范围")
                elif current_lr < 0.0001:
                    print(f"      学习率可能过低，建议提高到 0.0005-0.001")
            
            print(f"   🎯 建议启用更强的assign奖励bonus")
            print(f"   🔧 考虑调整action-balanced采样比例 (增加assign权重)")
            print(f"   📚 使用prioritized experience replay优先训练高价值样本")

            csv_file = self._export_recent_experiences_csv(experiences)
            
        except Exception as e:
            print(f"❌ 矛盾分析失败: {e}")
        
        return csv_file

    def _count_idle_vehicles(self):
        """
        统计当前处于idle状态的车辆数量
        Idle车辆定义：没有分配请求、没有乘客在车、没有在充电站充电
        """
        idle_count = 0
        for vehicle_id, vehicle in self.vehicles.items():
            is_idle = (
                vehicle.get('assigned_request') is None and
                vehicle.get('passenger_onboard') is None and
                vehicle.get('charging_station') is None and
                vehicle.get('battery_level', 1.0) > self.min_battery_level
            )
            if is_idle:
                idle_count += 1
        
        return idle_count
    
    def _get_idle_action_index(self):
        """
        获取idle动作对应的DQN动作索引
        这需要根据DQN动作空间的定义来确定
        假设idle动作是最后一个动作（索引为动作空间大小-1）
        """
        # 根据DQN动作空间定义，通常idle/wait动作在末尾
        # 这里假设动作空间大小为32，idle动作索引为31
        # 实际使用时需要根据具体的DQN实现调整
        return 31  # 或者根据实际的DQN动作定义返回正确的索引

    def _count_idle_vehicles(self):
        """
        统计当前处于idle状态的车辆数量
        Idle车辆定义：没有分配请求、没有乘客在车、没有在充电站充电
        """
        idle_count = 0
        for vehicle_id, vehicle in self.vehicles.items():
            is_idle = (
                vehicle.get('assigned_request') is None and
                vehicle.get('passenger_onboard') is None and
                vehicle.get('charging_station') is None and
                vehicle.get('battery_level', 1.0) > self.min_battery_level
            )
            if is_idle:
                idle_count += 1
        
        return idle_count
