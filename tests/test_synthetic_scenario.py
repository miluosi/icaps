from src.synthetic_scenario import (
    DEFAULT_AEV_INITIAL_BATTERY_SCALE,
    DEFAULT_CHARGE_DURATION,
    DEFAULT_CRITICAL_CHARGING_BATTERY,
    DEFAULT_EPISODE_DAYS,
    DEFAULT_GRID_SIZE,
    DEFAULT_NUM_STATIONS,
    DEFAULT_SIMULATION_PERIOD,
    DEFAULT_STATION_CAPACITY,
    DEFAULT_STATION_QUEUE_CAPACITY,
    DEFAULT_SYNTHETIC_DEMAND_PROFILE,
    DEFAULT_SYNTHETIC_DEMAND_SCALE,
    DEFAULT_WAIT_PENALTY_PER_STEP,
    synthetic_checkpoint_suffix,
)


def test_default_synthetic_scenario_matches_calibrated_wait05_preset():
    assert (
        DEFAULT_NUM_STATIONS,
        DEFAULT_STATION_CAPACITY,
        DEFAULT_STATION_QUEUE_CAPACITY,
        DEFAULT_CHARGE_DURATION,
        DEFAULT_SIMULATION_PERIOD,
        DEFAULT_EPISODE_DAYS,
        DEFAULT_WAIT_PENALTY_PER_STEP,
        DEFAULT_AEV_INITIAL_BATTERY_SCALE,
        DEFAULT_CRITICAL_CHARGING_BATTERY,
        DEFAULT_GRID_SIZE,
        DEFAULT_SYNTHETIC_DEMAND_PROFILE,
        DEFAULT_SYNTHETIC_DEMAND_SCALE,
    ) == (6, 2, 3, 14, 100, 2, 0.5, 0.68, 0.22, 24, "predictive", 0.97)

    assert synthetic_checkpoint_suffix(
        num_stations=DEFAULT_NUM_STATIONS,
        station_capacity=DEFAULT_STATION_CAPACITY,
        charge_duration=DEFAULT_CHARGE_DURATION,
        simulation_period=DEFAULT_SIMULATION_PERIOD,
        episode_days=DEFAULT_EPISODE_DAYS,
        charging_wait_penalty_per_step=DEFAULT_WAIT_PENALTY_PER_STEP,
        station_queue_capacity=DEFAULT_STATION_QUEUE_CAPACITY,
        aev_initial_battery_scale=DEFAULT_AEV_INITIAL_BATTERY_SCALE,
        critical_charging_battery=DEFAULT_CRITICAL_CHARGING_BATTERY,
        grid_size=DEFAULT_GRID_SIZE,
        synthetic_demand_profile=DEFAULT_SYNTHETIC_DEMAND_PROFILE,
        synthetic_demand_scale=DEFAULT_SYNTHETIC_DEMAND_SCALE,
    ) == "_synq_n6_c2_d14_h200_w0p5_q3_ab0p68_cb0p22_g24_ppredictive_ds0p97"


def test_legacy_synthetic_checkpoint_path_remains_compatible():
    assert synthetic_checkpoint_suffix(
        num_stations=3,
        station_capacity=4,
        charge_duration=8,
        simulation_period=50,
        episode_days=1,
        charging_wait_penalty_per_step=1.0,
    ) == ""


def test_queue_stress_checkpoint_has_explicit_scenario_tag():
    assert synthetic_checkpoint_suffix(
        num_stations=6,
        station_capacity=2,
        charge_duration=11,
        simulation_period=50,
        episode_days=2,
        charging_wait_penalty_per_step=0.5,
        station_queue_capacity=2,
        aev_initial_battery_scale=0.72,
        critical_charging_battery=0.22,
    ) == "_synq_n6_c2_d11_h100_w0p5_q2_ab0p72_cb0p22"


def test_current_checkpoint_tag_separates_map_and_demand_profile():
    assert synthetic_checkpoint_suffix(
        num_stations=9,
        station_capacity=2,
        charge_duration=14,
        simulation_period=100,
        episode_days=2,
        charging_wait_penalty_per_step=0.5,
        station_queue_capacity=3,
        aev_initial_battery_scale=0.72,
        critical_charging_battery=0.22,
        grid_size=30,
        synthetic_demand_profile="predictive",
    ) == "_synq_n9_c2_d14_h200_w0p5_q3_ab0p72_cb0p22_g30_ppredictive"
