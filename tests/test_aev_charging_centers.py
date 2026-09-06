from pathlib import Path

import pytest

import run_nyctrainer
import test_nyc_model
from src.NYCEnvironment import (
    AEV_CHARGING_CENTER_CAPACITY,
    NYCEnvironment,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SITES = {
    3: {"D1", "M1", "U1"},
    4: {"D1", "M1", "M2", "U1"},
    5: {"D1", "M1", "M2", "U1", "U2"},
}


def _environment(center_count: int) -> NYCEnvironment:
    return NYCEnvironment(
        num_vehicles=4,
        ev_num_vehicles=2,
        parquet_path=str(
            ROOT
            / "nyedata/nye_simulation/parquet/yellow_tripdata_2025-12-18_sample.parquet"
        ),
        station_csv=str(ROOT / "nyedata/nyc_all_charging_stations.csv"),
        start_date="2025-12-18",
        end_date="2025-12-18",
        start_hour=8.0,
        stop_hour=8.01,
        episode_length=2,
        random_seed=1,
        ifonlymanhatten=True,
        aev_charging_center_count=center_count,
    )


@pytest.fixture(scope="module")
def charging_scenarios():
    return {count: _environment(count) for count in (0, 3, 4, 5)}


@pytest.mark.parametrize("count", (3, 4, 5))
def test_scenarios_preserve_public_stations_and_add_only_expected_aev_centers(
    charging_scenarios,
    count,
):
    baseline = charging_scenarios[0]
    env = charging_scenarios[count]

    assert env.public_charging_station_ids == baseline.public_charging_station_ids
    assert len(env.aev_charging_station_ids) == count
    assert {
        env.charging_manager.stations[station_id].site_id
        for station_id in env.aev_charging_station_ids
    } == EXPECTED_SITES[count]
    assert all(
        env.charging_manager.stations[station_id].max_capacity
        == AEV_CHARGING_CENTER_CAPACITY
        for station_id in env.aev_charging_station_ids
    )
    assert all(
        env.charging_manager.stations[station_id].aev_exclusive
        for station_id in env.aev_charging_station_ids
    )
    for station_id in baseline.public_charging_station_ids:
        original = baseline.charging_manager.stations[station_id]
        retained = env.charging_manager.stations[station_id]
        assert retained.location == original.location
        assert retained.max_capacity == original.max_capacity
        assert not retained.aev_exclusive


def test_ev_and_aev_station_visibility_do_not_leak(charging_scenarios):
    env = charging_scenarios[5]
    ev_id = 0
    aev_id = 2

    assert env._charging_station_ids_for_vehicle(ev_id) == env.public_charging_station_ids
    assert env._charging_station_ids_for_vehicle(aev_id) == env.aev_charging_station_ids
    assert not set(env.public_charging_station_ids).intersection(
        env.aev_charging_station_ids
    )
    assert all(
        not env._can_reach_charging_station(ev_id, station_id)
        for station_id in env.aev_charging_station_ids
    )
    assert all(
        not env._can_reach_charging_station(aev_id, station_id)
        for station_id in env.public_charging_station_ids
    )


def test_charge_matrix_uses_aev_centers_without_ev_cross_fleet_edges(
    charging_scenarios,
):
    env = charging_scenarios[3]
    env.vehicles[0]["battery"] = 1.0
    env.vehicles[2]["battery"] = 1.0
    env.charge_action_range_km = None
    env.charge_top_k = None

    matrix = env.generate_vehicle_chargerange([0, 2])
    station_ids = list(env._last_expected_charge_expansion["station_ids"])
    public_columns = [station_ids.index(sid) for sid in env.public_charging_station_ids]
    aev_columns = [station_ids.index(sid) for sid in env.aev_charging_station_ids]

    assert matrix[0, aev_columns].sum() == 0.0
    assert matrix[1, public_columns].sum() == 0.0
    assert matrix[1, aev_columns].sum() > 0.0


@pytest.mark.parametrize("entrypoint", (run_nyctrainer, test_nyc_model))
def test_train_and_test_cli_accept_aev_center_scenario(entrypoint):
    args = entrypoint.parse_args(["--aev-charging-center-count", "5"])
    assert args.aev_charging_center_count == 5
    with pytest.raises(SystemExit):
        entrypoint.parse_args(["--aev-charging-center-count", "2"])
