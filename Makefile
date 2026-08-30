.PHONY: test smoke smoke-toy smoke-nyc smoke-adp-heu smoke-recourse smoke-recourse-toy smoke-recourse-nyc assignment-audit assignment-matrix-smoke assignment-runner-dry-run assignment-production-smoke

test:
	python -m pytest

smoke: smoke-toy smoke-nyc

smoke-toy:
	bash scripts/smoke_toy.sh

smoke-nyc:
	bash scripts/smoke_nyc.sh

smoke-adp-heu:
	bash scripts/smoke_adp_heuristic_train.sh

smoke-recourse: smoke-recourse-toy smoke-recourse-nyc

smoke-recourse-toy:
	bash scripts/smoke_recourse_toy.sh

smoke-recourse-nyc:
	bash scripts/smoke_recourse_nyc.sh

assignment-audit:
	python -m pytest -q tests/test_assignment_audit_repairs.py tests/test_all_nyc_models_runner.py tests/test_recourse_day.py

assignment-production-smoke:
	python -m pytest -q tests/test_assignment_method_production_matrix.py

assignment-runner-dry-run:
	python run_all_nyc_assignment_methods.py list
	python run_recourse_panel.py --train-days 2025-12-18 --test-days 2025-12-19 --seeds 71 --parquet-path nyedata/nye_simulation/parquet/yellow_tripdata_2025-12.parquet --dry-run

assignment-matrix-smoke: assignment-audit assignment-runner-dry-run assignment-production-smoke
