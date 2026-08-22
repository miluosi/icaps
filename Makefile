.PHONY: test smoke smoke-toy smoke-nyc smoke-adp-heu

test:
	python -m pytest

smoke: smoke-toy smoke-nyc

smoke-toy:
	bash scripts/smoke_toy.sh

smoke-nyc:
	bash scripts/smoke_nyc.sh

smoke-adp-heu:
	bash scripts/smoke_adp_heuristic_train.sh
