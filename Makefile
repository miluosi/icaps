.PHONY: test smoke smoke-toy smoke-nyc smoke-adp-heu smoke-recourse smoke-recourse-toy smoke-recourse-nyc

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
