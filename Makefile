.PHONY: install test lint fmt calibrate train sweep clean

install:
	pip install -e ".[dev]" || pip install -e . && pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# Run calibration to match empirical half-lives from stablecoin-contagion-network
calibrate:
	python -m stablesim.calibration.optimizer --config configs/base.yaml

# Train RL policies (PPO) for arbitrageur / redeemer agents
train:
	python -m stablesim.rl.ppo --config configs/base.yaml

# Sweep interventions x StressBench scenarios
sweep:
	python -m stablesim.experiments.sweep --config configs/interventions.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
