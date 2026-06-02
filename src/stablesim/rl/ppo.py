"""PPO training entry point using stable-baselines3.

Follows Gu et al. convention: largely default PPO hyperparameters are a
defensible starting point for market-agent training.

Usage:
    python -m stablesim.rl.ppo --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def train_ppo(
    env_kwargs: dict | None = None,
    ppo_kwargs: dict | None = None,
    total_timesteps: int = 500_000,
    save_path: str = "experiments/results/ppo_arb",
    agent_type: str = "arbitrageur",
    scenario_name: str | None = None,
) -> None:
    """Train a PPO policy on StablecoinEnv.

    Parameters
    ----------
    env_kwargs : dict
        Passed to StablecoinEnv.
    ppo_kwargs : dict
        Overrides for PPO hyperparameters.
    total_timesteps : int
        Training budget.
    save_path : str
        Where to save the final policy checkpoint.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from .env import StablecoinEnv
    from ..scenarios.loader import load_stressbench_scenarios

    scenarios = load_stressbench_scenarios()
    scenario = next((s for s in scenarios if s.name == scenario_name), None) if scenario_name else None

    env_kw = env_kwargs or {}
    env_kw.setdefault("agent_type", agent_type)
    env_kw["scenario"] = scenario

    vec_env = make_vec_env(lambda: StablecoinEnv(**env_kw), n_envs=4)

    default_ppo = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
    }
    default_ppo.update(ppo_kwargs or {})

    model = PPO("MlpPolicy", vec_env, **default_ppo)
    model.learn(total_timesteps=total_timesteps)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Saved PPO checkpoint to {save_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_ppo(
        env_kwargs=cfg.get("env", {}),
        ppo_kwargs=cfg.get("ppo", {}),
        total_timesteps=cfg.get("total_timesteps", 500_000),
        save_path=cfg.get("save_path", "experiments/results/ppo_arb"),
        agent_type=cfg.get("agent_type", "arbitrageur"),
        scenario_name=cfg.get("scenario_name"),
    )


if __name__ == "__main__":
    _cli()
