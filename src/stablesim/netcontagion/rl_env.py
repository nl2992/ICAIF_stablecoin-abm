"""
RL regulator environment over the calibrated networked-contagion engine.

A regulator allocates a reserve-protection budget across venues to minimise contagion.
Action = per-node reserve boost in [0,1] (multiplies that node's recovery speed kappa);
reward = contagion reduction - cost x budget used. One decision per episode (a contextual
bandit), so PPO trains quickly and stably.

Point of the experiment: a learning agent, told NOTHING about which node is causal, should
concentrate budget on the causal venues (origin USDC, relay DAI) and put ~0 on the GNN's
spurious correlational hub (BUSD) — independently rediscovering the causal ranking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None


if gym is not None:

    class RegulatorEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, net, origin: str, shock: float, kappa_boost: float = 20.0,
                     cost_weight: float = 0.35):
            super().__init__()
            self.net = net
            self.origin = origin
            self.shock = shock
            self.kappa_boost = kappa_boost
            self.cost_weight = cost_weight
            self.targets = list(net.nodes)
            self.N = len(self.targets)
            self.victims = [n for n in net.nodes if n != origin]
            self.base = net.contagion_over(origin, shock, self.victims)
            self._obs = self._features()
            self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.N * 3,), dtype=np.float32)
            self.action_space = spaces.Box(0.0, 1.0, shape=(self.N,), dtype=np.float32)

        def _features(self) -> np.ndarray:
            W = self.net.W
            out = W.sum(axis=0)  # out-transmission (sender)
            inn = W.sum(axis=1)  # in-transmission (receiver)
            feats = []
            for j, nd in enumerate(self.targets):
                feats += [float(out[j]), float(inn[j]), 1.0 if nd == self.origin else 0.0]
            return np.array(feats, dtype=np.float32)

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            return self._obs.copy(), {}

        def step(self, action):
            a = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
            kappa_scale = {self.targets[j]: 1.0 + a[j] * self.kappa_boost for j in range(self.N)}
            cont = self.net.contagion_over(self.origin, self.shock, self.victims,
                                           kappa_scale=kappa_scale)
            reduction = (self.base - cont) / self.base if self.base > 0 else 0.0
            cost = self.cost_weight * float(a.sum()) / max(self.N, 1)
            reward = reduction - cost
            info = {"contagion": cont, "reduction": reduction,
                    "alloc": {self.targets[j].split("/")[0]: round(float(a[j]), 3)
                              for j in range(self.N)}}
            return self._obs.copy(), float(reward), True, False, info
