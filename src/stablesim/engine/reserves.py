"""Reserve backing model for the stablecoin issuer.

Models the reserve ratio and optional transparency signal:
- true backing ratio r_t follows a mean-reverting process
- disclosed signal is released at configurable frequency with noise
"""

from __future__ import annotations

import numpy as np


class ReserveModel:
    """Stochastic reserve backing with controlled disclosure.

    Parameters
    ----------
    initial_ratio : float
        Starting r_0 (1.0 = fully backed).
    mean_ratio : float
        Long-run mean backing ratio (theta for OU).
    speed : float
        OU mean-reversion speed kappa.
    vol : float
        Volatility of the backing ratio.
    transparency_freq : int
        Disclose true ratio every N steps.  0 = never disclose.
    transparency_noise : float
        Std dev of Gaussian noise added to disclosed ratio.
    rng : np.random.Generator | None
    """

    def __init__(
        self,
        initial_ratio: float = 1.0,
        mean_ratio: float = 1.0,
        speed: float = 0.05,
        vol: float = 0.02,
        transparency_freq: int = 0,
        transparency_noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.ratio = float(initial_ratio)
        self.mean_ratio = float(mean_ratio)
        self.speed = float(speed)
        self.vol = float(vol)
        self.transparency_freq = transparency_freq
        self.transparency_noise = float(transparency_noise)
        self.rng = rng or np.random.default_rng()
        self._step = 0
        self._last_disclosed: float | None = None

    def step(self, dt: float = 1.0) -> None:
        """Advance the reserve ratio by one time step (Euler-Maruyama OU)."""
        dW = self.rng.normal(0, np.sqrt(dt))
        self.ratio += self.speed * (self.mean_ratio - self.ratio) * dt + self.vol * dW
        self.ratio = max(0.0, self.ratio)
        self._step += 1
        if self.transparency_freq > 0 and self._step % self.transparency_freq == 0:
            noise = self.rng.normal(0, self.transparency_noise) if self.transparency_noise > 0 else 0.0
            self._last_disclosed = float(np.clip(self.ratio + noise, 0, None))

    @property
    def disclosed_ratio(self) -> float | None:
        """Most recently disclosed backing ratio (None if never disclosed)."""
        return self._last_disclosed

    @property
    def perceived_backing(self) -> float:
        """Agent-observable backing: disclosed if available, else prior mean."""
        return self._last_disclosed if self._last_disclosed is not None else self.mean_ratio

    def state(self) -> dict:
        return {
            "ratio": self.ratio,
            "disclosed": self._last_disclosed,
            "perceived": self.perceived_backing,
            "step": self._step,
        }
