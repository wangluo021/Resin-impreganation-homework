"""Kamal-Sourour cure kinetics and cure-viscosity coupling."""

from __future__ import annotations

import math

import numpy as np

from config import CureConfig, ViscosityConfig


def cure_rate(alpha: np.ndarray | float, T_K: float, config: CureConfig | None = None) -> np.ndarray:
    """Calculate Kamal-Sourour degree-of-cure rate.

    Governing equation:
        d alpha / dt = [k1(T) + k2(T) alpha^m] (1 - alpha)^n

        k1 = A1 exp(-Ea / RT)
        k2 = A2 exp(-Ea / RT)

    Args:
        alpha: Degree of cure, unitless.
        T_K: Temperature in Kelvin.
        config: Cure constants from the supplied PU characterization study.

    Returns:
        Cure rate in 1/s.
    """
    config = config or CureConfig()
    if T_K <= 0.0:
        raise ValueError("T_K must be positive.")
    alpha_array = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)
    thermal = math.exp(-config.Ea_J_per_mol / (config.R_J_per_mol_K * T_K))
    k1 = config.A1 * thermal
    k2 = config.A2 * thermal
    rate = (k1 + k2 * np.power(alpha_array, config.m)) * np.power(1.0 - alpha_array, config.n)
    return np.maximum(rate, 0.0)


def update_degree_of_cure(
    alpha: np.ndarray,
    active_resin_mask: np.ndarray,
    T_K: float,
    dt_s: float,
    config: CureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance degree of cure with stable forward Euler substeps."""
    config = config or CureConfig()
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    updated = alpha.copy()
    last_rate = np.zeros_like(alpha, dtype=float)
    n_substeps = max(1, int(math.ceil(dt_s / config.max_cure_substep_s)))
    sub_dt = dt_s / n_substeps
    for _ in range(n_substeps):
        rate = np.zeros_like(alpha, dtype=float)
        rate[active_resin_mask] = cure_rate(updated[active_resin_mask], T_K, config)
        updated[active_resin_mask] = np.clip(
            updated[active_resin_mask] + rate[active_resin_mask] * sub_dt,
            0.0,
            1.0,
        )
        last_rate = rate
    return updated, last_rate


def calculate_cure_viscosity_factor(
    alpha: np.ndarray | float,
    C_alpha: float = 6.0,
) -> np.ndarray:
    """Calculate simplified educational cure-viscosity multiplier.

    Governing equation:
        f(alpha) = exp(C_alpha alpha)

    This is not a validated chemorheological equation for the supplied PU
    system; it is an educational coupling that makes viscosity rise as cure
    progresses.
    """
    if C_alpha < 0.0:
        raise ValueError("C_alpha must be non-negative.")
    alpha_array = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)
    return np.exp(C_alpha * alpha_array)


def gelled_mask(alpha: np.ndarray, resin_mask: np.ndarray, config: CureConfig | None = None) -> np.ndarray:
    """Return cells that have reached the gel degree of cure."""
    config = config or CureConfig()
    return resin_mask & (alpha >= config.alpha_gel)


def viscosity_with_cure(
    mu_temperature_only: float,
    alpha: np.ndarray,
    resin_mask: np.ndarray,
    gel_mask: np.ndarray,
    viscosity_config: ViscosityConfig | None = None,
) -> np.ndarray:
    """Calculate local viscosity from thermal viscosity and degree of cure."""
    viscosity_config = viscosity_config or ViscosityConfig()
    viscosity = np.full(alpha.shape, mu_temperature_only, dtype=float)
    viscosity[resin_mask] = (
        mu_temperature_only
        * calculate_cure_viscosity_factor(alpha[resin_mask], viscosity_config.C_alpha)
    )
    viscosity[gel_mask] = viscosity_config.mu_max_Pa_s
    return np.clip(viscosity, 1.0e-12, viscosity_config.mu_max_Pa_s)
