"""Temperature-dependent resin viscosity."""

from __future__ import annotations

import math


def _todo(function_name: str) -> None:
    raise NotImplementedError(
        f"TODO still incomplete: {function_name}. "
        "Use the equation in the docstring to complete this function."
    )


def calculate_viscosity_temperature_only(
    temperature_C: float,
    mu_ref_Pa_s: float = 1.0,
    T_ref_C: float = 80.0,
    E_mu_J_per_mol: float = 35000.0,
    R_J_per_mol_K: float = 8.314462618,
) -> float:
    """Calculate resin viscosity using an Arrhenius relation.

    Governing equation:
        mu(T) = mu_ref exp[(E_mu / R) (1 / T - 1 / T_ref)]

    Args:
        temperature_C: Temperature in degrees Celsius.
        mu_ref_Pa_s: Reference viscosity in Pa s.
        T_ref_C: Reference temperature in degrees Celsius.
        E_mu_J_per_mol: Activation energy in J/mol.
        R_J_per_mol_K: Gas constant in J/(mol K).

    Returns:
        Viscosity in Pa s.
    """
    temperature_K = temperature_C + 273.15
    reference_K = T_ref_C + 273.15
    if temperature_K <= 0.0 or reference_K <= 0.0:
        raise ValueError("Temperature must be positive in Kelvin.")
    if mu_ref_Pa_s <= 0.0 or E_mu_J_per_mol <= 0.0 or R_J_per_mol_K <= 0.0:
        raise ValueError("Viscosity parameters must be positive.")

    # TODO: evaluate the Arrhenius expression using the validated Kelvin temperatures.
    pass
    _todo("calculate_viscosity_temperature_only")


def calculate_viscosity(temperature_C: float) -> float:
    """Backward-compatible alias for the temperature-only viscosity."""
    return calculate_viscosity_temperature_only(temperature_C)
