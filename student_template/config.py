"""Configuration for the educational fiber impregnation model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    """2D representative domain and grid."""

    width_m: float = 1.0e-3
    height_m: float = 0.60e-3
    nx: int = 120
    ny: int = 72
    fiber_radius_m: float = 35.0e-6


@dataclass(frozen=True)
class ViscosityConfig:
    """Educational/calibration parameters for initial thermal viscosity."""

    mu_ref_Pa_s: float = 1.0
    T_ref_C: float = 80.0
    E_mu_J_per_mol: float = 35000.0
    R_J_per_mol_K: float = 8.314462618
    C_alpha: float = 6.0
    mu_max_Pa_s: float = 1.0e6


@dataclass(frozen=True)
class CureConfig:
    """Kamal-Sourour cure constants from the supplied PU characterization study."""

    A1: float = 1.002e-2
    A2: float = 3.516e4
    m: float = 0.110
    n: float = 1.6384
    Ea_J_per_mol: float = 46.12e3
    R_J_per_mol_K: float = 8.314
    alpha_initial: float = 0.0
    alpha_gel: float = 0.53
    max_cure_substep_s: float = 0.5


@dataclass(frozen=True)
class FlowConfig:
    """Pressure and local channel-mobility controls."""

    pressure_left_Pa: float = 500.0
    pressure_right_Pa: float = 0.0
    C_parallel: float = 1.0 / 25.0
    C_perp: float = 1.0 / 180.0
    min_channel_factor: float = 0.05
    max_channel_factor: float = 3.0


@dataclass(frozen=True)
class TimeConfig:
    """Common physical simulation time for all temperature cases."""

    total_time_s: float = 650.0
    dt_s: float = 2.0
    n_snapshots: int = 56


@dataclass(frozen=True)
class StudyConfig:
    """Required comparison cases."""

    patterns: tuple[str, ...] = ("square", "hexagonal", "random")
    vf_values: tuple[float, ...] = (0.40, 0.50, 0.60)
    temperatures_C: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0)
    pattern_study_temperature_C: float = 50.0
    pattern_study_vf: float = 0.50
    pattern_temperature_temperatures_C: tuple[float, ...] = (25.0, 75.0, 125.0)
    vf_study_pattern: str = "square"
    vf_study_temperature_C: float = 50.0
    temperature_study_pattern: str = "square"
    temperature_study_vf: float = 0.50


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    viscosity: ViscosityConfig = field(default_factory=ViscosityConfig)
    cure: CureConfig = field(default_factory=CureConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    study: StudyConfig = field(default_factory=StudyConfig)


def default_config() -> SimulationConfig:
    """Return the default homework configuration."""
    return SimulationConfig()
