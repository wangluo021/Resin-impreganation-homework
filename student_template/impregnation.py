"""Coupled impregnation, cure, viscosity, and gelation time stepping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CureConfig, TimeConfig, ViscosityConfig
from cure import gelled_mask, update_degree_of_cure, viscosity_with_cure
from flow_solver import FlowResult, velocity_from_viscosity
from viscosity import calculate_viscosity_temperature_only


def _todo(function_name: str) -> None:
    raise NotImplementedError(
        f"TODO still incomplete: {function_name}. "
        "Use the equation in the docstring to complete this function."
    )


@dataclass(frozen=True)
class ImpregnationResult:
    """Time histories and field snapshots for one simulation case."""

    time: np.ndarray
    fraction: np.ndarray
    average_alpha: np.ndarray
    matrix_average_alpha: np.ndarray
    gelled_fraction: np.ndarray
    average_viscosity: np.ndarray
    minimum_viscosity: np.ndarray
    average_velocity: np.ndarray
    maximum_velocity: np.ndarray
    average_cure_rate: np.ndarray
    final_saturation: np.ndarray
    final_alpha: np.ndarray
    final_gelled: np.ndarray
    final_viscosity: np.ndarray
    snapshot_times: np.ndarray
    saturation_snapshots: np.ndarray
    alpha_snapshots: np.ndarray
    gelled_snapshots: np.ndarray
    viscosity_snapshots: np.ndarray
    t25: float
    t50: float
    t75: float
    t90: float
    gel_time: float
    final_fraction: float
    final_average_alpha: float
    final_matrix_average_alpha: float
    final_gelled_fraction: float


def calculate_impregnation_time(
    viscosity_Pa_s: float,
    impregnation_distance_m: float,
    transverse_permeability_m2: float,
    delta_pressure_Pa: float,
    calibration_factor: float = 1.0,
) -> float:
    """Calculate a characteristic educational impregnation time.

    Governing equation:
        t_imp = C mu L_imp^2 / (K_perp DeltaP)

    Returns:
        Characteristic time in seconds.
    """
    if viscosity_Pa_s <= 0.0:
        raise ValueError("viscosity_Pa_s must be positive.")
    if impregnation_distance_m <= 0.0:
        raise ValueError("impregnation_distance_m must be positive.")
    if transverse_permeability_m2 <= 0.0:
        raise ValueError("transverse_permeability_m2 must be positive.")
    if delta_pressure_Pa <= 0.0:
        raise ValueError("delta_pressure_Pa must be positive.")
    if calibration_factor <= 0.0:
        raise ValueError("calibration_factor must be positive.")
    return (
        calibration_factor
        * viscosity_Pa_s
        * impregnation_distance_m**2
        / (transverse_permeability_m2 * delta_pressure_Pa)
    )


def calculate_impregnation_fraction(saturation: np.ndarray, matrix_mask: np.ndarray) -> float:
    """Calculate matrix-area impregnation fraction.

    Governing equation:
        I(t) = resin-filled matrix cells / total matrix cells

    Fiber cells are excluded from the denominator.
    """
    # TODO: count filled matrix cells and divide by total non-fiber matrix cells.
    pass
    _todo("calculate_impregnation_fraction")


def _candidate_front_speed(source_speed: np.ndarray, dry: np.ndarray) -> np.ndarray:
    """Map neighboring source-cell speeds onto dry front cells."""
    candidate_speed = np.zeros_like(source_speed, dtype=float)
    candidate_speed[1:, :] = np.maximum(candidate_speed[1:, :], source_speed[:-1, :])
    candidate_speed[:-1, :] = np.maximum(candidate_speed[:-1, :], source_speed[1:, :])
    candidate_speed[:, 1:] = np.maximum(candidate_speed[:, 1:], source_speed[:, :-1])
    candidate_speed[:, :-1] = np.maximum(candidate_speed[:, :-1], source_speed[:, 1:])
    return np.where(dry, candidate_speed, 0.0)


def _first_threshold_time(time: np.ndarray, fraction: np.ndarray, threshold: float) -> float:
    reached = np.flatnonzero(fraction >= threshold)
    if reached.size == 0:
        return float("nan")
    return float(time[int(reached[0])])


def run_coupled_impregnation(
    flow: FlowResult,
    temperature_C: float,
    time_config: TimeConfig | None = None,
    cure_config: CureConfig | None = None,
    viscosity_config: ViscosityConfig | None = None,
) -> ImpregnationResult:
    """Run coupled resin-front, cure, viscosity, gel, and velocity evolution.

    Time-step order:
        1. Move the resin front using current local velocity.
        2. Initialize alpha in newly filled cells.
        3. Update cure only in filled cells.
        4. Increase viscosity with alpha and cap gelled cells.
        5. Recompute local velocity for the next step.
    """
    time_config = time_config or TimeConfig()
    cure_config = cure_config or CureConfig()
    viscosity_config = viscosity_config or ViscosityConfig()
    if time_config.dt_s <= 0.0 or time_config.total_time_s <= 0.0:
        raise ValueError("Time step and total time must be positive.")

    T_K = temperature_C + 273.15
    mu_temperature = calculate_viscosity_temperature_only(
        temperature_C,
        viscosity_config.mu_ref_Pa_s,
        viscosity_config.T_ref_C,
        viscosity_config.E_mu_J_per_mol,
        viscosity_config.R_J_per_mol_K,
    )
    matrix = flow.active_matrix_mask
    saturation = np.zeros_like(matrix, dtype=bool)
    saturation[:, 0] = matrix[:, 0]
    alpha = np.zeros_like(flow.pressure, dtype=float)
    alpha[saturation] = cure_config.alpha_initial
    fill_progress = np.zeros_like(flow.pressure, dtype=float)
    gelled = gelled_mask(alpha, saturation, cure_config)
    viscosity = viscosity_with_cure(mu_temperature, alpha, saturation, gelled, viscosity_config)
    velocity_x, velocity_y, speed = velocity_from_viscosity(flow, viscosity, gelled)

    n_steps = int(np.ceil(time_config.total_time_s / time_config.dt_s))
    times = np.linspace(0.0, n_steps * time_config.dt_s, n_steps + 1)
    snapshot_indices = np.unique(
        np.linspace(0, n_steps, time_config.n_snapshots, dtype=int)
    )
    snapshot_lookup = {int(step): idx for idx, step in enumerate(snapshot_indices)}
    snap_shape = (len(snapshot_indices),) + saturation.shape
    saturation_snaps = np.zeros(snap_shape, dtype=bool)
    alpha_snaps = np.zeros(snap_shape, dtype=float)
    gelled_snaps = np.zeros(snap_shape, dtype=bool)
    viscosity_snaps = np.zeros(snap_shape, dtype=float)

    fraction_history = np.zeros(n_steps + 1, dtype=float)
    alpha_history = np.zeros(n_steps + 1, dtype=float)
    matrix_alpha_history = np.zeros(n_steps + 1, dtype=float)
    gel_history = np.zeros(n_steps + 1, dtype=float)
    viscosity_history = np.zeros(n_steps + 1, dtype=float)
    min_viscosity_history = np.zeros(n_steps + 1, dtype=float)
    average_velocity_history = np.zeros(n_steps + 1, dtype=float)
    maximum_velocity_history = np.zeros(n_steps + 1, dtype=float)
    cure_rate_history = np.zeros(n_steps + 1, dtype=float)
    gel_time = float("nan")

    cell_distance = min(flow.dx, flow.dy)
    matrix_count = max(int(np.count_nonzero(flow.matrix_mask)), 1)
    for step, time_s in enumerate(times):
        fraction_history[step] = calculate_impregnation_fraction(saturation, flow.matrix_mask)
        resin_count = max(int(np.count_nonzero(saturation)), 1)
        alpha_history[step] = float(np.sum(alpha[saturation]) / resin_count)
        matrix_alpha_history[step] = float(np.sum(alpha[flow.matrix_mask]) / matrix_count)
        gel_history[step] = float(np.count_nonzero(gelled) / matrix_count)
        viscosity_history[step] = float(np.mean(viscosity[saturation])) if np.any(saturation) else mu_temperature
        min_viscosity_history[step] = float(np.min(viscosity[saturation])) if np.any(saturation) else mu_temperature
        mobile_speed = speed[matrix & ~gelled]
        average_velocity_history[step] = float(np.mean(mobile_speed)) if mobile_speed.size else 0.0
        maximum_velocity_history[step] = float(np.max(mobile_speed)) if mobile_speed.size else 0.0
        if np.isnan(gel_time) and np.any(gelled):
            gel_time = float(time_s)

        if step in snapshot_lookup:
            idx = snapshot_lookup[step]
            saturation_snaps[idx] = saturation
            alpha_snaps[idx] = alpha
            gelled_snaps[idx] = gelled
            viscosity_snaps[idx] = viscosity

        if step == n_steps:
            break

        source = saturation & ~gelled & matrix
        dry = matrix & ~saturation
        source_speed = np.where(source, speed, 0.0)
        front_speed = _candidate_front_speed(source_speed, dry)
        fill_progress[dry] += time_config.dt_s * front_speed[dry] / max(cell_distance, 1.0e-30)
        newly_filled = dry & (fill_progress >= 1.0)
        saturation[newly_filled] = True
        alpha[newly_filled] = cure_config.alpha_initial

        active_resin = saturation
        alpha, cure_rate = update_degree_of_cure(
            alpha,
            active_resin,
            T_K,
            time_config.dt_s,
            cure_config,
        )
        cure_rate_history[step + 1] = (
            float(np.mean(cure_rate[active_resin])) if np.any(active_resin) else 0.0
        )
        gelled = gelled_mask(alpha, active_resin, cure_config)
        viscosity = viscosity_with_cure(mu_temperature, alpha, active_resin, gelled, viscosity_config)
        velocity_x, velocity_y, speed = velocity_from_viscosity(flow, viscosity, gelled)

    return ImpregnationResult(
        time=times,
        fraction=fraction_history,
        average_alpha=alpha_history,
        matrix_average_alpha=matrix_alpha_history,
        gelled_fraction=gel_history,
        average_viscosity=viscosity_history,
        minimum_viscosity=min_viscosity_history,
        average_velocity=average_velocity_history,
        maximum_velocity=maximum_velocity_history,
        average_cure_rate=cure_rate_history,
        final_saturation=saturation,
        final_alpha=alpha,
        final_gelled=gelled,
        final_viscosity=viscosity,
        snapshot_times=times[snapshot_indices],
        saturation_snapshots=saturation_snaps,
        alpha_snapshots=alpha_snaps,
        gelled_snapshots=gelled_snaps,
        viscosity_snapshots=viscosity_snaps,
        t25=_first_threshold_time(times, fraction_history, 0.25),
        t50=_first_threshold_time(times, fraction_history, 0.50),
        t75=_first_threshold_time(times, fraction_history, 0.75),
        t90=_first_threshold_time(times, fraction_history, 0.90),
        gel_time=gel_time,
        final_fraction=float(fraction_history[-1]),
        final_average_alpha=float(alpha_history[-1]),
        final_matrix_average_alpha=float(matrix_alpha_history[-1]),
        final_gelled_fraction=float(gel_history[-1]),
    )
