"""Lightweight pressure and velocity solve in the resin region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from config import FlowConfig
from geometry import (
    FIBER_DIAMETER_M,
    Microstructure,
    calculate_fiber_volume_fraction,
    inlet_connected_matrix,
    rasterize_fibers,
    spacing_metrics,
)


DELTA_PRESSURE_PA = FlowConfig().pressure_left_Pa - FlowConfig().pressure_right_Pa


@dataclass(frozen=True)
class FlowResult:
    """Pressure, channel mobility, and velocity fields for one microstructure."""

    X: np.ndarray
    Y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    fiber_mask: np.ndarray
    matrix_mask: np.ndarray
    active_matrix_mask: np.ndarray
    pressure: np.ndarray
    pressure_gradient_x: np.ndarray
    pressure_gradient_y: np.ndarray
    channel_factor: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    speed: np.ndarray
    permeability: float
    average_velocity: float
    maximum_velocity: float


def _kozeny_factor(Vf: float) -> float:
    if not (0.0 < Vf < 0.75):
        raise ValueError("Vf must be between 0 and 0.75 for this educational model.")
    porosity = 1.0 - Vf
    return porosity**3 / Vf**2


def calculate_longitudinal_permeability(
    Vf: float,
    spacing_correction: float = 1.0,
    fiber_diameter_m: float = FIBER_DIAMETER_M,
    config: FlowConfig | None = None,
) -> float:
    """Calculate educational longitudinal permeability.

    Governing equation:
        K_parallel = C_parallel d_f^2 ((1 - Vf)^3 / Vf^2) f_spacing

    Returns:
        Longitudinal permeability in m^2.
    """
    config = config or FlowConfig()
    if fiber_diameter_m <= 0.0 or spacing_correction <= 0.0:
        raise ValueError("fiber_diameter_m and spacing_correction must be positive.")
    return config.C_parallel * fiber_diameter_m**2 * _kozeny_factor(Vf) * spacing_correction


def calculate_transverse_permeability(
    Vf: float,
    spacing_correction: float = 1.0,
    fiber_diameter_m: float = FIBER_DIAMETER_M,
    config: FlowConfig | None = None,
) -> float:
    """Calculate educational transverse permeability.

    Governing equation:
        K_perp = C_perp d_f^2 ((1 - Vf)^3 / Vf^2) f_spacing

    Returns:
        Transverse permeability in m^2.
    """
    config = config or FlowConfig()
    if fiber_diameter_m <= 0.0 or spacing_correction <= 0.0:
        raise ValueError("fiber_diameter_m and spacing_correction must be positive.")
    return config.C_perp * fiber_diameter_m**2 * _kozeny_factor(Vf) * spacing_correction


def _harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _solve_pressure(
    active_mask: np.ndarray,
    channel_factor: np.ndarray,
    pressure_left: float,
    pressure_right: float,
) -> np.ndarray:
    """Solve a weighted finite-difference pressure problem in active matrix cells."""
    ny, nx = active_mask.shape
    ids = -np.ones_like(active_mask, dtype=int)
    active_indices = np.argwhere(active_mask)
    for unknown, (j, i) in enumerate(active_indices):
        ids[j, i] = unknown

    matrix = lil_matrix((len(active_indices), len(active_indices)), dtype=float)
    rhs = np.zeros(len(active_indices), dtype=float)

    for row, (j, i) in enumerate(active_indices):
        if i == 0:
            matrix[row, row] = 1.0
            rhs[row] = pressure_left
            continue
        if i == nx - 1:
            matrix[row, row] = 1.0
            rhs[row] = pressure_right
            continue

        diagonal = 0.0
        for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if 0 <= jj < ny and 0 <= ii < nx and active_mask[jj, ii]:
                conductance = _harmonic(channel_factor[j, i], channel_factor[jj, ii])
                matrix[row, ids[jj, ii]] = -conductance
                diagonal += conductance
        if diagonal <= 0.0:
            matrix[row, row] = 1.0
            rhs[row] = 0.5 * (pressure_left + pressure_right)
        else:
            matrix[row, row] = diagonal

    solution = spsolve(matrix.tocsr(), rhs)
    pressure = np.full(active_mask.shape, np.nan, dtype=float)
    for value, (j, i) in zip(solution, active_indices):
        pressure[j, i] = value
    return pressure


def _pressure_gradients(
    pressure: np.ndarray,
    active_mask: np.ndarray,
    dx: float,
    dy: float,
    pressure_left: float,
    pressure_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate cell-centered pressure gradients while respecting solid fibers."""
    ny, nx = active_mask.shape
    dpdx = np.zeros_like(pressure, dtype=float)
    dpdy = np.zeros_like(pressure, dtype=float)
    for j in range(ny):
        for i in range(nx):
            if not active_mask[j, i]:
                continue
            p_here = pressure[j, i]
            if i == 0:
                p_left, left_distance = pressure_left, 0.5 * dx
            elif active_mask[j, i - 1]:
                p_left, left_distance = pressure[j, i - 1], dx
            else:
                p_left, left_distance = p_here, dx

            if i == nx - 1:
                p_right, right_distance = pressure_right, 0.5 * dx
            elif active_mask[j, i + 1]:
                p_right, right_distance = pressure[j, i + 1], dx
            else:
                p_right, right_distance = p_here, dx

            if j > 0 and active_mask[j - 1, i]:
                p_bottom, bottom_distance = pressure[j - 1, i], dy
            else:
                p_bottom, bottom_distance = p_here, dy
            if j < ny - 1 and active_mask[j + 1, i]:
                p_top, top_distance = pressure[j + 1, i], dy
            else:
                p_top, top_distance = p_here, dy

            dpdx[j, i] = (p_right - p_left) / (left_distance + right_distance)
            dpdy[j, i] = (p_top - p_bottom) / (bottom_distance + top_distance)
    return dpdx, dpdy


def local_channel_factor(
    active_mask: np.ndarray,
    dx: float,
    dy: float,
    flow_config: FlowConfig | None = None,
) -> np.ndarray:
    """Calculate a simple local channel-width mobility factor.

    Approximation:
        f_channel = clip((d_nearest_solid / mean(d_nearest_solid))^2)

    Wider resin channels receive larger mobility and narrow gaps receive lower
    mobility. This is an interpretable educational approximation, not a
    Navier-Stokes solution.
    """
    flow_config = flow_config or FlowConfig()
    distance = distance_transform_edt(active_mask, sampling=(dy, dx))
    reference = max(float(np.mean(distance[active_mask])), 1.0e-30)
    factor = (distance / reference) ** 2
    factor = np.clip(factor, flow_config.min_channel_factor, flow_config.max_channel_factor)
    return np.where(active_mask, factor, 0.0)


def velocity_from_viscosity(
    flow: FlowResult,
    viscosity_field: np.ndarray,
    gelled: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute local Darcy velocity from current viscosity and gel state."""
    if viscosity_field.shape != flow.active_matrix_mask.shape:
        raise ValueError("viscosity_field shape does not match flow grid.")
    mobile = flow.active_matrix_mask.copy()
    if gelled is not None:
        mobile &= ~gelled
    mobility = np.zeros_like(viscosity_field, dtype=float)
    mobility[mobile] = (
        flow.permeability * flow.channel_factor[mobile] / viscosity_field[mobile]
    )
    velocity_x = -mobility * flow.pressure_gradient_x
    velocity_y = -mobility * flow.pressure_gradient_y
    speed = np.hypot(velocity_x, velocity_y)
    return velocity_x, velocity_y, speed


def solve_flow(
    microstructure: Microstructure,
    viscosity_Pa_s: float,
    nx: int = 120,
    ny: int = 72,
    pressure_left_Pa: float | None = None,
    pressure_right_Pa: float | None = None,
    flow_config: FlowConfig | None = None,
) -> FlowResult:
    """Solve pressure and initial velocity in resin cells around fibers."""
    flow_config = flow_config or FlowConfig()
    pressure_left = flow_config.pressure_left_Pa if pressure_left_Pa is None else pressure_left_Pa
    pressure_right = flow_config.pressure_right_Pa if pressure_right_Pa is None else pressure_right_Pa
    X, Y, x, y, dx, dy, fiber_mask = rasterize_fibers(microstructure, nx, ny)
    matrix_mask = ~fiber_mask
    active_mask = inlet_connected_matrix(matrix_mask)
    channel_factor = local_channel_factor(active_mask, dx, dy, flow_config)
    pressure = _solve_pressure(active_mask, channel_factor, pressure_left, pressure_right)
    dpdx, dpdy = _pressure_gradients(pressure, active_mask, dx, dy, pressure_left, pressure_right)
    vf = calculate_fiber_volume_fraction(microstructure.fibers, microstructure.width, microstructure.height)
    spacing = spacing_metrics(microstructure)
    permeability = calculate_transverse_permeability(vf, spacing.spacing_correction, config=flow_config)

    shell = FlowResult(
        X=X,
        Y=Y,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        fiber_mask=fiber_mask,
        matrix_mask=matrix_mask,
        active_matrix_mask=active_mask,
        pressure=pressure,
        pressure_gradient_x=dpdx,
        pressure_gradient_y=dpdy,
        channel_factor=channel_factor,
        velocity_x=np.zeros_like(pressure),
        velocity_y=np.zeros_like(pressure),
        speed=np.zeros_like(pressure),
        permeability=permeability,
        average_velocity=0.0,
        maximum_velocity=0.0,
    )
    viscosity_field = np.full(active_mask.shape, viscosity_Pa_s, dtype=float)
    velocity_x, velocity_y, speed = velocity_from_viscosity(shell, viscosity_field)
    active_speed = speed[active_mask]
    return FlowResult(
        X=X,
        Y=Y,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        fiber_mask=fiber_mask,
        matrix_mask=matrix_mask,
        active_matrix_mask=active_mask,
        pressure=pressure,
        pressure_gradient_x=dpdx,
        pressure_gradient_y=dpdy,
        channel_factor=channel_factor,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=speed,
        permeability=permeability,
        average_velocity=float(np.mean(active_speed)),
        maximum_velocity=float(np.max(active_speed)),
    )
