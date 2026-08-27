"""Fiber microstructure generation and spacing calculations."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


DOMAIN_WIDTH_M = 1.0e-3
DOMAIN_HEIGHT_M = 0.60e-3
FIBER_RADIUS_M = 35.0e-6
FIBER_DIAMETER_M = 2.0 * FIBER_RADIUS_M


@dataclass(frozen=True)
class Fiber:
    """One circular fiber in the 2D representative domain."""

    fiber_id: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Microstructure:
    """A rectangular resin domain containing impermeable circular fibers."""

    pattern: str
    Vf_target: float
    width: float
    height: float
    fibers: tuple[Fiber, ...]


@dataclass(frozen=True)
class SpacingMetrics:
    """Nearest-neighbor spacing quantities."""

    average_spacing: float
    minimum_spacing: float
    resin_channel_width: float
    spacing_correction: float


def _todo(function_name: str) -> None:
    raise NotImplementedError(
        f"TODO still incomplete: {function_name}. "
        "Use the equation in the docstring to complete this function."
    )


def calculate_fiber_volume_fraction(
    fibers: Sequence[Fiber],
    domain_width_m: float,
    domain_height_m: float,
) -> float:
    """Calculate the 2D fiber area fraction.

    Governing equation:
        Vf = sum_i(pi r_i^2) / A_domain

    Args:
        fibers: Circular fibers in the domain.
        domain_width_m: Domain width in meters.
        domain_height_m: Domain height in meters.

    Returns:
        Area fraction used as the fiber volume fraction for a uniform
        unidirectional microstructure.
    """
    # TODO: sum pi*r^2 for all fibers and divide by rectangular domain area.
    pass
    _todo("calculate_fiber_volume_fraction")


def calculate_characteristic_spacing(fibers: Sequence[Fiber]) -> float:
    """Calculate average nearest surface-to-surface fiber spacing.

    Governing equation:
        s_ij = ||x_i - x_j|| - r_i - r_j

    The characteristic spacing is the average nearest-neighbor surface spacing.

    Args:
        fibers: Circular fibers.

    Returns:
        Average nearest surface-to-surface spacing in meters.
    """
    # TODO: for each fiber, find its nearest surface-to-surface spacing,
    # then return the average of those nearest-neighbor spacings.
    pass
    _todo("calculate_characteristic_spacing")


def _target_fiber_count(Vf_target: float, radius: float) -> int:
    if not (0.0 < Vf_target < 0.75):
        raise ValueError("Vf_target must be between 0 and 0.75.")
    fiber_area = math.pi * radius**2
    return max(2, int(round(Vf_target * DOMAIN_WIDTH_M * DOMAIN_HEIGHT_M / fiber_area)))


def _grid_shape(fiber_count: int) -> tuple[int, int]:
    rows = max(2, int(round(math.sqrt(fiber_count * DOMAIN_HEIGHT_M / DOMAIN_WIDTH_M))))
    columns = int(math.ceil(fiber_count / rows))
    while rows * columns < fiber_count:
        columns += 1
    return rows, columns


def _complete_lattice_shape_and_radius(Vf_target: float, radius: float) -> tuple[int, int, float]:
    """Choose a complete lattice and adjust fiber radius to match target Vf."""
    estimated_count = _target_fiber_count(Vf_target, radius)
    rows, columns = _grid_shape(estimated_count)
    full_count = rows * columns
    adjusted_radius = math.sqrt(Vf_target * DOMAIN_WIDTH_M * DOMAIN_HEIGHT_M / (full_count * math.pi))
    if adjusted_radius <= 0.0:
        raise ValueError("Adjusted fiber radius must be positive.")
    return rows, columns, adjusted_radius


def _validate_microstructure(microstructure: Microstructure) -> None:
    tolerance = 1.0e-12
    for fiber in microstructure.fibers:
        if fiber.radius <= 0.0:
            raise ValueError(f"{fiber.fiber_id} has non-positive radius.")
        if not (
            fiber.radius - tolerance <= fiber.x <= microstructure.width - fiber.radius + tolerance
        ):
            raise ValueError(f"{fiber.fiber_id} crosses an x boundary.")
        if not (
            fiber.radius - tolerance <= fiber.y <= microstructure.height - fiber.radius + tolerance
        ):
            raise ValueError(f"{fiber.fiber_id} crosses a y boundary.")

    for i, first in enumerate(microstructure.fibers):
        for second in microstructure.fibers[i + 1 :]:
            distance = math.hypot(first.x - second.x, first.y - second.y)
            if distance < first.radius + second.radius - tolerance:
                raise ValueError(f"{first.fiber_id} overlaps {second.fiber_id}.")


def generate_square_packing(Vf_target: float, radius: float = FIBER_RADIUS_M) -> Microstructure:
    """Generate complete square packing with the requested Vf."""
    rows, columns, radius = _complete_lattice_shape_and_radius(Vf_target, radius)
    x_values = np.linspace(radius, DOMAIN_WIDTH_M - radius, columns)
    y_values = np.linspace(radius, DOMAIN_HEIGHT_M - radius, rows)
    fibers: list[Fiber] = []
    for y in y_values:
        for x in x_values:
            fibers.append(Fiber(f"sq_{len(fibers) + 1:03d}", float(x), float(y), radius))
    microstructure = Microstructure("square", Vf_target, DOMAIN_WIDTH_M, DOMAIN_HEIGHT_M, tuple(fibers))
    _validate_microstructure(microstructure)
    return microstructure


def generate_hexagonal_packing(Vf_target: float, radius: float = FIBER_RADIUS_M) -> Microstructure:
    """Generate complete staggered hexagonal-like packing with the requested Vf."""
    rows, columns, radius = _complete_lattice_shape_and_radius(Vf_target, radius)
    x_pitch = (DOMAIN_WIDTH_M - 2.0 * radius) / max(columns - 0.5, 1.0)
    y_values = np.linspace(radius, DOMAIN_HEIGHT_M - radius, rows)
    fibers: list[Fiber] = []
    for row, y in enumerate(y_values):
        offset = 0.5 * x_pitch if row % 2 else 0.0
        for column in range(columns):
            x = radius + column * x_pitch + offset
            if x <= DOMAIN_WIDTH_M - radius + 1.0e-12:
                fibers.append(Fiber(f"hex_{len(fibers) + 1:03d}", float(x), float(y), radius))
    microstructure = Microstructure("hexagonal", Vf_target, DOMAIN_WIDTH_M, DOMAIN_HEIGHT_M, tuple(fibers))
    _validate_microstructure(microstructure)
    return microstructure


def generate_random_packing(
    Vf_target: float,
    radius: float = FIBER_RADIUS_M,
    seed: int = 8,
) -> Microstructure:
    """Generate randomized non-overlapping fiber positions."""
    rng = np.random.default_rng(seed)
    base = generate_square_packing(Vf_target, radius)
    fibers = [Fiber(f"rnd_{i + 1:03d}", fiber.x, fiber.y, fiber.radius) for i, fiber in enumerate(base.fibers)]
    average_spacing = calculate_characteristic_spacing(fibers)
    step = max(0.2 * average_spacing, 0.5e-6)

    for _ in range(5000):
        index = int(rng.integers(0, len(fibers)))
        old = fibers[index]
        candidate = Fiber(
            old.fiber_id,
            float(np.clip(old.x + rng.normal(0.0, step), old.radius, DOMAIN_WIDTH_M - old.radius)),
            float(np.clip(old.y + rng.normal(0.0, step), old.radius, DOMAIN_HEIGHT_M - old.radius)),
            old.radius,
        )
        ok = True
        for other_index, other in enumerate(fibers):
            if other_index == index:
                continue
            if math.hypot(candidate.x - other.x, candidate.y - other.y) < 2.02 * radius:
                ok = False
                break
        if ok:
            fibers[index] = candidate

    microstructure = Microstructure("random", Vf_target, DOMAIN_WIDTH_M, DOMAIN_HEIGHT_M, tuple(fibers))
    _validate_microstructure(microstructure)
    return microstructure


def generate_microstructure(pattern: str, Vf_target: float) -> Microstructure:
    """Generate one of the supported microstructures."""
    key = pattern.strip().lower()
    if key in {"square", "sq"}:
        return generate_square_packing(Vf_target)
    if key in {"hexagonal", "hex"}:
        return generate_hexagonal_packing(Vf_target)
    if key in {"random", "randomized", "rnd"}:
        return generate_random_packing(Vf_target)
    raise ValueError(f"Unsupported pattern: {pattern}")


def spacing_metrics(microstructure: Microstructure) -> SpacingMetrics:
    """Calculate spacing metrics and a simple permeability correction."""
    fibers = microstructure.fibers
    all_spacings: list[float] = []
    for i, first in enumerate(fibers):
        for second in fibers[i + 1 :]:
            center_distance = math.hypot(first.x - second.x, first.y - second.y)
            all_spacings.append(center_distance - first.radius - second.radius)
    average_spacing = calculate_characteristic_spacing(fibers)
    minimum_spacing = float(min(all_spacings))
    resin_channel_width = 0.5 * (average_spacing + minimum_spacing)
    uniformity = max(0.0, min(1.0, minimum_spacing / max(average_spacing, 1.0e-30)))
    reference_spacing = max(average_spacing, 1.0e-30)
    spacing_correction = max(
        0.20,
        min(1.20, (resin_channel_width / reference_spacing) * math.sqrt(uniformity)),
    )
    return SpacingMetrics(
        average_spacing=average_spacing,
        minimum_spacing=minimum_spacing,
        resin_channel_width=resin_channel_width,
        spacing_correction=spacing_correction,
    )


def rasterize_fibers(
    microstructure: Microstructure,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Rasterize circular fibers onto a Cartesian grid."""
    dx = microstructure.width / nx
    dy = microstructure.height / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    solid = np.zeros((ny, nx), dtype=bool)
    for fiber in microstructure.fibers:
        solid |= (X - fiber.x) ** 2 + (Y - fiber.y) ** 2 <= fiber.radius**2
    return X, Y, x, y, dx, dy, solid


def inlet_connected_matrix(matrix_mask: np.ndarray) -> np.ndarray:
    """Return matrix cells connected to the inlet side."""
    ny, nx = matrix_mask.shape
    connected = np.zeros_like(matrix_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for j in range(ny):
        if matrix_mask[j, 0]:
            connected[j, 0] = True
            queue.append((j, 0))
    while queue:
        j, i = queue.popleft()
        for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if 0 <= jj < ny and 0 <= ii < nx and matrix_mask[jj, ii] and not connected[jj, ii]:
                connected[jj, ii] = True
                queue.append((jj, ii))
    return connected
