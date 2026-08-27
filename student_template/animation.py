"""Animation of resin front, cure, gelation, and moving flow tracers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from flow_solver import FlowResult, velocity_from_viscosity
from geometry import Microstructure, calculate_fiber_volume_fraction
from impregnation import ImpregnationResult


def _sample_velocity(
    flow: FlowResult,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    blocked: np.ndarray,
    x_pos: float,
    y_pos: float,
) -> tuple[float, float, bool]:
    i = int(np.clip(x_pos / flow.dx, 0, flow.active_matrix_mask.shape[1] - 1))
    j = int(np.clip(y_pos / flow.dy, 0, flow.active_matrix_mask.shape[0] - 1))
    if not flow.active_matrix_mask[j, i] or blocked[j, i]:
        return 0.0, 0.0, False
    return float(velocity_x[j, i]), float(velocity_y[j, i]), True


def _reset_tracer(flow: FlowResult, rng: np.random.Generator, blocked: np.ndarray) -> tuple[float, float]:
    inlet_width = max(2, flow.active_matrix_mask.shape[1] // 20)
    candidates = np.argwhere(flow.active_matrix_mask[:, :inlet_width] & ~blocked[:, :inlet_width])
    if candidates.size == 0:
        candidates = np.argwhere(flow.active_matrix_mask & ~blocked)
    if candidates.size == 0:
        candidates = np.argwhere(flow.active_matrix_mask)
    local_j, local_i = candidates[int(rng.integers(0, len(candidates)))]
    return float((local_i + rng.random()) * flow.dx), float((local_j + rng.random()) * flow.dy)


def _nearest_snapshot(impregnation: ImpregnationResult, elapsed: float) -> int:
    return int(np.argmin(np.abs(impregnation.snapshot_times - elapsed)))


def save_impregnation_animation(
    path: str | Path,
    microstructure: Microstructure,
    flow: FlowResult,
    impregnation: ImpregnationResult,
    temperature_C: float,
    frame_count: int = 56,
    tracer_count: int = 36,
    try_mp4: bool = True,
) -> None:
    """Save an animation with fibers, front, cure, gel, and moving tracers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    time_end = float(impregnation.time[-1])
    dt = time_end / max(frame_count - 1, 1)
    blocked0 = impregnation.gelled_snapshots[0] | ~flow.active_matrix_mask
    tracer_xy = np.array([_reset_tracer(flow, rng, blocked0) for _ in range(tracer_count)])
    trail_xy = tracer_xy.copy()

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    saturation0 = np.where(flow.matrix_mask, 0.0, np.nan)
    saturation_image = ax.imshow(
        saturation0,
        origin="lower",
        extent=(0.0, microstructure.width * 1.0e6, 0.0, microstructure.height * 1.0e6),
        vmin=0.0,
        vmax=1.0,
        alpha=0.50,
        aspect="equal",
    )
    gel_image = ax.imshow(
        np.full_like(saturation0, np.nan),
        origin="lower",
        extent=(0.0, microstructure.width * 1.0e6, 0.0, microstructure.height * 1.0e6),
        vmin=0.0,
        vmax=1.0,
        alpha=0.35,
        aspect="equal",
    )
    for fiber in microstructure.fibers:
        ax.add_patch(
            Circle(
                (fiber.x * 1.0e6, fiber.y * 1.0e6),
                fiber.radius * 1.0e6,
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
            )
        )
    quiver = ax.quiver(
        tracer_xy[:, 0] * 1.0e6,
        tracer_xy[:, 1] * 1.0e6,
        np.ones(tracer_count) * 12.0,
        np.zeros(tracer_count),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    trails = ax.scatter(trail_xy[:, 0] * 1.0e6, trail_xy[:, 1] * 1.0e6, s=7, alpha=0.35)
    status = ax.text(
        0.015,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 3},
    )
    ax.set_xlim(0.0, microstructure.width * 1.0e6)
    ax.set_ylim(0.0, microstructure.height * 1.0e6)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(False)
    vf = calculate_fiber_volume_fraction(microstructure.fibers, microstructure.width, microstructure.height)

    def update(frame: int):
        elapsed = frame * dt
        snap_idx = _nearest_snapshot(impregnation, elapsed)
        saturation = impregnation.saturation_snapshots[snap_idx]
        gelled = impregnation.gelled_snapshots[snap_idx]
        viscosity = impregnation.viscosity_snapshots[snap_idx]
        velocity_x, velocity_y, _speed = velocity_from_viscosity(flow, viscosity, gelled)
        saturation_image.set_array(np.where(flow.matrix_mask, saturation.astype(float), np.nan))
        gel_image.set_array(np.where(gelled, 1.0, np.nan))

        arrow_u = np.zeros(tracer_count)
        arrow_v = np.zeros(tracer_count)
        blocked = gelled | ~flow.active_matrix_mask
        for idx in range(tracer_count):
            trail_xy[idx] = tracer_xy[idx]
            vx, vy, ok = _sample_velocity(flow, velocity_x, velocity_y, blocked, tracer_xy[idx, 0], tracer_xy[idx, 1])
            if not ok or tracer_xy[idx, 0] >= microstructure.width:
                tracer_xy[idx] = _reset_tracer(flow, rng, blocked)
                vx, vy, _ok = _sample_velocity(flow, velocity_x, velocity_y, blocked, tracer_xy[idx, 0], tracer_xy[idx, 1])
            tracer_xy[idx, 0] += vx * dt
            tracer_xy[idx, 1] += vy * dt
            vx2, vy2, ok2 = _sample_velocity(flow, velocity_x, velocity_y, blocked, tracer_xy[idx, 0], tracer_xy[idx, 1])
            if not ok2 or tracer_xy[idx, 0] < 0.0 or tracer_xy[idx, 0] > microstructure.width:
                tracer_xy[idx] = _reset_tracer(flow, rng, blocked)
                vx2, vy2, _ok2 = _sample_velocity(flow, velocity_x, velocity_y, blocked, tracer_xy[idx, 0], tracer_xy[idx, 1])
            direction_norm = max(float(np.hypot(vx2, vy2)), 1.0e-30)
            arrow_u[idx] = 16.0 * vx2 / direction_norm
            arrow_v[idx] = 16.0 * vy2 / direction_norm

        quiver.set_offsets(tracer_xy * 1.0e6)
        quiver.set_UVC(arrow_u, arrow_v)
        trails.set_offsets(trail_xy * 1.0e6)
        hist_idx = int(np.argmin(np.abs(impregnation.time - elapsed)))
        status.set_text(
            f"Time: {impregnation.time[hist_idx]:.0f} s\n"
            f"T: {temperature_C:.0f} C, Vf: {vf:.3f}\n"
            f"Avg mu: {impregnation.average_viscosity[hist_idx]:.2e} Pa s\n"
            f"Avg DoC: {impregnation.average_alpha[hist_idx]:.2f}\n"
            f"I: {impregnation.fraction[hist_idx]:.2f}\n"
            f"Gelled: {impregnation.gelled_fraction[hist_idx]:.2f}"
        )
        return saturation_image, gel_image, quiver, trails, status

    animation = FuncAnimation(fig, update, frames=frame_count, interval=90, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=12))
    if try_mp4 and FFMpegWriter.isAvailable():
        try:
            animation.save(output_path.with_suffix(".mp4"), writer=FFMpegWriter(fps=12))
        except Exception:
            pass
    plt.close(fig)
