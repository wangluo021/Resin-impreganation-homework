"""Plot and CSV output helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from geometry import Microstructure


def _smoothed(values: list[float], window: int) -> np.ndarray:
    """Return a centered moving-average copy for visual readability."""
    array = np.asarray(values, dtype=float)
    if window <= 1 or array.size < 3:
        return array
    window = min(window, array.size)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(array, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write a flat summary CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_fiber_patterns(path: str | Path, microstructures: list[Microstructure]) -> None:
    """Plot circular fiber geometry for several microstructures."""
    fig, axes = plt.subplots(len(microstructures), 1, figsize=(7.0, 6.0), constrained_layout=True)
    if len(microstructures) == 1:
        axes = [axes]
    for ax, microstructure in zip(axes, microstructures):
        for fiber in microstructure.fibers:
            ax.add_patch(
                Circle(
                    (fiber.x * 1.0e6, fiber.y * 1.0e6),
                    fiber.radius * 1.0e6,
                    fill=False,
                    linewidth=0.7,
                )
            )
        ax.set_xlim(0.0, microstructure.width * 1.0e6)
        ax.set_ylim(0.0, microstructure.height * 1.0e6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel(f"{microstructure.pattern}\ny (um)")
        ax.grid(False)
    axes[-1].set_xlabel("x (um)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_viscosity(path: str | Path, temperatures_C: list[float], viscosities: list[float]) -> None:
    """Plot viscosity versus temperature."""
    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    ax.plot(temperatures_C, viscosities, marker="o")
    ax.set_xlabel("temperature (deg C)")
    ax.set_ylabel("viscosity (Pa s)")
    ax.grid(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_temperature_metric(
    path: str | Path,
    temperatures_C: list[float],
    values: list[float],
    y_label: str,
) -> None:
    """Plot a metric versus temperature."""
    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    ax.plot(temperatures_C, values, marker="o")
    ax.set_xlabel("temperature (deg C)")
    ax.set_ylabel(y_label)
    ax.grid(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_impregnation_curves(
    path: str | Path,
    curves: list[tuple[str, list[float], list[float]]],
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    inset_xlim: tuple[float, float] | None = None,
    inset_ylim: tuple[float, float] | None = None,
    legend_outside: bool = False,
) -> None:
    """Plot I(t) curves."""
    figsize = (7.2, 4.1) if legend_outside else (5.8, 3.9)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for label, time, fraction in curves:
        ax.plot(time, fraction, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("impregnation fraction, I(t) (-)")
    ax.set_ylim(*(ylim or (0.0, 1.02)))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if legend_outside:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend()
    ax.grid(False)
    if inset_xlim is not None and inset_ylim is not None:
        inset = ax.inset_axes([0.54, 0.14, 0.39, 0.38])
        for _label, time, fraction in curves:
            inset.plot(time, fraction)
        inset.set_xlim(*inset_xlim)
        inset.set_ylim(*inset_ylim)
        inset.set_xlabel("time (s)", fontsize=7)
        inset.set_ylabel("I(t)", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.grid(False)
        ax.indicate_inset_zoom(inset, edgecolor="0.35", linewidth=0.8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_time_curves(
    path: str | Path,
    curves: list[tuple[str, list[float], list[float]]],
    y_label: str,
    y_scale: str = "linear",
    inset_xlim: tuple[float, float] | None = None,
    inset_ylim: tuple[float, float] | None = None,
    smooth_window: int = 1,
    legend_outside: bool = False,
) -> None:
    """Plot generic time-history curves."""
    use_zoom_panel = inset_xlim is not None and inset_ylim is not None
    if use_zoom_panel:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.4, 5.6) if legend_outside else (7.3, 5.6),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        ax, zoom_ax = axes
    else:
        fig, ax = plt.subplots(figsize=(7.6, 3.9) if legend_outside else (6.8, 3.9), constrained_layout=True)
        zoom_ax = None

    for label, time, values in curves:
        ax.plot(time, _smoothed(values, smooth_window), label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(y_label)
    ax.set_yscale(y_scale)
    if legend_outside:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(False)
    if zoom_ax is not None:
        for _label, time, values in curves:
            zoom_ax.plot(time, _smoothed(values, smooth_window))
        zoom_ax.set_xlim(*inset_xlim)
        zoom_ax.set_ylim(*inset_ylim)
        zoom_ax.set_xlabel("time (s)")
        zoom_ax.set_ylabel(y_label)
        zoom_ax.grid(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_spacing_vs_vf(path: str | Path, vf_values: list[float], average: list[float], minimum: list[float]) -> None:
    """Plot spacing trends versus target Vf."""
    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    ax.plot(vf_values, [value * 1.0e6 for value in average], marker="o", label="average spacing")
    ax.plot(vf_values, [value * 1.0e6 for value in minimum], marker="s", label="minimum spacing")
    ax.set_xlabel("target fiber volume fraction, Vf (-)")
    ax.set_ylabel("surface-to-surface spacing (um)")
    ax.legend()
    ax.grid(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)
