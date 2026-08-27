"""Run the instructor solution for the fiber/cure impregnation homework."""

from __future__ import annotations

import inspect
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animation import save_impregnation_animation
from config import SimulationConfig, default_config
from cure import cure_rate
from flow_solver import (
    DELTA_PRESSURE_PA,
    calculate_longitudinal_permeability,
    solve_flow,
    velocity_from_viscosity,
)
from geometry import (
    Microstructure,
    calculate_fiber_volume_fraction,
    generate_microstructure,
    spacing_metrics,
)
from impregnation import (
    ImpregnationResult,
    calculate_impregnation_time,
    run_coupled_impregnation,
)
from postprocess import (
    plot_fiber_patterns,
    plot_impregnation_curves,
    plot_spacing_vs_vf,
    plot_temperature_metric,
    plot_time_curves,
    plot_viscosity,
    write_summary_csv,
)
from viscosity import calculate_viscosity_temperature_only


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TODO_FUNCTIONS = [
    ("geometry", "calculate_fiber_volume_fraction"),
    ("geometry", "calculate_characteristic_spacing"),
    ("viscosity", "calculate_viscosity_temperature_only"),
    ("impregnation", "calculate_impregnation_fraction"),
]


@dataclass(frozen=True)
class CaseResult:
    """All computed outputs for one simulation case."""

    study: str
    label: str
    pattern: str
    Vf_target: float
    temperature_C: float
    microstructure: Microstructure
    viscosity_initial: float
    average_spacing: float
    minimum_spacing: float
    resin_channel_width: float
    spacing_correction: float
    flow: object
    impregnation: ImpregnationResult
    characteristic_time: float
    longitudinal_permeability: float


def check_for_unfinished_todos() -> None:
    """Stop with a clear message if any TODO function still contains pass."""
    import cure
    import geometry
    import impregnation
    import viscosity

    modules = {
        "geometry": geometry,
        "viscosity": viscosity,
        "cure": cure,
        "impregnation": impregnation,
    }
    for module_name, function_name in TODO_FUNCTIONS:
        function = getattr(modules[module_name], function_name)
        source = inspect.getsource(function)
        has_pass_line = re.search(r"^\s*pass\s*(#.*)?$", source, re.MULTILINE)
        has_todo_call = "_todo(" in source
        if has_pass_line or has_todo_call:
            raise NotImplementedError(
                f"{function_name} in {module_name}.py is still incomplete. "
                "Complete this TODO function before running the homework."
            )


def _threshold(value: float) -> str | float:
    """Report thresholds not reached by the final physical simulation time."""
    if not np.isfinite(value):
        return "NOT REACHED"
    return float(value)


def run_case(
    study: str,
    pattern: str,
    Vf_target: float,
    temperature_C: float,
    label: str,
    config: SimulationConfig,
) -> CaseResult:
    """Run one geometry, static flow, and coupled cure/impregnation simulation."""
    microstructure = generate_microstructure(pattern, Vf_target)
    spacing = spacing_metrics(microstructure)
    viscosity_initial = calculate_viscosity_temperature_only(
        temperature_C,
        config.viscosity.mu_ref_Pa_s,
        config.viscosity.T_ref_C,
        config.viscosity.E_mu_J_per_mol,
        config.viscosity.R_J_per_mol_K,
    )
    flow = solve_flow(
        microstructure,
        viscosity_initial,
        nx=config.domain.nx,
        ny=config.domain.ny,
        pressure_left_Pa=config.flow.pressure_left_Pa,
        pressure_right_Pa=config.flow.pressure_right_Pa,
        flow_config=config.flow,
    )
    impregnation = run_coupled_impregnation(
        flow,
        temperature_C,
        time_config=config.time,
        cure_config=config.cure,
        viscosity_config=config.viscosity,
    )
    vf = calculate_fiber_volume_fraction(microstructure.fibers, microstructure.width, microstructure.height)
    longitudinal = calculate_longitudinal_permeability(vf, spacing.spacing_correction, config=config.flow)
    characteristic_time = calculate_impregnation_time(
        viscosity_initial,
        microstructure.width,
        flow.permeability,
        DELTA_PRESSURE_PA,
    )
    return CaseResult(
        study=study,
        label=label,
        pattern=pattern,
        Vf_target=Vf_target,
        temperature_C=temperature_C,
        microstructure=microstructure,
        viscosity_initial=viscosity_initial,
        average_spacing=spacing.average_spacing,
        minimum_spacing=spacing.minimum_spacing,
        resin_channel_width=spacing.resin_channel_width,
        spacing_correction=spacing.spacing_correction,
        flow=flow,
        impregnation=impregnation,
        characteristic_time=characteristic_time,
        longitudinal_permeability=longitudinal,
    )


def case_summary_row(case: CaseResult) -> dict[str, object]:
    """Create one summary CSV row."""
    achieved_vf = calculate_fiber_volume_fraction(
        case.microstructure.fibers,
        case.microstructure.width,
        case.microstructure.height,
    )
    min_viscosity_idx = int(np.argmin(case.impregnation.average_viscosity))
    return {
        "study": case.study,
        "case": case.label,
        "pattern": case.pattern,
        "Vf_target": case.Vf_target,
        "Vf_achieved": achieved_vf,
        "temperature_C": case.temperature_C,
        "average_spacing_um": case.average_spacing * 1.0e6,
        "minimum_spacing_um": case.minimum_spacing * 1.0e6,
        "resin_channel_width_um": case.resin_channel_width * 1.0e6,
        "initial_viscosity_Pa_s": case.viscosity_initial,
        "minimum_average_viscosity_Pa_s": float(np.min(case.impregnation.average_viscosity)),
        "time_of_minimum_viscosity_s": float(case.impregnation.time[min_viscosity_idx]),
        "effective_transverse_permeability_m2": case.flow.permeability,
        "effective_longitudinal_permeability_m2": case.longitudinal_permeability,
        "average_resin_velocity_initial_m_per_s": case.flow.average_velocity,
        "maximum_resin_velocity_initial_m_per_s": case.flow.maximum_velocity,
        "characteristic_impregnation_time_s": case.characteristic_time,
        "I25_time_s": _threshold(case.impregnation.t25),
        "I50_time_s": _threshold(case.impregnation.t50),
        "I75_time_s": _threshold(case.impregnation.t75),
        "I90_time_s": _threshold(case.impregnation.t90),
        "final_I": case.impregnation.final_fraction,
        "final_average_DoC": case.impregnation.final_average_alpha,
        "final_matrix_average_DoC": case.impregnation.final_matrix_average_alpha,
        "final_gelled_fraction": case.impregnation.final_gelled_fraction,
        "gel_time_s": _threshold(case.impregnation.gel_time),
    }


def _curves(cases: list[CaseResult], attribute: str) -> list[tuple[str, list[float], list[float]]]:
    curves = []
    for case in cases:
        y = getattr(case.impregnation, attribute)
        curves.append((case.label, case.impregnation.time.tolist(), y.tolist()))
    return curves


def run_physical_checks(
    pattern_cases: list[CaseResult],
    vf_cases: list[CaseResult],
    temperature_cases: list[CaseResult],
    config: SimulationConfig,
) -> None:
    """Print automatic physical checks."""
    vf_values = [
        calculate_fiber_volume_fraction(case.microstructure.fibers, case.microstructure.width, case.microstructure.height)
        for case in pattern_cases
    ]
    permeabilities = [case.flow.permeability for case in vf_cases]
    viscosities = [case.viscosity_initial for case in temperature_cases]
    longitudinal = [case.longitudinal_permeability for case in vf_cases]
    alpha_valid = all(
        np.all(case.impregnation.final_alpha >= 0.0) and np.all(case.impregnation.final_alpha <= 1.0)
        for case in temperature_cases
    )
    rate_low = float(cure_rate(0.2, 50.0 + 273.15, config.cure))
    rate_high = float(cure_rate(0.2, 100.0 + 273.15, config.cure))
    mu_field = np.full(pattern_cases[0].flow.active_matrix_mask.shape, 10.0)
    vx_slow, vy_slow, speed_slow = velocity_from_viscosity(pattern_cases[0].flow, mu_field)
    fiber_impermeable = np.all(pattern_cases[0].flow.speed[pattern_cases[0].flow.fiber_mask] == 0.0)
    fraction_valid = all(
        np.all(case.impregnation.fraction >= 0.0) and np.all(case.impregnation.fraction <= 1.0)
        for case in temperature_cases + vf_cases + pattern_cases
    )
    gel_alpha_ok = all(
        np.all(case.impregnation.final_alpha[case.impregnation.final_gelled] >= config.cure.alpha_gel)
        for case in temperature_cases
    )
    checks = [
        ("achieved Vf is between 0 and 1", all(0.0 < vf < 1.0 for vf in vf_values)),
        (
            "average fiber spacing decreases as Vf increases",
            all(a.average_spacing > b.average_spacing for a, b in zip(vf_cases, vf_cases[1:])),
        ),
        (
            "uncured viscosity decreases as temperature increases",
            all(a > b for a, b in zip(viscosities, viscosities[1:])),
        ),
        (
            "Kamal-Sourour cure rate increases with temperature over representative conditions",
            rate_high > rate_low,
        ),
        ("degree of cure remains between 0 and 1", alpha_valid),
        ("gelation occurs when alpha reaches approximately 0.53", gel_alpha_ok),
        (
            "increasing cure increases resin viscosity",
            np.exp(config.viscosity.C_alpha * 0.5) > np.exp(config.viscosity.C_alpha * 0.1),
        ),
        (
            "flow velocity decreases as local viscosity increases",
            float(np.mean(speed_slow[pattern_cases[0].flow.active_matrix_mask])) < pattern_cases[0].flow.average_velocity,
        ),
        ("fiber cells remain impermeable", fiber_impermeable),
        ("impregnation fraction remains between 0 and 1", fraction_valid),
        (
            "permeability decreases with increasing fiber volume fraction",
            all(a > b for a, b in zip(permeabilities, permeabilities[1:])),
        ),
        (
            "K_parallel is greater than K_perp",
            all(kp > kt for kp, kt in zip(longitudinal, permeabilities)),
        ),
    ]
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {label}")


def _animation_pattern_name(pattern: str) -> str:
    """Use compact names for required animation filenames."""
    if pattern == "hexagonal":
        return "hex"
    return pattern


def main(run_animations: bool = True) -> None:
    """Run all requested studies and generate student outputs."""
    check_for_unfinished_todos()
    config = default_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pattern_cases = [
        run_case(
            "pattern",
            pattern,
            config.study.pattern_study_vf,
            config.study.pattern_study_temperature_C,
            f"{pattern}, Vf=0.50, T=50 C",
            config,
        )
        for pattern in config.study.patterns
    ]
    vf_cases = [
        run_case(
            "Vf",
            config.study.vf_study_pattern,
            vf,
            config.study.vf_study_temperature_C,
            f"square, Vf={vf:.2f}, T=50 C",
            config,
        )
        for vf in config.study.vf_values
    ]
    temperature_cases = [
        run_case(
            "temperature",
            config.study.temperature_study_pattern,
            config.study.temperature_study_vf,
            temperature,
            f"square, Vf=0.50, T={temperature:.0f} C",
            config,
        )
        for temperature in config.study.temperatures_C
    ]
    pattern_temperature_cases = [
        run_case(
            "pattern_temperature",
            pattern,
            config.study.pattern_study_vf,
            temperature,
            f"{pattern}, Vf=0.50, T={temperature:.0f} C",
            config,
        )
        for pattern in config.study.patterns
        for temperature in config.study.pattern_temperature_temperatures_C
    ]

    plot_fiber_patterns(RESULTS_DIR / "fiber_patterns.png", [case.microstructure for case in pattern_cases])
    plot_spacing_vs_vf(
        RESULTS_DIR / "spacing_vs_Vf.png",
        list(config.study.vf_values),
        [case.average_spacing for case in vf_cases],
        [case.minimum_spacing for case in vf_cases],
    )
    plot_viscosity(
        RESULTS_DIR / "viscosity_vs_temperature.png",
        list(config.study.temperatures_C),
        [case.viscosity_initial for case in temperature_cases],
    )
    plot_time_curves(RESULTS_DIR / "cure_rate_vs_time.png", _curves(temperature_cases, "average_cure_rate"), "average cure rate (1/s)")
    plot_time_curves(
        RESULTS_DIR / "degree_of_cure_vs_time.png",
        _curves(temperature_cases, "matrix_average_alpha"),
        "matrix-averaged degree of cure (-)",
        smooth_window=9,
    )
    plot_time_curves(
        RESULTS_DIR / "degree_of_cure_vs_time_pattern_temperature.png",
        _curves(pattern_temperature_cases, "matrix_average_alpha"),
        "matrix-averaged degree of cure (-)",
        smooth_window=9,
        legend_outside=True,
    )
    plot_time_curves(
        RESULTS_DIR / "viscosity_vs_time.png",
        _curves(temperature_cases, "average_viscosity"),
        "average viscosity (Pa s)",
        y_scale="log",
        inset_xlim=(0.0, 160.0),
        inset_ylim=(0.0, 12.0),
    )
    plot_impregnation_curves(RESULTS_DIR / "impregnation_vs_time_pattern.png", _curves(pattern_cases, "fraction"))
    plot_impregnation_curves(
        RESULTS_DIR / "impregnation_vs_time_pattern_temperature.png",
        _curves(pattern_temperature_cases, "fraction"),
        legend_outside=True,
    )
    plot_impregnation_curves(
        RESULTS_DIR / "impregnation_vs_time_Vf.png",
        _curves(vf_cases, "fraction"),
        inset_xlim=(0.0, 260.0),
        inset_ylim=(0.0, 0.35),
    )
    plot_impregnation_curves(
        RESULTS_DIR / "impregnation_vs_time_Vf_zoom.png",
        _curves(vf_cases, "fraction"),
        xlim=(0.0, 300.0),
        ylim=(0.0, 0.38),
    )
    plot_impregnation_curves(RESULTS_DIR / "impregnation_vs_time_temperature.png", _curves(temperature_cases, "fraction"))
    plot_temperature_metric(
        RESULTS_DIR / "final_impregnation_vs_temperature.png",
        list(config.study.temperatures_C),
        [case.impregnation.final_fraction for case in temperature_cases],
        "final impregnation fraction (-)",
    )
    plot_temperature_metric(
        RESULTS_DIR / "gel_time_vs_temperature.png",
        list(config.study.temperatures_C),
        [case.impregnation.gel_time if np.isfinite(case.impregnation.gel_time) else np.nan for case in temperature_cases],
        "gel time (s)",
    )

    rows = [
        case_summary_row(case)
        for case in pattern_cases + vf_cases + temperature_cases + pattern_temperature_cases
    ]
    write_summary_csv(RESULTS_DIR / "results_summary.csv", rows)
    write_summary_csv(RESULTS_DIR / "expected_results.csv", rows)

    if run_animations:
        animation_cases = pattern_cases + vf_cases + temperature_cases + pattern_temperature_cases
        seen: set[str] = set()
        for case in animation_cases:
            pattern_name = _animation_pattern_name(case.pattern)
            filename = f"animation_{pattern_name}_Vf{int(round(case.Vf_target * 100)):03d}_T{int(case.temperature_C)}.gif"
            if filename in seen:
                continue
            seen.add(filename)
            save_impregnation_animation(
                RESULTS_DIR / filename,
                case.microstructure,
                case.flow,
                case.impregnation,
                case.temperature_C,
            )
            if case.pattern == "hexagonal":
                alias = f"animation_hexagonal_Vf{int(round(case.Vf_target * 100)):03d}_T{int(case.temperature_C)}.gif"
                shutil.copyfile(RESULTS_DIR / filename, RESULTS_DIR / alias)

    run_physical_checks(pattern_cases, vf_cases, temperature_cases, config)
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
