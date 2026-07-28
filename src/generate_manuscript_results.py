#!/usr/bin/env python3
"""Generate the numerical tables and figures used in the V3.2 manuscript."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boolean_phase_core import (
    ccz_robustness_coefficient,
    uniform_robustness_coefficient,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def composed_channel_fidelity(state_fidelity_squared: float) -> float:
    r"""Compose two equal test fidelities by the Bures-angle inequality.

    .. math::
       F_{\rm ch}\ge
       \max\{0,\cos(2\arccos\sqrt{F_{\rm ext}^2})\}.
    """

    if not 0.0 <= state_fidelity_squared <= 1.0:
        raise ValueError("state_fidelity_squared must lie in [0, 1]")
    return max(0.0, 2.0 * state_fidelity_squared - 1.0)


def diamond_distance_upper_bound(
    channel_root_fidelity: float,
    dimension: int,
) -> float:
    r"""Return \(\min\{2,2d\sqrt{1-F_{\rm ch}^2}\}\)."""

    if not 0.0 <= channel_root_fidelity <= 1.0:
        raise ValueError("channel_root_fidelity must lie in [0, 1]")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return min(
        2.0,
        2.0
        * dimension
        * math.sqrt(max(0.0, 1.0 - channel_root_fidelity**2)),
    )


def visibility_rows(
    visibilities: Iterable[float],
    slope: float,
    number_of_qubits: int = 3,
) -> list[dict[str, float | bool]]:
    r"""Evaluate the affine state and channel bounds for equal visibility.

    Each test has
    \(\beta(v)=2\sqrt2\,n v\) and
    the plotted affine curve is
    \(\max\{0,1-s(2\sqrt2 n-\beta)\}\).
    """

    if number_of_qubits < 1:
        raise ValueError("number_of_qubits must be positive")
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("slope must be finite and positive")
    quantum_bound = 2.0 * math.sqrt(2.0) * number_of_qubits
    dimension = 2**number_of_qubits
    rows: list[dict[str, float | bool]] = []
    for visibility in visibilities:
        visibility = float(visibility)
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("visibilities must lie in [0, 1]")
        deficit = quantum_bound * (1.0 - visibility)
        state_fidelity_squared = max(0.0, 1.0 - slope * deficit)
        channel_fidelity = composed_channel_fidelity(
            state_fidelity_squared
        )
        rows.append(
            {
                "visibility": visibility,
                "bell_value_each_test": quantum_bound * visibility,
                "bell_deficit_each_test": deficit,
                "bell_violation": visibility > 1.0 / math.sqrt(2.0),
                "state_fidelity_squared_lower": state_fidelity_squared,
                "standalone_state_fidelity_squared_lower": max(
                    1.0 / dimension,
                    state_fidelity_squared,
                ),
                "channel_root_fidelity_lower": channel_fidelity,
                "channel_entanglement_fidelity_lower": (
                    channel_fidelity**2
                ),
                "diamond_distance_upper": diamond_distance_upper_bound(
                    channel_fidelity,
                    dimension,
                ),
            }
        )
    return rows


def visibility_thresholds(
    slope: float,
    number_of_qubits: int = 3,
) -> dict[str, float]:
    """Return the four visibility thresholds quoted in the manuscript."""

    if number_of_qubits < 1:
        raise ValueError("number_of_qubits must be positive")
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("slope must be finite and positive")
    dimension = 2**number_of_qubits
    scale = 2.0 * math.sqrt(2.0) * number_of_qubits
    diamond_target = math.sqrt(1.0 - 1.0 / dimension**2)
    return {
        "bell_violation": 1.0 / math.sqrt(2.0),
        "state_positive": 1.0 - 1.0 / (slope * scale),
        "state_beats_floor": 1.0
        - (1.0 - 1.0 / dimension) / (slope * scale),
        "channel_positive": 1.0 - 1.0 / (2.0 * slope * scale),
        "diamond_nontrivial": 1.0
        - (1.0 - diamond_target) / (2.0 * slope * scale),
    }


def write_csv(
    path: Path,
    rows: Sequence[dict[str, float | bool]],
) -> None:
    """Write a table with stable column order."""

    if not rows:
        raise ValueError("rows must be nonempty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_ccz_bounds(
    rows: Sequence[dict[str, float | bool]],
    fidelity_path: Path,
    diamond_path: Path,
) -> None:
    """Plot the main fidelity bounds and appendix diamond-distance bound."""

    visibility = np.asarray([row["visibility"] for row in rows], dtype=float)
    state = np.asarray(
        [row["state_fidelity_squared_lower"] for row in rows],
        dtype=float,
    )
    standalone = np.asarray(
        [
            row["standalone_state_fidelity_squared_lower"]
            for row in rows
        ],
        dtype=float,
    )
    channel_squared = np.asarray(
        [row["channel_entanglement_fidelity_lower"] for row in rows],
        dtype=float,
    )
    diamond = np.asarray(
        [row["diamond_distance_upper"] for row in rows],
        dtype=float,
    )

    fidelity_figure, fidelity_axis = plt.subplots(figsize=(4.2, 2.75))
    fidelity_axis.plot(
        visibility,
        standalone,
        label=r"standalone state $F^2$",
    )
    fidelity_axis.plot(
        visibility,
        state,
        linestyle=":",
        label=r"common-map affine $F^2$",
    )
    fidelity_axis.plot(
        visibility,
        channel_squared,
        linestyle="--",
        label=r"channel $F_{\rm Choi}^2$",
    )
    fidelity_axis.set(
        xlabel="visibility in each test",
        ylabel="certified squared fidelity",
        xlim=(float(visibility.min()), float(visibility.max())),
        ylim=(-0.02, 1.02),
    )
    fidelity_axis.spines[["top", "right"]].set_visible(False)
    fidelity_axis.grid(alpha=0.2)
    fidelity_axis.legend(frameon=False, fontsize=8)
    fidelity_figure.tight_layout()
    fidelity_path.parent.mkdir(parents=True, exist_ok=True)
    fidelity_figure.savefig(
        fidelity_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fidelity_figure)

    diamond_figure, diamond_axis = plt.subplots(figsize=(4.2, 2.55))
    diamond_axis.plot(visibility, diamond, color="#A23E48")
    diamond_axis.axhline(
        2.0,
        color="0.45",
        linewidth=0.8,
        linestyle=":",
    )
    diamond_axis.set(
        xlabel="visibility in each test",
        ylabel=r"diamond-distance upper bound",
        xlim=(max(0.995, float(visibility.min())), float(visibility.max())),
        ylim=(-0.04, 2.04),
    )
    diamond_axis.spines[["top", "right"]].set_visible(False)
    diamond_axis.grid(alpha=0.2)
    diamond_figure.tight_layout()
    diamond_path.parent.mkdir(parents=True, exist_ok=True)
    diamond_figure.savefig(
        diamond_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(diamond_figure)


def generate_all(output_root: Path) -> None:
    """Generate the V3.2 result files and manuscript figures."""

    results_directory = output_root / "results"
    figure_directory = (
        output_root / "manuscript" / "quantum_submission" / "figures"
    )
    uniform_slope = float(uniform_robustness_coefficient())
    ccz_slope = float(ccz_robustness_coefficient())
    ccz_rows = visibility_rows(
        np.linspace(0.80, 1.0, 1001),
        ccz_slope,
    )
    write_csv(results_directory / "analytic_di_bounds.csv", ccz_rows)

    thresholds = {
        "uniform_boolean_phase": {
            "coefficient_exact": "3(1+sqrt(2))/8",
            "coefficient_decimal": uniform_slope,
            **visibility_thresholds(uniform_slope),
        },
        "ccz_fixed_channel": {
            "coefficient_exact": "(4+5sqrt(2))/16",
            "coefficient_decimal": ccz_slope,
            **visibility_thresholds(ccz_slope),
        },
    }
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / "analytic_di_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_ccz_bounds(
        ccz_rows,
        figure_directory / "di_robustness.pdf",
        figure_directory / "di_diamond.pdf",
    )


def parse_args() -> argparse.Namespace:
    """Parse the optional V3.2 root directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="V3.2 directory that contains results/ and manuscript/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_all(parse_args().output_root.resolve())
