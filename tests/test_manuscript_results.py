"""Regressions for the numerical values and figure generator."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from generate_manuscript_results import (  # noqa: E402
    generate_all,
    visibility_rows,
    visibility_thresholds,
)


def test_thresholds_match_the_values_quoted_in_the_manuscript() -> None:
    """Both theorem coefficients reproduce the four printed thresholds."""

    uniform = visibility_thresholds(3.0 * (1.0 + math.sqrt(2.0)) / 8.0)
    ccz = visibility_thresholds((4.0 + 5.0 * math.sqrt(2.0)) / 16.0)

    assert np.allclose(
        [
            uniform["state_positive"],
            uniform["state_beats_floor"],
            uniform["channel_positive"],
            uniform["diamond_nontrivial"],
        ],
        [0.869825, 0.886097, 0.934913, 0.999490],
        atol=5e-7,
        rtol=0.0,
    )
    assert np.allclose(
        [
            ccz["state_positive"],
            ccz["state_beats_floor"],
            ccz["channel_positive"],
            ccz["diamond_nontrivial"],
        ],
        [0.829681, 0.850970, 0.914840, 0.999332],
        atol=5e-7,
        rtol=0.0,
    )


def test_equal_test_composition_has_the_expected_closed_form() -> None:
    r"""Two equal Bures angles give \(F_{\rm ch}=[2F^2-1]_+\)."""

    slope = (4.0 + 5.0 * math.sqrt(2.0)) / 16.0
    rows = visibility_rows((0.8, 0.95, 1.0), slope)

    for row in rows:
        expected = max(
            0.0,
            2.0 * float(row["state_fidelity_squared_lower"]) - 1.0,
        )
        assert abs(float(row["channel_root_fidelity_lower"]) - expected) < 1e-15
        assert abs(
            float(row["channel_entanglement_fidelity_lower"])
            - expected**2
        ) < 1e-15


def test_generator_writes_the_reproducibility_surface(tmp_path: Path) -> None:
    """The public entry point writes data and separate main/appendix figures."""

    generate_all(tmp_path)
    threshold_path = tmp_path / "results" / "analytic_di_thresholds.json"
    data = json.loads(threshold_path.read_text(encoding="utf-8"))

    assert set(data) == {"uniform_boolean_phase", "ccz_fixed_channel"}
    assert (tmp_path / "results" / "analytic_di_bounds.csv").stat().st_size > 1000
    assert (
        tmp_path
        / "manuscript"
        / "quantum_submission"
        / "figures"
        / "di_robustness.pdf"
    ).stat().st_size > 1000
    assert (
        tmp_path
        / "manuscript"
        / "quantum_submission"
        / "figures"
        / "di_diamond.pdf"
    ).stat().st_size > 1000
