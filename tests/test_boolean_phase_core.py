"""Reference-model regressions for the Boolean-phase Bell construction."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from boolean_phase_core import (  # noqa: E402
    ccz_phase,
    ccz_robustness_coefficient,
    derivative,
    local_deterministic_bound,
    nonsymmetric_phase,
    reference_checks,
    reference_state_and_stabilizers,
    uniform_robustness_coefficient,
)


def test_boolean_derivative_matches_ccz_control_product() -> None:
    """For CCZ, each derivative is the product of the other two bits."""

    for bit_string in (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ):
        for site in range(3):
            expected = math.prod(
                bit for index, bit in enumerate(bit_string) if index != site
            )
            assert derivative(ccz_phase, bit_string, site) == expected


def test_reference_state_is_the_unique_bell_maximizer() -> None:
    """The generalized stabilizers reproduce the ideal Bell spectrum."""

    state, stabilizers = reference_state_and_stabilizers(3, ccz_phase)
    eigenvalues = reference_checks(3, ccz_phase)

    assert np.isclose(np.linalg.norm(state), 1.0)
    assert len(stabilizers) == 6
    assert abs(eigenvalues[-1] - 6.0 * math.sqrt(2.0)) < 1e-11
    assert np.count_nonzero(np.isclose(eigenvalues, eigenvalues[-1])) == 1


def test_local_bound_is_six_for_distinct_boolean_phases() -> None:
    """The deterministic bound is independent of the Boolean derivative."""

    for phase_function in (ccz_phase, nonsymmetric_phase):
        bound, maximizers = local_deterministic_bound(3, phase_function)
        assert bound == 6
        assert maximizers > 0


def test_two_robustness_coefficients_are_exact_and_ordered() -> None:
    """The CCZ certificate is sharper than the uniform certificate."""

    uniform = uniform_robustness_coefficient()
    ccz = ccz_robustness_coefficient()

    assert uniform == 3 * (1 + sp.sqrt(2)) / 8
    assert ccz == (4 + 5 * sp.sqrt(2)) / 16
    assert float(ccz) < float(uniform) < 1.0

