"""Validated global-certificate regressions for the CCZ STOPI polynomial."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ccz_stopi_interval_certificate import (  # noqa: E402
    DyadicCell,
    _exact_bernstein_restriction,
    build_root_bernstein_data,
    cell_is_inside_exact_local_region,
    evaluate_power_tensor,
    verified_cholesky_floor,
)
from ccz_stopi_optimal import corrected_coupling_gap  # noqa: E402


def test_root_power_tensor_replays_the_physical_numerator() -> None:
    """The exact half-angle polynomial agrees with the physical operator."""

    data = build_root_bernstein_data(("Z",) * 6)
    x = np.array((0.17, 0.31, 0.43, 0.59, 0.71, 0.83))
    numerator = evaluate_power_tensor(data.power_matrices, x)
    t = math.sqrt(2.0) - 1.0
    z = t * x
    angles = math.pi / 4.0 - 2.0 * np.arctan(z)
    physical = corrected_coupling_gap(
        angles[:3],
        angles[3:],
        eta=0.5,
    )
    physical *= np.prod(1.0 + z * z)

    assert data.power_matrices.shape == (3,) * 6 + (64, 64)
    assert data.bernstein_matrices.shape == (3,) * 6 + (64, 64)
    assert np.linalg.norm(numerator - physical) < 3e-12
    assert float(np.max(data.bernstein_errors)) < 1e-9


def test_verified_cholesky_floor_accounts_for_input_error() -> None:
    """The residual audit proves a floor and rejects an indefinite matrix."""

    positive = np.diag(np.linspace(0.25, 2.0, 64))
    floor = verified_cholesky_floor(
        positive,
        input_error=1e-14,
        proposed_shift=0.1,
    )
    indefinite = positive.copy()
    indefinite[0, 0] = -0.01

    assert floor > 0.09
    assert (
        verified_cholesky_floor(
            indefinite,
            input_error=1e-14,
            proposed_shift=1e-12,
        )
        == 0.0
    )
    asymmetric = positive.copy()
    asymmetric[0, 1] = 1e-8
    with pytest.raises(ArithmeticError, match="exactly symmetric"):
        verified_cholesky_floor(
            asymmetric,
            input_error=1e-14,
            proposed_shift=0.1,
        )


def test_exact_bernstein_restriction_replays_a_quadratic() -> None:
    """The rational subcell transform preserves degree-two polynomials."""

    lower = Fraction(3, 16)
    upper = Fraction(11, 16)
    transform = _exact_bernstein_restriction(lower, upper)
    coefficients = (Fraction(2, 7), Fraction(-1, 5), Fraction(9, 11))
    restricted = tuple(
        sum(
            weight * coefficient
            for weight, coefficient in zip(row, coefficients)
        )
        for row in transform
    )

    def evaluate_bernstein(
        values: tuple[Fraction, Fraction, Fraction],
        coordinate: Fraction,
    ) -> Fraction:
        return (
            values[0] * (1 - coordinate) ** 2
            + 2 * values[1] * coordinate * (1 - coordinate)
            + values[2] * coordinate**2
        )

    for local in (Fraction(0), Fraction(5, 13), Fraction(1)):
        global_coordinate = lower + (upper - lower) * local
        assert evaluate_bernstein(restricted, local) == (
            evaluate_bernstein(coefficients, global_coordinate)
        )


def test_dyadic_local_handoffs_use_the_exact_l1_regions() -> None:
    """Ideal and endpoint corner cells are accepted by exact surd tests."""

    ideal = DyadicCell(
        levels=(12,) * 6,
        indices=(0,) * 6,
    )
    endpoint = DyadicCell(
        levels=(12,) * 6,
        indices=(2**12 - 1, 0, 0, 2**12 - 1, 0, 0),
    )
    root = DyadicCell(
        levels=(0,) * 6,
        indices=(0,) * 6,
    )

    assert cell_is_inside_exact_local_region(ideal, ("Z",) * 6)
    assert cell_is_inside_exact_local_region(endpoint, ("Z",) * 6)
    assert not cell_is_inside_exact_local_region(root, ("Z",) * 6)
    assert ideal.upper(0) == Fraction(1, 2**12)
