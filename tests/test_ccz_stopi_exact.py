"""Exact-algebra regressions for the CCZ STOPI endpoint certificate."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ccz_stopi_exact import (  # noqa: E402
    QuadraticSurd,
    canonical_endpoint_kernel_vectors,
    endpoint_complement_unit_lower_bound_audit,
    endpoint_quadratic_positivity_audit,
    exact_endpoint_full_numerator_coefficients,
    exact_endpoint_local_certificate,
    exact_endpoint_numerator_coefficients,
    exact_ideal_full_numerator_coefficients,
    exact_ideal_local_certificate,
    exact_ideal_schur_quadratic,
    ideal_complement_two_fifths_lower_bound_audit,
    exact_endpoint_schur_quadratic,
    exact_pauli_action,
    exact_target_pauli_coefficients,
    endpoint_adjacent_branch_axes,
    local_branch_orbit_representatives,
    pauli_dictionary_matrix,
)
from ccz_stopi_local_certificate import (  # noqa: E402
    _cleared_gap_coefficients,
)
from ccz_stopi_optimal import (  # noqa: E402
    corrected_coupling_gap,
    target_projector,
)


def test_quadratic_surd_field_arithmetic_is_exact() -> None:
    """The custom field implements rational arithmetic in ``Q(sqrt(2))``."""

    root_two = QuadraticSurd(0, 1)
    inverse = QuadraticSurd(0, Fraction(1, 2))

    assert root_two * root_two == QuadraticSurd(2, 0)
    assert root_two * inverse == QuadraticSurd(1, 0)
    assert (QuadraticSurd(1, 1)).inverse() == QuadraticSurd(-1, 1)


def test_local_branch_orbits_cover_all_low_high_patterns() -> None:
    """The exact-local audit uses the same 20 orbits as the global proof."""

    representatives = local_branch_orbit_representatives()

    assert len(representatives) == 20
    assert sum(
        int(representative["orbit_size"])
        for representative in representatives
    ) == 64
    assert len(
        {representative["axes"] for representative in representatives}
    ) == 20


def test_endpoint_adjacent_branch_list_has_sixteen_patterns() -> None:
    """The canonical endpoint has four independently chosen good axes."""

    branches = endpoint_adjacent_branch_axes()

    assert len(branches) == 16
    assert len(set(branches)) == 16
    assert all(axes[0] == "Z" and axes[3] == "Z" for axes in branches)


def test_exact_target_pauli_expansion_replays_the_rotated_projector() -> None:
    """The 783 nonzero Pauli coefficients reproduce the target exactly."""

    coefficients = exact_target_pauli_coefficients()
    replay = pauli_dictionary_matrix(coefficients)

    assert len(coefficients) == 783
    assert np.linalg.norm(replay - target_projector()) < 3e-15


def test_exact_endpoint_degree_two_numerator_matches_float_reconstruction() -> None:
    """The exact Pauli polynomial independently reconstructs every low degree."""

    axes = ("Z",) * 6
    exact = exact_endpoint_numerator_coefficients(axes)
    floating = _cleared_gap_coefficients(
        axes,
        bad_site=0,
        bob_endpoint=0.0,
        eta=0.5,
    )

    assert len(exact) == 28
    for exponent, pauli_coefficients in exact.items():
        replay = pauli_dictionary_matrix(pauli_coefficients)
        assert np.linalg.norm(replay - floating[exponent]) < 2e-12


def test_full_exact_endpoint_numerator_extends_the_low_degree_data() -> None:
    """The full tensor has all ``3^6`` exponents and preserves low degrees."""

    axes = ("Z",) * 6
    low_degree = exact_endpoint_numerator_coefficients(axes)
    full = exact_endpoint_full_numerator_coefficients(axes)

    assert len(full) == 3**6
    for exponent, coefficients in low_degree.items():
        assert full[exponent] == coefficients


def test_exact_ideal_numerator_reproduces_the_coupling_equality_matrix() -> None:
    """The all-ideal constructor has 729 coefficients and the correct constant."""

    axes = ("Z",) * 6
    full = exact_ideal_full_numerator_coefficients(axes)
    equality = pauli_dictionary_matrix(full[(0,) * 6])
    angles = (math.pi / 4.0,) * 3
    floating = corrected_coupling_gap(
        angles,
        angles,
        eta=0.5,
    )

    assert len(full) == 3**6
    assert np.linalg.norm(equality - floating) < 2e-12


def test_ideal_schur_quadratic_is_exactly_diagonal_and_positive() -> None:
    """The ideal kernel curvature has no mixed terms and a ``1/4`` floor."""

    quadratic = exact_ideal_schur_quadratic(("Z",) * 6)
    zero = QuadraticSurd()

    for exponent, coefficient in quadratic.items():
        active = [degree for degree in exponent if degree]
        if active == [2]:
            assert coefficient.is_positive()
        else:
            assert coefficient == zero
    for coordinate in range(3):
        exponent = tuple(
            2 if index == coordinate else 0 for index in range(6)
        )
        assert quadratic[exponent] == QuadraticSurd(Fraction(1, 4))


def test_ideal_complement_and_local_remainder_close_exactly() -> None:
    """The ideal orbit has an exact complement floor and a rational radius."""

    complement = ideal_complement_two_fifths_lower_bound_audit(("Z",) * 6)
    certificate = exact_ideal_local_certificate(("Z",) * 6)

    assert complement["passed"] is True
    assert complement["nullity"] == 1
    assert certificate["radius"] == Fraction(29, 60000)
    assert certificate["l1_radius"] == Fraction(29, 10000)
    assert certificate["passed"] is True
    assert float(certificate["quadratic_simplex_lower_bound"]) > 0.062
    assert float(certificate["remainder_bound"]) < 0.06
    assert certificate["bound_method"] == "projected kernel/complement norms"


def test_canonical_endpoint_kernel_is_exact_and_orthogonal() -> None:
    """The two matched-sector vectors span the exact endpoint nullspace."""

    numerator = exact_endpoint_numerator_coefficients(("Z",) * 6)
    kernel = canonical_endpoint_kernel_vectors()
    zero = QuadraticSurd()

    assert len(kernel) == 2
    assert sum(
        left * right for left, right in zip(kernel[0], kernel[1])
    ) == zero
    assert sum(value * value for value in kernel[0]) == sum(
        value * value for value in kernel[1]
    )
    for vector in kernel:
        assert all(
            value == zero
            for value in exact_pauli_action(
                numerator[(0,) * 6],
                vector,
            )
        )


def test_endpoint_schur_quadratic_has_a_sparse_exact_positive_form() -> None:
    """Only the two bad angles couple, and a direct Young bound is positive."""

    quadratic = exact_endpoint_schur_quadratic(("Z",) * 6)
    zero = QuadraticSurd()
    coupled_exponent = (1, 0, 0, 1, 0, 0)
    diagonal_exponents = {
        tuple(2 if coordinate == index else 0 for coordinate in range(6))
        for index in range(6)
    }

    for exponent, matrix in quadratic.items():
        if exponent in diagonal_exponents or exponent == coupled_exponent:
            continue
        assert all(entry == zero for row in matrix for entry in row)

    audit = endpoint_quadratic_positivity_audit(("Z",) * 6)
    assert audit["coupled_exponent"] == coupled_exponent
    assert audit["passed"] is True
    assert float(audit["young_lower_bound"]) > 0.25


def test_endpoint_complement_is_exactly_bounded_below_by_identity() -> None:
    """On the kernel complement, the equality operator obeys ``C0 >= I``."""

    audit = endpoint_complement_unit_lower_bound_audit(("Z",) * 6)

    assert audit["passed"] is True
    assert audit["matched_nullities"] == (1, 1)
    assert audit["mismatched_nullities"] == (0, 0)
    assert audit["arithmetic"] == "exact Q(sqrt(2)) pivoted LDL"


def test_endpoint_local_certificate_closes_every_alice_branch_exactly() -> None:
    """A rational radius closes the Schur remainder and complement motion."""

    for alice_good_axes in (("Z", "Z"), ("Z", "X"), ("X", "Z"), ("X", "X")):
        axes = ("Z",) + alice_good_axes + ("Z", "Z", "Z")
        certificate = exact_endpoint_local_certificate(axes)

        assert certificate["radius"] == Fraction(29, 60000)
        assert certificate["l1_radius"] == Fraction(29, 10000)
        assert certificate["passed"] is True
        assert float(certificate["quadratic_simplex_lower_bound"]) > 0.14
        assert float(certificate["remainder_bound"]) < 0.11
        assert float(certificate["complement_perturbation_bound"]) < 0.025
        assert certificate["bound_method"] == "projected kernel/complement norms"
        assert certificate["arithmetic"] == "exact Q(sqrt(2)) and rational bounds"
