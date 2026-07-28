#!/usr/bin/env python3
r"""Exact :math:`\mathbb Q(\sqrt2)` algebra for the CCZ STOPI endpoint.

The corrected coupling gap at :math:`\eta=1/2` is

.. math::
   C_{1/2}
   =K_{\rm CCZ}-\frac12\sum_i\widetilde K_i
    +\frac12I
    +\frac{c_{\rm CCZ}}2
      \left(6\sqrt2I-\mathcal B_{\rm CCZ}\right).

For inward half-angle variables :math:`z_i=\tan(\delta_i/2)`, write

.. math::
   C_{1/2}(z)=\frac{N(z)}{\prod_i(1+z_i^2)}.

This module constructs every coefficient of :math:`N` through total degree
two exactly in the Pauli-string basis.  All coefficients lie in
:math:`\mathbb Q(\sqrt2)`.  The construction is independent of the
``complex128`` interpolation used by the discovery audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from numbers import Rational
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PauliString = tuple[int, int, int, int, int, int]
Exponent = tuple[int, int, int, int, int, int]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CERTIFICATE_ROOT = _PROJECT_ROOT / "results" / "ccz_stopi_certificate"
_DEFAULT_EXACT_RECORD = (
    _CERTIFICATE_ROOT
    / "endpoint_local_exact_certificate.json"
)
_DEFAULT_LOCAL_AGGREGATE = (
    _CERTIFICATE_ROOT
    / "local_exact_certificate.json"
)


def local_branch_orbit_representatives() -> list[dict[str, object]]:
    """Return the 20 low/high branch orbits under site permutations."""

    site_types = ("ZZ", "ZX", "XZ", "XX")
    output = []
    for indices in itertools.combinations_with_replacement(range(4), 3):
        orbit_size = math.factorial(3)
        for index in set(indices):
            orbit_size //= math.factorial(indices.count(index))
        labels = tuple(site_types[index] for index in indices)
        axes = tuple(label[0] for label in labels) + tuple(
            label[1] for label in labels
        )
        output.append(
            {
                "site_types": labels,
                "axes": axes,
                "orbit_size": orbit_size,
            }
        )
    return output


def endpoint_adjacent_branch_axes() -> list[
    tuple[str, str, str, str, str, str]
]:
    """Return the 16 adjacent branches at the canonical bad-pair endpoint."""

    output = []
    for good_axes in itertools.product(("Z", "X"), repeat=4):
        axes = ["Z"] * 6
        for coordinate, axis in zip((1, 2, 4, 5), good_axes):
            axes[coordinate] = axis
        output.append(tuple(axes))  # type: ignore[arg-type]
    return output


@dataclass(frozen=True)
class QuadraticSurd:
    r"""Represent :math:`a+b\sqrt2` with exact rational coefficients."""

    rational: Fraction = Fraction(0)
    root_two: Fraction = Fraction(0)

    def __init__(
        self,
        rational: Rational | Fraction = 0,
        root_two: Rational | Fraction = 0,
    ) -> None:
        object.__setattr__(self, "rational", Fraction(rational))
        object.__setattr__(self, "root_two", Fraction(root_two))

    @staticmethod
    def _coerce(value: object) -> "QuadraticSurd":
        if isinstance(value, QuadraticSurd):
            return value
        if isinstance(value, Rational):
            return QuadraticSurd(value, 0)
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: object) -> "QuadraticSurd":
        parsed = self._coerce(other)
        if parsed is NotImplemented:
            return NotImplemented
        return QuadraticSurd(
            self.rational + parsed.rational,
            self.root_two + parsed.root_two,
        )

    __radd__ = __add__

    def __neg__(self) -> "QuadraticSurd":
        return QuadraticSurd(-self.rational, -self.root_two)

    def __sub__(self, other: object) -> "QuadraticSurd":
        parsed = self._coerce(other)
        if parsed is NotImplemented:
            return NotImplemented
        return self + (-parsed)

    def __rsub__(self, other: object) -> "QuadraticSurd":
        parsed = self._coerce(other)
        if parsed is NotImplemented:
            return NotImplemented
        return parsed - self

    def __mul__(self, other: object) -> "QuadraticSurd":
        parsed = self._coerce(other)
        if parsed is NotImplemented:
            return NotImplemented
        return QuadraticSurd(
            self.rational * parsed.rational
            + 2 * self.root_two * parsed.root_two,
            self.rational * parsed.root_two
            + self.root_two * parsed.rational,
        )

    __rmul__ = __mul__

    def inverse(self) -> "QuadraticSurd":
        r"""Return :math:`(a+b\sqrt2)^{-1}` exactly."""

        norm = self.rational**2 - 2 * self.root_two**2
        if norm == 0:
            raise ZeroDivisionError("quadratic surd is zero")
        return QuadraticSurd(
            self.rational / norm,
            -self.root_two / norm,
        )

    def __truediv__(self, other: object) -> "QuadraticSurd":
        parsed = self._coerce(other)
        if parsed is NotImplemented:
            return NotImplemented
        return self * parsed.inverse()

    def __float__(self) -> float:
        return float(self.rational) + math.sqrt(2.0) * float(self.root_two)

    def sign(self) -> int:
        """Return the exact sign in the real embedding with ``sqrt(2)>0``."""

        a = self.rational
        b = self.root_two
        if a == 0:
            return (b > 0) - (b < 0)
        if b == 0 or (a > 0 and b > 0) or (a < 0 and b < 0):
            return (a > 0) - (a < 0)
        rational_square = a * a
        radical_square = 2 * b * b
        if rational_square == radical_square:
            return 0
        if a > 0:
            return 1 if rational_square > radical_square else -1
        return -1 if rational_square > radical_square else 1

    def is_positive(self) -> bool:
        """Return whether the represented real algebraic number is positive."""

        return self.sign() > 0


_ZERO = QuadraticSurd()
_ONE = QuadraticSurd(1)
_ROOT_TWO = QuadraticSurd(0, 1)
_Q = QuadraticSurd(0, Fraction(1, 2))
_DENOMINATOR = (_ONE, _ZERO, _ONE)
_ZERO_EXPONENT: Exponent = (0, 0, 0, 0, 0, 0)
def _pell_root_two_enclosure(
    iterations: int = 36,
) -> tuple[Fraction, Fraction]:
    """Return consecutive Pell convergents enclosing ``sqrt(2)``."""

    numerator = 1
    denominator = 1
    lower = Fraction(1)
    upper = Fraction(3, 2)
    for _ in range(iterations):
        numerator, denominator = (
            numerator + 2 * denominator,
            numerator + denominator,
        )
        candidate = Fraction(numerator, denominator)
        if numerator * numerator < 2 * denominator * denominator:
            lower = candidate
        else:
            upper = candidate
    return lower, upper


_ROOT_TWO_LOWER, _ROOT_TWO_UPPER = _pell_root_two_enclosure()

if not (
    _ROOT_TWO_LOWER**2 < 2 < _ROOT_TWO_UPPER**2
):
    raise ArithmeticError("invalid rational enclosure of sqrt(2)")


def _apply_pauli_string(
    pauli_string: PauliString,
    basis_index: int,
) -> tuple[int, complex]:
    """Apply one Pauli string to a computational-basis index."""

    output_index = basis_index
    phase = 1.0 + 0.0j
    for site, pauli in enumerate(pauli_string):
        shift = 5 - site
        bit = (basis_index >> shift) & 1
        if pauli == 1:
            output_index ^= 1 << shift
        elif pauli == 2:
            output_index ^= 1 << shift
            phase *= 1.0j * (-1.0 if bit else 1.0)
        elif pauli == 3:
            phase *= -1.0 if bit else 1.0
    return output_index, phase


def _ccz_choi_sign(basis_index: int) -> int:
    """Return the unnormalized real CCZ-Choi amplitude."""

    alice = basis_index >> 3
    bob = basis_index & 7
    if alice != bob:
        return 0
    bits = ((alice >> 2) & 1, (alice >> 1) & 1, alice & 1)
    return -1 if bits[0] * bits[1] * bits[2] else 1


@lru_cache(maxsize=1)
def exact_target_pauli_coefficients(
) -> dict[PauliString, QuadraticSurd]:
    """Return the exact Pauli expansion of the rotated CCZ Choi projector."""

    unrotated: dict[PauliString, QuadraticSurd] = {}
    for pauli_string in itertools.product(range(4), repeat=6):
        total = 0.0 + 0.0j
        for basis_index in range(64):
            input_sign = _ccz_choi_sign(basis_index)
            if not input_sign:
                continue
            output_index, phase = _apply_pauli_string(
                pauli_string,
                basis_index,
            )
            total += (
                input_sign
                * _ccz_choi_sign(output_index)
                * phase
            )
        if abs(total) < 1e-12:
            continue
        if abs(total.imag) > 1e-12:
            raise ArithmeticError("a Hermitian Pauli expectation was not real")
        unrotated[pauli_string] = QuadraticSurd(
            Fraction(int(round(total.real)), 512)
        )

    # Conjugation by W=Z R_y(-pi/4) on each Bob qubit:
    # X -> (-X+Z)/sqrt(2), Y -> -Y, Z -> (X+Z)/sqrt(2).
    rotation = {
        0: ((0, _ONE),),
        1: ((1, -_Q), (3, _Q)),
        2: ((2, -_ONE),),
        3: ((1, _Q), (3, _Q)),
    }
    output: dict[PauliString, QuadraticSurd] = {}
    for pauli_string, coefficient in unrotated.items():
        partial: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            QuadraticSurd,
        ] = {(pauli_string[:3], ()): coefficient}
        for bob_pauli in pauli_string[3:]:
            updated: dict[
                tuple[tuple[int, ...], tuple[int, ...]],
                QuadraticSurd,
            ] = {}
            for (alice_part, bob_part), partial_coefficient in partial.items():
                for rotated_pauli, weight in rotation[bob_pauli]:
                    key = (alice_part, bob_part + (rotated_pauli,))
                    updated[key] = (
                        updated.get(key, _ZERO)
                        + partial_coefficient * weight
                    )
            partial = updated
        for (alice_part, bob_part), partial_coefficient in partial.items():
            key = alice_part + bob_part
            output[key] = output.get(key, _ZERO) + partial_coefficient
    return {
        pauli_string: coefficient
        for pauli_string, coefficient in output.items()
        if coefficient != _ZERO
    }


def pauli_dictionary_matrix(
    coefficients: Mapping[PauliString, QuadraticSurd],
) -> np.ndarray:
    """Evaluate an exact Pauli dictionary as a ``complex128`` matrix."""

    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    pauli_y = np.array(
        [[0.0, -1.0j], [1.0j, 0.0]],
        dtype=np.complex128,
    )
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    paulis = (identity, pauli_x, pauli_y, pauli_z)
    output = np.zeros((64, 64), dtype=np.complex128)
    for pauli_string, coefficient in coefficients.items():
        term = np.array([[1.0]], dtype=np.complex128)
        for pauli in pauli_string:
            term = np.kron(term, paulis[pauli])
        output += float(coefficient) * term
    return output


def exact_pauli_action(
    coefficients: Mapping[PauliString, QuadraticSurd],
    vector: Sequence[QuadraticSurd],
) -> tuple[QuadraticSurd, ...]:
    """Apply an exact Hermitian Pauli dictionary to a real exact vector."""

    if len(vector) != 64:
        raise ValueError("the vector must have dimension 64")
    real_output = [_ZERO for _ in range(64)]
    imaginary_output = [_ZERO for _ in range(64)]
    for pauli_string, coefficient in coefficients.items():
        for column, vector_coefficient in enumerate(vector):
            if vector_coefficient == _ZERO:
                continue
            row, phase = _apply_pauli_string(pauli_string, column)
            weighted = coefficient * vector_coefficient
            real_phase = int(round(phase.real))
            imaginary_phase = int(round(phase.imag))
            if real_phase:
                real_output[row] = (
                    real_output[row] + real_phase * weighted
                )
            if imaginary_phase:
                imaginary_output[row] = (
                    imaginary_output[row] + imaginary_phase * weighted
                )
    if any(value != _ZERO for value in imaginary_output):
        raise ArithmeticError("the exact real operator produced an imaginary vector")
    return tuple(real_output)


@lru_cache(maxsize=1)
def canonical_endpoint_kernel_vectors(
) -> tuple[
    tuple[QuadraticSurd, ...],
    tuple[QuadraticSurd, ...],
]:
    r"""Return unnormalized exact kernel vectors at ``a1=b1=0``.

    The two vectors occupy the matched ``Z_A1=Z_B1`` sectors.  Their
    nonzero entries are expressed through
    :math:`t=\sqrt2-1` and :math:`t^2=3-2\sqrt2`.
    """

    t = QuadraticSurd(-1, 1)
    t_squared = QuadraticSurd(3, -2)
    low_sector = (
        _ONE,
        t,
        t,
        t_squared,
        t,
        -_ONE,
        t_squared,
        -t,
        t,
        t_squared,
        -_ONE,
        -t,
        t_squared,
        -t,
        -t,
        _ONE,
    )
    high_sector = (
        _ONE,
        t,
        t,
        t_squared,
        t,
        -_ONE,
        t_squared,
        -t,
        t,
        t_squared,
        -_ONE,
        -t,
        -t_squared,
        t,
        t,
        -_ONE,
    )
    output = []
    for bit, sector_vector in enumerate((low_sector, high_sector)):
        vector = [_ZERO for _ in range(64)]
        sector_indices = [
            index
            for index in range(64)
            if ((index >> 5) & 1) == bit
            and ((index >> 2) & 1) == bit
        ]
        for index, coefficient in zip(sector_indices, sector_vector):
            vector[index] = coefficient
        output.append(tuple(vector))
    return output[0], output[1]


def _anticommutes(pauli: int, axis: str) -> bool:
    """Return whether one Pauli label anticommutes with the branch axis."""

    if axis == "Z":
        return pauli in (1, 2)
    return pauli in (2, 3)


def _endpoint_scalar_polynomials(
    axes: Sequence[str],
) -> tuple[
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
]:
    """Return cleared cosine, sine, and visibility numerators."""

    bad_cosine = (_ONE, _ZERO, -_ONE)
    bad_sine = (_ZERO, 2 * _ONE, _ZERO)
    low_cosine = (_Q, _ROOT_TWO, -_Q)
    low_sine = (_Q, -_ROOT_TWO, -_Q)
    good_visibility = (
        _ONE,
        _ZERO,
        -QuadraticSurd(3, 2),
    )
    bad_visibility_weight = 2 * (_ONE + _ROOT_TWO)
    bad_visibility = (
        _ZERO,
        bad_visibility_weight,
        -bad_visibility_weight,
    )

    cosines = []
    sines = []
    visibilities = []
    for coordinate, axis in enumerate(axes):
        if coordinate in (0, 3):
            cosines.append(bad_cosine)
            sines.append(bad_sine)
            visibilities.append(bad_visibility)
        elif axis == "Z":
            cosines.append(low_cosine)
            sines.append(low_sine)
            visibilities.append(good_visibility)
        else:
            cosines.append(low_sine)
            sines.append(low_cosine)
            visibilities.append(good_visibility)
    return tuple(cosines), tuple(sines), tuple(visibilities)


def _ideal_scalar_polynomials(
    axes: Sequence[str],
) -> tuple[
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
]:
    """Return ideal-point half-angle numerators on selected branches."""

    low_cosine = (_Q, _ROOT_TWO, -_Q)
    low_sine = (_Q, -_ROOT_TWO, -_Q)
    visibility = (
        _ONE,
        _ZERO,
        -QuadraticSurd(3, 2),
    )
    cosines = []
    sines = []
    visibilities = []
    for axis in axes:
        if axis == "Z":
            cosines.append(low_cosine)
            sines.append(low_sine)
        else:
            cosines.append(low_sine)
            sines.append(low_cosine)
        visibilities.append(visibility)
    return tuple(cosines), tuple(sines), tuple(visibilities)


def _add_separable_term(
    output: dict[
        Exponent,
        dict[PauliString, QuadraticSurd],
    ],
    pauli_string: PauliString,
    scalar: QuadraticSurd,
    factors: Mapping[int, Sequence[QuadraticSurd]],
    *,
    maximum_total_degree: int,
) -> None:
    """Add one cleared separable term with a total-degree truncation."""

    polynomial: dict[Exponent, QuadraticSurd] = {
        _ZERO_EXPONENT: scalar
    }
    for coordinate in range(6):
        local_polynomial = factors.get(coordinate, _DENOMINATOR)
        updated: dict[Exponent, QuadraticSurd] = {}
        for exponent, coefficient in polynomial.items():
            for degree, weight in enumerate(local_polynomial):
                if (
                    weight == _ZERO
                    or sum(exponent) + degree > maximum_total_degree
                ):
                    continue
                new_exponent = list(exponent)
                new_exponent[coordinate] += degree
                key = tuple(new_exponent)
                updated[key] = (
                    updated.get(key, _ZERO) + coefficient * weight
                )
        polynomial = updated

    for exponent, coefficient in polynomial.items():
        pauli_coefficients = output.setdefault(exponent, {})
        pauli_coefficients[pauli_string] = (
            pauli_coefficients.get(pauli_string, _ZERO) + coefficient
        )


def _pair_target_coefficients(
) -> tuple[
    dict[tuple[int, int], QuadraticSurd],
    dict[tuple[int, int], QuadraticSurd],
]:
    """Return exact Pauli coefficients of the two effective CHSH targets."""

    quarter = QuadraticSurd(Fraction(1, 4))
    q_quarter = _Q * Fraction(1, 4)
    plus = {
        (0, 0): quarter,
        (1, 1): -q_quarter,
        (1, 3): q_quarter,
        (2, 2): quarter,
        (3, 1): q_quarter,
        (3, 3): q_quarter,
    }
    minus = {
        (0, 0): quarter,
        (1, 1): q_quarter,
        (1, 3): -q_quarter,
        (2, 2): -quarter,
        (3, 1): q_quarter,
        (3, 3): q_quarter,
    }
    return plus, minus


def _remote_derivative_terms(
    site: int,
    cosines: Sequence[Sequence[QuadraticSurd]],
    sines: Sequence[Sequence[QuadraticSurd]],
) -> list[
    tuple[
        dict[int, int],
        QuadraticSurd,
        dict[int, Sequence[QuadraticSurd]],
    ]
]:
    r"""Return the Pauli expansion of
    \((I+B_{j,0}+B_{k,0}-B_{j,0}B_{k,0})/2\).
    """

    other_sites = [index for index in range(3) if index != site]
    output = [({}, QuadraticSurd(Fraction(1, 2)), {})]
    for other in other_sites:
        coordinate = 3 + other
        for pauli, polynomial in (
            (3, cosines[coordinate]),
            (1, sines[coordinate]),
        ):
            output.append(
                (
                    {coordinate: pauli},
                    QuadraticSurd(Fraction(1, 2)),
                    {coordinate: polynomial},
                )
            )
    left, right = other_sites
    left_coordinate = 3 + left
    right_coordinate = 3 + right
    for left_pauli, left_polynomial in (
        (3, cosines[left_coordinate]),
        (1, sines[left_coordinate]),
    ):
        for right_pauli, right_polynomial in (
            (3, cosines[right_coordinate]),
            (1, sines[right_coordinate]),
        ):
            output.append(
                (
                    {
                        left_coordinate: left_pauli,
                        right_coordinate: right_pauli,
                    },
                    QuadraticSurd(Fraction(-1, 2)),
                    {
                        left_coordinate: left_polynomial,
                        right_coordinate: right_polynomial,
                    },
                )
            )
    return output


@lru_cache(maxsize=64)
def _exact_local_numerator_coefficients(
    axes: tuple[str, str, str, str, str, str],
    equality_kind: str,
    maximum_total_degree: int,
) -> dict[Exponent, dict[PauliString, QuadraticSurd]]:
    r"""Return all coefficients of the endpoint numerator through degree two.

    The current exact constructor covers the canonical endpoint
    :math:`a_1=b_1=0`.  Other bad sites and the :math:`b_1=\pi/2` endpoint
    follow from the site-permutation and Bob-axis symmetries.
    """

    if len(axes) != 6 or any(axis not in ("X", "Z") for axis in axes):
        raise ValueError("axes must contain six X/Z labels")
    if maximum_total_degree < 0 or maximum_total_degree > 12:
        raise ValueError("maximum_total_degree must lie between 0 and 12")
    if equality_kind == "endpoint":
        if axes[0] != "Z" or axes[3] != "Z":
            raise ValueError("the canonical endpoint requires Z at A1 and B1")
        cosines, sines, visibilities = _endpoint_scalar_polynomials(axes)
    elif equality_kind == "ideal":
        cosines, sines, visibilities = _ideal_scalar_polynomials(axes)
    else:
        raise ValueError("equality_kind must be endpoint or ideal")
    output: dict[
        Exponent,
        dict[PauliString, QuadraticSurd],
    ] = {}

    for pauli_string, coefficient in (
        exact_target_pauli_coefficients().items()
    ):
        factors = {
            coordinate: visibilities[coordinate]
            for coordinate in range(6)
            if _anticommutes(pauli_string[coordinate], axes[coordinate])
        }
        _add_separable_term(
            output,
            pauli_string,
            coefficient,
            factors,
            maximum_total_degree=maximum_total_degree,
        )

    plus_target, minus_target = _pair_target_coefficients()
    for site in range(3):
        derivative_terms = _remote_derivative_terms(
            site,
            cosines,
            sines,
        )
        for pair_pauli in set(plus_target) | set(minus_target):
            plus_coefficient = plus_target.get(pair_pauli, _ZERO)
            minus_coefficient = minus_target.get(pair_pauli, _ZERO)
            average = (plus_coefficient + minus_coefficient) / 2
            difference = (plus_coefficient - minus_coefficient) / 2
            base_pauli = [0] * 6
            base_pauli[site] = pair_pauli[0]
            base_pauli[3 + site] = pair_pauli[1]
            pair_factors: dict[int, Sequence[QuadraticSurd]] = {}
            if _anticommutes(pair_pauli[0], axes[site]):
                pair_factors[site] = visibilities[site]
            if _anticommutes(pair_pauli[1], axes[3 + site]):
                pair_factors[3 + site] = visibilities[3 + site]

            _add_separable_term(
                output,
                tuple(base_pauli),
                -average / 2,
                pair_factors,
                maximum_total_degree=maximum_total_degree,
            )
            for assignment, derivative_coefficient, derivative_factors in (
                derivative_terms
            ):
                pauli_string = list(base_pauli)
                for coordinate, pauli in assignment.items():
                    pauli_string[coordinate] = pauli
                _add_separable_term(
                    output,
                    tuple(pauli_string),
                    -difference * derivative_coefficient / 2,
                    pair_factors | derivative_factors,
                    maximum_total_degree=maximum_total_degree,
                )

    identity_coefficient = QuadraticSurd(
        Fraction(19, 8),
        Fraction(3, 4),
    )
    _add_separable_term(
        output,
        (0, 0, 0, 0, 0, 0),
        identity_coefficient,
        {},
        maximum_total_degree=maximum_total_degree,
    )

    bell_factor = QuadraticSurd(
        Fraction(-1, 8),
        Fraction(-5, 32),
    )
    for site in range(3):
        derivative_terms = _remote_derivative_terms(
            site,
            cosines,
            sines,
        )
        for bob_pauli, bob_polynomial in (
            (3, cosines[3 + site]),
            (1, sines[3 + site]),
        ):
            pauli_string = [0] * 6
            pauli_string[site] = 3
            pauli_string[3 + site] = bob_pauli
            _add_separable_term(
                output,
                tuple(pauli_string),
                2 * bell_factor,
                {
                    site: cosines[site],
                    3 + site: bob_polynomial,
                },
                maximum_total_degree=maximum_total_degree,
            )

        for bob_pauli, bob_polynomial, sign in (
            (3, cosines[3 + site], 1),
            (1, sines[3 + site], -1),
        ):
            for assignment, derivative_coefficient, derivative_factors in (
                derivative_terms
            ):
                pauli_string = [0] * 6
                pauli_string[site] = 1
                pauli_string[3 + site] = bob_pauli
                for coordinate, pauli in assignment.items():
                    pauli_string[coordinate] = pauli
                _add_separable_term(
                    output,
                    tuple(pauli_string),
                    2
                    * sign
                    * bell_factor
                    * derivative_coefficient,
                    {
                        site: sines[site],
                        3 + site: bob_polynomial,
                    }
                    | derivative_factors,
                    maximum_total_degree=maximum_total_degree,
                )

    cleaned: dict[
        Exponent,
        dict[PauliString, QuadraticSurd],
    ] = {}
    for exponent in itertools.product(range(3), repeat=6):
        if sum(exponent) > maximum_total_degree:
            continue
        pauli_coefficients = output.get(exponent, {})
        nonzero = {
            pauli_string: coefficient
            for pauli_string, coefficient in pauli_coefficients.items()
            if coefficient != _ZERO
        }
        cleaned[exponent] = nonzero
    return cleaned


def exact_endpoint_numerator_coefficients(
    axes: tuple[str, str, str, str, str, str],
) -> dict[Exponent, dict[PauliString, QuadraticSurd]]:
    """Return exact endpoint-numerator coefficients through total degree two."""

    return _exact_local_numerator_coefficients(axes, "endpoint", 2)


def exact_endpoint_full_numerator_coefficients(
    axes: tuple[str, str, str, str, str, str],
) -> dict[Exponent, dict[PauliString, QuadraticSurd]]:
    """Return the complete degree-two-per-variable endpoint numerator."""

    return _exact_local_numerator_coefficients(axes, "endpoint", 12)


def exact_ideal_full_numerator_coefficients(
    axes: tuple[str, str, str, str, str, str],
) -> dict[Exponent, dict[PauliString, QuadraticSurd]]:
    """Return the complete ideal-point numerator on one branch orthant."""

    return _exact_local_numerator_coefficients(axes, "ideal", 12)


def _dot(
    left: Sequence[QuadraticSurd],
    right: Sequence[QuadraticSurd],
) -> QuadraticSurd:
    """Return an exact real Euclidean inner product."""

    return sum(
        (left_value * right_value for left_value, right_value in zip(left, right)),
        start=_ZERO,
    )


def _invert_matrix(
    matrix: Sequence[Sequence[QuadraticSurd]],
) -> tuple[tuple[QuadraticSurd, ...], ...]:
    """Invert a small nonsingular matrix by exact Gauss--Jordan elimination."""

    dimension = len(matrix)
    if dimension == 0 or any(len(row) != dimension for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    augmented = [
        list(row)
        + [
            _ONE if row_index == column else _ZERO
            for column in range(dimension)
        ]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] != _ZERO
            ),
            None,
        )
        if pivot is None:
            raise np.linalg.LinAlgError("exact matrix is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        pivot_inverse = augmented[column][column].inverse()
        augmented[column] = [
            value * pivot_inverse for value in augmented[column]
        ]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == _ZERO:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                )
            ]
    return tuple(
        tuple(row[dimension:]) for row in augmented
    )


def _exact_sector_block(
    coefficients: Mapping[PauliString, QuadraticSurd],
    indices: Sequence[int],
) -> tuple[tuple[QuadraticSurd, ...], ...]:
    """Return one exact computational-basis sector block."""

    positions = {basis_index: position for position, basis_index in enumerate(indices)}
    real = [
        [_ZERO for _ in indices]
        for _ in indices
    ]
    imaginary = [
        [_ZERO for _ in indices]
        for _ in indices
    ]
    for pauli_string, coefficient in coefficients.items():
        for column, basis_index in enumerate(indices):
            output_index, phase = _apply_pauli_string(
                pauli_string,
                basis_index,
            )
            row = positions.get(output_index)
            if row is None:
                if coefficient != _ZERO:
                    raise ArithmeticError(
                        "the endpoint equality operator left its Z sector"
                    )
                continue
            real_phase = int(round(phase.real))
            imaginary_phase = int(round(phase.imag))
            if real_phase:
                real[row][column] = (
                    real[row][column] + real_phase * coefficient
                )
            if imaginary_phase:
                imaginary[row][column] = (
                    imaginary[row][column]
                    + imaginary_phase * coefficient
                )
    if any(
        value != _ZERO
        for row in imaginary
        for value in row
    ):
        raise ArithmeticError("the exact sector block was not real")
    return tuple(tuple(row) for row in real)


@lru_cache(maxsize=16)
def _endpoint_sector_inverses(
    axes: tuple[str, str, str, str, str, str],
) -> tuple[
    tuple[
        tuple[int, ...],
        tuple[tuple[QuadraticSurd, ...], ...],
    ],
    ...,
]:
    """Return exact inverses of ``C0`` augmented on its matched kernels."""

    equality = exact_endpoint_numerator_coefficients(axes)[_ZERO_EXPONENT]
    kernel = canonical_endpoint_kernel_vectors()
    norm = _dot(kernel[0], kernel[0])
    sectors = []
    for alice_bit, bob_bit in itertools.product((0, 1), repeat=2):
        indices = tuple(
            index
            for index in range(64)
            if ((index >> 5) & 1) == alice_bit
            and ((index >> 2) & 1) == bob_bit
        )
        block = [list(row) for row in _exact_sector_block(equality, indices)]
        if alice_bit == bob_bit:
            vector = tuple(kernel[alice_bit][index] for index in indices)
            for row in range(16):
                for column in range(16):
                    block[row][column] = (
                        block[row][column]
                        + vector[row] * vector[column] / norm
                    )
        sectors.append((indices, _invert_matrix(block)))
    return tuple(sectors)


def _matrix_vector_product(
    matrix: Sequence[Sequence[QuadraticSurd]],
    vector: Sequence[QuadraticSurd],
) -> tuple[QuadraticSurd, ...]:
    """Multiply an exact square matrix and vector."""

    return tuple(
        sum(
            (
                matrix_entry * vector_entry
                for matrix_entry, vector_entry in zip(row, vector)
            ),
            start=_ZERO,
        )
        for row in matrix
    )


def _endpoint_resolvent_action(
    axes: tuple[str, str, str, str, str, str],
    vector: Sequence[QuadraticSurd],
) -> tuple[QuadraticSurd, ...]:
    r"""Apply \(R_0=(QC(0)Q)^{-1}\) exactly."""

    kernel = canonical_endpoint_kernel_vectors()
    norm = _dot(kernel[0], kernel[0])
    projected = list(vector)
    for kernel_vector in kernel:
        overlap = _dot(kernel_vector, projected) / norm
        if overlap == _ZERO:
            continue
        projected = [
            value - overlap * kernel_value
            for value, kernel_value in zip(projected, kernel_vector)
        ]

    output = [_ZERO for _ in range(64)]
    for indices, inverse in _endpoint_sector_inverses(axes):
        sector_vector = tuple(projected[index] for index in indices)
        sector_output = _matrix_vector_product(inverse, sector_vector)
        for index, value in zip(indices, sector_output):
            output[index] = value
    return tuple(output)


@lru_cache(maxsize=16)
def exact_endpoint_schur_quadratic(
    axes: tuple[str, str, str, str, str, str],
) -> dict[
    Exponent,
    tuple[
        tuple[QuadraticSurd, QuadraticSurd],
        tuple[QuadraticSurd, QuadraticSurd],
    ],
]:
    r"""Return the exact normalized quadratic Schur polynomial at the endpoint.

    The fixed comparison inverse is multiplied by
    :math:`\kappa=11/10`, matching the successful local certificate design.
    """

    numerator = exact_endpoint_numerator_coefficients(axes)
    kernel = canonical_endpoint_kernel_vectors()
    norm = _dot(kernel[0], kernel[0])
    kappa = QuadraticSurd(Fraction(11, 10))

    linear_exponents = []
    first_actions = []
    resolvent_actions = []
    for coordinate in range(6):
        exponent = tuple(
            1 if index == coordinate else 0
            for index in range(6)
        )
        linear_exponents.append(exponent)
        coordinate_actions = tuple(
            exact_pauli_action(numerator[exponent], vector)
            for vector in kernel
        )
        first_actions.append(coordinate_actions)
        resolvent_actions.append(
            tuple(
                _endpoint_resolvent_action(axes, action)
                for action in coordinate_actions
            )
        )

    output = {}
    for exponent in itertools.product(range(3), repeat=6):
        if sum(exponent) != 2:
            continue
        active = [
            coordinate
            for coordinate, degree in enumerate(exponent)
            for _ in range(degree)
        ]
        left, right = active
        second_actions = tuple(
            exact_pauli_action(numerator[exponent], vector)
            for vector in kernel
        )
        rows = []
        for row in range(2):
            entries = []
            for column in range(2):
                direct = _dot(kernel[row], second_actions[column])
                if left == right:
                    correction = _dot(
                        first_actions[left][row],
                        resolvent_actions[left][column],
                    )
                else:
                    correction = (
                        _dot(
                            first_actions[left][row],
                            resolvent_actions[right][column],
                        )
                        + _dot(
                            first_actions[right][row],
                            resolvent_actions[left][column],
                        )
                    )
                entries.append((direct - kappa * correction) / norm)
            rows.append(tuple(entries))
        output[exponent] = tuple(rows)
    return output


def _absolute_surd(value: QuadraticSurd) -> QuadraticSurd:
    """Return the exact absolute value in the real embedding."""

    return value if value.sign() >= 0 else -value


def _minimum_surd(
    values: Sequence[QuadraticSurd],
) -> QuadraticSurd:
    """Return the exact minimum in the real embedding."""

    if not values:
        raise ValueError("at least one value is required")
    output = values[0]
    for value in values[1:]:
        if (value - output).sign() < 0:
            output = value
    return output


def endpoint_quadratic_positivity_audit(
    axes: tuple[str, str, str, str, str, str],
) -> dict[str, object]:
    r"""Certify the sparse endpoint quadratic form by a direct Young bound."""

    quadratic = exact_endpoint_schur_quadratic(axes)
    coupled_exponent: Exponent = (1, 0, 0, 1, 0, 0)
    bad_alice_square: Exponent = (2, 0, 0, 0, 0, 0)
    bad_bob_square: Exponent = (0, 0, 0, 2, 0, 0)
    alice_matrix = quadratic[bad_alice_square]
    bob_matrix = quadratic[bad_bob_square]
    coupled_matrix = quadratic[coupled_exponent]
    if (
        alice_matrix[0][1] != _ZERO
        or alice_matrix[1][0] != _ZERO
        or alice_matrix[0][0] != alice_matrix[1][1]
        or bob_matrix[0][1] != _ZERO
        or bob_matrix[1][0] != _ZERO
        or bob_matrix[0][0] != bob_matrix[1][1]
        or coupled_matrix[0][0] != _ZERO
        or coupled_matrix[1][1] != _ZERO
        or coupled_matrix[0][1] != coupled_matrix[1][0]
    ):
        raise ArithmeticError("the endpoint quadratic did not have its sparse form")

    coupling = _absolute_surd(coupled_matrix[0][1])
    coordinate_bounds = [
        alice_matrix[0][0] - coupling / 2,
        bob_matrix[0][0] - coupling / 2,
    ]
    for coordinate in (1, 2, 4, 5):
        exponent = tuple(
            2 if index == coordinate else 0
            for index in range(6)
        )
        matrix = quadratic[exponent]
        if matrix[0][1] != _ZERO or matrix[1][0] != _ZERO:
            raise ArithmeticError("an uncoupled square block was not diagonal")
        coordinate_bounds.append(
            _minimum_surd((matrix[0][0], matrix[1][1]))
        )
    harmonic_simplex_lower_bound = sum(
        (bound.inverse() for bound in coordinate_bounds),
        start=_ZERO,
    ).inverse()
    candidates = tuple(coordinate_bounds)
    lower_bound = _minimum_surd(candidates)
    return {
        "coupled_exponent": coupled_exponent,
        "young_lower_bound": lower_bound,
        "coordinate_bounds": candidates,
        "harmonic_simplex_lower_bound": harmonic_simplex_lower_bound,
        "candidate_bounds": candidates,
        "passed": all(candidate.is_positive() for candidate in candidates),
        "arithmetic": "exact Q(sqrt(2))",
    }


def _pivoted_ldl_psd_audit(
    matrix: Sequence[Sequence[QuadraticSurd]],
) -> dict[str, object]:
    """Run exact symmetric elimination with positive diagonal pivoting."""

    reduced = [list(row) for row in matrix]
    dimension = len(reduced)
    pivots: list[QuadraticSurd] = []
    active = 0
    while active < dimension:
        if any(
            reduced[index][index].sign() < 0
            for index in range(active, dimension)
        ):
            return {
                "passed": False,
                "pivots": tuple(pivots),
                "nullity": 0,
            }
        pivot = next(
            (
                index
                for index in range(active, dimension)
                if reduced[index][index].is_positive()
            ),
            None,
        )
        if pivot is None:
            trailing_is_zero = all(
                reduced[row][column] == _ZERO
                for row in range(active, dimension)
                for column in range(active, dimension)
            )
            return {
                "passed": trailing_is_zero,
                "pivots": tuple(pivots),
                "nullity": dimension - active if trailing_is_zero else 0,
            }
        if pivot != active:
            reduced[active], reduced[pivot] = (
                reduced[pivot],
                reduced[active],
            )
            for row in range(dimension):
                reduced[row][active], reduced[row][pivot] = (
                    reduced[row][pivot],
                    reduced[row][active],
                )
        diagonal = reduced[active][active]
        pivots.append(diagonal)
        for row in range(active + 1, dimension):
            for column in range(row, dimension):
                updated = (
                    reduced[row][column]
                    - reduced[row][active]
                    * reduced[column][active]
                    / diagonal
                )
                reduced[row][column] = updated
                reduced[column][row] = updated
        for index in range(active + 1, dimension):
            reduced[index][active] = _ZERO
            reduced[active][index] = _ZERO
        active += 1
    return {
        "passed": True,
        "pivots": tuple(pivots),
        "nullity": 0,
    }


def endpoint_complement_unit_lower_bound_audit(
    axes: tuple[str, str, str, str, str, str],
) -> dict[str, object]:
    r"""Prove \(C(0)\succeq I-P\) by exact sectorwise pivoted LDL."""

    equality = exact_endpoint_numerator_coefficients(axes)[_ZERO_EXPONENT]
    kernel = canonical_endpoint_kernel_vectors()
    norm = _dot(kernel[0], kernel[0])
    records = []
    for alice_bit, bob_bit in itertools.product((0, 1), repeat=2):
        indices = tuple(
            index
            for index in range(64)
            if ((index >> 5) & 1) == alice_bit
            and ((index >> 2) & 1) == bob_bit
        )
        block = [list(row) for row in _exact_sector_block(equality, indices)]
        for index in range(16):
            block[index][index] = block[index][index] - _ONE
        matched = alice_bit == bob_bit
        if matched:
            vector = tuple(kernel[alice_bit][index] for index in indices)
            for row in range(16):
                for column in range(16):
                    block[row][column] = (
                        block[row][column]
                        + vector[row] * vector[column] / norm
                    )
        audit = _pivoted_ldl_psd_audit(block)
        records.append(
            {
                "sector": (alice_bit, bob_bit),
                "matched": matched,
                **audit,
            }
        )
    matched_nullities = tuple(
        record["nullity"] for record in records if record["matched"]
    )
    mismatched_nullities = tuple(
        record["nullity"] for record in records if not record["matched"]
    )
    return {
        "passed": all(record["passed"] for record in records),
        "matched_nullities": matched_nullities,
        "mismatched_nullities": mismatched_nullities,
        "records": tuple(records),
        "arithmetic": "exact Q(sqrt(2)) pivoted LDL",
    }


def _rational_absolute_upper(value: QuadraticSurd) -> Fraction:
    r"""Bound \(|a+b\sqrt2|\) rationally using \(\sqrt2<3/2\)."""

    return abs(value.rational) + Fraction(3, 2) * abs(value.root_two)


def _rational_surd_upper(value: QuadraticSurd) -> Fraction:
    """Return a tight rational upper bound on one real quadratic surd."""

    root_bound = (
        _ROOT_TWO_UPPER
        if value.root_two >= 0
        else _ROOT_TWO_LOWER
    )
    return value.rational + value.root_two * root_bound


def _rational_absolute_upper_tight(value: QuadraticSurd) -> Fraction:
    """Return a tight rational upper bound on an absolute quadratic surd."""

    return _rational_surd_upper(_absolute_surd(value))


def _rational_square_root_upper(value: QuadraticSurd) -> Fraction:
    """Return a rational upper bound on the square root of a positive surd."""

    if value.sign() < 0:
        raise ValueError("cannot take the square root of a negative surd")
    rational_upper = _rational_surd_upper(value)
    if rational_upper < 0:
        raise ArithmeticError("the rational surd enclosure was inconsistent")
    numerator = rational_upper.numerator
    denominator = rational_upper.denominator
    root_numerator = math.isqrt(numerator * denominator)
    if root_numerator * root_numerator < numerator * denominator:
        root_numerator += 1
    return Fraction(root_numerator, denominator)


def _multinomial(exponent: Sequence[int]) -> int:
    """Return the exact multinomial coefficient."""

    output = math.factorial(sum(exponent))
    for degree in exponent:
        output //= math.factorial(degree)
    return output


def _endpoint_projected_homogeneous_bounds(
    full: Mapping[
        Exponent,
        Mapping[PauliString, QuadraticSurd],
    ],
) -> tuple[dict[int, Fraction], dict[int, Fraction]]:
    r"""Bound the endpoint kernel block and kernel--complement coupling.

    For the normalized endpoint kernel isometry \(V\), the two returned
    sequences bound

    .. math::
       \|V^\dagger N_\alpha V\|,\qquad
       \|Q N_\alpha V\|_{\rm F},

    after division by the homogeneous multinomial coefficient.  The first
    norm uses an exact row-sum bound; the second is obtained from the exact
    squared Frobenius norm before taking a rational upper square root.
    """

    kernel = canonical_endpoint_kernel_vectors()
    norm = _dot(kernel[0], kernel[0])
    kernel_bounds: dict[int, Fraction] = {}
    coupling_bounds: dict[int, Fraction] = {}
    for exponent, pauli_coefficients in full.items():
        degree = sum(exponent)
        multinomial = _multinomial(exponent)
        actions = tuple(
            exact_pauli_action(pauli_coefficients, vector)
            for vector in kernel
        )
        block = tuple(
            tuple(
                _dot(kernel[row], actions[column]) / norm
                for column in range(2)
            )
            for row in range(2)
        )
        block_bound = max(
            sum(
                (
                    _rational_absolute_upper_tight(entry)
                    for entry in row
                ),
                start=Fraction(0),
            )
            for row in block
        ) / multinomial

        coupling_squared = sum(
            (_dot(action, action) for action in actions),
            start=_ZERO,
        ) / norm
        coupling_squared = coupling_squared - sum(
            (
                entry * entry
                for row in block
                for entry in row
            ),
            start=_ZERO,
        )
        if coupling_squared.sign() < 0:
            raise ArithmeticError(
                "the endpoint projected coupling norm became negative"
            )
        coupling_bound = (
            _rational_square_root_upper(coupling_squared)
            / multinomial
        )
        kernel_bounds[degree] = max(
            kernel_bounds.get(degree, Fraction(0)),
            block_bound,
        )
        coupling_bounds[degree] = max(
            coupling_bounds.get(degree, Fraction(0)),
            coupling_bound,
        )
    return kernel_bounds, coupling_bounds


def _homogeneous_complement_motion_bounds(
    full: Mapping[
        Exponent,
        Mapping[PauliString, QuadraticSurd],
    ],
) -> dict[int, Fraction]:
    r"""Bound \(N(z)-D(z)N(0)\) on a nonnegative \(\ell_1\) ball."""

    equality_coefficients = full[_ZERO_EXPONENT]
    output: dict[int, Fraction] = {}
    for exponent, pauli_coefficients in full.items():
        difference = dict(pauli_coefficients)
        if all(degree in (0, 2) for degree in exponent):
            for pauli_string, coefficient in equality_coefficients.items():
                difference[pauli_string] = (
                    difference.get(pauli_string, _ZERO) - coefficient
                )
        coefficient_bound = sum(
            (
                _rational_absolute_upper_tight(coefficient)
                for coefficient in difference.values()
            ),
            start=Fraction(0),
        ) / _multinomial(exponent)
        degree = sum(exponent)
        output[degree] = max(
            output.get(degree, Fraction(0)),
            coefficient_bound,
        )
    return output


def exact_endpoint_local_certificate(
    axes: tuple[str, str, str, str, str, str],
    *,
    radius: Fraction = Fraction(29, 60000),
) -> dict[str, object]:
    r"""Return a fully exact local Schur certificate at one endpoint branch.

    On the simplex \(y_i=z_i/\sum_jz_j\), let \(a_d\) bound the normalized
    kernel block of the degree-\(d\) numerator and let \(b_d\) bound its
    normalized kernel--complement coupling.  Since \(C(0)\succeq I-P\), the
    fixed complement resolvent obeys \(\|R_0\|\le1\).  Therefore the
    degree-\(d\) Schur remainder is bounded by

    .. math::
       L_d=
       \sum_{j\ge0}a_{d-2j}
       +\frac{11}{10}\sum_{p=1}^{d-1}b_pb_{d-p},

    where the first sum starts at numerator degree two because the kernel
    blocks of degrees zero and one vanish exactly.
    """

    radius = Fraction(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    full = exact_endpoint_full_numerator_coefficients(axes)
    kernel = canonical_endpoint_kernel_vectors()
    zero = QuadraticSurd()

    for exponent, pauli_coefficients in full.items():
        if sum(exponent) > 1:
            continue
        actions = tuple(
            exact_pauli_action(pauli_coefficients, vector)
            for vector in kernel
        )
        if any(
            _dot(kernel[row], actions[column]) != zero
            for row in range(2)
            for column in range(2)
        ):
            raise ArithmeticError(
                "a degree-zero or degree-one kernel block did not vanish"
            )

    (
        homogeneous_kernel_bounds,
        homogeneous_coupling_bounds,
    ) = _endpoint_projected_homogeneous_bounds(full)

    schur_homogeneous_bounds: dict[int, Fraction] = {}
    kappa = Fraction(11, 10)
    for degree in range(3, 25):
        denominator_times_kernel = sum(
            (
                homogeneous_kernel_bounds.get(
                    degree - 2 * order,
                    Fraction(0),
                )
                for order in range(7)
                if degree - 2 * order >= 2
            ),
            start=Fraction(0),
        )
        coupling_product = sum(
            (
                homogeneous_coupling_bounds.get(
                    left_degree,
                    Fraction(0),
                )
                * homogeneous_coupling_bounds.get(
                    degree - left_degree,
                    Fraction(0),
                )
                for left_degree in range(1, degree)
            ),
            start=Fraction(0),
        )
        schur_homogeneous_bounds[degree] = (
            denominator_times_kernel + kappa * coupling_product
        )

    simplex_radius = 6 * radius
    remainder_bound = sum(
        (
            bound * simplex_radius ** (degree - 2)
            for degree, bound in schur_homogeneous_bounds.items()
        ),
        start=Fraction(0),
    )
    quadratic_audit = endpoint_quadratic_positivity_audit(axes)
    quadratic_simplex_lower_bound = quadratic_audit[
        "harmonic_simplex_lower_bound"
    ]

    homogeneous_complement_bounds = (
        _homogeneous_complement_motion_bounds(full)
    )
    complement_perturbation_bound = sum(
        (
            bound * simplex_radius**degree
            for degree, bound in homogeneous_complement_bounds.items()
        ),
        start=Fraction(0),
    )

    complement_audit = endpoint_complement_unit_lower_bound_audit(axes)
    remainder_margin = (
        quadratic_simplex_lower_bound
        - QuadraticSurd(remainder_bound)
    )
    complement_margin = (
        Fraction(1, 11) - complement_perturbation_bound
    )
    return {
        "axes": "".join(axes),
        "radius": radius,
        "l1_radius": simplex_radius,
        "maximum_angle_radius_upper": 2 * radius,
        "schur_factor": Fraction(11, 10),
        "quadratic_simplex_lower_bound": quadratic_simplex_lower_bound,
        "remainder_bound": remainder_bound,
        "remainder_margin": remainder_margin,
        "complement_perturbation_bound": complement_perturbation_bound,
        "complement_margin": complement_margin,
        "homogeneous_kernel_bounds": homogeneous_kernel_bounds,
        "homogeneous_coupling_bounds": homogeneous_coupling_bounds,
        "homogeneous_complement_bounds": homogeneous_complement_bounds,
        "schur_homogeneous_bounds": schur_homogeneous_bounds,
        "bound_method": "projected kernel/complement norms",
        "passed": bool(
            quadratic_audit["passed"]
            and complement_audit["passed"]
            and remainder_margin.is_positive()
            and complement_margin > 0
        ),
        "arithmetic": "exact Q(sqrt(2)) and rational bounds",
    }


def _add_matrices(
    left: Sequence[Sequence[QuadraticSurd]],
    right: Sequence[Sequence[QuadraticSurd]],
) -> tuple[tuple[QuadraticSurd, ...], ...]:
    """Add two exact matrices."""

    return tuple(
        tuple(
            left_value + right_value
            for left_value, right_value in zip(left_row, right_row)
        )
        for left_row, right_row in zip(left, right)
    )


def _subtract_matrices(
    left: Sequence[Sequence[QuadraticSurd]],
    right: Sequence[Sequence[QuadraticSurd]],
) -> tuple[tuple[QuadraticSurd, ...], ...]:
    """Subtract two exact matrices."""

    return tuple(
        tuple(
            left_value - right_value
            for left_value, right_value in zip(left_row, right_row)
        )
        for left_row, right_row in zip(left, right)
    )


def _multiply_matrices(
    left: Sequence[Sequence[QuadraticSurd]],
    right: Sequence[Sequence[QuadraticSurd]],
) -> tuple[tuple[QuadraticSurd, ...], ...]:
    """Multiply exact square matrices, skipping zero factors."""

    dimension = len(left)
    output = [
        [_ZERO for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for row in range(dimension):
        for shared in range(dimension):
            left_value = left[row][shared]
            if left_value == _ZERO:
                continue
            for column in range(dimension):
                right_value = right[shared][column]
                if right_value == _ZERO:
                    continue
                output[row][column] = (
                    output[row][column] + left_value * right_value
                )
    return tuple(tuple(row) for row in output)


def _trace_product(
    left: Sequence[Sequence[QuadraticSurd]],
    right: Sequence[Sequence[QuadraticSurd]],
) -> QuadraticSurd:
    """Return ``Tr(left @ right)`` exactly."""

    dimension = len(left)
    return sum(
        (
            left[row][column] * right[column][row]
            for row in range(dimension)
            for column in range(dimension)
        ),
        start=_ZERO,
    )


@lru_cache(maxsize=64)
def _ideal_equality_matrices(
    axes: tuple[str, str, str, str, str, str],
) -> tuple[
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
    tuple[tuple[QuadraticSurd, ...], ...],
]:
    """Return ``C0``, its rank-one kernel projector, and ``R0``."""

    indices = tuple(range(64))
    equality = _exact_sector_block(
        exact_ideal_full_numerator_coefficients(axes)[_ZERO_EXPONENT],
        indices,
    )
    projector = _exact_sector_block(
        exact_target_pauli_coefficients(),
        indices,
    )
    augmented_inverse = _invert_matrix(
        _add_matrices(equality, projector)
    )
    resolvent = _subtract_matrices(augmented_inverse, projector)
    if any(
        value != _ZERO
        for row in _multiply_matrices(equality, projector)
        for value in row
    ):
        raise ArithmeticError("the ideal target projector was not in the kernel")
    return equality, projector, resolvent


@lru_cache(maxsize=64)
def exact_ideal_schur_quadratic(
    axes: tuple[str, str, str, str, str, str],
) -> dict[Exponent, QuadraticSurd]:
    """Return the exact scalar ideal-point Schur quadratic."""

    full = exact_ideal_full_numerator_coefficients(axes)
    _, projector, resolvent = _ideal_equality_matrices(axes)
    first_matrices = []
    projector_times_first = []
    resolvent_times_first = []
    for coordinate in range(6):
        exponent = tuple(
            1 if index == coordinate else 0
            for index in range(6)
        )
        matrix = _exact_sector_block(
            full[exponent],
            tuple(range(64)),
        )
        first_matrices.append(matrix)
        projector_times_first.append(
            _multiply_matrices(projector, matrix)
        )
        resolvent_times_first.append(
            _multiply_matrices(resolvent, matrix)
        )

    kappa = QuadraticSurd(Fraction(11, 10))
    output = {}
    for exponent in itertools.product(range(3), repeat=6):
        if sum(exponent) != 2:
            continue
        active = [
            coordinate
            for coordinate, degree in enumerate(exponent)
            for _ in range(degree)
        ]
        left, right = active
        second = _exact_sector_block(
            full[exponent],
            tuple(range(64)),
        )
        direct = _trace_product(projector, second)
        if left == right:
            correction = _trace_product(
                projector_times_first[left],
                resolvent_times_first[left],
            )
        else:
            correction = (
                _trace_product(
                    projector_times_first[left],
                    resolvent_times_first[right],
                )
                + _trace_product(
                    projector_times_first[right],
                    resolvent_times_first[left],
                )
            )
        output[exponent] = direct - kappa * correction
    return output


def ideal_complement_two_fifths_lower_bound_audit(
    axes: tuple[str, str, str, str, str, str],
) -> dict[str, object]:
    r"""Prove \(C(0)\succeq\frac25(I-P)\) by exact pivoted LDL."""

    equality, projector, _ = _ideal_equality_matrices(axes)
    two_fifths = QuadraticSurd(Fraction(2, 5))
    matrix = []
    for row in range(64):
        output_row = []
        for column in range(64):
            value = equality[row][column] + two_fifths * projector[row][column]
            if row == column:
                value = value - two_fifths
            output_row.append(value)
        matrix.append(output_row)
    audit = _pivoted_ldl_psd_audit(matrix)
    return {
        **audit,
        "lower_bound": Fraction(2, 5),
        "arithmetic": "exact Q(sqrt(2)) pivoted LDL",
    }


@lru_cache(maxsize=None)
def _commuting_pauli_product(
    left: PauliString,
    right: PauliString,
) -> tuple[PauliString, int] | None:
    """Return the signed product of commuting Hermitian Pauli strings."""

    local_products = {
        (1, 2): (3, 1j),
        (2, 1): (3, -1j),
        (2, 3): (1, 1j),
        (3, 2): (1, -1j),
        (3, 1): (2, 1j),
        (1, 3): (2, -1j),
    }
    output = []
    phase = 1.0 + 0.0j
    for left_pauli, right_pauli in zip(left, right):
        if left_pauli == 0:
            output.append(right_pauli)
        elif right_pauli == 0:
            output.append(left_pauli)
        elif left_pauli == right_pauli:
            output.append(0)
        else:
            product, local_phase = local_products[
                (left_pauli, right_pauli)
            ]
            output.append(product)
            phase *= local_phase
    if abs(phase.imag) > 0.5:
        return None
    return tuple(output), int(round(phase.real))


def _ideal_target_expectation_of_square(
    coefficients: Mapping[PauliString, QuadraticSurd],
) -> QuadraticSurd:
    r"""Return \(\operatorname{Tr}(P_{\rm CCZ}N^2)\) exactly."""

    target = exact_target_pauli_coefficients()
    items = tuple(coefficients.items())
    output = sum(
        (coefficient * coefficient for _, coefficient in items),
        start=_ZERO,
    )
    for left_index, (left_pauli, left_coefficient) in enumerate(items):
        for right_pauli, right_coefficient in items[left_index + 1:]:
            product = _commuting_pauli_product(left_pauli, right_pauli)
            if product is None:
                continue
            product_pauli, sign = product
            target_coefficient = target.get(product_pauli, _ZERO)
            if target_coefficient == _ZERO:
                continue
            output = (
                output
                + 128
                * sign
                * left_coefficient
                * right_coefficient
                * target_coefficient
            )
    return output


def _ideal_projected_homogeneous_bounds(
    full: Mapping[
        Exponent,
        Mapping[PauliString, QuadraticSurd],
    ],
) -> tuple[dict[int, Fraction], dict[int, Fraction]]:
    r"""Bound the ideal kernel block and kernel--complement coupling."""

    target = exact_target_pauli_coefficients()
    kernel_bounds: dict[int, Fraction] = {}
    coupling_bounds: dict[int, Fraction] = {}
    for exponent, pauli_coefficients in full.items():
        degree = sum(exponent)
        multinomial = _multinomial(exponent)
        kernel_coefficient = 64 * sum(
            (
                target.get(pauli_string, _ZERO) * coefficient
                for pauli_string, coefficient in pauli_coefficients.items()
            ),
            start=_ZERO,
        )
        square_expectation = _ideal_target_expectation_of_square(
            pauli_coefficients
        )
        coupling_squared = (
            square_expectation
            - kernel_coefficient * kernel_coefficient
        )
        if coupling_squared.sign() < 0:
            raise ArithmeticError(
                "the ideal projected coupling norm became negative"
            )
        kernel_bound = (
            _rational_absolute_upper_tight(kernel_coefficient)
            / multinomial
        )
        coupling_bound = (
            _rational_square_root_upper(coupling_squared)
            / multinomial
        )
        kernel_bounds[degree] = max(
            kernel_bounds.get(degree, Fraction(0)),
            kernel_bound,
        )
        coupling_bounds[degree] = max(
            coupling_bounds.get(degree, Fraction(0)),
            coupling_bound,
        )
    return kernel_bounds, coupling_bounds


def exact_ideal_local_certificate(
    axes: tuple[str, str, str, str, str, str],
    *,
    radius: Fraction = Fraction(29, 60000),
) -> dict[str, object]:
    """Return an exact local certificate around the all-ideal equality point."""

    radius = Fraction(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    full = exact_ideal_full_numerator_coefficients(axes)
    quadratic = exact_ideal_schur_quadratic(axes)
    diagonal = []
    for coordinate in range(6):
        exponent = tuple(
            2 if index == coordinate else 0
            for index in range(6)
        )
        diagonal.append(quadratic[exponent])
    for exponent, coefficient in quadratic.items():
        if sum(1 for degree in exponent if degree) > 1 and coefficient != _ZERO:
            raise ArithmeticError("the ideal quadratic had a mixed term")
    quadratic_simplex_lower_bound = sum(
        (coefficient.inverse() for coefficient in diagonal),
        start=_ZERO,
    ).inverse()

    (
        homogeneous_kernel_bounds,
        homogeneous_coupling_bounds,
    ) = _ideal_projected_homogeneous_bounds(full)

    schur_homogeneous_bounds: dict[int, Fraction] = {}
    # (11/10) * ||R0|| <= (11/10) * (5/2) = 11/4.
    coupling_factor = Fraction(11, 4)
    for degree in range(3, 25):
        denominator_times_kernel = sum(
            (
                homogeneous_kernel_bounds.get(
                    degree - 2 * order,
                    Fraction(0),
                )
                for order in range(7)
                if degree - 2 * order >= 2
            ),
            start=Fraction(0),
        )
        coupling_product = sum(
            (
                homogeneous_coupling_bounds.get(left, Fraction(0))
                * homogeneous_coupling_bounds.get(
                    degree - left,
                    Fraction(0),
                )
                for left in range(1, degree)
            ),
            start=Fraction(0),
        )
        schur_homogeneous_bounds[degree] = (
            denominator_times_kernel
            + coupling_factor * coupling_product
        )
    simplex_radius = 6 * radius
    remainder_bound = sum(
        (
            bound * simplex_radius ** (degree - 2)
            for degree, bound in schur_homogeneous_bounds.items()
        ),
        start=Fraction(0),
    )

    homogeneous_complement_bounds = (
        _homogeneous_complement_motion_bounds(full)
    )
    complement_perturbation_bound = sum(
        (
            bound * simplex_radius**degree
            for degree, bound in homogeneous_complement_bounds.items()
        ),
        start=Fraction(0),
    )

    complement_audit = ideal_complement_two_fifths_lower_bound_audit(axes)
    remainder_margin = (
        quadratic_simplex_lower_bound
        - QuadraticSurd(remainder_bound)
    )
    # C0 >= (2/5)Q and kappa=11/10 permit ||C-C0|| < 2/55.
    complement_margin = (
        Fraction(2, 55) - complement_perturbation_bound
    )
    return {
        "axes": "".join(axes),
        "radius": radius,
        "l1_radius": simplex_radius,
        "maximum_angle_radius_upper": 2 * radius,
        "schur_factor": Fraction(11, 10),
        "quadratic_simplex_lower_bound": quadratic_simplex_lower_bound,
        "remainder_bound": remainder_bound,
        "remainder_margin": remainder_margin,
        "complement_perturbation_bound": complement_perturbation_bound,
        "complement_margin": complement_margin,
        "homogeneous_kernel_bounds": homogeneous_kernel_bounds,
        "homogeneous_coupling_bounds": homogeneous_coupling_bounds,
        "homogeneous_complement_bounds": homogeneous_complement_bounds,
        "schur_homogeneous_bounds": schur_homogeneous_bounds,
        "bound_method": "projected kernel/complement norms",
        "passed": bool(
            complement_audit["passed"]
            and remainder_margin.is_positive()
            and complement_margin > 0
        ),
        "arithmetic": "exact Q(sqrt(2)) and rational bounds",
    }


def _fraction_string(value: Fraction) -> str:
    """Serialize a rational number canonically."""

    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _surd_record(value: QuadraticSurd) -> dict[str, object]:
    """Serialize one exact quadratic surd and its display approximation."""

    return {
        "rational": _fraction_string(value.rational),
        "root_two": _fraction_string(value.root_two),
        "decimal": float(value),
    }


def generate_exact_ideal_orbit_record(
    orbit_index: int,
    output_path: Path | str,
) -> dict[str, object]:
    """Generate one exact all-ideal local certificate orbit record."""

    representatives = local_branch_orbit_representatives()
    if not 0 <= orbit_index < len(representatives):
        raise ValueError("orbit index must lie between 0 and 19")
    representative = representatives[orbit_index]
    certificate = exact_ideal_local_certificate(
        representative["axes"]  # type: ignore[arg-type]
    )
    record = {
        "orbit_index": orbit_index,
        "site_types": representative["site_types"],
        "axes": certificate["axes"],
        "orbit_size": representative["orbit_size"],
        "passed": certificate["passed"],
        "radius": _fraction_string(certificate["radius"]),
        "l1_radius": _fraction_string(certificate["l1_radius"]),
        "quadratic_simplex_lower_bound": _surd_record(
            certificate["quadratic_simplex_lower_bound"]
        ),
        "remainder_bound": _fraction_string(
            certificate["remainder_bound"]
        ),
        "remainder_bound_decimal": float(
            certificate["remainder_bound"]
        ),
        "complement_perturbation_bound": _fraction_string(
            certificate["complement_perturbation_bound"]
        ),
        "complement_perturbation_bound_decimal": float(
            certificate["complement_perturbation_bound"]
        ),
        "arithmetic": certificate["arithmetic"],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def aggregate_exact_local_records(
    ideal_directory: Path | str,
    endpoint_path: Path | str = _DEFAULT_EXACT_RECORD,
    output_path: Path | str = _DEFAULT_LOCAL_AGGREGATE,
) -> dict[str, object]:
    """Validate and aggregate the ideal and endpoint local certificates."""

    import hashlib

    source_directory = Path(ideal_directory)
    ideal_records = []
    digests = {}
    for orbit_index in range(20):
        path = source_directory / f"ideal_orbit_{orbit_index}.json"
        payload = path.read_bytes()
        record = json.loads(payload)
        if record.get("orbit_index") != orbit_index:
            raise ArithmeticError("ideal local orbit index mismatch")
        if record.get("passed") is not True:
            raise ArithmeticError(
                f"ideal local orbit {orbit_index} did not pass"
            )
        ideal_records.append(record)
        digests[path.name] = hashlib.sha256(payload).hexdigest()

    endpoint_file = Path(endpoint_path)
    endpoint_payload = endpoint_file.read_bytes()
    endpoint_record = json.loads(endpoint_payload)
    if endpoint_record.get("all_passed") is not True:
        raise ArithmeticError("endpoint local certificate did not pass")
    if endpoint_record.get("covered_adjacent_branches") != 16:
        raise ArithmeticError("endpoint branch count mismatch")
    digests[endpoint_file.name] = hashlib.sha256(
        endpoint_payload
    ).hexdigest()

    physical_branches = sum(
        int(record["orbit_size"]) for record in ideal_records
    )
    if physical_branches != 64:
        raise ArithmeticError(
            "ideal local orbit multiplicities did not sum to 64"
        )
    aggregate = {
        "status": "passed_exact_local_certificate",
        "candidate_slope": "(4 + 5*sqrt(2))/16",
        "eta": "1/2",
        "l1_radius": ideal_records[0]["l1_radius"],
        "ideal_branch_orbits": len(ideal_records),
        "ideal_physical_branches": physical_branches,
        "endpoint_adjacent_branches": 16,
        "other_endpoint_orbits": (
            "Simultaneous site permutations and the canonical Bob axis "
            "swap cover the remaining one-bad-pair equality points."
        ),
        "arithmetic": (
            "exact Q(sqrt(2)) pivoted LDL and projected rational "
            "kernel/complement bounds"
        ),
        "sha256": digests,
        "ideal_records": ideal_records,
        "endpoint_record": endpoint_record,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _endpoint_certificate_summary(
    certificate: Mapping[str, object],
) -> dict[str, object]:
    """Serialize one exact canonical-endpoint branch certificate."""

    return {
        "axes": certificate["axes"],
        "passed": certificate["passed"],
        "quadratic_simplex_lower_bound": _surd_record(
            certificate["quadratic_simplex_lower_bound"]  # type: ignore[arg-type]
        ),
        "remainder_bound": _fraction_string(
            certificate["remainder_bound"]  # type: ignore[arg-type]
        ),
        "remainder_bound_decimal": float(certificate["remainder_bound"]),
        "remainder_margin": _surd_record(
            certificate["remainder_margin"]  # type: ignore[arg-type]
        ),
        "complement_perturbation_bound": _fraction_string(
            certificate["complement_perturbation_bound"]  # type: ignore[arg-type]
        ),
        "complement_perturbation_bound_decimal": float(
            certificate["complement_perturbation_bound"]
        ),
        "complement_margin": _fraction_string(
            certificate["complement_margin"]  # type: ignore[arg-type]
        ),
    }


def generate_exact_endpoint_branch_record(
    branch_index: int,
    output_path: Path | str,
) -> dict[str, object]:
    """Generate one canonical-endpoint adjacent-branch record."""

    branches = endpoint_adjacent_branch_axes()
    if not 0 <= branch_index < len(branches):
        raise ValueError("endpoint branch index must lie between 0 and 15")
    certificate = exact_endpoint_local_certificate(branches[branch_index])
    record = {
        "branch_index": branch_index,
        "radius": _fraction_string(certificate["radius"]),
        "l1_radius": _fraction_string(certificate["l1_radius"]),
        **_endpoint_certificate_summary(certificate),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def aggregate_exact_endpoint_branch_records(
    directory: Path | str,
    output_path: Path | str = _DEFAULT_EXACT_RECORD,
) -> dict[str, object]:
    """Aggregate the 16 independently generated endpoint branch records."""

    source_directory = Path(directory)
    records = []
    for branch_index in range(16):
        path = source_directory / f"endpoint_branch_{branch_index}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("branch_index") != branch_index:
            raise ArithmeticError("endpoint branch index mismatch")
        if record.get("passed") is not True:
            raise ArithmeticError(
                f"endpoint branch {branch_index} did not pass"
            )
        records.append(record)
    record = {
        "status": "passed_exact_local_endpoint_certificate",
        "candidate_slope": "(4 + 5*sqrt(2))/16",
        "eta": "1/2",
        "radius": records[0]["radius"],
        "l1_radius": records[0]["l1_radius"],
        "maximum_angle_radius_upper": _fraction_string(
            2 * Fraction(records[0]["radius"])
        ),
        "schur_factor": "11/10",
        "covered_adjacent_branches": len(records),
        "all_passed": True,
        "other_equality_orbits": (
            "The other bad sites follow by simultaneous site permutations; "
            "the Bob pi/2 endpoint follows by the canonical local axis swap."
        ),
        "arithmetic": "exact Q(sqrt(2)) plus rational projected-norm bounds",
        "records": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def generate_exact_endpoint_certificate_record(
    output_path: Path | str = _DEFAULT_EXACT_RECORD,
) -> dict[str, object]:
    """Generate the exact local certificate record for all 16 branches."""

    records = []
    for axes in endpoint_adjacent_branch_axes():
        certificate = exact_endpoint_local_certificate(axes)
        records.append(_endpoint_certificate_summary(certificate))
    record = {
        "status": "passed_exact_local_endpoint_certificate",
        "candidate_slope": "(4 + 5*sqrt(2))/16",
        "eta": "1/2",
        "radius": _fraction_string(certificate["radius"]),
        "l1_radius": _fraction_string(
            certificate["l1_radius"]
        ),
        "maximum_angle_radius_upper": _fraction_string(
            certificate["maximum_angle_radius_upper"]
        ),
        "schur_factor": "11/10",
        "covered_adjacent_branches": len(records),
        "all_passed": all(record["passed"] for record in records),
        "other_equality_orbits": (
            "The other bad sites follow by simultaneous site permutations; "
            "the Bob pi/2 endpoint follows by the canonical local axis swap."
        ),
        "arithmetic": "exact Q(sqrt(2)) plus rational Pauli-norm bounds",
        "records": records,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    """Generate the default exact endpoint certificate record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ideal-orbit-index",
        type=int,
        default=None,
        help="generate one all-ideal branch-orbit record instead",
    )
    parser.add_argument(
        "--endpoint-branch-index",
        type=int,
        default=None,
        help="generate one canonical-endpoint adjacent-branch record",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_EXACT_RECORD,
    )
    arguments = parser.parse_args()
    if (
        arguments.ideal_orbit_index is not None
        and arguments.endpoint_branch_index is not None
    ):
        raise ValueError(
            "choose either an ideal orbit or an endpoint branch"
        )
    if arguments.ideal_orbit_index is not None:
        record = generate_exact_ideal_orbit_record(
            arguments.ideal_orbit_index,
            arguments.output,
        )
    elif arguments.endpoint_branch_index is not None:
        record = generate_exact_endpoint_branch_record(
            arguments.endpoint_branch_index,
            arguments.output,
        )
    else:
        record = generate_exact_endpoint_certificate_record(arguments.output)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
