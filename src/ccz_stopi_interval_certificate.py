#!/usr/bin/env python3
r"""Validated global certificate for the CCZ STOPI coupling polynomial.

On a fixed low/high extraction branch, write

.. math::
   z_i=(\sqrt2-1)x_i,\qquad 0\le x_i\le1,

and

.. math::
   C_{1/2}(x)
   =\frac{N(x)}{\prod_i(1+z_i^2)}.

The denominator is positive.  This module therefore certifies
``N(x) >= 0``.  It combines two rigorous ingredients:

1. tensor-product Bernstein positivity on dyadic boxes;
2. the exact local Schur certificates at the seven equality orbits.

Every floating matrix carries an explicit operator-norm enclosure of its
exact :math:`\mathbb Q(\sqrt2)` value.  Positive definiteness is accepted
only after a Cholesky factor is replayed through a residual norm bound.
A center-and-derivative calculation is used only to select a subdivision
coordinate and is not an acceptance criterion.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ccz_stopi_exact import (
    Exponent,
    PauliString,
    QuadraticSurd,
    _absolute_surd,
    _apply_pauli_string,
    _rational_absolute_upper_tight,
    _rational_square_root_upper,
    _ROOT_TWO_LOWER,
    _ROOT_TWO_UPPER,
    exact_ideal_full_numerator_coefficients,
    local_branch_orbit_representatives,
)

_DIMENSION = 64
_ROOT_TWO_MINUS_ONE = QuadraticSurd(-1, 1)
_LOCAL_L1_RADIUS = Fraction(29, 10000)
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
_POWER_TO_BERNSTEIN = np.array(
    (
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)
_LEFT_SUBDIVISION = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.25, 0.5, 0.25),
    ),
    dtype=np.float64,
)
_RIGHT_SUBDIVISION = np.array(
    (
        (0.25, 0.5, 0.25),
        (0.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def _gamma(operation_count: int) -> float:
    """Return the standard IEEE-754 ``gamma_n`` error factor."""

    product = operation_count * _UNIT_ROUNDOFF
    if product >= 1.0:
        raise ArithmeticError("floating error factor left its valid range")
    return product / (1.0 - product)


def _upward(value: float) -> float:
    """Round one nonnegative floating bound toward positive infinity."""

    if value < 0.0 or not math.isfinite(value):
        raise ArithmeticError("expected a finite nonnegative bound")
    return float(np.nextafter(value, math.inf))


def _downward(value: float) -> float:
    """Round one finite floating bound toward negative infinity."""

    if not math.isfinite(value):
        raise ArithmeticError("expected a finite bound")
    return float(np.nextafter(value, -math.inf))


def _positive_product_up(left: float, right: float) -> float:
    """Multiply nonnegative upper bounds with outward rounding."""

    return _upward(left * right)


def _positive_product_down(left: float, right: float) -> float:
    """Multiply nonnegative lower bounds with outward rounding."""

    return _downward(left * right)


def _positive_sum_up(values: Sequence[float]) -> float:
    """Sum nonnegative upper bounds with outward rounding."""

    output = 0.0
    for value in values:
        output = _upward(output + value)
    return output


@dataclass(frozen=True)
class RootBernsteinData:
    """Power and Bernstein tensors with rigorous norm-error enclosures."""

    axes: tuple[str, str, str, str, str, str]
    exponents: tuple[Exponent, ...]
    power_matrices: np.ndarray
    power_errors: np.ndarray
    power_magnitudes: np.ndarray
    coefficient_norm_bounds: tuple[float, ...]
    bernstein_matrices: np.ndarray
    bernstein_errors: np.ndarray
    bernstein_magnitudes: np.ndarray


@dataclass(frozen=True)
class DyadicCell:
    """A product of six dyadic intervals in normalized half-angle space."""

    levels: tuple[int, int, int, int, int, int]
    indices: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.levels) != 6 or len(self.indices) != 6:
            raise ValueError("a dyadic cell requires six levels and indices")
        for level, index in zip(self.levels, self.indices):
            if level < 0 or index < 0 or index >= 2**level:
                raise ValueError("invalid dyadic interval")

    @property
    def depth(self) -> int:
        """Return the total number of coordinate bisections."""

        return sum(self.levels)

    def lower(self, coordinate: int) -> Fraction:
        """Return one exact normalized lower endpoint."""

        return Fraction(
            self.indices[coordinate],
            2 ** self.levels[coordinate],
        )

    def upper(self, coordinate: int) -> Fraction:
        """Return one exact normalized upper endpoint."""

        return Fraction(
            self.indices[coordinate] + 1,
            2 ** self.levels[coordinate],
        )

    def split(self, coordinate: int) -> tuple["DyadicCell", "DyadicCell"]:
        """Bisect one coordinate and return the left and right cells."""

        levels = list(self.levels)
        indices = list(self.indices)
        levels[coordinate] += 1
        left_indices = list(indices)
        right_indices = list(indices)
        left_indices[coordinate] = 2 * indices[coordinate]
        right_indices[coordinate] = 2 * indices[coordinate] + 1
        return (
            DyadicCell(tuple(levels), tuple(left_indices)),
            DyadicCell(tuple(levels), tuple(right_indices)),
        )


@lru_cache(maxsize=1)
def _pauli_actions() -> tuple[
    tuple[PauliString, ...],
    Mapping[PauliString, int],
    np.ndarray,
    np.ndarray,
]:
    """Return real matrix-entry maps for all six-qubit Pauli strings."""

    pauli_strings = tuple(itertools.product(range(4), repeat=6))
    positions = {pauli_string: index for index, pauli_string in enumerate(
        pauli_strings
    )}
    flat_indices = np.empty((4**6, _DIMENSION), dtype=np.int32)
    signs = np.empty((4**6, _DIMENSION), dtype=np.int8)
    for pauli_index, pauli_string in enumerate(pauli_strings):
        for column in range(_DIMENSION):
            row, phase = _apply_pauli_string(pauli_string, column)
            if abs(phase.imag) > 0.5:
                signs[pauli_index, column] = 0
            else:
                signs[pauli_index, column] = int(round(phase.real))
            flat_indices[pauli_index, column] = (
                row * _DIMENSION + column
            )
    return pauli_strings, positions, flat_indices, signs


def _verified_operator_norm_upper(
    matrix: np.ndarray,
    input_error: float,
    fallback: float,
) -> float:
    r"""Prove \(\|A\|_2\le m\) through \(mI\pm A\succeq0\)."""

    approximate = np.asarray(matrix, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(approximate)
    estimate = float(max(abs(eigenvalues[0]), abs(eigenvalues[-1])))
    matrix_magnitude = np.linalg.norm(approximate, ord=np.inf)
    identity = np.eye(_DIMENSION, dtype=np.float64)
    for multiplier in (64.0, 256.0, 1024.0):
        margin = max(
            multiplier * input_error,
            multiplier * _UNIT_ROUNDOFF * (1.0 + estimate),
        )
        bound = _upward(estimate + margin)
        candidate_error = _upward(
            input_error
            + _gamma(2) * (matrix_magnitude + bound)
        )
        proposed_shift = margin / 4.0
        lower_plus = verified_cholesky_floor(
            bound * identity + approximate,
            input_error=candidate_error,
            proposed_shift=proposed_shift,
        )
        lower_minus = verified_cholesky_floor(
            bound * identity - approximate,
            input_error=candidate_error,
            proposed_shift=proposed_shift,
        )
        if lower_plus > 0.0 and lower_minus > 0.0:
            return bound
    return _upward(fallback)


def _scaled_power_tensors(
    coefficients: Mapping[
        Exponent,
        Mapping[PauliString, QuadraticSurd],
    ],
) -> tuple[
    tuple[Exponent, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, ...],
]:
    """Convert exact scaled power coefficients to enclosed float matrices."""

    _, positions, flat_indices, signs = _pauli_actions()
    exponents = tuple(coefficients)
    matrices = np.zeros(
        (3,) * 6 + (_DIMENSION, _DIMENSION),
        dtype=np.float64,
    )
    errors = np.zeros((3,) * 6, dtype=np.float64)
    magnitudes = np.zeros((3,) * 6, dtype=np.float64)
    frobenius_bounds: list[Fraction] = []
    gamma_sum = _gamma(_DIMENSION)
    for exponent in exponents:
        scale = _ROOT_TWO_MINUS_ONE
        exact_scale = QuadraticSurd(1)
        for _ in range(sum(exponent)):
            exact_scale = exact_scale * scale
        flat_matrix = matrices[exponent].reshape(-1)
        conversion_error = Fraction(0)
        magnitude_bound = Fraction(0)
        frobenius_squared = QuadraticSurd()
        for pauli_string, coefficient in coefficients[exponent].items():
            scaled = coefficient * exact_scale
            approximate = float(scaled)
            exact_approximate = QuadraticSurd(
                Fraction.from_float(approximate)
            )
            conversion_error += _rational_absolute_upper_tight(
                scaled - exact_approximate
            )
            magnitude_bound += _rational_absolute_upper_tight(scaled)
            frobenius_squared = (
                frobenius_squared + 64 * scaled * scaled
            )
            pauli_index = positions[pauli_string]
            real_signs = signs[pauli_index]
            if not np.any(real_signs):
                raise ArithmeticError(
                    "a real coefficient tensor contained an odd-Y Pauli"
                )
            flat_matrix[flat_indices[pauli_index]] += (
                approximate * real_signs
            )
        magnitude_float = _upward(float(magnitude_bound))
        error = (
            _upward(float(conversion_error))
            + gamma_sum * magnitude_float
        )
        errors[exponent] = _upward(error)
        magnitudes[exponent] = _upward(
            magnitude_float + errors[exponent]
        )
        frobenius_bounds.append(
            _rational_square_root_upper(frobenius_squared)
        )
        if not np.array_equal(matrices[exponent], matrices[exponent].T):
            raise ArithmeticError("a power coefficient lost exact symmetry")
    norm_bounds = []
    for exponent, fallback in zip(exponents, frobenius_bounds):
        norm_bounds.append(
            _verified_operator_norm_upper(
                matrices[exponent],
                float(errors[exponent]),
                float(fallback),
            )
        )
    return (
        exponents,
        matrices,
        errors,
        magnitudes,
        tuple(norm_bounds),
    )


def _positive_axis_transform(
    matrices: np.ndarray,
    errors: np.ndarray,
    magnitudes: np.ndarray,
    transform: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a nonnegative three-term transform with norm-error propagation."""

    transformed_matrices = np.moveaxis(
        np.tensordot(transform, matrices, axes=(1, axis)),
        0,
        axis,
    )
    linear_errors = np.moveaxis(
        np.tensordot(transform, errors, axes=(1, axis)),
        0,
        axis,
    )
    linear_magnitudes = np.moveaxis(
        np.tensordot(transform, magnitudes, axes=(1, axis)),
        0,
        axis,
    )
    rounding = _gamma(3) * linear_magnitudes
    transformed_errors = np.nextafter(
        linear_errors + rounding,
        math.inf,
    )
    transformed_magnitudes = np.nextafter(
        linear_magnitudes + transformed_errors,
        math.inf,
    )
    return (
        transformed_matrices,
        transformed_errors,
        transformed_magnitudes,
    )


@lru_cache(maxsize=64)
def build_root_bernstein_data(
    axes: tuple[str, str, str, str, str, str],
) -> RootBernsteinData:
    """Build the enclosed root Bernstein tensor for one extraction branch."""

    if len(axes) != 6 or any(axis not in ("X", "Z") for axis in axes):
        raise ValueError("axes must contain six X/Z labels")
    exact = exact_ideal_full_numerator_coefficients(axes)
    (
        exponents,
        power_matrices,
        power_errors,
        power_magnitudes,
        norm_bounds,
    ) = _scaled_power_tensors(exact)
    bernstein_matrices = power_matrices
    bernstein_errors = power_errors
    bernstein_magnitudes = power_magnitudes
    for axis in range(6):
        (
            bernstein_matrices,
            bernstein_errors,
            bernstein_magnitudes,
        ) = _positive_axis_transform(
            bernstein_matrices,
            bernstein_errors,
            bernstein_magnitudes,
            _POWER_TO_BERNSTEIN,
            axis,
        )
    return RootBernsteinData(
        axes=axes,
        exponents=exponents,
        power_matrices=power_matrices,
        power_errors=power_errors,
        power_magnitudes=power_magnitudes,
        coefficient_norm_bounds=norm_bounds,
        bernstein_matrices=bernstein_matrices,
        bernstein_errors=bernstein_errors,
        bernstein_magnitudes=bernstein_magnitudes,
    )


def evaluate_power_tensor(
    power_matrices: np.ndarray,
    coordinates: Sequence[float],
) -> np.ndarray:
    """Evaluate a degree-two-per-variable power tensor."""

    if len(coordinates) != 6:
        raise ValueError("six coordinates are required")
    weights = np.array((1.0,), dtype=np.float64)
    for coordinate in coordinates:
        value = float(coordinate)
        weights = np.kron(weights, (1.0, value, value * value))
    return (
        weights @ power_matrices.reshape(3**6, _DIMENSION**2)
    ).reshape(_DIMENSION, _DIMENSION)


def _float_operator_norm_upper(matrix: np.ndarray) -> float:
    r"""Upper-bound \(\|A\|_2\) through enclosed row and column sums."""

    absolute = np.abs(np.asarray(matrix, dtype=np.float64))
    row_sum = float(np.max(np.sum(absolute, axis=1)))
    column_sum = float(np.max(np.sum(absolute, axis=0)))
    row_upper = _upward(row_sum / (1.0 - _gamma(_DIMENSION)))
    column_upper = _upward(
        column_sum / (1.0 - _gamma(_DIMENSION))
    )
    product_upper = _positive_product_up(row_upper, column_upper)
    return _upward(math.sqrt(product_upper))


def verified_cholesky_floor(
    matrix: np.ndarray,
    *,
    input_error: float,
    proposed_shift: float,
) -> float:
    r"""Prove \(A_{\rm exact}\succeq\mu I\) from a Cholesky residual."""

    approximate = np.asarray(matrix, dtype=np.float64)
    if approximate.shape != (_DIMENSION, _DIMENSION):
        raise ValueError("matrix must be 64 by 64")
    if not np.array_equal(approximate, approximate.T):
        raise ArithmeticError(
            "verified Cholesky input must be exactly symmetric"
        )
    if input_error < 0.0 or not math.isfinite(input_error):
        raise ValueError("input_error must be finite and nonnegative")
    shift = float(proposed_shift)
    if shift <= 0.0 or not math.isfinite(shift):
        return 0.0
    identity = np.eye(_DIMENSION, dtype=np.float64)
    shifted = approximate - shift * identity
    try:
        factor = np.linalg.cholesky(shifted)
    except np.linalg.LinAlgError:
        return 0.0
    product = factor @ factor.T
    residual = shifted - product

    gamma_rows = _gamma(_DIMENSION)
    residual_norm = _float_operator_norm_upper(residual)
    absolute_product = np.abs(factor) @ np.abs(factor).T
    # The helper encloses the row/column sums; this extra factor encloses
    # the preceding matrix product that formed ``absolute_product``.
    product_magnitude = _float_operator_norm_upper(absolute_product)
    product_magnitude = _upward(
        product_magnitude / (1.0 - gamma_rows)
    )
    product_roundoff = gamma_rows * product_magnitude

    matrix_magnitude = _float_operator_norm_upper(approximate)
    computed_product_magnitude = _float_operator_norm_upper(product)
    shifted_roundoff = _gamma(2) * (
        matrix_magnitude + shift
    )
    subtraction_roundoff = _gamma(2) * (
        matrix_magnitude
        + shift
        + computed_product_magnitude
    )
    total_error = _upward(
        input_error
        + residual_norm
        + product_roundoff
        + shifted_roundoff
        + subtraction_roundoff
    )
    lower_bound = float(np.nextafter(shift - total_error, -math.inf))
    return lower_bound if lower_bound > 0.0 else 0.0


def _scaled_coordinate(value: Fraction) -> QuadraticSurd:
    """Return ``(sqrt(2)-1) * value`` exactly."""

    return QuadraticSurd(-value, value)


def _endpoint_inward_coordinate(value: Fraction) -> QuadraticSurd:
    r"""Return \(\tan(\theta/2)\) from the ideal-centered coordinate."""

    z = _scaled_coordinate(value)
    numerator = _ROOT_TWO_MINUS_ONE - z
    denominator = QuadraticSurd(1) + _ROOT_TWO_MINUS_ONE * z
    return numerator / denominator


def cell_is_inside_exact_local_region(
    cell: DyadicCell,
    axes: tuple[str, str, str, str, str, str],
) -> bool:
    """Test the exact ideal or endpoint local-Schur handoff conditions."""

    if len(axes) != 6:
        raise ValueError("six branch axes are required")
    upper_floats = tuple(
        (index + 1) / 2**level
        for level, index in zip(cell.levels, cell.indices)
    )
    lower_floats = tuple(
        index / 2**level
        for level, index in zip(cell.levels, cell.indices)
    )
    root_offset_upper = _upward(float(_ROOT_TWO_UPPER - 1))
    root_offset_lower = _downward(float(_ROOT_TWO_LOWER - 1))
    local_radius_lower = _downward(float(_LOCAL_L1_RADIUS))

    ideal_upper = _positive_product_up(
        root_offset_upper,
        _positive_sum_up(upper_floats),
    )
    if ideal_upper <= local_radius_lower:
        return True

    for site in range(3):
        if axes[site] != "Z":
            continue
        good_upper = _positive_product_up(
            root_offset_upper,
            _positive_sum_up(
                [
                    upper_floats[index]
                    for index in range(6)
                    if index not in (site, 3 + site)
                ]
            ),
        )
        inward_terms = [good_upper]
        for index in (site, 3 + site):
            one_minus = _upward(1.0 - lower_floats[index])
            numerator_upper = _positive_product_up(
                root_offset_upper,
                one_minus,
            )
            root_square_lower = _positive_product_down(
                root_offset_lower,
                root_offset_lower,
            )
            denominator_lower = _downward(
                1.0
                + _positive_product_down(
                    root_square_lower,
                    lower_floats[index],
                )
            )
            inward_terms.append(
                _upward(numerator_upper / denominator_lower)
            )
        if _positive_sum_up(inward_terms) <= local_radius_lower:
            return True
    return False


@dataclass
class _CertificateState:
    """Sparse unresolved Bernstein data plus certified scalar floors."""

    residual_matrices: dict[Exponent, np.ndarray]
    residual_errors: dict[Exponent, float]
    residual_magnitudes: dict[Exponent, float]
    floors: np.ndarray


def _initial_certificate_state(
    data: RootBernsteinData,
) -> tuple[_CertificateState, dict[str, float]]:
    """Verify the root coefficients and retain only unresolved residuals."""

    residual_matrices: dict[Exponent, np.ndarray] = {}
    residual_errors: dict[Exponent, float] = {}
    residual_magnitudes: dict[Exponent, float] = {}
    floors = np.zeros((3,) * 6, dtype=np.float64)
    minimum_floor = math.inf
    smallest_estimate = math.inf
    verified = 0
    for key in itertools.product(range(3), repeat=6):
        matrix = data.bernstein_matrices[key]
        input_error = float(data.bernstein_errors[key])
        estimate = float(np.linalg.eigvalsh(matrix)[0])
        smallest_estimate = min(smallest_estimate, estimate)
        floor = 0.0
        if estimate > 4.0 * input_error:
            floor = verified_cholesky_floor(
                matrix,
                input_error=input_error,
                proposed_shift=estimate / 2.0,
            )
        if floor > 0.0:
            floors[key] = floor
            minimum_floor = min(minimum_floor, floor)
            verified += 1
        else:
            residual_matrices[key] = matrix.copy()
            residual_errors[key] = input_error
            residual_magnitudes[key] = float(
                data.bernstein_magnitudes[key]
            )
    return (
        _CertificateState(
            residual_matrices=residual_matrices,
            residual_errors=residual_errors,
            residual_magnitudes=residual_magnitudes,
            floors=floors,
        ),
        {
            "verified_coefficients": float(verified),
            "unresolved_coefficients": float(len(residual_matrices)),
            "minimum_verified_floor": (
                minimum_floor if math.isfinite(minimum_floor) else 0.0
            ),
            "minimum_floating_estimate": smallest_estimate,
        },
    )


def _downward_floor_transform(
    floors: np.ndarray,
    transform: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Transform certified identity floors with downward rounding."""

    raw = np.moveaxis(
        np.tensordot(transform, floors, axes=(1, axis)),
        0,
        axis,
    )
    rounding = _gamma(3) * np.moveaxis(
        np.tensordot(transform, np.abs(floors), axes=(1, axis)),
        0,
        axis,
    )
    return np.maximum(
        0.0,
        np.nextafter(raw - rounding, -math.inf),
    )


def _sparse_midpoint_transform(
    state: _CertificateState,
    axis: int,
    transform: np.ndarray,
) -> _CertificateState:
    """Subdivide one sparse Bernstein state and verify new coefficients."""

    matrices: dict[Exponent, np.ndarray] = {}
    linear_errors: dict[Exponent, float] = {}
    linear_magnitudes: dict[Exponent, float] = {}
    for key, matrix in state.residual_matrices.items():
        old_degree = key[axis]
        parent_error = state.residual_errors[key]
        parent_magnitude = state.residual_magnitudes[key]
        for new_degree in range(3):
            weight = float(transform[new_degree, old_degree])
            if weight == 0.0:
                continue
            new_key = list(key)
            new_key[axis] = new_degree
            parsed_key = tuple(new_key)
            weighted = weight * matrix
            if parsed_key in matrices:
                matrices[parsed_key] = matrices[parsed_key] + weighted
            else:
                matrices[parsed_key] = weighted.copy()
            linear_errors[parsed_key] = (
                linear_errors.get(parsed_key, 0.0)
                + weight * parent_error
            )
            linear_magnitudes[parsed_key] = (
                linear_magnitudes.get(parsed_key, 0.0)
                + weight * parent_magnitude
            )

    floors = _downward_floor_transform(
        state.floors,
        transform,
        axis,
    )
    residual_matrices: dict[Exponent, np.ndarray] = {}
    residual_errors: dict[Exponent, float] = {}
    residual_magnitudes: dict[Exponent, float] = {}
    identity = np.eye(_DIMENSION, dtype=np.float64)
    for key, matrix in matrices.items():
        magnitude = _upward(linear_magnitudes[key])
        input_error = _upward(
            linear_errors[key] + _gamma(3) * magnitude
        )
        scalar_floor = float(floors[key])
        candidate = matrix + scalar_floor * identity
        addition_error = _gamma(2) * (
            np.linalg.norm(matrix, ord=np.inf) + scalar_floor
        )
        candidate_error = _upward(input_error + addition_error)

        proposed_shift = max(
            scalar_floor / 2.0,
            32.0 * candidate_error,
        )
        floor = verified_cholesky_floor(
            candidate,
            input_error=candidate_error,
            proposed_shift=proposed_shift,
        )
        if floor == 0.0:
            # A cheap Cholesky screen avoids an eigenvalue call for matrices
            # that are visibly indefinite.
            try:
                np.linalg.cholesky(
                    candidate
                    + max(32.0 * candidate_error, 1e-15) * identity
                )
            except np.linalg.LinAlgError:
                pass
            else:
                estimate = float(np.linalg.eigvalsh(candidate)[0])
                if estimate > 4.0 * candidate_error:
                    floor = verified_cholesky_floor(
                        candidate,
                        input_error=candidate_error,
                        proposed_shift=estimate / 2.0,
                    )
        if floor > 0.0:
            floors[key] = floor
        else:
            residual_matrices[key] = matrix
            residual_errors[key] = input_error
            residual_magnitudes[key] = _upward(
                magnitude + input_error
            )
    return _CertificateState(
        residual_matrices=residual_matrices,
        residual_errors=residual_errors,
        residual_magnitudes=residual_magnitudes,
        floors=floors,
    )


def _cell_center_data(
    cell: DyadicCell,
    data: RootBernsteinData,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Return an enclosed center matrix and exact box-variation bound."""

    if max(cell.levels) > 50:
        raise ArithmeticError(
            "the fast dyadic-center enclosure requires levels at most 50"
        )
    centers = np.array(
        [
            (2 * index + 1) / 2 ** (level + 1)
            for level, index in zip(cell.levels, cell.indices)
        ],
        dtype=np.float64,
    )
    half_widths = np.array(
        [2.0 ** (-(level + 1)) for level in cell.levels],
        dtype=np.float64,
    )
    uppers = np.nextafter(
        np.array(
            [
                (index + 1) / 2**level
                for level, index in zip(cell.levels, cell.indices)
            ],
            dtype=np.float64,
        ),
        math.inf,
    )
    exponents = np.asarray(data.exponents, dtype=np.int8)
    weights = np.prod(
        np.power(centers[None, :], exponents),
        axis=1,
    )
    power_errors = np.array(
        [data.power_errors[exponent] for exponent in data.exponents]
    )
    power_magnitudes = np.array(
        [data.power_magnitudes[exponent] for exponent in data.exponents]
    )
    evaluation_magnitude = float(
        np.dot(weights, power_magnitudes)
    )
    evaluation_error = float(np.dot(weights, power_errors))
    evaluation_error = _upward(
        evaluation_error
        + (_gamma(6) + _gamma(3**6)) * evaluation_magnitude
    )
    matrix = (
        np.asarray(weights)
        @ data.power_matrices.reshape(3**6, _DIMENSION**2)
    ).reshape(_DIMENSION, _DIMENSION)

    coefficient_bounds = np.nextafter(
        np.array(
            [float(value) for value in data.coefficient_norm_bounds],
            dtype=np.float64,
        ),
        math.inf,
    )
    variation_contributions = np.zeros(6, dtype=np.float64)
    for coordinate in range(6):
        active = exponents[:, coordinate] > 0
        adjusted = exponents[active].copy()
        adjusted[:, coordinate] -= 1
        monomials = np.prod(
            np.power(uppers[None, :], adjusted),
            axis=1,
        )
        variation_contributions[coordinate] = (
            half_widths[coordinate]
            * float(
                np.dot(
                    coefficient_bounds[active]
                    * exponents[active, coordinate],
                    monomials,
                )
            )
        )
    variation = float(np.sum(variation_contributions))
    variation = _upward(
        variation
        / (1.0 - _gamma(6) - _gamma(3**6) - _gamma(6))
    )
    variation_contributions = np.nextafter(
        variation_contributions
        / (1.0 - _gamma(6) - _gamma(3**6)),
        math.inf,
    )
    return (
        matrix,
        evaluation_error,
        variation,
        variation_contributions,
    )


def _cell_center_weyl_certificate(
    cell: DyadicCell,
    data: RootBernsteinData,
) -> tuple[bool, float]:
    """Certify a complete cell from its center and a derivative norm."""

    matrix, input_error, exact_variation, _ = _cell_center_data(cell, data)
    estimate = float(np.linalg.eigvalsh(matrix)[0])
    if estimate <= 4.0 * input_error:
        return False, 0.0
    center_floor = verified_cholesky_floor(
        matrix,
        input_error=input_error,
        proposed_shift=estimate / 2.0,
    )
    variation = exact_variation
    margin = float(
        np.nextafter(center_floor - variation, -math.inf)
    )
    return margin > 0.0, max(0.0, margin)


def _cell_center_weyl_decision(
    cell: DyadicCell,
    data: RootBernsteinData,
) -> tuple[bool, float, int]:
    """Return the Weyl decision and the dominant variation coordinate."""

    (
        matrix,
        input_error,
        variation,
        contributions,
    ) = _cell_center_data(cell, data)
    estimate = float(np.linalg.eigvalsh(matrix)[0])
    margin = 0.0
    if estimate > 4.0 * input_error:
        center_floor = verified_cholesky_floor(
            matrix,
            input_error=input_error,
            proposed_shift=estimate / 2.0,
        )
        margin = float(
            np.nextafter(center_floor - variation, -math.inf)
        )
    coordinate = max(
        range(6),
        key=lambda index: (
            contributions[index],
            -((index - cell.depth) % 6),
        ),
    )
    return margin > 0.0, max(0.0, margin), coordinate


def _exact_bernstein_restriction(
    lower: Fraction,
    upper: Fraction,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    """Return the exact degree-two Bernstein restriction matrix."""

    if not Fraction(0) <= lower < upper <= Fraction(1):
        raise ValueError("require 0 <= lower < upper <= 1")
    width = upper - lower
    one_minus_lower = 1 - lower
    one_minus_upper = 1 - upper
    return (
        (
            one_minus_lower**2,
            2 * lower * one_minus_lower,
            lower**2,
        ),
        (
            one_minus_lower**2 - width * one_minus_lower,
            2 * lower * one_minus_lower + width * (1 - 2 * lower),
            lower**2 + width * lower,
        ),
        (
            one_minus_upper**2,
            2 * upper * one_minus_upper,
            upper**2,
        ),
    )


def _enclosed_axis_restriction(
    matrices: np.ndarray,
    errors: np.ndarray,
    magnitudes: np.ndarray,
    exact_transform: tuple[
        tuple[Fraction, Fraction, Fraction],
        ...,
    ],
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an exact positive Bernstein restriction through float enclosures."""

    approximate = np.array(
        [[float(value) for value in row] for row in exact_transform],
        dtype=np.float64,
    )
    upper_weights = np.nextafter(approximate, math.inf)
    weight_errors = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            difference = abs(
                exact_transform[row][column]
                - Fraction.from_float(approximate[row, column])
            )
            weight_errors[row, column] = _upward(float(difference))

    transformed_matrices = np.moveaxis(
        np.tensordot(approximate, matrices, axes=(1, axis)),
        0,
        axis,
    )
    propagated_errors = np.moveaxis(
        np.tensordot(upper_weights, errors, axes=(1, axis)),
        0,
        axis,
    )
    weight_error = np.moveaxis(
        np.tensordot(weight_errors, magnitudes, axes=(1, axis)),
        0,
        axis,
    )
    linear_magnitudes = np.moveaxis(
        np.tensordot(upper_weights, magnitudes, axes=(1, axis)),
        0,
        axis,
    )
    transformed_errors = np.nextafter(
        propagated_errors
        + weight_error
        + _gamma(3) * linear_magnitudes,
        math.inf,
    )
    transformed_magnitudes = np.nextafter(
        linear_magnitudes + transformed_errors,
        math.inf,
    )
    return (
        transformed_matrices,
        transformed_errors,
        transformed_magnitudes,
    )


def _materialize_cell_bernstein(
    cell: DyadicCell,
    data: RootBernsteinData,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize enclosed Bernstein coefficients on one dyadic cell."""

    matrices = data.bernstein_matrices
    errors = data.bernstein_errors
    magnitudes = data.bernstein_magnitudes
    for axis in range(6):
        if cell.levels[axis] == 0:
            continue
        transform = _exact_bernstein_restriction(
            cell.lower(axis),
            cell.upper(axis),
        )
        matrices, errors, magnitudes = _enclosed_axis_restriction(
            matrices,
            errors,
            magnitudes,
            transform,
            axis,
        )
    return matrices, errors


def _cell_bernstein_certificate(
    cell: DyadicCell,
    data: RootBernsteinData,
) -> tuple[bool, float]:
    """Certify a cell when all enclosed Bernstein matrices are positive."""

    matrices, errors = _materialize_cell_bernstein(cell, data)
    flattened = matrices.reshape(3**6, _DIMENSION, _DIMENSION)
    flattened_errors = errors.reshape(3**6)
    estimates = np.linalg.eigvalsh(flattened)[:, 0]
    if np.any(estimates <= 4.0 * flattened_errors):
        return False, 0.0
    minimum_floor = math.inf
    for matrix, input_error, estimate in zip(
        flattened,
        flattened_errors,
        estimates,
    ):
        floor = verified_cholesky_floor(
            matrix,
            input_error=float(input_error),
            proposed_shift=float(estimate) / 2.0,
        )
        if floor == 0.0:
            return False, 0.0
        minimum_floor = min(minimum_floor, floor)
    return True, minimum_floor


def _split_coordinate(cell: DyadicCell) -> int:
    """Choose the widest coordinate with a deterministic cyclic tie break."""

    return min(
        range(6),
        key=lambda coordinate: (
            cell.levels[coordinate],
            (coordinate - cell.depth) % 6,
        ),
    )


def branch_orbit_representatives() -> list[dict[str, object]]:
    """Return the 20 low/high branch orbits under site permutations."""

    return local_branch_orbit_representatives()


def audit_branch(
    axes: tuple[str, str, str, str, str, str],
    *,
    maximum_depth: int = 180,
    maximum_nodes: int = 500000,
) -> dict[str, object]:
    """Run the deterministic validated certificate for one branch."""

    if maximum_depth < 0 or maximum_nodes <= 0:
        raise ValueError("invalid audit limits")
    started = time.time()
    data = build_root_bernstein_data(axes)
    root_estimates = np.linalg.eigvalsh(
        data.bernstein_matrices.reshape(3**6, _DIMENSION, _DIMENSION)
    )[:, 0]
    root_stats = {
        "minimum_floating_estimate": float(np.min(root_estimates)),
        "maximum_input_error": float(np.max(data.bernstein_errors)),
        "positive_estimates": int(np.count_nonzero(root_estimates > 0.0)),
    }
    root = DyadicCell((0,) * 6, (0,) * 6)
    stack = [root]
    stats = {
        "nodes": 0,
        "bernstein_leaves": 0,
        "local_schur_handoffs": 0,
        "center_weyl_leaves": 0,
        "unresolved_cells": 0,
        "maximum_depth": 0,
    }
    minimum_bernstein_margin = math.inf
    minimum_center_margin = math.inf
    while stack:
        cell = stack.pop()
        stats["nodes"] += 1
        stats["maximum_depth"] = max(
            stats["maximum_depth"],
            cell.depth,
        )
        if cell_is_inside_exact_local_region(cell, axes):
            stats["local_schur_handoffs"] += 1
            continue
        _, _, coordinate = _cell_center_weyl_decision(
            cell,
            data,
        )
        bernstein_certified, bernstein_margin = (
            _cell_bernstein_certificate(cell, data)
        )
        if bernstein_certified:
            stats["bernstein_leaves"] += 1
            minimum_bernstein_margin = min(
                minimum_bernstein_margin,
                bernstein_margin,
            )
            continue
        if (
            cell.depth >= maximum_depth
            or stats["nodes"] >= maximum_nodes
        ):
            stats["unresolved_cells"] += len(stack) + 1
            break

        left_cell, right_cell = cell.split(coordinate)
        stack.append(right_cell)
        stack.append(left_cell)

    return {
        "axes": "".join(axes),
        "status": (
            "passed"
            if stats["unresolved_cells"] == 0
            else "unresolved"
        ),
        "arithmetic": (
            "exact Q(sqrt(2)) coefficients; outward norm errors; "
            "Cholesky residual verification"
        ),
        "local_l1_radius": str(_LOCAL_L1_RADIUS),
        "minimum_center_weyl_margin": (
            minimum_center_margin
            if math.isfinite(minimum_center_margin)
            else None
        ),
        "minimum_bernstein_margin": (
            minimum_bernstein_margin
            if math.isfinite(minimum_bernstein_margin)
            else None
        ),
        "elapsed_seconds": time.time() - started,
        "root": root_stats,
        **stats,
    }


def aggregate_orbit_records(
    directory: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Validate and aggregate the 20 independently generated orbit records."""

    import hashlib

    source_directory = Path(directory)
    records = []
    digests = {}
    for orbit_index in range(20):
        path = source_directory / f"orbit_{orbit_index}.json"
        payload = path.read_bytes()
        record = json.loads(payload)
        if record.get("orbit_index") != orbit_index:
            raise ArithmeticError("orbit record index mismatch")
        if record.get("status") != "passed":
            raise ArithmeticError(f"orbit {orbit_index} did not pass")
        if record.get("unresolved_cells") != 0:
            raise ArithmeticError(f"orbit {orbit_index} has unresolved cells")
        records.append(record)
        digests[path.name] = hashlib.sha256(payload).hexdigest()

    aggregate = {
        "status": "passed_global_interval_certificate",
        "candidate_slope": "(4 + 5*sqrt(2))/16",
        "eta": "1/2",
        "branch_orbits": len(records),
        "physical_branches_covered_by_symmetry": sum(
            int(record["orbit_size"]) for record in records
        ),
        "nodes": sum(int(record["nodes"]) for record in records),
        "bernstein_leaves": sum(
            int(record["bernstein_leaves"]) for record in records
        ),
        "local_schur_handoffs": sum(
            int(record["local_schur_handoffs"]) for record in records
        ),
        "center_weyl_leaves": sum(
            int(record["center_weyl_leaves"]) for record in records
        ),
        "unresolved_cells": 0,
        "maximum_depth": max(
            int(record["maximum_depth"]) for record in records
        ),
        "minimum_bernstein_margin": min(
            float(record["minimum_bernstein_margin"])
            for record in records
            if record["minimum_bernstein_margin"] is not None
        ),
        "minimum_center_weyl_margin": min(
            (
                float(record["minimum_center_weyl_margin"])
                for record in records
                if record["minimum_center_weyl_margin"] is not None
            ),
            default=None,
        ),
        "arithmetic": (
            "exact Q(sqrt(2)) source coefficients; dyadic Bernstein "
            "restriction with outward operator-norm errors; Cholesky "
            "residual verification; exact local Schur handoffs"
        ),
        "orbit_sha256": digests,
        "orbits": records,
    }
    if aggregate["physical_branches_covered_by_symmetry"] != 64:
        raise ArithmeticError("branch-orbit multiplicities did not sum to 64")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def main() -> None:
    """Run one validated branch-orbit certificate."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit-index", type=int, default=0)
    parser.add_argument("--maximum-depth", type=int, default=180)
    parser.add_argument("--maximum-nodes", type=int, default=500000)
    arguments = parser.parse_args()
    representatives = branch_orbit_representatives()
    if not 0 <= arguments.orbit_index < len(representatives):
        raise ValueError("orbit-index must lie between 0 and 19")
    representative = representatives[arguments.orbit_index]
    result = audit_branch(
        representative["axes"],
        maximum_depth=arguments.maximum_depth,
        maximum_nodes=arguments.maximum_nodes,
    )
    result["orbit_index"] = arguments.orbit_index
    result["site_types"] = representative["site_types"]
    result["orbit_size"] = representative["orbit_size"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
