#!/usr/bin/env python3
r"""Endpoint Taylor--Pólya audit for the CCZ STOPI coupling gap.

At a one-bad-pair equality point, write the corrected coupling gap as

.. math::
   C(z)=\frac{N(z)}{D(z)},\qquad
   D(z)=\prod_{i=1}^6(1+z_i^2),

where :math:`z_i=\tan(\delta_i/2)` are inward half-angle deviations.
The numerator :math:`N` has degree at most two in each variable.  Relative
to the equality kernel :math:`P` and its complement, this module audits the
fixed-reference Schur sufficient condition

.. math::
   DPNP-\kappa PNR_0NP\succeq0,\qquad
   R_0=(QC(0)Q)^{-1},\qquad \kappa=\frac{11}{10}.

The homogeneous quadratic part is bounded on the nonnegative simplex by
degree-elevated Pólya--Bernstein coefficients.  Higher orders and complement
motion are bounded by homogeneous Bernstein coefficient norms.

All matrix construction and eigenvalue decisions in this module currently
use ``complex128``.  The generated record is therefore a reproducible
certificate-design audit, not yet a publication-grade interval certificate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ccz_stopi_optimal import corrected_coupling_gap_coordinates

_DEFAULT_RECORD = (
    Path(__file__).resolve().parents[3]
    / ".research"
    / "plans"
    / "ccz-stopi-optimal-proof"
    / "endpoint_local_taylor_polya_eta_half.json"
)
_INTERPOLATION_NODES = np.array((-1.0, 0.0, 1.0))
_INVERSE_VANDERMONDE = np.linalg.inv(
    np.vander(_INTERPOLATION_NODES, 3, increasing=True)
)


def _compositions(
    total: int,
    variables: int = 6,
    prefix: tuple[int, ...] = (),
) -> Iterator[tuple[int, ...]]:
    """Yield weak compositions of ``total`` into ``variables`` parts."""

    if variables == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from _compositions(
            total - value,
            variables - 1,
            prefix + (value,),
        )


def _multinomial(exponents: Sequence[int]) -> int:
    """Return the exact multinomial coefficient for ``exponents``."""

    output = math.factorial(sum(exponents))
    for exponent in exponents:
        output //= math.factorial(exponent)
    return output


def endpoint_half_angle_coordinates(
    deviations: Sequence[float],
    axes: Sequence[str],
    *,
    bad_site: int = 0,
    bob_endpoint: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    r"""Return exact rational trigonometric coordinates for one endpoint box.

    For :math:`z=\tan(\delta/2)`,

    .. math::
       (\cos\delta,\sin\delta)
       =\frac{(1-z^2,2z)}{1+z^2}.

    Good coordinates use :math:`\pi/4-\delta` on the low ``Z`` branch and
    :math:`\pi/4+\delta` on the high ``X`` branch.  The bad Alice angle is
    zero; the bad Bob angle may be zero or :math:`\pi/2`.
    """

    if len(deviations) != 6 or len(axes) != 6:
        raise ValueError("six deviations and six branch axes are required")
    if bad_site not in (0, 1, 2):
        raise ValueError("bad_site must be 0, 1, or 2")
    if not (
        math.isclose(bob_endpoint, 0.0, abs_tol=1e-15)
        or math.isclose(
            bob_endpoint,
            math.pi / 2.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("bob_endpoint must be 0 or pi/2")
    if any(axis not in ("X", "Z") for axis in axes):
        raise ValueError("axes must be X or Z")
    parsed_deviations = tuple(float(value) for value in deviations)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in parsed_deviations
    ):
        raise ValueError(
            "half-angle deviations must be finite and nonnegative"
        )

    bad_coordinates = (bad_site, 3 + bad_site)
    if axes[bad_site] != "Z":
        raise ValueError("the bad Alice endpoint must use the Z branch")
    expected_bob_axis = "Z" if bob_endpoint == 0.0 else "X"
    if axes[3 + bad_site] != expected_bob_axis:
        raise ValueError("the bad Bob axis does not match its endpoint")

    return _endpoint_half_angle_coordinates_signed(
        parsed_deviations,
        axes,
        bad_site=bad_site,
        bob_endpoint=bob_endpoint,
    )


def _endpoint_half_angle_coordinates_signed(
    deviations: Sequence[float],
    axes: Sequence[str],
    *,
    bad_site: int,
    bob_endpoint: float,
) -> tuple[tuple[float, float], ...]:
    """Evaluate the rational coordinate formula for signed interpolation nodes."""

    root_two_inverse = 1.0 / math.sqrt(2.0)
    bad_coordinates = (bad_site, 3 + bad_site)
    output = []
    for coordinate, (deviation, axis) in enumerate(
        zip(deviations, axes)
    ):
        z = float(deviation)
        denominator = 1.0 + z * z
        endpoint_cosine = (1.0 - z * z) / denominator
        endpoint_sine = 2.0 * z / denominator
        if coordinate in bad_coordinates:
            if coordinate == 3 + bad_site and bob_endpoint != 0.0:
                output.append((endpoint_sine, endpoint_cosine))
            else:
                output.append((endpoint_cosine, endpoint_sine))
            continue
        sign = 1.0 if axis == "Z" else -1.0
        output.append(
            (
                root_two_inverse
                * (1.0 - z * z + sign * 2.0 * z)
                / denominator,
                root_two_inverse
                * (1.0 - z * z - sign * 2.0 * z)
                / denominator,
            )
        )
    return tuple(output)


def _cleared_gap_coefficients(
    axes: Sequence[str],
    *,
    bad_site: int,
    bob_endpoint: float,
    eta: float,
) -> np.ndarray:
    """Interpolate the degree-two-per-variable numerator ``N``."""

    values = np.empty((3,) * 6 + (64, 64), dtype=np.complex128)
    for node_indices in itertools.product(range(3), repeat=6):
        deviations = _INTERPOLATION_NODES[list(node_indices)]
        coordinates = _endpoint_half_angle_coordinates_signed(
            deviations,
            axes,
            bad_site=bad_site,
            bob_endpoint=bob_endpoint,
        )
        denominator = float(np.prod(1.0 + deviations * deviations))
        values[node_indices] = (
            corrected_coupling_gap_coordinates(
                coordinates,
                axes,
                eta=eta,
            )
            * denominator
        )

    coefficients = values
    for coordinate in range(6):
        coefficients = np.moveaxis(
            np.tensordot(
                _INVERSE_VANDERMONDE,
                coefficients,
                axes=(1, coordinate),
            ),
            0,
            coordinate,
        )
    return coefficients


def _polya_quadratic_margin(
    quadratic: dict[tuple[int, ...], np.ndarray],
    power: int,
) -> float:
    """Return the least degree-elevated simplex Bernstein eigenvalue."""

    degree = power + 2
    denominator = float(degree * (degree - 1))
    diagonal = []
    mixed = []
    for left in range(6):
        exponent = [0] * 6
        exponent[left] = 2
        diagonal.append(quadratic[tuple(exponent)])
        for right in range(left + 1, 6):
            exponent = [0] * 6
            exponent[left] = 1
            exponent[right] = 1
            mixed.append((left, right, quadratic[tuple(exponent)]))

    minimum = math.inf
    batch: list[np.ndarray] = []
    for beta in _compositions(degree):
        coefficient = np.zeros((2, 2), dtype=np.complex128)
        for coordinate, matrix in enumerate(diagonal):
            coefficient += (
                beta[coordinate]
                * (beta[coordinate] - 1)
                / denominator
                * matrix
            )
        for left, right, matrix in mixed:
            coefficient += (
                beta[left] * beta[right] / denominator * matrix
            )
        batch.append(coefficient)
        if len(batch) == 8192:
            matrices = np.asarray(batch)
            minimum = min(
                minimum,
                float(np.linalg.eigvalsh(matrices)[:, 0].min()),
            )
            batch.clear()
    if batch:
        matrices = np.asarray(batch)
        minimum = min(
            minimum,
            float(np.linalg.eigvalsh(matrices)[:, 0].min()),
        )
    return minimum


def _homogeneous_bernstein_norm_bounds(
    coefficients: np.ndarray,
    *,
    minimum_degree: int,
    maximum_degree: int,
    maximum_coordinate_degree: int,
) -> dict[int, float]:
    """Bound each homogeneous component on the standard simplex."""

    bounds: dict[int, float] = {}
    for degree in range(minimum_degree, maximum_degree + 1):
        maximum = 0.0
        for exponent in _compositions(degree):
            if max(exponent) > maximum_coordinate_degree:
                continue
            coefficient = coefficients[exponent] / _multinomial(exponent)
            maximum = max(
                maximum,
                float(np.linalg.norm(coefficient, ord=2)),
            )
        if maximum:
            bounds[degree] = maximum
    return bounds


def endpoint_local_taylor_audit(
    axes: Sequence[str],
    *,
    bad_site: int = 0,
    bob_endpoint: float = 0.0,
    eta: float = 0.5,
    half_angle_radius: float = 1.0 / 200.0,
    schur_factor: float = 11.0 / 10.0,
    polya_power: int = 30,
) -> dict[str, object]:
    """Audit one adjacent endpoint branch by Taylor--Pólya bounds."""

    if half_angle_radius <= 0.0:
        raise ValueError("half_angle_radius must be positive")
    if schur_factor <= 1.0:
        raise ValueError("schur_factor must exceed one")
    if polya_power < 0:
        raise ValueError("polya_power must be nonnegative")

    numerator = _cleared_gap_coefficients(
        axes,
        bad_site=bad_site,
        bob_endpoint=bob_endpoint,
        eta=eta,
    )
    equality = numerator[(0,) * 6]
    eigenvalues, eigenvectors = np.linalg.eigh(equality)
    kernel = eigenvectors[:, :2]
    complement = eigenvectors[:, 2:]
    complement_eigenvalues = eigenvalues[2:]
    inverse_complement = np.diag(1.0 / complement_eigenvalues)
    inverse_sqrt_complement = np.diag(
        1.0 / np.sqrt(complement_eigenvalues)
    )

    kernel_block = np.einsum(
        "ai,...ab,bj->...ij",
        kernel.conj(),
        numerator,
        kernel,
        optimize=True,
    )
    coupling_block = np.einsum(
        "ai,...ab,bj->...ij",
        kernel.conj(),
        numerator,
        complement,
        optimize=True,
    )
    coupling_times_inverse = np.einsum(
        "...ij,jk->...ik",
        coupling_block,
        inverse_complement,
        optimize=True,
    )

    schur_polynomial = np.zeros(
        (5,) * 6 + (2, 2),
        dtype=np.complex128,
    )
    for denominator_exponents in itertools.product((0, 1), repeat=6):
        slices = tuple(
            slice(2 * exponent, 2 * exponent + 3)
            for exponent in denominator_exponents
        )
        schur_polynomial[slices] += kernel_block
    degree_two_exponents = list(itertools.product(range(3), repeat=6))
    for left_exponents in degree_two_exponents:
        left = coupling_times_inverse[left_exponents]
        for right_exponents in degree_two_exponents:
            output_exponents = tuple(
                left_exponents[index] + right_exponents[index]
                for index in range(6)
            )
            schur_polynomial[output_exponents] -= (
                schur_factor
                * left
                @ coupling_block[right_exponents].conj().T
            )
    schur_polynomial = (
        schur_polynomial
        + schur_polynomial.swapaxes(-1, -2).conj()
    ) / 2.0

    quadratic = {
        exponent: schur_polynomial[exponent]
        for exponent in _compositions(2)
    }
    quadratic_margin = _polya_quadratic_margin(
        quadratic,
        polya_power,
    )
    remainder_norms = _homogeneous_bernstein_norm_bounds(
        schur_polynomial,
        minimum_degree=3,
        maximum_degree=24,
        maximum_coordinate_degree=4,
    )
    simplex_radius = 6.0 * half_angle_radius
    remainder_bound = sum(
        bound * simplex_radius ** (degree - 2)
        for degree, bound in remainder_norms.items()
    )

    complement_block = np.einsum(
        "ai,...ab,bj->...ij",
        complement.conj(),
        numerator,
        complement,
        optimize=True,
    )
    normalized_complement = np.einsum(
        "ij,...jk,kl->...il",
        inverse_sqrt_complement,
        complement_block,
        inverse_sqrt_complement,
        optimize=True,
    )
    identity = np.eye(62, dtype=np.complex128)
    for denominator_exponents in itertools.product((0, 1), repeat=6):
        exponent = tuple(
            2 * value for value in denominator_exponents
        )
        normalized_complement[exponent] -= identity
    complement_norms = _homogeneous_bernstein_norm_bounds(
        normalized_complement,
        minimum_degree=1,
        maximum_degree=12,
        maximum_coordinate_degree=2,
    )
    complement_relative_bound = sum(
        bound * simplex_radius**degree
        for degree, bound in complement_norms.items()
    )
    complement_allowance = 1.0 - 1.0 / schur_factor

    return {
        "axes": "".join(axes),
        "bad_site": bad_site,
        "bob_endpoint": bob_endpoint,
        "eta": eta,
        "half_angle_radius": half_angle_radius,
        "maximum_angle_radius": 2.0 * math.atan(half_angle_radius),
        "schur_factor": schur_factor,
        "polya_power": polya_power,
        "quadratic_margin": quadratic_margin,
        "remainder_bound": remainder_bound,
        "schur_margin": quadratic_margin - remainder_bound,
        "complement_relative_bound": complement_relative_bound,
        "complement_allowance": complement_allowance,
        "minimum_positive_equality_eigenvalue": float(
            complement_eigenvalues[0]
        ),
        "passed": bool(
            quadratic_margin > remainder_bound
            and complement_relative_bound < complement_allowance
        ),
        "arithmetic": "complex128 discovery audit",
    }


def summarize_endpoint_local_bounds(
    record_path: Path | str = _DEFAULT_RECORD,
) -> dict[str, object]:
    """Return the branch-independent summary from a generated record."""

    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    branches = record["records"]
    return {
        "half_angle_radius": record["half_angle_radius"],
        "schur_factor": record["schur_factor"],
        "polya_power": record["polya_power"],
        "covered_branch_patterns": record["covered_branch_patterns"],
        "minimum_quadratic_margin": min(
            branch["quadratic_margin"] for branch in branches
        ),
        "maximum_remainder_bound": max(
            branch["remainder_bound"] for branch in branches
        ),
        "maximum_complement_relative_bound": max(
            branch["complement_relative_bound"] for branch in branches
        ),
        "all_passed": all(branch["passed"] for branch in branches),
        "arithmetic": record["arithmetic"],
    }


def generate_endpoint_record(
    output_path: Path | str = _DEFAULT_RECORD,
) -> dict[str, object]:
    """Generate all 16 adjacent good-angle branch audits."""

    representatives: dict[tuple[str, str], dict[str, object]] = {}
    for alice_good_axes in itertools.product(("Z", "X"), repeat=2):
        axes = ["Z"] * 6
        for coordinate, axis in zip((1, 2), alice_good_axes):
            axes[coordinate] = axis
        representatives[alice_good_axes] = endpoint_local_taylor_audit(
            tuple(axes)
        )

    records = []
    for good_axis_bits in itertools.product(("Z", "X"), repeat=4):
        alice_good_axes = good_axis_bits[:2]
        representative = dict(representatives[alice_good_axes])
        axes = ["Z"] * 6
        for coordinate, axis in zip((1, 2, 4, 5), good_axis_bits):
            axes[coordinate] = axis
        representative["axes"] = "".join(axes)
        representative["symmetry_representative"] = (
            representatives[alice_good_axes]["axes"]
        )
        records.append(representative)

    record = {
        "status": "passed_floating_certificate_design",
        "candidate_slope": "(4 + 5*sqrt(2))/16",
        "eta": 0.5,
        "half_angle_radius": 1.0 / 200.0,
        "maximum_angle_radius": 2.0 * math.atan(1.0 / 200.0),
        "schur_factor": 11.0 / 10.0,
        "polya_power": 30,
        "covered_branch_patterns": 16,
        "endpoint_symmetry": (
            "The pi/2 Bob endpoint and the other bad sites are related by "
            "the canonical axis swap and simultaneous site permutations. "
            "At the equality endpoint, changing either good Bob branch is "
            "a local axis/outcome conjugation, so four Alice representatives "
            "cover all 16 adjacent patterns."
        ),
        "arithmetic": "complex128 discovery audit; interval lift pending",
        "records": records,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    """Generate the default endpoint record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_RECORD,
    )
    arguments = parser.parse_args()
    record = generate_endpoint_record(arguments.output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
