"""Regression tests for the full-projector CCZ STOPI candidate."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ccz_stopi_optimal import (  # noqa: E402
    bell_operator,
    arc_barycentric_weights,
    arc_outer_triangle,
    canonical_dual_weights,
    corner_audit,
    corrected_coupling_gap,
    corrected_coupling_gap_coordinates,
    effective_chsh_gap_sum,
    extracted_target_projector,
    gap_operator,
    ideal_schur_hessian_audit,
    mismatched_channel_counterexample,
    one_bad_pair_audit,
    one_bad_pair_schur_hessian_audit,
    required_slope,
    c_ccz,
    target_projector,
)


def test_ccz_coefficient_has_the_exact_closed_form() -> None:
    """The CCZ affine coefficient has the claimed exact value."""

    assert abs(c_ccz() - (4.0 + 5.0 * math.sqrt(2.0)) / 16.0) < 1e-15


def test_ideal_point_aligns_projector_extraction_and_bell_score() -> None:
    """All coordinate conventions coincide at the six-party ideal point."""

    angles = (math.pi / 4.0,) * 3
    projector = target_projector()
    extracted = extracted_target_projector(angles, angles)
    bell = bell_operator(angles, angles)
    target = np.linalg.eigh(projector)[1][:, -1]

    assert abs(np.trace(projector).real - 1.0) < 1e-12
    assert np.linalg.norm(projector @ projector - projector) < 1e-12
    assert np.linalg.norm(extracted - projector) < 2e-12
    assert abs(np.vdot(target, bell @ target).real - 6.0 * math.sqrt(2.0)) < 2e-12
    assert np.linalg.norm(gap_operator(c_ccz(), angles, angles) @ target) < 2e-12


def test_channel_branch_is_covariant_with_the_v2_axis_convention() -> None:
    """The canonical ``Z``/``X`` swap is applied to Kaniewski's map."""

    low = canonical_dual_weights(0.0)
    high = canonical_dual_weights(math.pi / 2.0)

    assert low == {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 1.0}
    assert high == {"I": 1.0, "X": 1.0, "Y": 0.0, "Z": 0.0}


def test_old_refutation_is_specific_to_the_axis_mismatched_channel() -> None:
    """The archived negative witness diagnoses the convention error."""

    audit = mismatched_channel_counterexample(c_ccz())
    expected = -(2.0 + math.sqrt(2.0)) / 8.0

    assert abs(audit["rayleigh_quotient"] - expected) < 2e-12
    assert audit["minimum_eigenvalue"] < -0.42


def test_corrected_asymmetric_corner_is_psd_at_the_ccz_coefficient() -> None:
    """The formerly refuting corner has a strict positive margin."""

    alice_angles = (math.pi / 4.0, 0.0, math.pi / 4.0)
    bob_angles = (math.pi / 4.0,) * 3
    audit = corner_audit(c_ccz(), alice_angles, bob_angles)

    assert abs(
        audit["minimum_eigenvalue"] - (2.0 - math.sqrt(2.0)) / 8.0
    ) < 2e-12


def test_corrected_corner_reaches_the_universal_endpoint_floor() -> None:
    """Its fixed-channel threshold is ``(1+sqrt(2))/4``."""

    alice_angles = (math.pi / 4.0, 0.0, math.pi / 4.0)
    bob_angles = (math.pi / 4.0,) * 3
    threshold = required_slope(alice_angles, bob_angles)

    assert abs(threshold - (1.0 + math.sqrt(2.0)) / 4.0) < 2e-10


def test_one_bad_pair_boundary_has_the_exact_chsh_zero_mode() -> None:
    """The tight orbit splits into matched and mismatched ``Z`` sectors."""

    audit = one_bad_pair_audit(site=0)
    matched = audit["matched_sector"]
    mismatched = audit["mismatched_sector"]

    assert audit["commutator_norms"]["A_bad_axis"] < 1e-13
    assert audit["commutator_norms"]["B_bad_axis"] < 1e-13
    assert matched["nullity"] == 1
    assert matched["projector_alignment_error"] < 2e-12
    assert abs(
        matched["extracted_weight"] - (2.0 + math.sqrt(2.0)) / 8.0
    ) < 2e-12
    assert abs(matched["minimum_eigenvalue"]) < 2e-12
    assert mismatched["nullity"] == 0
    assert abs(
        mismatched["unextracted_floor"]
        - (3.0 / 4.0 + 9.0 * math.sqrt(2.0) / 8.0)
    ) < 2e-12


def test_second_one_bad_pair_endpoint_has_the_same_zero_mode() -> None:
    """The Bob ``pi/2`` endpoint is the axis-swapped tight orbit."""

    audit = one_bad_pair_audit(site=0, bob_endpoint=math.pi / 2.0)

    assert audit["commutator_norms"]["A_bad_axis"] < 1e-13
    assert audit["commutator_norms"]["B_bad_axis"] < 1e-13
    assert audit["matched_sector"]["nullity"] == 1
    assert abs(audit["matched_sector"]["minimum_eigenvalue"]) < 2e-12
    assert audit["mismatched_sector"]["minimum_eigenvalue"] > 2.3


def test_all_equality_orbits_have_positive_effective_hessians() -> None:
    """Numerical Schur diagnostics isolate all seven observed zero points."""

    ideal = ideal_schur_hessian_audit()
    endpoint_zero = one_bad_pair_schur_hessian_audit(
        site=0,
        bob_endpoint=0.0,
    )
    endpoint_half_pi = one_bad_pair_schur_hessian_audit(
        site=0,
        bob_endpoint=math.pi / 2.0,
    )

    assert ideal["minimum_effective_hessian_eigenvalue"] > 0.249
    assert endpoint_zero["minimum_block_hessian_eigenvalue"] > 0.47
    assert endpoint_half_pi["minimum_block_hessian_eigenvalue"] > 0.47
    assert ideal["evidence"] == "analytic-derivative local Schur audit"
    assert endpoint_zero["evidence"] == "analytic-derivative local Schur audit"
    assert (
        0.476
        < endpoint_zero["minimum_block_hessian_eigenvalue"]
        < 0.478
    )


def test_effective_chsh_decomposition_is_exact_and_psd_at_key_points() -> None:
    """Controlled CHSH sectors isolate a smaller coupling remainder."""

    eta = 1.0 / 12.0
    points = [
        ((math.pi / 4.0,) * 3, (math.pi / 4.0,) * 3),
        (
            (0.0, math.pi / 4.0, math.pi / 4.0),
            (math.pi / 2.0, math.pi / 4.0, math.pi / 4.0),
        ),
        (
            (math.pi / 4.0, math.pi / 4.0, math.pi / 2.0),
            (math.pi / 4.0, math.pi / 4.0, 0.0),
        ),
    ]
    for alice_angles, bob_angles in points:
        full = gap_operator(c_ccz(), alice_angles, bob_angles)
        chsh_sum = effective_chsh_gap_sum(alice_angles, bob_angles)
        coupling = corrected_coupling_gap(
            alice_angles,
            bob_angles,
            eta=eta,
        )

        assert np.linalg.eigvalsh(chsh_sum)[0] > -2e-12
        assert np.linalg.eigvalsh(coupling)[0] > -2e-12
        assert np.linalg.norm(
            full - ((1.0 - eta) * chsh_sum + coupling)
        ) < 2e-12


def test_outer_triangle_coordinates_preserve_the_multiaffine_gap() -> None:
    """Arc points are convex combinations of proof-vertex matrices."""

    angles = np.array([0.31, 0.58, 0.74, 0.22, 0.49, 0.69])
    coordinates = [
        (math.cos(float(angle)), math.sin(float(angle)))
        for angle in angles
    ]
    axes = ("Z",) * 6
    physical = corrected_coupling_gap(
        angles[:3],
        angles[3:],
        eta=0.5,
    )
    coordinate_value = corrected_coupling_gap_coordinates(
        coordinates,
        axes,
        eta=0.5,
    )

    lower = 0.2
    upper = 0.4
    triangle = arc_outer_triangle(lower, upper)
    weights = arc_barycentric_weights(angles[0], lower, upper)
    interpolated = np.zeros((64, 64), dtype=np.complex128)
    for weight, vertex in zip(weights, triangle):
        vertex_coordinates = list(coordinates)
        vertex_coordinates[0] = vertex
        interpolated += weight * corrected_coupling_gap_coordinates(
            vertex_coordinates,
            axes,
            eta=0.5,
        )

    assert min(weights) >= -1e-14
    assert abs(sum(weights) - 1.0) < 1e-14
    assert np.linalg.norm(physical - coordinate_value) < 3e-12
    assert np.linalg.norm(physical - interpolated) < 3e-12
