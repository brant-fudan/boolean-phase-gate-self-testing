#!/usr/bin/env python3
r"""Full-projector CCZ STOPI operator under product local extraction.

The proof target is

.. math::
   T_s(\mathbf a,\mathbf b)
   =K_{\rm CCZ}(\mathbf a,\mathbf b)-I
    +s\left(6\sqrt2 I-\mathcal B_{\rm CCZ}(\mathbf a,\mathbf b)\right)
   \succeq0,

where :math:`K_{\rm CCZ}=\Lambda^\dagger(P_{\rm CCZ})` and every local
channel is the Kaniewski dephasing channel associated with one Jordan block.
This module distinguishes exact fixed-point refutations from floating-point
global diagnostics.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.diag([1.0, -1.0]).astype(np.complex128)
_I64 = np.eye(64, dtype=np.complex128)
_SQRT_TWO = math.sqrt(2.0)


def _kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    """Return the Kronecker product in ``A1 A2 A3 B1 B2 B3`` order."""

    output = np.array([[1.0]], dtype=np.complex128)
    for operator in operators:
        output = np.kron(output, operator)
    return output


def _embed(operator: np.ndarray, party: int) -> np.ndarray:
    """Embed one-qubit operator at a zero-based party index."""

    operators = [_I2] * 6
    operators[party] = operator
    return _kron_all(operators)


def _validate_angles(angles: Sequence[float]) -> tuple[float, float, float]:
    """Validate three canonical Jordan angles in ``[0, pi/2]``."""

    if len(angles) != 3:
        raise ValueError("exactly three Jordan angles are required")
    output = tuple(float(angle) for angle in angles)
    if any(angle < 0.0 or angle > math.pi / 2.0 for angle in output):
        raise ValueError("Jordan angles must lie in [0, pi/2]")
    return output  # type: ignore[return-value]


def c_ccz() -> float:
    r"""Return \(c_{\rm CCZ}=(4+5\sqrt2)/16\)."""

    return (4.0 + 5.0 * _SQRT_TWO) / 16.0


def _rotation_y(angle: float) -> np.ndarray:
    r"""Return \(R_y(\theta)=\exp(-i\theta Y/2)\)."""

    return (
        math.cos(angle / 2.0) * _I2
        - 1.0j * math.sin(angle / 2.0) * _Y
    )


@lru_cache(maxsize=1)
def target_projector() -> np.ndarray:
    r"""Return the canonical rotated CCZ Choi projector.

    With

    .. math::
       |\Psi_{\rm CCZ}\rangle
       =2^{-3/2}\sum_x(-1)^{x_1x_2x_3}|x\rangle_A|x\rangle_B

    and :math:`W_B=Z R_y(-\pi/4)`, the Jordan-frame target is

    .. math::
       |\widetilde\Psi_{\rm CCZ}\rangle
       =(I_A\otimes W_B^{\otimes3})|\Psi_{\rm CCZ}\rangle.
    """

    state = np.zeros(64, dtype=np.complex128)
    for value in range(8):
        bits = (
            (value >> 2) & 1,
            (value >> 1) & 1,
            value & 1,
        )
        phase = -1.0 if bits[0] * bits[1] * bits[2] else 1.0
        state[(value << 3) | value] = phase / math.sqrt(8.0)

    bob_rotation = _Z @ _rotation_y(-math.pi / 4.0)
    global_rotation = _kron_all(
        (_I2, _I2, _I2, bob_rotation, bob_rotation, bob_rotation)
    )
    rotated_state = global_rotation @ state
    return np.outer(rotated_state, rotated_state.conj())


def _kaniewski_visibility(angle: float) -> float:
    r"""Return \(g(\theta)=(1+\sqrt2)(\cos\theta+\sin\theta-1)\)."""

    visibility = (1.0 + _SQRT_TWO) * (
        math.cos(angle) + math.sin(angle) - 1.0
    )
    if visibility < -1e-14 or visibility > 1.0 + 1e-14:
        raise ValueError("Kaniewski visibility left [0, 1]")
    return min(1.0, max(0.0, visibility))


def _apply_local_dual(
    operator: np.ndarray,
    angle: float,
    party: int,
    axis_rule: str,
) -> np.ndarray:
    r"""Apply the local Kaniewski dephasing dual.

    .. math::
       \Lambda_\theta^\dagger(O)
       =\frac{1+g(\theta)}2O
        +\frac{1-g(\theta)}2\Gamma(\theta)O\Gamma(\theta),

    In Kaniewski's coordinates the low-angle axis is ``X``, but his
    observables are :math:`\cos\theta X\pm\sin\theta Z`.  The manuscript
    coordinates exchange ``X`` and ``Z``.  Hence ``covariant`` preserves
    ``Z`` at low angles and ``X`` at high angles.  ``literal`` retains the
    untransformed branch only to replay the archived convention-error
    counterexample.
    """

    visibility = _kaniewski_visibility(angle)
    if axis_rule == "covariant":
        preserved_axis = _Z if angle <= math.pi / 4.0 else _X
    elif axis_rule == "literal":
        preserved_axis = _X if angle <= math.pi / 4.0 else _Z
    else:
        raise ValueError(f"unknown axis rule: {axis_rule}")
    embedded_axis = _embed(preserved_axis, party)
    identity_weight = (1.0 + visibility) / 2.0
    conjugation_weight = (1.0 - visibility) / 2.0
    return (
        identity_weight * operator
        + conjugation_weight
        * embedded_axis
        @ operator
        @ embedded_axis
    )


def canonical_dual_weights(angle: float) -> dict[str, float]:
    r"""Return the Pauli weights of the covariant local dual.

    In the manuscript convention

    .. math::
       M_{0,1}=\cos\theta Z\pm\sin\theta X,

    the correct Kaniewski channel preserves ``Z`` below :math:`\pi/4` and
    ``X`` above :math:`\pi/4`; the two definitions coincide at the ideal
    point because :math:`g(\pi/4)=1`.
    """

    if angle < 0.0 or angle > math.pi / 2.0:
        raise ValueError("Jordan angle must lie in [0, pi/2]")
    visibility = _kaniewski_visibility(angle)
    if angle <= math.pi / 4.0:
        return {
            "I": 1.0,
            "X": visibility,
            "Y": visibility,
            "Z": 1.0,
        }
    return {
        "I": 1.0,
        "X": 1.0,
        "Y": visibility,
        "Z": visibility,
    }


def extracted_target_projector(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
    axis_rule: str = "covariant",
) -> np.ndarray:
    r"""Return \(K_{\rm CCZ}=\Lambda^\dagger(P_{\rm CCZ})\)."""

    alice = _validate_angles(alice_angles)
    bob = _validate_angles(bob_angles)
    output = target_projector().copy()
    for party, angle in enumerate(alice + bob):
        output = _apply_local_dual(output, angle, party, axis_rule)
    return (output + output.conj().T) / 2.0


def _canonical_observables(angle: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return \(M_{0,1}=\cos\theta Z\pm\sin\theta X\)."""

    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine * _Z + sine * _X, cosine * _Z - sine * _X


def _bell_terms_from_observables(
    alice: Sequence[tuple[np.ndarray, np.ndarray]],
    bob: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """Return the three lifted CHSH terms."""

    terms = []
    for site in range(3):
        other_sites = [index for index in range(3) if index != site]
        remote_zero = [
            _embed(bob[index][0], 3 + index) for index in other_sites
        ]
        derivative = (
            _I64
            + remote_zero[0]
            + remote_zero[1]
            - remote_zero[0] @ remote_zero[1]
        ) / 2.0
        r_zero = _embed(bob[site][0], 3 + site)
        r_one = _embed(bob[site][1], 3 + site) @ derivative
        a_zero = _embed(alice[site][0], site)
        a_one = _embed(alice[site][1], site)
        term = a_zero @ (r_zero + r_one)
        term += a_one @ (r_zero - r_one)
        terms.append((term + term.conj().T) / 2.0)
    return terms


def _bell_from_observables(
    alice: Sequence[tuple[np.ndarray, np.ndarray]],
    bob: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return the CCZ Bell matrix from six observable pairs."""

    return sum(
        _bell_terms_from_observables(alice, bob),
        start=np.zeros((64, 64), dtype=np.complex128),
    )


def bell_operator(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
) -> np.ndarray:
    r"""Return the Boolean-derivative CCZ Bell operator.

    .. math::
       \mathcal B_{\rm CCZ}
       =\sum_i A_{i,0}(R_{i,0}+R_{i,1})
        +A_{i,1}(R_{i,0}-R_{i,1}),

    with :math:`R_{i,0}=B_{i,0}` and
    :math:`R_{i,1}=B_{i,1}(I+B_{j,0}+B_{k,0}-B_{j,0}B_{k,0})/2`.
    """

    alice_angles = _validate_angles(alice_angles)
    bob_angles = _validate_angles(bob_angles)
    alice = [_canonical_observables(angle) for angle in alice_angles]
    bob = [_canonical_observables(angle) for angle in bob_angles]
    return _bell_from_observables(alice, bob)


def gap_operator(
    slope: float,
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
    axis_rule: str = "covariant",
) -> np.ndarray:
    r"""Return \(K-I+s(6\sqrt2 I-\mathcal B_{\rm CCZ})\)."""

    extracted = extracted_target_projector(
        alice_angles,
        bob_angles,
        axis_rule=axis_rule,
    )
    bell = bell_operator(alice_angles, bob_angles)
    output = (
        extracted
        - _I64
        + slope * (6.0 * _SQRT_TWO * _I64 - bell)
    )
    return (output + output.conj().T) / 2.0


@lru_cache(maxsize=1)
def _effective_chsh_pair_targets() -> tuple[np.ndarray, np.ndarray]:
    r"""Return the two target projectors controlled by a derivative sign.

    In the \(D_i=+1\) sector the effective Bob reflections are
    \((B_{i,0},B_{i,1})\).  In the \(D_i=-1\) sector they are
    \((B_{i,0},-B_{i,1})\), which is obtained by conjugating the canonical
    Bob qubit by a Hadamard.
    """

    bell_state = np.array(
        [1.0, 0.0, 0.0, 1.0],
        dtype=np.complex128,
    ) / _SQRT_TWO
    bob_rotation = _Z @ _rotation_y(-math.pi / 4.0)
    plus_state = np.kron(_I2, bob_rotation) @ bell_state
    plus_projector = np.outer(plus_state, plus_state.conj())
    hadamard = (_X + _Z) / _SQRT_TWO
    bob_hadamard = np.kron(_I2, hadamard)
    minus_projector = (
        bob_hadamard @ plus_projector @ bob_hadamard
    )
    return plus_projector, minus_projector


def _apply_pair_dual(
    operator: np.ndarray,
    angle: float,
    pair_party: int,
) -> np.ndarray:
    """Apply one physical extraction dual to a two-qubit pair operator."""

    visibility = _kaniewski_visibility(angle)
    axis = _Z if angle <= math.pi / 4.0 else _X
    embedded_axis = (
        np.kron(axis, _I2)
        if pair_party == 0
        else np.kron(_I2, axis)
    )
    return (
        (1.0 + visibility) / 2.0 * operator
        + (1.0 - visibility)
        / 2.0
        * embedded_axis
        @ operator
        @ embedded_axis
    )


def _embed_pair_operator(operator: np.ndarray, site: int) -> np.ndarray:
    """Embed a two-qubit operator on ``A_site B_site``."""

    paulis = (_I2, _X, _Y, _Z)
    output = np.zeros((64, 64), dtype=np.complex128)
    for alice_pauli in paulis:
        for bob_pauli in paulis:
            pair_pauli = np.kron(alice_pauli, bob_pauli)
            coefficient = np.trace(pair_pauli @ operator) / 4.0
            if abs(coefficient) < 1e-15:
                continue
            factors = [_I2] * 6
            factors[site] = alice_pauli
            factors[3 + site] = bob_pauli
            output += coefficient * _kron_all(factors)
    return (output + output.conj().T) / 2.0


def _effective_chsh_extracted_operators(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
) -> list[np.ndarray]:
    r"""Return derivative-controlled pair extraction operators.

    The remote CCZ derivative

    .. math::
       D_i=\frac{I+B_{j,0}+B_{k,0}-B_{j,0}B_{k,0}}2

    is a reflection commuting with pair \(i\).  Its two eigenspaces reduce
    the lifted Bell term to two ordinary CHSH Jordan blocks related by a Bob
    Hadamard.  The physical Kaniewski dephasing axis is the same in both
    descriptions.
    """

    alice_angles = _validate_angles(alice_angles)
    bob_angles = _validate_angles(bob_angles)
    bob_observables = [
        _canonical_observables(angle) for angle in bob_angles
    ]
    plus_target, minus_target = _effective_chsh_pair_targets()
    output = []
    for site in range(3):
        plus_extracted = _apply_pair_dual(
            _apply_pair_dual(
                plus_target,
                alice_angles[site],
                pair_party=0,
            ),
            bob_angles[site],
            pair_party=1,
        )
        minus_extracted = _apply_pair_dual(
            _apply_pair_dual(
                minus_target,
                alice_angles[site],
                pair_party=0,
            ),
            bob_angles[site],
            pair_party=1,
        )
        other_sites = [index for index in range(3) if index != site]
        remote_zero = [
            _embed(
                bob_observables[index][0],
                3 + index,
            )
            for index in other_sites
        ]
        derivative = (
            _I64
            + remote_zero[0]
            + remote_zero[1]
            - remote_zero[0] @ remote_zero[1]
        ) / 2.0
        plus_sector = (_I64 + derivative) / 2.0
        minus_sector = (_I64 - derivative) / 2.0
        controlled = (
            _embed_pair_operator(plus_extracted, site) @ plus_sector
            + _embed_pair_operator(minus_extracted, site) @ minus_sector
        )
        output.append((controlled + controlled.conj().T) / 2.0)
    return output


def effective_chsh_gap_sum(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
) -> np.ndarray:
    r"""Return the sum of three exact derivative-controlled CHSH gaps.

    Each summand is positive semidefinite by the ordinary two-qubit
    Kaniewski STOPI inequality at :func:`c_ccz`.
    """

    alice_angles = _validate_angles(alice_angles)
    bob_angles = _validate_angles(bob_angles)
    alice = [
        _canonical_observables(angle) for angle in alice_angles
    ]
    bob = [_canonical_observables(angle) for angle in bob_angles]
    bell_terms = _bell_terms_from_observables(alice, bob)
    extracted_terms = _effective_chsh_extracted_operators(
        alice_angles,
        bob_angles,
    )
    output = np.zeros((64, 64), dtype=np.complex128)
    for extracted, bell_term in zip(extracted_terms, bell_terms):
        output += (
            extracted
            - _I64
            + c_ccz()
            * (2.0 * _SQRT_TWO * _I64 - bell_term)
        )
    return (output + output.conj().T) / 2.0


def corrected_coupling_gap(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
    eta: float = 1.0 / 12.0,
) -> np.ndarray:
    r"""Return the residual after reserving ``1-eta`` of the CHSH SOS.

    .. math::
       C_\eta
       =T_{\rm CCZ}-(1-\eta)\sum_i T_i^{\rm eff}.

    If \(C_\eta\succeq0\), the full CCZ STOPI follows immediately because
    every effective CHSH gap is positive semidefinite.  The default
    \(\eta=1/12\) has strong finite-dimensional evidence but is not yet a
    continuous-domain certificate.
    """

    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta must lie in [0, 1]")
    full = gap_operator(
        c_ccz(),
        alice_angles,
        bob_angles,
    )
    chsh_sum = effective_chsh_gap_sum(alice_angles, bob_angles)
    output = full - (1.0 - eta) * chsh_sum
    return (output + output.conj().T) / 2.0


def arc_outer_triangle(
    lower: float,
    upper: float,
) -> tuple[tuple[float, float], ...]:
    r"""Return the tangent outer triangle of a unit-circle subarc.

    For midpoint \(m=(\ell+u)/2\) and half-width \(h=(u-\ell)/2\), the
    tangent intersection is

    .. math::
       \sec(h)(\cos m,\sin m).
    """

    lower = float(lower)
    upper = float(upper)
    if lower < 0.0 or upper > math.pi / 2.0 or lower >= upper:
        raise ValueError("require 0 <= lower < upper <= pi/2")
    if lower < math.pi / 4.0 < upper:
        raise ValueError("an outer triangle cannot cross a channel branch")
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    radius = 1.0 / math.cos(half_width)
    return (
        (math.cos(lower), math.sin(lower)),
        (math.cos(upper), math.sin(upper)),
        (
            radius * math.cos(midpoint),
            radius * math.sin(midpoint),
        ),
    )


def arc_barycentric_weights(
    angle: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    """Return the outer-triangle weights of one physical arc point."""

    angle = float(angle)
    if angle < lower or angle > upper:
        raise ValueError("angle must lie in [lower, upper]")
    triangle = arc_outer_triangle(lower, upper)
    system = np.array(
        [
            [vertex[0] for vertex in triangle],
            [vertex[1] for vertex in triangle],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    target = np.array(
        [math.cos(angle), math.sin(angle), 1.0],
        dtype=float,
    )
    weights = np.linalg.solve(system, target)
    return tuple(float(weight) for weight in weights)


def corrected_coupling_gap_coordinates(
    coordinates: Sequence[tuple[float, float]],
    axes: Sequence[str],
    eta: float = 0.5,
) -> np.ndarray:
    r"""Evaluate the multiaffine coupling gap off the physical arc.

    ``coordinates`` contains six \((c_i,s_i)\) pairs and ``axes`` fixes the
    adjacent extraction branch (``"Z"`` below the ideal angle, ``"X"``
    above it).  The visibility is extended affinely as

    .. math::
       g_i=(1+\sqrt2)(c_i+s_i-1).

    This extension is used only for convex outer certificates; its vertices
    need not describe physical observables or quantum channels.
    """

    if len(coordinates) != 6 or len(axes) != 6:
        raise ValueError("six coordinates and six axes are required")
    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta must lie in [0, 1]")

    parsed_coordinates = [
        (float(cosine), float(sine))
        for cosine, sine in coordinates
    ]
    if any(
        not math.isfinite(value)
        for coordinate in parsed_coordinates
        for value in coordinate
    ):
        raise ValueError("coordinates must be finite")
    if any(axis not in ("X", "Z") for axis in axes):
        raise ValueError("axes must be X or Z")

    visibilities = [
        (1.0 + _SQRT_TWO) * (cosine + sine - 1.0)
        for cosine, sine in parsed_coordinates
    ]
    axis_matrices = [_X if axis == "X" else _Z for axis in axes]

    def apply_coordinate_dual(
        operator: np.ndarray,
        coordinate: int,
    ) -> np.ndarray:
        embedded_axis = _embed(
            axis_matrices[coordinate],
            coordinate,
        )
        visibility = visibilities[coordinate]
        return (
            (1.0 + visibility) / 2.0 * operator
            + (1.0 - visibility)
            / 2.0
            * embedded_axis
            @ operator
            @ embedded_axis
        )

    def apply_coordinate_pair_dual(
        operator: np.ndarray,
        coordinate: int,
        pair_party: int,
    ) -> np.ndarray:
        pair_axis = (
            np.kron(axis_matrices[coordinate], _I2)
            if pair_party == 0
            else np.kron(_I2, axis_matrices[coordinate])
        )
        visibility = visibilities[coordinate]
        return (
            (1.0 + visibility) / 2.0 * operator
            + (1.0 - visibility)
            / 2.0
            * pair_axis
            @ operator
            @ pair_axis
        )

    observables = [
        (
            cosine * _Z + sine * _X,
            cosine * _Z - sine * _X,
        )
        for cosine, sine in parsed_coordinates
    ]
    alice_observables = observables[:3]
    bob_observables = observables[3:]
    bell_terms = _bell_terms_from_observables(
        alice_observables,
        bob_observables,
    )

    extracted = target_projector().copy()
    for coordinate in range(6):
        extracted = apply_coordinate_dual(extracted, coordinate)
    bell = sum(
        bell_terms,
        start=np.zeros((64, 64), dtype=np.complex128),
    )
    full_gap = (
        extracted
        - _I64
        + c_ccz() * (6.0 * _SQRT_TWO * _I64 - bell)
    )

    plus_target, minus_target = _effective_chsh_pair_targets()
    chsh_sum = np.zeros((64, 64), dtype=np.complex128)
    for site in range(3):
        plus_extracted = apply_coordinate_pair_dual(
            apply_coordinate_pair_dual(
                plus_target,
                site,
                pair_party=0,
            ),
            3 + site,
            pair_party=1,
        )
        minus_extracted = apply_coordinate_pair_dual(
            apply_coordinate_pair_dual(
                minus_target,
                site,
                pair_party=0,
            ),
            3 + site,
            pair_party=1,
        )
        other_sites = [index for index in range(3) if index != site]
        remote_zero = [
            _embed(
                bob_observables[index][0],
                3 + index,
            )
            for index in other_sites
        ]
        derivative = (
            _I64
            + remote_zero[0]
            + remote_zero[1]
            - remote_zero[0] @ remote_zero[1]
        ) / 2.0
        effective_extracted = (
            _embed_pair_operator(plus_extracted, site)
            @ ((_I64 + derivative) / 2.0)
            + _embed_pair_operator(minus_extracted, site)
            @ ((_I64 - derivative) / 2.0)
        )
        chsh_sum += (
            effective_extracted
            - _I64
            + c_ccz()
            * (2.0 * _SQRT_TWO * _I64 - bell_terms[site])
        )

    output = full_gap - (1.0 - eta) * chsh_sum
    return (output + output.conj().T) / 2.0


def _counterexample_vector() -> np.ndarray:
    r"""Return the explicit asymmetric-corner Rayleigh vector.

    Its computational-basis coefficient is

    .. math::
       (-1)^{1+a_2+a_1b_1+a_2b_2+a_3b_3+a_1a_2a_3}
       (\sqrt2-1)^{|a\oplus b|}.
    """

    output = np.zeros(64, dtype=np.complex128)
    radial = _SQRT_TWO - 1.0
    for alice_value in range(8):
        alice_bits = (
            (alice_value >> 2) & 1,
            (alice_value >> 1) & 1,
            alice_value & 1,
        )
        for bob_value in range(8):
            bob_bits = (
                (bob_value >> 2) & 1,
                (bob_value >> 1) & 1,
                bob_value & 1,
            )
            parity = (
                1
                + alice_bits[1]
                + alice_bits[0] * bob_bits[0]
                + alice_bits[1] * bob_bits[1]
                + alice_bits[2] * bob_bits[2]
                + alice_bits[0] * alice_bits[1] * alice_bits[2]
            ) % 2
            distance = sum(
                left ^ right
                for left, right in zip(alice_bits, bob_bits)
            )
            coefficient = (-1.0 if parity else 1.0) * radial**distance
            output[(alice_value << 3) | bob_value] = coefficient
    return output


def corner_audit(
    slope: float,
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
    axis_rule: str = "covariant",
) -> dict[str, object]:
    """Return the minimum eigenvalue at one fixed Jordan point."""

    alice_angles = _validate_angles(alice_angles)
    bob_angles = _validate_angles(bob_angles)
    gap = gap_operator(
        slope,
        alice_angles,
        bob_angles,
        axis_rule=axis_rule,
    )
    return {
        "slope": float(slope),
        "alice_angles": alice_angles,
        "bob_angles": bob_angles,
        "axis_rule": axis_rule,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(gap)[0]),
        "evidence": "fixed-point floating-point matrix audit",
    }


def mismatched_channel_counterexample(slope: float) -> dict[str, object]:
    r"""Replay the corner refuting the axis-mismatched channel."""

    alice_angles = (math.pi / 4.0, 0.0, math.pi / 4.0)
    bob_angles = (math.pi / 4.0,) * 3
    gap = gap_operator(
        slope,
        alice_angles,
        bob_angles,
        axis_rule="literal",
    )
    vector = _counterexample_vector()
    norm_squared = float(np.vdot(vector, vector).real)
    expectation = float(np.vdot(vector, gap @ vector).real)
    return {
        "slope": float(slope),
        "alice_angles": alice_angles,
        "bob_angles": bob_angles,
        "axis_rule": "literal",
        "norm_squared": norm_squared,
        "expectation": expectation,
        "rayleigh_quotient": expectation / norm_squared,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(gap)[0]),
        "evidence": "fixed-point floating evaluation of an exact algebraic witness",
    }


def one_bad_pair_audit(
    site: int = 0,
    bob_endpoint: float = 0.0,
) -> dict[str, object]:
    r"""Audit the exact boundary with one endpoint Jordan pair.

    Set :math:`a_i=0`, :math:`b_i\in\{0,\pi/2\}`, and all other angles to
    :math:`\pi/4`.  The gap then commutes with the locally preserved axes
    on the endpoint pair.  In a sector
    with eigenvalues :math:`\alpha,\beta\in\{\pm1\}`, the Bell block has
    the form

    .. math::
       \mathcal B_{\alpha\beta}
       =2\alpha\beta I+\sqrt2\sum_{r=1}^4G_r,

    for four commuting reflections.  In a matched sector the rank-one
    extracted operator is supported on their joint ``+1`` eigenspace with
    weight :math:`(2+\sqrt2)/8`; this cancels the unique negative eigenvalue
    of the unextracted affine gap at :func:`c_ccz`.

    The returned floating-point quantities audit that exact algebraic
    identity.  They do not certify a neighborhood of the boundary.
    """

    if site not in (0, 1, 2):
        raise ValueError("site must be 0, 1, or 2")
    if not (
        math.isclose(bob_endpoint, 0.0, abs_tol=1e-15)
        or math.isclose(bob_endpoint, math.pi / 2.0, abs_tol=1e-15)
    ):
        raise ValueError("bob_endpoint must be 0 or pi/2")

    alice_angles = [math.pi / 4.0] * 3
    bob_angles = [math.pi / 4.0] * 3
    alice_angles[site] = 0.0
    bob_angles[site] = float(bob_endpoint)
    alice_tuple = tuple(alice_angles)
    bob_tuple = tuple(bob_angles)
    extracted = extracted_target_projector(alice_tuple, bob_tuple)
    bell = bell_operator(alice_tuple, bob_tuple)
    gap = gap_operator(c_ccz(), alice_tuple, bob_tuple)
    alice_z = _embed(_Z, site)
    bob_axis = _Z if bob_endpoint == 0.0 else _X
    bob_endpoint_axis = _embed(bob_axis, 3 + site)
    alice_commutator_norm = float(
        np.linalg.norm(gap @ alice_z - alice_z @ gap)
    )
    bob_commutator_norm = float(
        np.linalg.norm(
            gap @ bob_endpoint_axis - bob_endpoint_axis @ gap
        )
    )

    if bob_endpoint == math.pi / 2.0:
        hadamard = (_X + _Z) / _SQRT_TWO
        basis_rotation = _embed(hadamard, 3 + site)
        extracted = basis_rotation @ extracted @ basis_rotation
        bell = basis_rotation @ bell @ basis_rotation
        gap = basis_rotation @ gap @ basis_rotation

    def sector_indices(alice_bit: int, bob_bit: int) -> list[int]:
        alice_shift = 5 - site
        bob_shift = 2 - site
        return [
            index
            for index in range(64)
            if ((index >> alice_shift) & 1) == alice_bit
            and ((index >> bob_shift) & 1) == bob_bit
        ]

    def sector_data(alice_bit: int, bob_bit: int) -> dict[str, float | int]:
        indices = sector_indices(alice_bit, bob_bit)
        extracted_block = extracted[np.ix_(indices, indices)]
        bell_block = bell[np.ix_(indices, indices)]
        gap_block = gap[np.ix_(indices, indices)]
        unextracted_block = (
            -np.eye(16, dtype=np.complex128)
            + c_ccz()
            * (
                6.0 * _SQRT_TWO * np.eye(16, dtype=np.complex128)
                - bell_block
            )
        )
        gap_eigenvalues = np.linalg.eigvalsh(gap_block)
        bell_eigenvalues, bell_eigenvectors = np.linalg.eigh(bell_block)
        top_projector = np.outer(
            bell_eigenvectors[:, -1],
            bell_eigenvectors[:, -1].conj(),
        )
        extracted_weight = float(np.trace(extracted_block).real)
        return {
            "minimum_eigenvalue": float(gap_eigenvalues[0]),
            "nullity": int(np.count_nonzero(np.abs(gap_eigenvalues) < 1e-10)),
            "extracted_weight": extracted_weight,
            "projector_alignment_error": float(
                np.linalg.norm(
                    extracted_block - extracted_weight * top_projector
                )
            ),
            "unextracted_floor": float(
                np.linalg.eigvalsh(unextracted_block)[0]
            ),
        }

    return {
        "site": site,
        "alice_angles": alice_tuple,
        "bob_angles": bob_tuple,
        "slope": c_ccz(),
        "commutator_norms": {
            "A_bad_axis": alice_commutator_norm,
            "B_bad_axis": bob_commutator_norm,
        },
        "matched_sector": sector_data(0, 0),
        "mismatched_sector": sector_data(0, 1),
        "evidence": "boundary floating-point audit of an exact sector identity",
    }


def _gap_at_six_angles(angles: np.ndarray) -> np.ndarray:
    """Evaluate the candidate gap from ``[a1,a2,a3,b1,b2,b3]``."""

    return gap_operator(c_ccz(), angles[:3], angles[3:])


def _preserved_axis_for_direction(
    angle: float,
    direction_sign: float,
) -> np.ndarray:
    """Return the extraction axis in one adjacent analytic branch."""

    if angle < math.pi / 4.0:
        return _Z
    if angle > math.pi / 4.0:
        return _X
    return _Z if direction_sign < 0.0 else _X


def _dual_even_odd_parts(
    operator: np.ndarray,
    party: int,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return \(E_\Gamma(O)\) and \(F_\Gamma(O)\).

    The local dual is

    .. math::
       \Lambda_g^\dagger(O)=E_\Gamma(O)+gF_\Gamma(O),
       \quad
       E_\Gamma(O)=\frac{O+\Gamma O\Gamma}{2},
       \quad
       F_\Gamma(O)=\frac{O-\Gamma O\Gamma}{2}.
    """

    embedded_axis = _embed(axis, party)
    conjugated = embedded_axis @ operator @ embedded_axis
    return (operator + conjugated) / 2.0, (operator - conjugated) / 2.0


def _gap_derivative_matrices(
    equality_angles: np.ndarray,
    direction_signs: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Return analytic first and second directional derivatives of the gap."""

    base_observables = [
        _canonical_observables(float(angle)) for angle in equality_angles
    ]
    zero_pair = (
        np.zeros((2, 2), dtype=np.complex128),
        np.zeros((2, 2), dtype=np.complex128),
    )
    first_observables = []
    second_observables = []
    for angle, sign in zip(equality_angles, direction_signs):
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        first_observables.append(
            (
                sign * (-sine * _Z + cosine * _X),
                sign * (-sine * _Z - cosine * _X),
            )
        )
        base_zero, base_one = _canonical_observables(float(angle))
        second_observables.append((-base_zero, -base_one))

    def bell_value(
        observable_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        return _bell_from_observables(
            observable_pairs[:3],
            observable_pairs[3:],
        )

    bell_first: list[np.ndarray] = []
    bell_second: list[np.ndarray] = []
    for coordinate in range(6):
        zeroed = list(base_observables)
        zeroed[coordinate] = zero_pair
        differentiated = list(base_observables)
        differentiated[coordinate] = first_observables[coordinate]
        twice_differentiated = list(base_observables)
        twice_differentiated[coordinate] = second_observables[coordinate]
        zeroed_value = bell_value(zeroed)
        bell_first.append(bell_value(differentiated) - zeroed_value)
        bell_second.append(
            bell_value(twice_differentiated) - zeroed_value
        )

    bell_mixed: dict[tuple[int, int], np.ndarray] = {}
    for left in range(6):
        for right in range(left + 1, 6):
            both = list(base_observables)
            both[left] = first_observables[left]
            both[right] = first_observables[right]
            left_only = list(base_observables)
            left_only[left] = first_observables[left]
            left_only[right] = zero_pair
            right_only = list(base_observables)
            right_only[left] = zero_pair
            right_only[right] = first_observables[right]
            neither = list(base_observables)
            neither[left] = zero_pair
            neither[right] = zero_pair
            bell_mixed[left, right] = (
                bell_value(both)
                - bell_value(left_only)
                - bell_value(right_only)
                + bell_value(neither)
            )

    axes = [
        _preserved_axis_for_direction(float(angle), float(sign))
        for angle, sign in zip(equality_angles, direction_signs)
    ]
    visibilities = [
        _kaniewski_visibility(float(angle)) for angle in equality_angles
    ]
    visibility_first = [
        float(sign)
        * (1.0 + _SQRT_TWO)
        * (math.cos(float(angle)) - math.sin(float(angle)))
        for angle, sign in zip(equality_angles, direction_signs)
    ]
    visibility_second = [
        -(1.0 + _SQRT_TWO)
        * (math.cos(float(angle)) + math.sin(float(angle)))
        for angle in equality_angles
    ]

    def apply_base_duals(
        operator: np.ndarray,
        skipped: set[int],
    ) -> np.ndarray:
        output = operator
        for party in range(6):
            if party in skipped:
                continue
            even, odd = _dual_even_odd_parts(
                output,
                party,
                axes[party],
            )
            output = even + visibilities[party] * odd
        return output

    extracted_first: list[np.ndarray] = []
    extracted_second: list[np.ndarray] = []
    for coordinate in range(6):
        reduced = apply_base_duals(target_projector(), {coordinate})
        _, odd = _dual_even_odd_parts(
            reduced,
            coordinate,
            axes[coordinate],
        )
        extracted_first.append(visibility_first[coordinate] * odd)
        extracted_second.append(visibility_second[coordinate] * odd)

    extracted_mixed: dict[tuple[int, int], np.ndarray] = {}
    for left in range(6):
        for right in range(left + 1, 6):
            reduced = apply_base_duals(
                target_projector(),
                {left, right},
            )
            _, left_odd = _dual_even_odd_parts(
                reduced,
                left,
                axes[left],
            )
            _, both_odd = _dual_even_odd_parts(
                left_odd,
                right,
                axes[right],
            )
            extracted_mixed[left, right] = (
                visibility_first[left]
                * visibility_first[right]
                * both_odd
            )

    slope = c_ccz()
    first = [
        extracted_first[index] - slope * bell_first[index]
        for index in range(6)
    ]
    second = [
        extracted_second[index] - slope * bell_second[index]
        for index in range(6)
    ]
    mixed = {
        pair: extracted_mixed[pair] - slope * bell_mixed[pair]
        for pair in extracted_mixed
    }
    return first, second, mixed


def _local_schur_block_hessian(
    equality_angles: np.ndarray,
    kernel_dimension: int,
    direction_signs: Sequence[float],
    step: float,
) -> np.ndarray:
    r"""Return the analytic-derivative effective Schur Hessian.

    Let :math:`P` project onto the kernel of the equality-point gap and
    :math:`R=(QT_0Q)^{-1}` on its orthogonal support.  The ``(i,j)`` block
    is

    .. math::
       P T_{ij}P-P T_i R T_jP-P T_j R T_iP.

    Matrix construction uses analytic angle derivatives.  Eigenspaces and
    inversion are evaluated in ``complex128``.  Positive definiteness is a
    local diagnostic; a theorem still requires outward rounding and a
    rigorous remainder bound.
    """

    if step <= 0.0:
        raise ValueError("step must be positive")
    signs = np.asarray(direction_signs, dtype=float)
    if signs.shape != (6,) or np.any(np.abs(signs) != 1.0):
        raise ValueError("direction_signs must contain six signs")
    _ = step

    base = _gap_at_six_angles(equality_angles)
    eigenvalues, eigenvectors = np.linalg.eigh(base)
    kernel = eigenvectors[:, :kernel_dimension]
    complement = eigenvectors[:, kernel_dimension:]
    inverse_complement = (
        complement / eigenvalues[kernel_dimension:]
    ) @ complement.conj().T
    first_derivatives, second_derivatives, mixed_derivatives = (
        _gap_derivative_matrices(equality_angles, signs)
    )

    block_size = kernel_dimension
    hessian = np.zeros(
        (6 * block_size, 6 * block_size),
        dtype=np.complex128,
    )
    for left in range(6):
        diagonal_block = (
            kernel.conj().T @ second_derivatives[left] @ kernel
            - 2.0
            * kernel.conj().T
            @ first_derivatives[left]
            @ inverse_complement
            @ first_derivatives[left]
            @ kernel
        )
        left_slice = slice(
            block_size * left,
            block_size * (left + 1),
        )
        hessian[left_slice, left_slice] = diagonal_block

        for right in range(left + 1, 6):
            mixed_block = (
                kernel.conj().T
                @ mixed_derivatives[left, right]
                @ kernel
                - kernel.conj().T
                @ first_derivatives[left]
                @ inverse_complement
                @ first_derivatives[right]
                @ kernel
                - kernel.conj().T
                @ first_derivatives[right]
                @ inverse_complement
                @ first_derivatives[left]
                @ kernel
            )
            right_slice = slice(
                block_size * right,
                block_size * (right + 1),
            )
            hessian[left_slice, right_slice] = mixed_block
            hessian[right_slice, left_slice] = mixed_block.conj().T

    return (hessian + hessian.conj().T) / 2.0


def ideal_schur_hessian_audit(step: float = 2e-4) -> dict[str, object]:
    r"""Audit all 64 branch orthants at the ideal equality point."""

    equality_angles = np.full(6, math.pi / 4.0)
    minima = []
    for signs in np.ndindex(*(2,) * 6):
        direction_signs = tuple(-1.0 if sign == 0 else 1.0 for sign in signs)
        hessian = _local_schur_block_hessian(
            equality_angles,
            kernel_dimension=1,
            direction_signs=direction_signs,
            step=step,
        )
        minima.append(float(np.linalg.eigvalsh(hessian)[0]))
    return {
        "step": float(step),
        "orthants": 64,
        "minimum_effective_hessian_eigenvalue": min(minima),
        "maximum_of_orthant_minima": max(minima),
        "evidence": "analytic-derivative local Schur audit",
    }


def one_bad_pair_schur_hessian_audit(
    site: int = 0,
    bob_endpoint: float = 0.0,
    step: float = 2e-4,
) -> dict[str, object]:
    r"""Audit the 16 local branch orthants around a tight endpoint."""

    if site not in (0, 1, 2):
        raise ValueError("site must be 0, 1, or 2")
    if not (
        math.isclose(bob_endpoint, 0.0, abs_tol=1e-15)
        or math.isclose(bob_endpoint, math.pi / 2.0, abs_tol=1e-15)
    ):
        raise ValueError("bob_endpoint must be 0 or pi/2")

    equality_angles = np.full(6, math.pi / 4.0)
    equality_angles[site] = 0.0
    equality_angles[3 + site] = float(bob_endpoint)
    good_coordinates = [
        coordinate
        for coordinate in range(6)
        if coordinate not in (site, 3 + site)
    ]
    minima = []
    for good_sign_bits in np.ndindex(*(2,) * 4):
        direction_signs = np.ones(6)
        direction_signs[3 + site] = (
            1.0 if bob_endpoint == 0.0 else -1.0
        )
        for coordinate, sign_bit in zip(
            good_coordinates,
            good_sign_bits,
        ):
            direction_signs[coordinate] = (
                -1.0 if sign_bit == 0 else 1.0
            )
        hessian = _local_schur_block_hessian(
            equality_angles,
            kernel_dimension=2,
            direction_signs=direction_signs,
            step=step,
        )
        minima.append(float(np.linalg.eigvalsh(hessian)[0]))
    return {
        "site": site,
        "bob_endpoint": float(bob_endpoint),
        "step": float(step),
        "orthants": 16,
        "minimum_block_hessian_eigenvalue": min(minima),
        "maximum_of_orthant_minima": max(minima),
        "evidence": "analytic-derivative local Schur audit",
    }


def required_slope(
    alice_angles: Sequence[float],
    bob_angles: Sequence[float],
    axis_rule: str = "covariant",
) -> float:
    r"""Return the smallest \(s\ge0\) for which the local gap is PSD.

    The Bell deficit is positive semidefinite, so the minimum gap eigenvalue
    is monotone nondecreasing in ``s``.  Bisection identifies the fixed-point
    threshold; it is not a continuous-domain certificate.
    """

    alice_angles = _validate_angles(alice_angles)
    bob_angles = _validate_angles(bob_angles)

    def minimum_eigenvalue(slope: float) -> float:
        return float(
            np.linalg.eigvalsh(
                gap_operator(
                    slope,
                    alice_angles,
                    bob_angles,
                    axis_rule=axis_rule,
                )
            )[0]
        )

    lower = 0.0
    upper = 1.0
    while minimum_eigenvalue(upper) < -1e-13:
        upper *= 2.0
        if upper > 1024.0:
            raise RuntimeError("failed to bracket a feasible slope")
    for _ in range(90):
        midpoint = (lower + upper) / 2.0
        if minimum_eigenvalue(midpoint) < -1e-13:
            lower = midpoint
        else:
            upper = midpoint
    return upper
