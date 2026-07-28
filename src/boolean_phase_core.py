#!/usr/bin/env python3
"""Reference-model checks for Boolean-phase Bell self-tests.

The routines verify the finite-dimensional identities used to define the
target state and Bell expression.  They are diagnostics, not a substitute
for the arbitrary-dimensional self-testing proofs in the paper.

All matrices use ``complex128`` and subsystem order
``A_0, ..., A_{n-1}, B_0, ..., B_{n-1}``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp


BitString = tuple[int, ...]
PhaseFunction = Callable[[BitString], int]

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    """Return the Kronecker product in the listed subsystem order."""

    result = np.array([[1.0]], dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def op_on(number_of_qubits: int, site: int, operator: np.ndarray) -> np.ndarray:
    """Embed a single-qubit operator at ``site``."""

    if not 0 <= site < number_of_qubits:
        raise ValueError("site must index an existing qubit")
    operators = [I] * number_of_qubits
    operators[site] = operator
    return kron_all(operators)


def ccz_phase(bit_string: BitString) -> int:
    r"""Return the CCZ phase \(f(x)=x_0x_1x_2\)."""

    if len(bit_string) != 3:
        raise ValueError("ccz_phase requires exactly three bits")
    return bit_string[0] & bit_string[1] & bit_string[2]


def nonsymmetric_phase(bit_string: BitString) -> int:
    r"""Return \(x_0x_1x_2\oplus x_0x_1\oplus x_1x_2\oplus x_2\)."""

    if len(bit_string) != 3:
        raise ValueError("nonsymmetric_phase requires exactly three bits")
    return (
        (bit_string[0] & bit_string[1] & bit_string[2])
        ^ (bit_string[0] & bit_string[1])
        ^ (bit_string[1] & bit_string[2])
        ^ bit_string[2]
    )


def derivative(
    phase_function: PhaseFunction,
    bit_string: BitString,
    site: int,
) -> int:
    r"""Evaluate the Boolean derivative.

    .. math::
       \partial_i f(x_{\bar i})
       =f(x_i=0,x_{\bar i})\oplus f(x_i=1,x_{\bar i}).
    """

    if not 0 <= site < len(bit_string):
        raise ValueError("site must index the bit string")
    zero = list(bit_string)
    one = list(bit_string)
    zero[site] = 0
    one[site] = 1
    return int(phase_function(tuple(zero))) ^ int(
        phase_function(tuple(one))
    )


def computational_bitstrings(number_of_qubits: int) -> list[BitString]:
    """List computational-basis bit strings in matrix-index order."""

    if number_of_qubits < 1:
        raise ValueError("number_of_qubits must be positive")
    return list(itertools.product((0, 1), repeat=number_of_qubits))


def reference_state_and_stabilizers(
    number_of_qubits: int,
    phase_function: PhaseFunction,
) -> tuple[np.ndarray, list[np.ndarray]]:
    r"""Construct the Boolean-phase Choi state and its \(2n\) stabilizers.

    .. math::
       |\Psi_f\rangle
       =2^{-n/2}\sum_x(-1)^{f(x)}|x\rangle_A|x\rangle_B,

    .. math::
       S_i^Z=Z_{A_i}Z_{B_i},\qquad
       S_i^X=X_{A_i}X_{B_i}D_i^f(Z_{B_{\bar i}}).
    """

    bitstrings = computational_bitstrings(number_of_qubits)
    dimension = 2**number_of_qubits
    total_qubits = 2 * number_of_qubits
    phases = np.asarray(
        [(-1) ** int(phase_function(x)) for x in bitstrings],
        dtype=np.complex128,
    )
    state = np.zeros(dimension**2, dtype=np.complex128)
    for index, phase in enumerate(phases):
        state[index * dimension + index] = phase / math.sqrt(dimension)

    stabilizers: list[np.ndarray] = []
    for site in range(number_of_qubits):
        z_stabilizer = op_on(total_qubits, site, Z) @ op_on(
            total_qubits,
            number_of_qubits + site,
            Z,
        )
        derivative_diagonal = np.asarray(
            [
                (-1) ** derivative(phase_function, bit_string, site)
                for bit_string in bitstrings
            ],
            dtype=np.complex128,
        )
        derivative_operator = np.kron(
            np.eye(dimension, dtype=np.complex128),
            np.diag(derivative_diagonal),
        )
        x_stabilizer = (
            op_on(total_qubits, site, X)
            @ op_on(total_qubits, number_of_qubits + site, X)
            @ derivative_operator
        )
        stabilizers.extend((z_stabilizer, x_stabilizer))
    return state, stabilizers


def reference_checks(
    number_of_qubits: int,
    phase_function: PhaseFunction,
    tolerance: float = 1e-10,
) -> np.ndarray:
    r"""Verify the ideal stabilizers and Bell spectrum.

    .. math::
       \mathcal B_f^{\rm ideal}
       =\sqrt2\sum_{i=1}^n(S_i^Z+S_i^X).
    """

    state, stabilizers = reference_state_and_stabilizers(
        number_of_qubits,
        phase_function,
    )
    for stabilizer in stabilizers:
        if np.linalg.norm(stabilizer @ state - state) >= tolerance:
            raise AssertionError("reference state is not stabilized")

    bell_operator = math.sqrt(2.0) * np.sum(stabilizers, axis=0)
    eigenvalues = np.linalg.eigvalsh(bell_operator)
    quantum_bound = 2.0 * math.sqrt(2.0) * number_of_qubits
    expected_second = quantum_bound - 2.0 * math.sqrt(2.0)
    if not math.isclose(eigenvalues[-1], quantum_bound, abs_tol=tolerance):
        raise AssertionError("incorrect largest Bell eigenvalue")
    if not math.isclose(
        eigenvalues[-2],
        expected_second,
        abs_tol=tolerance,
    ):
        raise AssertionError("incorrect Bell spectral gap")
    if np.sum(np.isclose(eigenvalues, eigenvalues[-1], atol=tolerance)) != 1:
        raise AssertionError("largest Bell eigenvalue must be nondegenerate")
    return eigenvalues


def local_deterministic_bound(
    number_of_qubits: int,
    phase_function: PhaseFunction,
) -> tuple[int, int]:
    r"""Enumerate the deterministic values of the lifted CHSH expression.

    .. math::
       \sum_i a_{i,0}(r_{i,0}+r_{i,1})
       +a_{i,1}(r_{i,0}-r_{i,1}).
    """

    best = -math.inf
    maximizers = 0
    for values in itertools.product(
        (-1, 1),
        repeat=4 * number_of_qubits,
    ):
        a_zero = values[0:number_of_qubits]
        a_one = values[number_of_qubits : 2 * number_of_qubits]
        b_z = values[2 * number_of_qubits : 3 * number_of_qubits]
        b_x = values[3 * number_of_qubits : 4 * number_of_qubits]
        bits = tuple(0 if value == 1 else 1 for value in b_z)
        score = 0
        for site in range(number_of_qubits):
            sign = (-1) ** derivative(phase_function, bits, site)
            r_zero = b_z[site]
            r_one = b_x[site] * sign
            score += a_zero[site] * (r_zero + r_one)
            score += a_one[site] * (r_zero - r_one)
        if score > best:
            best = score
            maximizers = 1
        elif score == best:
            maximizers += 1

    if best != 2 * number_of_qubits:
        raise AssertionError("deterministic enumeration violates the bound")
    return int(best), maximizers


def uniform_robustness_coefficient() -> sp.Expr:
    r"""Return \(c_Z=3(1+\sqrt2)/8\) exactly."""

    return 3 * (1 + sp.sqrt(2)) / 8


def ccz_robustness_coefficient() -> sp.Expr:
    r"""Return \(c_{\rm CCZ}=(4+5\sqrt2)/16\) exactly."""

    return (4 + 5 * sp.sqrt(2)) / 16
