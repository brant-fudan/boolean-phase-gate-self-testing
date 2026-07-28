# Proof archive for Boolean-phase gate self-testing

This archive contains the mechanically replayable layer behind the paper
*Robust Device-Independent Certification of Boolean-Phase Gates*.  The paper
keeps the analytic arguments that establish the arbitrary-dimensional
self-testing and channel-certification statements.  The source and records
here verify finite symbolic identities, exact polynomial certificates, and
the validated continuous-domain cover.

The distinction is deliberate:

- symbolic checks supplement the analytic exact self-testing proof;
- the uniform local state--operator inequality is an exact
  computer-assisted certificate over `Q(sqrt(2))`;
- the CCZ coupling inequality is a rigorous computer-assisted certificate
  combining exact local algebra with outward-rounded interval validation;
- neither code nor finite matrices replace the direct-integral lift to
  arbitrary local Hilbert-space dimension.

## Claim--evidence map

| Paper result | Analytic part retained in the paper | Repository evidence |
|---|---|---|
| Bell bounds and exact Choi-state self-test | CHSH equality relations, label matching, Boolean-cube phase propagation, and SWAP extraction | `src/boolean_phase_core.py` and `tests/test_boolean_phase_core.py` check target identities, spectra, and finite reference realizations |
| Uniform robust coefficient `3(1+sqrt(2))/8` | Conditional pair inequality interface, exact preservation of remote `Z` reflections, commuting-projector union bound, and arbitrary-dimensional lift | `src/verify_boolean_phase_robustness.py` reconstructs all four determinant polynomials and checks the exact Bernstein/local certificates |
| CCZ coefficient `(4+5sqrt(2))/16` | Fixed extraction, derivative-sector covariance, pairwise CHSH gaps, coupling decomposition, and arbitrary-dimensional lift | `src/ccz_stopi_exact.py`, `src/ccz_stopi_local_certificate.py`, and `src/ccz_stopi_interval_certificate.py` verify the exact local and validated global certificates |
| Fixed-extraction tightness | Endpoint obstruction and scope statement | `src/ccz_stopi_optimal.py` and its tests reproduce the endpoint equality and reject smaller affine slopes |
| Visibility and channel consequences | Analytic Bures-angle composition and standard fidelity conversions | `src/generate_manuscript_results.py` deterministically regenerates the tables and figures |

## Detailed certificate notes

### Uniform coefficient

After Jordan reduction, one Alice block and one Bob block are described by
angles `a,b in [0,pi/2]`.  The dual extraction channels use

```text
g(t) = (1 + sqrt(2)) (sin(t) + cos(t) - 1).
```

Resolving the remote derivative reflection and the eigenvalue
`r in {+1,-1}` of the commuting product `XX * ZZ` reduces positivity of the
pair gap to four scalar determinants,

```text
D_r = (3(1+sqrt(2)) + r p h)^2
      - (u + r v - 3(1+sqrt(2))(cos(a) + r sin(a) sin(2b)))^2
      - 9(1+sqrt(2))^2 sin(a)^2 cos(2b)^2.
```

The low/high Alice branches and the two values of `r` give four polynomials
over `Q(sqrt(2))` after tangent-half-angle substitution and multiplication
by a positive denominator.  The verifier proves nonnegativity on the full
unit domains using exact Bernstein coefficients away from equality corners
and exact quadratic-remainder bounds inside those corners.  Every comparison
is exact; floating-point arithmetic does not decide a sign.

The endpoint `a=pi/2, b=0` gives `K=I/4` and `C=2 X tensor Z`.  On the
`+1` eigenspace, any affine slope `s` for the fixed channels must satisfy

```text
1/4 >= 1 - s(2sqrt(2)-2),
```

which yields `s >= 3(1+sqrt(2))/8`.

### CCZ coefficient

For each of the 64 low/high channel branches, the six Jordan angles are
mapped to a unit cube by a tangent-half-angle substitution.  Clearing the
strictly positive denominator produces a symmetric `64 x 64` matrix
polynomial of coordinate degree at most two over `Q(sqrt(2))`.

The certificate covers its seven equality points by exact local
Taylor--Schur analysis: one all-ideal point and two one-bad-pair endpoints
for each conditional CHSH pair.  On the compact complement, each dyadic cell
uses the degree-two tensor Bernstein representation.  All 729 coefficient
matrices are accepted only after a Cholesky factor, residual norm, input
rounding error, and spectral shift jointly leave a strictly positive lower
bound.  Floating eigenvalues propose shifts but never establish positivity.

The frozen record hierarchy contains:

- 20 symmetry-reduced branch orbits covering all 64 physical branches;
- 17,836 visited dyadic cells;
- 8,818 Bernstein--Cholesky terminal cells;
- 110 exact-local handoffs;
- zero unresolved cells;
- a smallest verified Bernstein margin of
  `3.493158352255358e-09`.

These implementation-level quantities are recorded in
`results/ccz_stopi_certificate/` and are intentionally not printed in the
paper.

## Replay

From the public repository root, run:

```bash
bash scripts/verify_all.sh
```

The final line must be:

```text
verification_status=PASS
```

Targeted replay commands and the expected aggregate values are documented in
`results/ccz_stopi_certificate/README.md`.

## Scope

- The CCZ coefficient is optimal only for the stated fixed extraction and
  affine state--operator inequality.
- The archive does not claim global optimality over extraction channels or
  nonlinear tradeoffs.
- The global CCZ certificate is validated, not a heuristic numerical scan.
- The independent-source assumption is used only in the gate-level product
  injection reduction; it is not part of the state self-testing theorem.
