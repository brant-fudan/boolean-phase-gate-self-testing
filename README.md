# Boolean-Phase Gate Self-Testing

This repository contains the compact verification code and frozen
certificates for *Boolean-Phase Gates: Robust Device-Independent
Certification*.  It follows a strict evidence boundary: the paper contains
the arbitrary-dimensional analytic self-testing and channel-composition
arguments, while this repository verifies the finite symbolic identities,
exact polynomial certificates, and validated continuous-domain cover.

The target family is

$$
U_f=\sum_{x\in\{0,1\}^n}(-1)^{f(x)}|x\rangle\langle x|,
$$

with the controlled-controlled-$Z$ gate as the principal example.

## Verified results

| Result | Verification role | Evidence |
|---|---|---|
| Bell bounds and exact Choi-state self-test | Finite checks of target identities, spectra, and reference measurements | Supplements the analytic Boolean-cube/SWAP proof |
| Uniform coefficient $3(1+\sqrt2)/8$ | Four exact determinant-polynomial certificates over $\mathbb Q(\sqrt2)$ | Exact computer-assisted local inequality |
| CCZ coefficient $(4+5\sqrt2)/16$ | Exact local Taylor--Schur certificates and outward-rounded Bernstein--Cholesky cover | Rigorous computer-assisted coupling inequality |
| Fixed-extraction tightness | Exact one-bad-pair endpoint obstruction | Exact finite check |
| Visibility thresholds and figures | Deterministic formula evaluation | Derived consequence |

The finite checks do not replace the arbitrary-dimensional analytic
arguments.  In particular, the direct-integral lift and the compatible
input--output channel composition remain proofs in the paper.

## Reproduce all checks

The frozen environment uses Python 3.11.11, NumPy 2.4.1, SymPy 1.14.0,
Matplotlib 3.8.4, and pytest 8.4.1.  From the repository root, run:

```bash
bash scripts/verify_all.sh
```

The final line must be:

```text
verification_status=PASS
```

## Repository map

- `src/boolean_phase_core.py`: Boolean derivatives, Choi targets, Bell
  operators, and exact finite-dimensional reference checks.
- `src/verify_boolean_phase_robustness.py`: exact
  $\mathbb Q(\sqrt2)$ verifier for the uniform coefficient.
- `src/ccz_stopi_exact.py`: exact CCZ numerator and local certificates.
- `src/ccz_stopi_local_certificate.py`: auxiliary coefficient and
  Taylor--Schur construction used by the exact verifier; its standalone
  floating-point design audit is not used to decide the theorem.
- `src/ccz_stopi_interval_certificate.py`: outward-rounded global cover.
- `src/ccz_stopi_optimal.py`: fixed-extraction endpoint obstruction.
- `src/generate_manuscript_results.py`: deterministic thresholds and figures.
- `results/ccz_stopi_certificate/`: frozen exact and validated certificate
  hierarchy.
- `proof/README.md`: proof--code interface and the mechanical details omitted
  from the paper.
- `tests/`: independent regression checks for the public verification
  surface.
- `verification/README.md`: frozen release-validation record.
- `MANIFEST.sha256`: hashes of every versioned release file.

The historical `stopi` filename prefix is retained for compatibility.  The
paper uses the descriptive term *affine state--operator inequality*.

## Scope

- The CCZ coefficient is optimal only for the stated fixed product extraction
  and affine state--operator inequality.
- No global optimality over all extraction channels or nonlinear tradeoffs is
  claimed.
- The validated CCZ cover is a rigorous certificate, not a numerical search.
- Source independence is needed for the product input-injection reduction at
  the gate level, not for Choi-state self-testing.

## Citation and license

Citation metadata are provided in `CITATION.cff`.  The verification software
is released under the MIT License.
