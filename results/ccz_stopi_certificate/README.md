# CCZ robust self-testing certificate

This directory contains the machine-readable certificate for

```text
K_CCZ - I
  + ((4 + 5 sqrt(2))/16) (6 sqrt(2) I - B_CCZ) >= 0
```

under the fixed product Kaniewski extraction used in the manuscript.

## Recorded proof coverage

- 20 exact all-ideal branch orbits, covering 64 physical low/high branches;
- 16 exact branches adjacent to the canonical one-bad-pair endpoint;
- 20 validated global branch-orbit records, again covering all 64 branches;
- 17,836 audited dyadic cells;
- 8,818 Bernstein--Cholesky leaves;
- 110 exact-local Schur handoffs;
- zero center-Weyl leaves and zero unresolved cells;
- maximum subdivision depth 61;
- minimum verified Bernstein margin
  `3.493158352255358e-09`.

The exact local calculations use arithmetic in `Q(sqrt(2))`, pivoted
`LDL^T`, and rational projected kernel/complement bounds.  The global
calculation starts from exact Pauli coefficients, propagates outward
operator-norm errors through every Bernstein restriction, and accepts a
matrix only after replaying a Cholesky residual bound.  Floating eigenvalues
only propose shifts; they do not decide positivity.

## Files

- `local_exact_certificate.json`: aggregate exact-local record and hashes;
- `endpoint_local_exact_certificate.json`: canonical endpoint aggregate;
- `ideal_local_orbits/`: 20 exact ideal-orbit records;
- `endpoint_local_branches/`: 16 exact endpoint-branch records;
- `global_interval_certificate.json`: aggregate global record and hashes;
- `global_orbits/`: 20 validated global orbit records.

## Replay

From the repository root, one global orbit can be recomputed with

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python Paper/V3.2/src/ccz_stopi_interval_certificate.py \
  --orbit-index 0 --maximum-depth 220 --maximum-nodes 500000
```

One exact ideal orbit and one endpoint branch can be recomputed with

```bash
python Paper/V3.2/src/ccz_stopi_exact.py \
  --ideal-orbit-index 0 --output /tmp/ideal_orbit_0.json

python Paper/V3.2/src/ccz_stopi_exact.py \
  --endpoint-branch-index 0 --output /tmp/endpoint_branch_0.json
```

The regression suite independently checks the exact target reconstruction,
the numerator polynomial, ideal and endpoint Schur bounds, dyadic local
handoffs, exact Bernstein restriction, input-error accounting, and verified
Cholesky floors:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  Paper/V3.2/tests/test_ccz_stopi_exact.py \
  Paper/V3.2/tests/test_ccz_stopi_interval_certificate.py
```

The stored records can also be re-aggregated without recomputing the
continuous cover:

```bash
PYTHONPATH=Paper/V3.2/src python -c "
from ccz_stopi_interval_certificate import aggregate_orbit_records
from ccz_stopi_exact import (
    aggregate_exact_endpoint_branch_records,
    aggregate_exact_local_records,
)
g = aggregate_orbit_records(
    'Paper/V3.2/results/ccz_stopi_certificate/global_orbits',
    '/tmp/ccz_global_replay.json',
)
e = aggregate_exact_endpoint_branch_records(
    'Paper/V3.2/results/ccz_stopi_certificate/endpoint_local_branches',
    '/tmp/ccz_endpoint_replay.json',
)
l = aggregate_exact_local_records(
    'Paper/V3.2/results/ccz_stopi_certificate/ideal_local_orbits',
    '/tmp/ccz_endpoint_replay.json',
    '/tmp/ccz_local_replay.json',
)
print(g['status'], g['unresolved_cells'], g['physical_branches_covered_by_symmetry'])
print(l['status'], l['ideal_physical_branches'], e['covered_adjacent_branches'])
"
```

The expected output is
`passed_global_interval_certificate 0 64` followed by
`passed_exact_local_certificate 64 16`.

The endpoint obstruction proves that the coefficient is necessary for this
fixed extraction and affine state-operator inequality.  The certificate does
not claim global extractability optimality over all channels or nonlinear
tradeoffs.
