# Release verification

This file records the checks performed for release `v1.0.0`.

Status: PASS.

Checks completed on 2026-07-28:

- exact uniform-certificate verification: PASS;
- complete public regression suite: 40 passed;
- stored certificate child-record hash validation: PASS;
- secret scan: no credential pattern found;
- large-file scan: no file above 25 MB;
- release size before Git metadata: below 1 MB;
- repository manifest validation: performed after the frozen manifest was
  generated.

The exact verifier prints
`The four determinant polynomials are nonnegative on their full domains.`
The complete entry point terminates with `verification_status=PASS`.
