#!/usr/bin/env python3
"""Independent exact sparse-action oracle, upstream matrix check, Lean CSV bridge."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import boolean_phase_core as upstream


def bits(n, k):
    return tuple((k >> (n - 1 - i)) & 1 for i in range(n))


def phase(n, table):
    # Truth-table bit k is f(binary k); site 0 is the most significant qubit.
    return lambda x: (table >> int(''.join(map(str, x)), 2)) & 1


def sparse_images(n, table, site):
    """Forward action on basis kets; independent of Lean's pullback formula."""
    d = 1 << n
    mask = 1 << (n - 1 - site)
    state, z, x = {}, {}, {}
    for a in range(d):
        value = 1 - 2*((table >> a) & 1)
        state[a, a] = value
        z[a, a] = value * (1 - 2*bits(n, a)[site])**2
        changed = a ^ mask
        derivative = ((table >> a) ^ (table >> changed)) & 1
        x[changed, changed] = value * (1 - 2*derivative)
    return state, z, x


def validate(lean_csv=None):
    summary = {'upstream_commit': '6c28d91163bc728999fef7b04ba54ae773fe58e9',
               'upstream_sha256': hashlib.sha256((ROOT/'src/boolean_phase_core.py').read_bytes()).hexdigest(),
               'ccz_cases': 0, 'exact_operator_cases': 0, 'matrix_functions': 0,
               'lean_rows_compared': 0, 'lean_bridge': 'NOT_RUN'}
    for x in itertools.product((0, 1), repeat=3):
        for site in range(3):
            expected = math.prod(x[j] for j in range(3) if j != site)
            assert upstream.derivative(upstream.ccz_phase, x, site) == expected
            summary['ccz_cases'] += 1
    cache = {}
    for n in (1, 2, 3):
        d = 1 << n
        for table in range(1 << d):
            f = phase(n, table)
            for site in range(n):
                state, z, x = sparse_images(n, table, site)
                assert state == z == x
                cache[n, table, site] = (state, z, x)
                summary['exact_operator_cases'] += 1
                for a in range(d):
                    changed = a ^ (1 << (n-1-site))
                    expected = ((table >> a) ^ (table >> changed)) & 1
                    assert upstream.derivative(f, bits(n, a), site) == expected
        # Exhaust n<=2; a fixed transparent sample at n=3, including CCZ table 128.
        tables = range(1 << d) if n < 3 else [0, 1, 7, 23, 42, 85, 127, 128, 170, 213, 254, 255]
        for table in tables:
            v, ops = upstream.reference_state_and_stabilizers(n, phase(n, table))
            assert np.isclose(np.linalg.norm(v), 1)
            for site in range(n):
                exact = cache[n, table, site]
                for values, numeric in zip(exact, [v, ops[2*site]@v, ops[2*site+1]@v]):
                    dense = np.array([values.get((a,b),0) for a in range(d) for b in range(d)])
                    np.testing.assert_allclose(dense/math.sqrt(d), numeric, atol=1e-12, rtol=0)
            summary['matrix_functions'] += 1
    if lean_csv is not None:
        seen = set()
        with open(lean_csv, newline='') as stream:
            for row in csv.DictReader(stream):
                n,t,i,a,b,der,amp,z,x = [int(row[k]) for k in
                    ('n','table','site','a','b','derivative','amplitude','z_image','x_image')]
                key = (n,t,i,a,b)
                assert key not in seen, ('duplicate', key)
                seen.add(key)
                assert n in (1,2,3) and 0 <= i < n and 0 <= a < 2**n and 0 <= b < 2**n
                exact = cache[n,t,i]
                assert [amp,z,x] == [v.get((a,b),0) for v in exact], key
                assert der == upstream.derivative(phase(n,t), bits(n,b), i), key
        expected_rows = sum((1 << (1 << n))*n*(1 << (2*n)) for n in (1,2,3))
        assert len(seen) == expected_rows, (len(seen), expected_rows)
        summary['lean_rows_compared'] = len(seen)
        summary['lean_bridge'] = 'PASS'
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lean-csv', type=Path)
    p.add_argument('--output', type=Path, default=Path('evidence/cross_validation.json'))
    args = p.parse_args()
    result = validate(args.lean_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
