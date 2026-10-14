import importlib.util
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parents[1]/'scripts/cross_validate.py'
spec = importlib.util.spec_from_file_location('cross', p)
cross = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross)
u = cross.upstream


def test_full_ccz_truth_table():
    for a in range(8):
        x = cross.bits(3,a)
        for i in range(3):
            others = [x[j] for j in range(3) if j != i]
            assert u.derivative(u.ccz_phase,x,i) == others[0]*others[1]


def test_missing_factor_counterexample():
    x = (0,1,0)
    assert u.derivative(u.ccz_phase,x,0) == 0
    assert x[1] == 1


def test_zero_derivative_counterexample():
    assert u.derivative(u.ccz_phase,(0,1,1),0) == 1


def test_ordinary_difference_is_not_boolean_derivative():
    f = lambda x: x[0]
    assert u.derivative(f,(1,),0) == 1
    assert f((0,))-f((1,)) == -1


def test_missing_phase_operator_fails():
    v,_ = u.reference_state_and_stabilizers(3,u.ccz_phase)
    wrong = u.op_on(6,0,u.X) @ u.op_on(6,3,u.X)
    assert np.linalg.norm(wrong@v-v) > 0.5


def test_single_z_is_not_stabilizer():
    v,_ = u.reference_state_and_stabilizers(3,u.ccz_phase)
    assert np.linalg.norm(u.op_on(6,0,u.Z)@v-v) > 0.5


def test_endianness_bug_detected_by_asymmetric_phase():
    # CCZ is symmetric, so it cannot expose all indexing swaps.
    f = lambda x: x[0]
    assert u.derivative(f,(0,0,0),0) == 1
    assert u.derivative(f,(0,0,0),2) == 0


def test_missing_normalization_detected():
    state,_,_ = cross.sparse_images(3,128,0)
    assert sum(v*v for v in state.values()) == 8
    v,_ = u.reference_state_and_stabilizers(3,u.ccz_phase)
    assert np.isclose(np.vdot(v,v),1)


def test_derivative_phase_commutes_with_flips():
    # Swapping D and X at the same site is VALID, not a negative control.
    for table in range(256):
        f = cross.phase(3,table)
        for a in range(8):
            for i in range(3):
                b = a ^ (1 << (2-i))
                assert u.derivative(f,cross.bits(3,a),i) == u.derivative(f,cross.bits(3,b),i)
