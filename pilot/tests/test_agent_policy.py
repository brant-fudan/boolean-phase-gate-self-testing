import importlib.util
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/agent_loop.py'
spec=importlib.util.spec_from_file_location('agent',p)
agent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

@pytest.mark.parametrize('body',[
    'sorry', 'admit', 'exact Classical.choice (by sorry)',
    'native_decide', 'axiom fake : False', 'exact Pilot.ccz_d0 a b c',
    'rfl\nend Candidate\naxiom fake : False',
    'run_tac unsafeIO (IO.println "hello")',
    'simp\nset_option debug.skipKernelTC true',
])
def test_untrusted_tokens_rejected(body):
    with pytest.raises(ValueError): agent.validate_proof(body)


def test_misformalization_hash_rejected():
    result=agent.check('ccz_d0',{'statement_sha256':'changed','proof':'rfl'})
    assert result['status']=='CONTRACT_REJECTED'


def test_added_assumption_rejected():
    result=agent.check('ccz_d0',{'statement_sha256':agent.contract('ccz_d0')['statement_sha256'],
        'statement':'True','proof':'rfl'})
    assert result['status']=='CONTRACT_REJECTED'


def test_kernel_axiom_whitelist():
    assert agent.parse_axioms("'Candidate.target' depends on axioms: [propext, Quot.sound]") == ['propext','Quot.sound']
    assert agent.parse_axioms("'Candidate.target' does not depend on any axioms") == []
    with pytest.raises(ValueError):
        agent.parse_axioms("'Candidate.target' depends on axioms: [sorryAx]")
    with pytest.raises(ValueError):
        agent.parse_axioms("'Candidate.target' depends on axioms: [Lean.ofReduceBool]")
    with pytest.raises(ValueError): agent.parse_axioms('compilation successful')


def test_valid_replay_bodies_pass_policy():
    for task in agent.TASKS:
        agent.validate_proof(agent.replay(task,1)['proof'])
