#!/usr/bin/env python3
"""Bounded proof-body agent. External providers receive JSON on stdin.

The replay provider is a deterministic integration fixture, NOT an LLM experiment.
The current closed vocabulary is an educational constraint, not an OS sandbox.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'ccz_d0': ('Pilot.Definitions', '(a b c : Bool)', 'Bool.xor (ccz false b c) (ccz true b c) = (b && c)'),
    'ccz_d1': ('Pilot.Definitions', '(a b c : Bool)', 'Bool.xor (ccz a false c) (ccz a true c) = (a && c)'),
    'ccz_d2': ('Pilot.Definitions', '(a b c : Bool)', 'Bool.xor (ccz a b false) (ccz a b true) = (a && b)'),
    'wrong_factor': ('Pilot.Definitions', '(a b c : Bool)', 'Bool.xor (ccz false b c) (ccz true b c) = b'),
    'wrong_zero': ('Pilot.Definitions', '(a b c : Bool)', 'Bool.xor (ccz false b c) (ccz true b c) = false'),
    'stabilizer_z': ('Pilot.Basic', '{n : Nat} (f : Bits n → Bool) (i : Fin n)', 'stabilizerZ i (choi f) = choi f'),
    'stabilizer_x': ('Pilot.Basic', '{n : Nat} (f : Bits n → Bool) (i : Fin n)', 'stabilizerX f i (choi f) = choi f'),
}
# Fixed reviewed imports and definitions; the target theorem is not imported.
VOCAB = set('''by cases decide rfl simp simpa only exact funext by_cases subst have
  fun using intro rw apply congrArg false true Bool xor ccz a b c n f i h hf hh
  stabilizerZ stabilizerX choi sign sign_square sign_cancel_left flip flip_flip
  derivative derivative_flip_form flip_injective if_pos not_not Eq refl
  nonexistent_ccz_lemma'''.split())
ALLOWED_AXIOMS = {'propext', 'Classical.choice', 'Quot.sound'}


def contract(task):
    imp,binders,statement = TASKS[task]
    payload = f'{imp}\n{binders}\n{statement}'
    return {'task_id':task, 'import':imp, 'binders':binders, 'statement':statement,
            'statement_sha256':hashlib.sha256(payload.encode()).hexdigest()}


def validate_proof(body):
    if not isinstance(body,str) or not body.strip() or len(body)>12000:
        raise ValueError('proof must be nonempty text <=12000 characters')
    # Closed language: no macros, commands, comments, quoted names, strings, options,
    # unicode-escaped identifiers, arbitrary imports, eval, axioms, or declarations.
    tokens = re.findall(r'[A-Za-z_][A-Za-z_0-9]*',body)
    unknown = set(tokens)-VOCAB
    if unknown:
        raise ValueError('unapproved proof identifiers: '+','.join(sorted(unknown)))
    residual = re.sub(r'[A-Za-z_][A-Za-z_0-9]*','',body)
    if re.search(r'[^\s()\[\]{},.:=<>;|!0-9·→≠]',residual):
        raise ValueError('unapproved proof punctuation')
    if '--' in body or '/-' in body or '#' in body or '"' in body or '`' in body:
        raise ValueError('commands/comments/quoted identifiers are not permitted')
    return body


def parse_axioms(output, theorem='Candidate.target'):
    if re.search(r"'"+re.escape(theorem)+r"' does not depend on any axioms",output):
        return []
    match = re.search(r"'"+re.escape(theorem)+r"' depends on axioms:\s*\[([^\]]*)\]",output,re.S)
    if not match:
        raise ValueError('missing axiom audit')
    axioms = [a.strip() for a in match.group(1).split(',') if a.strip()]
    if set(axioms)-ALLOWED_AXIOMS:
        raise ValueError('unapproved axioms: '+','.join(axioms))
    return axioms


def check(task,proposal,timeout=20):
    expected = contract(task)
    if proposal.get('statement_sha256') != expected['statement_sha256']:
        return {'status':'CONTRACT_REJECTED','feedback':'original statement hash required'}
    # Providers are never allowed to replace the statement or add hypotheses.
    if set(proposal)-{'statement_sha256','proof'}:
        return {'status':'CONTRACT_REJECTED','feedback':'only proof and statement_sha256 are allowed'}
    try:
        proof = validate_proof(proposal.get('proof'))
    except ValueError as err:
        return {'status':'POLICY_REJECTED','feedback':str(err)}
    source = (f"import {expected['import']}\nopen Pilot\nnamespace Candidate\n"
              f"set_option maxHeartbeats 200000\n"
              f"theorem target {expected['binders']} : {expected['statement']} := by\n"+
              '\n'.join('  '+line for line in proof.splitlines())+
              '\nend Candidate\n#print axioms Candidate.target\n')
    with tempfile.TemporaryDirectory(prefix='lean-agent-') as directory:
        path=Path(directory)/'Candidate.lean'; path.write_text(source)
        try:
            result=subprocess.run(['lake','env','lean',str(path)],cwd=ROOT,
                                  capture_output=True,text=True,timeout=timeout)
        except FileNotFoundError:
            return {'status':'ENVIRONMENT_ERROR','feedback':'lake is not installed'}
        except subprocess.TimeoutExpired:
            return {'status':'TIMEOUT','feedback':f'compiler timeout {timeout}s'}
    output=result.stdout+result.stderr
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'feedback':output[-16000:], 'source':source}
    if result.returncode:
        # Syntax/type failure does not imply mathematical falsity.
        status='ENVIRONMENT_ERROR' if any(s in output for s in
            ['failed to locate application','could not detect','unknown module prefix']) else 'LEAN_REJECTED'
        return {**report,'status':status}
    try:
        axioms=parse_axioms(output)
    except ValueError as err:
        return {**report,'status':'AUDIT_REJECTED','feedback':str(err)+'\n'+output}
    if 'declaration uses' in output:
        return {**report,'status':'AUDIT_REJECTED'}
    return {**report,'status':'VERIFIED','axioms':axioms}


def replay(task,attempt):
    if attempt==0:
        proof='exact nonexistent_ccz_lemma'
    elif task.startswith('stabilizer_'):
        # Test fixture replays a curated reference proof, explicitly not a model sample.
        text=(ROOT/'Pilot/Stabilizers.lean').read_text()
        name='choi_stabilizerZ' if task.endswith('_z') else 'choi_stabilizerX'
        chunk=text.split('theorem '+name,1)[1].split(' := by\n',1)[1]
        proof=chunk.split('\ntheorem ',1)[0].split('\n--',1)[0].split('\nend Pilot',1)[0].strip('\n')
        proof='\n'.join(line[2:] if line.startswith('  ') else line for line in proof.splitlines())
    else:
        proof='cases a <;> cases b <;> cases c <;> decide'
    return {'statement_sha256':contract(task)['statement_sha256'],'proof':proof}


def run(task,provider_cmd=None,max_attempts=3,timeout=20,output=None):
    if max_attempts<1 or max_attempts>20:
        raise ValueError('max_attempts must be 1..20')
    trace={'task':contract(task),'provider':'external' if provider_cmd else 'replay_fixture',
           'attempts':[], 'status':'UNRESOLVED'}
    for attempt in range(max_attempts):
        request={'contract':contract(task),'attempt':attempt,
                 'feedback':trace['attempts'][-1]['result'] if trace['attempts'] else None,
                 'instruction':'Return JSON with statement_sha256 and proof only. Preserve the theorem.',
                 'allowed_identifiers':sorted(VOCAB)}
        start=time.monotonic()
        try:
            if provider_cmd:
                response=subprocess.run(provider_cmd,input=json.dumps(request),text=True,
                                        capture_output=True,timeout=120,check=True)
                proposal=json.loads(response.stdout)
            else:
                proposal=replay(task,attempt)
            result=check(task,proposal,timeout)
        except (subprocess.SubprocessError,ValueError,OSError) as err:
            proposal=None
            result={'status':'PROVIDER_ERROR','feedback':type(err).__name__+': '+str(err)[:2000]}
        trace['attempts'].append({'number':attempt+1,'request':request,'proposal':proposal,
                                 'result':result,'seconds':round(time.monotonic()-start,4)})
        if result['status'] in ('VERIFIED','ENVIRONMENT_ERROR'):
            trace['status']=result['status']; break
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
    return trace


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,default='ccz_d0')
    p.add_argument('--provider-cmd-json',help='JSON array, e.g. ["python", "scripts/llm_provider.py"]')
    p.add_argument('--max-attempts',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('evidence/agent_trace.json'))
    args=p.parse_args()
    cmd=json.loads(args.provider_cmd_json) if args.provider_cmd_json else None
    trace=run(args.task,cmd,args.max_attempts,output=args.output)
    print(json.dumps({'task':args.task,'provider':trace['provider'],'status':trace['status'],
                      'attempts':len(trace['attempts'])},indent=2))
    raise SystemExit(0 if trace['status']=='VERIFIED' else 2)
