#!/usr/bin/env python3
"""One-command fail-closed verification. Run from pilot/."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import agent_loop
from cross_validate import validate

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'
EVIDENCE.mkdir(exist_ok=True)
os.chdir(ROOT)
os.environ['OPENBLAS_NUM_THREADS']='1'
report={'status':'RUNNING','stages':[], 'live_llm_experiment':'NOT_RUN'}


def command(name,args,timeout=180):
    start=time.monotonic()
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    output=result.stdout+result.stderr
    (EVIDENCE/(name+'.log')).write_text(output)
    report['stages'].append({'stage':name,'returncode':result.returncode,'seconds':round(time.monotonic()-start,4)})
    print(name+': '+('PASS' if result.returncode==0 else 'FAIL'),flush=True)
    if result.returncode:
        raise RuntimeError(name+' failed:\n'+output[-6000:])
    return output


try:
    version=command('lean_version',['lean','--version'])
    if 'version 4.19.0,' not in version:
        raise RuntimeError('unexpected Lean toolchain')
    report['lean_version']=version.strip()
    sources=list((ROOT/'Pilot').glob('*.lean'))+[ROOT/'Pilot.lean',ROOT/'Audit.lean',ROOT/'ExportRows.lean']
    # Static check plus axiom audit; no native computation or introduced axioms.
    forbidden=r'\b(sorry|admit|axiom|unsafe|native_decide|implemented_by)\b|skipKernelTC'
    for path in sources:
        source=path.read_text()
        if re.search(forbidden,source):
            raise RuntimeError('forbidden token in '+str(path))
    command('clean',['lake','clean'])
    command('build',['lake','build'])
    audit=command('axioms',['lake','env','lean','Audit.lean'])
    names=['ccz_derivative0','ccz_derivative1','ccz_derivative2','derivative_flip_form',
           'choi_stabilizerZ','choi_stabilizerX','ccz_wrong_omits_factor','ccz_wrong_zero_derivative']
    report['axioms']={name:agent_loop.parse_axioms(audit,'Pilot.'+name) for name in names}
    rows=command('export',[str(ROOT/'.lake/build/bin/export_rows')])
    (EVIDENCE/'lean_rows.csv').write_text(rows)
    report['cross_validation']=validate(EVIDENCE/'lean_rows.csv')
    (EVIDENCE/'cross_validation.json').write_text(json.dumps(report['cross_validation'],indent=2)+'\n')
    command('pytest',[sys.executable,'-m','pytest','tests','../tests/test_boolean_phase_core.py','-q'])
    report['agent_smoke']={}
    for task in agent_loop.TASKS:
        trace=agent_loop.run(task,max_attempts=2,output=EVIDENCE/('agent_'+task+'.json'))
        statuses=[attempt['result']['status'] for attempt in trace['attempts']]
        expected=['LEAN_REJECTED','LEAN_REJECTED'] if task.startswith('wrong_') else ['LEAN_REJECTED','VERIFIED']
        if statuses!=expected:
            raise RuntimeError(f'{task}: expected {expected}, got {statuses}')
        report['agent_smoke'][task]={'provider':'replay_fixture','statuses':statuses}
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report['python']=sys.version
    report['status']='PASS'
except Exception as err:
    report['status']='FAIL'
    report['error']=str(err)
finally:
    (EVIDENCE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
