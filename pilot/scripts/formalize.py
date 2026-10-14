#!/usr/bin/env python3
"""Natural-language task cards and strict reference-statement matching.
Equivalent but differently written statements require review; they are not false.
"""
import argparse
import json
import subprocess
from pathlib import Path
from agent_loop import contract, TASKS

NL={
 'ccz_d0':'对 f(a,b,c)=a∧b∧c，证明 f(0,b,c) XOR f(1,b,c)=b∧c，对所有布尔变量成立。',
 'ccz_d1':'对 f(a,b,c)=a∧b∧c，证明 f(a,0,c) XOR f(a,1,c)=a∧c。',
 'ccz_d2':'对 f(a,b,c)=a∧b∧c，证明 f(a,b,0) XOR f(a,b,1)=a∧b。',
 'wrong_factor':'检验命题：CCZ 关于第一个变量的布尔导数始终等于第二个变量。',
 'wrong_zero':'检验命题：CCZ 关于第一个变量的布尔导数恒为零。',
 'stabilizer_z':'对任意自然数 n、布尔函数 f 和 i:Fin n，未归一化整数振幅 Ψ(a,b)=δ(a,b)(-1)^f(a) 满足 Z_Ai Z_Bi Ψ=Ψ。',
 'stabilizer_x':'对任意自然数 n、布尔函数 f 和 i:Fin n，未归一化整数振幅 Ψ(a,b)=δ(a,b)(-1)^f(a) 满足 X_Ai X_Bi D_i^f Ψ=Ψ；D 的符号由布尔导数决定。',
}


def review(task,proposal):
    target=contract(task)
    fields=['import','binders','statement']
    differences=[field for field in fields if proposal.get(field)!=target[field]]
    return {'task_id':task, 'status':'REVIEW_REQUIRED' if differences else 'REFERENCE_MATCH',
            'differing_fields':differences, 'contract':target if not differences else None,
            'note':'Exact reference match is not an automatic proof of NL semantic faithfulness. Independent human review is required for new tasks.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--proposal',type=Path)
    p.add_argument('--provider-cmd-json')
    p.add_argument('--output',type=Path,default=Path('evidence/formalization_proposal.json'))
    args=p.parse_args()
    if args.provider_cmd_json:
        request={'stage':'formalize','task_id':args.task,'natural_language':NL[args.task],
                 'instruction':'Return JSON with import, binders, statement. Use Pilot.Definitions for CCZ or Pilot.Basic for stabilizers. Do not add assumptions or proofs.'}
        response=subprocess.run(json.loads(args.provider_cmd_json),input=json.dumps(request,ensure_ascii=False),
                                text=True,capture_output=True,timeout=120,check=True)
        proposal=json.loads(response.stdout)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')
        args.proposal=args.output
    if args.proposal:
        result=review(args.task,json.loads(args.proposal.read_text()))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(0 if result['status']=='REFERENCE_MATCH' else 2)
    print(json.dumps({'task_id':args.task,'natural_language':NL[args.task],
        'expected_output_fields':['import','binders','statement'],
        'conventions':'Bool; XOR; zero-based site; Bits n=Fin n→Bool; exact unnormalized Int amplitudes',
        'definition_file':'Pilot/Definitions.lean'},ensure_ascii=False,indent=2))
