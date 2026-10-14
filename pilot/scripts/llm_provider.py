#!/usr/bin/env python3
"""Optional chat-completions-compatible HTTP adapter; no API key in source/logs.
Not exercised in the initial pilot. Endpoint/model selection belongs to the user.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

request=json.load(sys.stdin)
endpoint=os.environ['LLM_ENDPOINT']
if not endpoint.startswith('https://'):
    raise ValueError('LLM_ENDPOINT must use HTTPS')
contract=request.get('contract',{})
root=Path(__file__).resolve().parents[1]
# Send definitions and permitted lemmas, never the target solution module.
context=(root/'Pilot/Definitions.lean').read_text()
if contract.get('import')=='Pilot.Basic':
    context+='\n'+(root/'Pilot/Basic.lean').read_text()
system='Prove the supplied Lean 4.19.0 theorem. Return a JSON object containing only proof (the body after by) and statement_sha256. Use only permitted identifiers. No markdown fences. Do not change the statement.'
if request.get('stage')=='formalize':
    system='Translate the natural-language task to Lean 4.19.0. Return JSON containing import, binders, statement only. No proof or markdown. Preserve all quantifiers; add no hypotheses. Use the supplied definitions.'
payload={'model':os.environ['LLM_MODEL'], 'messages':[
    {'role':'system','content':system+'\n'+context},
    {'role':'user','content':json.dumps(request,ensure_ascii=False)}],
    'max_tokens':2048}
http=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={
    'Content-Type':'application/json','Authorization':'Bearer '+os.environ['LLM_API_KEY']})
with urllib.request.urlopen(http,timeout=100) as response:
    result=json.load(response)
proposal=json.loads(result['choices'][0]['message']['content'])
# Metadata goes to a separate local file only if explicitly requested.
if os.environ.get('LLM_USAGE_LOG'):
    with open(os.environ['LLM_USAGE_LOG'],'a') as log:
        log.write(json.dumps({'model':payload['model'],'usage':result.get('usage'),
                              'response_id':result.get('id')})+'\n')
print(json.dumps(proposal))
