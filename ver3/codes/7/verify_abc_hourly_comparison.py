"""Verify exported A/B/C payload and selector/plot logic without browser control."""
import re
import json
import gzip
import base64
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, write_json

page=(V3/'output/07/full/03_abc_comparison/summary.html').read_text(encoding='utf-8')
js=re.findall(r'<script>(.*?)</script>',page,re.S)[-1]
payload=json.loads(gzip.decompress(base64.b64decode(re.search(r"atob\('([^']+)'\)",js).group(1))))
assert len(payload['pairs'])==5866 and len({p['id'] for p in payload['pairs']})==5866
for p in payload['pairs']:
    aa,bb,cc=[payload['series'][p[t]] for t in ['a','b','c']]
    assert aa['end']==cc['end'] and aa['stn']==bb['stn']
    for e in [aa,bb,cc]:
        assert len(e['v'])==len(e['q'])==360
        assert pd.Timestamp(e['end']).hour==0 and pd.Timestamp(e['end']).minute==0
        assert all(v is None or np.isfinite(v) and v>=0 for v in e['v'])
# Compare C 360-hour totals against the previously saved linkage features.
c=pd.read_parquet(V3/'output/07/full/02_rainfall/c_rainfall_features.parquet').set_index('A_REFERENCE_ID')
for p in payload['pairs']:
    values=payload['series'][p['c']]['v'];valid=[v for v in values if v is not None];r=c.loc[p['id']]
    assert len(valid)==r.RAIN_15D_VALID_HOURS
    assert (not valid and pd.isna(r.RAIN_15D_OBSERVED_SUM_MM)) or np.isclose(sum(valid),r.RAIN_15D_OBSERVED_SUM_MM)
harness=r'''
const assert=require('node:assert/strict');global.window=global;
const elements=new Map(),plots=new Map();
function el(id){return {id,value:'',children:[],append(...v){this.children.push(...v);if(this.children.length===v.length&&v[0]?.value!==undefined)this.value=String(v[0].value);},replaceChildren(){this.children=[];this.value='';}};}
global.document={getElementById(id){if(!elements.has(id))elements.set(id,el(id));return elements.get(id);},createElement(){return el('');}};
global.Plotly={react:async(id,traces,layout)=>{plots.set(id,{traces,layout});}};
'''
checks=r'''
(async()=>{
for(let i=0;i<500&&!viewerReadyFlag();i++)await new Promise(r=>setTimeout(r,10));
assert.equal(window.viewerError,undefined);assert.equal(window.viewerReady,true);
assert.equal(viewerState.hours,15);
for(const id of ['aPlot','bPlot','cPlot'])assert.equal(plots.get(id).traces[0].y.length,15);
assert.deepEqual(plots.get('aPlot').layout.yaxis.range,plots.get('cPlot').layout.yaxis.range);
elements.get('window').value='360';await elements.get('window').onchange();
assert.equal(viewerState.hours,360);
for(const id of ['aPlot','bPlot','cPlot'])assert.equal(plots.get(id).traces[0].y.length,360);
assert.equal(plots.get('meanPlot').traces.length,3);
let options=elements.get('unit').children;elements.get('unit').value=String(options.at(-1).value);await changeUnit();
assert.ok(viewerState.pair);assert.equal(measures([0,null,2]).missing,1);assert.equal(measures([null]).sum,null);
console.log(JSON.stringify({status:'PASS',triplets:5866,all_c_360h_totals_match_saved_features:true,js_15h_360h_and_selectors:true,visual_render_verified:false}));
})().catch(e=>{console.error(e);process.exitCode=1;});
function viewerReadyFlag(){return window.viewerReady||window.viewerError;}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'test.cjs';path.write_text(harness+js+checks,encoding='utf-8')
    result=subprocess.run([shutil.which('node'),str(path)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    qc=json.loads(result.stdout.strip().splitlines()[-1])
write_json(V3/'log/07/full/03_abc_comparison/verification.json',qc)
print(json.dumps(qc))
