"""Offline checks for the hourly viewer; no source-data writes or browser control."""
import base64
import gzip
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

spec=importlib.util.spec_from_file_location('viewer',Path(__file__).with_name('04_build_hourly_comparison.py'))
viewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(viewer)
idx=pd.date_range('2024-01-01',periods=5,freq='h')
frame=pd.DataFrame(dict(VALUE=[0,5,None,6,7],USABLE=[True,True,True,False,True],
    LAT=[37,37,37,37,38],LON=[127]*5,SOURCE=['AWS','ASOS','MISSING','AWS','AWS']),index=idx)
values,status=viewer.sequence(frame,idx[0],idx[-1]+pd.Timedelta(hours=2),37,127)
assert values==[0,5,None,None,None,None]
assert status==[0,1,2,3,4,2]
html=(viewer.V3/'output/06/full/04_hourly_rainfall_comparison.html').read_text(encoding='utf-8')
scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
js=scripts[-1]
encoded=re.search(r"atob\('([^']+)'\)",js).group(1)
payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
assert len(payload['pairs'])==5878
assert len({p['stn'] for p in payload['pairs']})==298
for pair in payload['pairs']:
    for key in ['a','b']:
        event=payload['events'][pair[key]]
        assert event['stn']==pair['stn']
        assert event['lat']==pair['lat'] and event['lon']==pair['lon']
        assert len(event['v'])==len(event['q'])
        assert all(v is None or v>=0 for v in event['v'])
        assert pd.Timestamp(event['start'])<=pd.Timestamp(event['anchor'])-pd.Timedelta(days=14)
        assert pd.Timestamp(event['end'])>=pd.Timestamp(event['anchor'])+pd.Timedelta(days=1)
assert payload['pairs'] and '__PAYLOAD__' not in html and '__PLOTLY_JS__' not in html
# Execute the page's selection and trace-generation logic with DOM/Plotly stubs.
# This verifies data/JS behavior, not browser rendering or visual layout.
harness=r'''
const assert=require('node:assert/strict');
global.window=global;const elements=new Map(),plots=new Map();
function element(id){return {id,value:id==='mode'?'event':id==='scale'?'common':'',hidden:false,children:[],append(...items){this.children.push(...items);if(this.children.length===items.length&&items[0]?.value!==undefined)this.value=String(items[0].value);},replaceChildren(){this.children=[];this.value='';}};}
global.document={getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},createElement(){return element('');}};
global.Plotly={react:async(id,traces,layout)=>{plots.set(id,{traces,layout});}};
'''
checks=r'''
(async()=>{
for(let i=0;i<200&&!window.viewerReady&&!window.viewerError;i++)await new Promise(r=>setTimeout(r,10));
assert.equal(window.viewerError,undefined);assert.equal(window.viewerReady,true);
assert.equal(elements.get('station').children.length,298);
for(const stn of ['119','485','522','588']){
 elements.get('station').value=stn;await changeStation();assert.equal(viewerState.station,+stn);
 elements.get('mode').value='days';await renderPair();await histogram();
 assert.equal(viewerState.aHours,360);assert.equal(viewerState.bHours,360);
 for(const id of ['aPlot','bPlot'])assert.equal(plots.get(id).traces[0].y.length,360);
 elements.get('mode').value='event';await renderPair();await histogram();
}
console.log(JSON.stringify({status:'PASS',stations:298,pairs:5878,synthetic_missing_zero_checks:true,js_selection_and_traces:true,visual_render_verified:false}));
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
node=shutil.which('node')
assert node,'Node.js needed for JavaScript logic verification'
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'verify.cjs';path.write_text(harness+js+checks,encoding='utf-8')
    result=subprocess.run([node,str(path)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    qc=json.loads(result.stdout.strip().splitlines()[-1])
viewer.write_json(viewer.V3/'log/06/full/04_hourly_viewer/code_verification.json',qc)
print(json.dumps(qc))
