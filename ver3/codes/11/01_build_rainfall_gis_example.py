"""GIS example: actual ABC administrative boundaries, fusion grid coordinates and hourly rain."""
from pathlib import Path
import importlib.util
import gzip
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/11/full/01_rainfall_summary'
LOG=V3/'log/11/full/01_rainfall_summary'
GROUP='ARG_002370d5ed8a54f45fc8'

def main():
    start=time.time()
    samples=pd.read_parquet(OUT/'abc_fusion_rainfall.parquet')
    samples=samples[samples.A_REFERENCE_ID.eq(GROUP)].sort_values('SAMPLE_TYPE')
    assert len(samples)==3
    links=pd.read_parquet(V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet')
    grid=pd.read_parquet('/data/RAIN_DATAS/융합기상/INT/grid.parquet')
    registry=gpd.read_parquet(V3/'output/04/full/spatial_units_5186.parquet').to_crs(5186)
    spec=importlib.util.spec_from_file_location('unit_verified_decoder',V3/'codes/10/01_collect_fusion_hourly_fixed.py')
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    ids=np.sort(links.loc[links.SPATIAL_UNIT_ID.isin(samples.SPATIAL_UNIT_ID),'GRID_ID'].unique())
    cache={};source_files=[]
    dates=sorted(set(d for end in samples.END_DATE for d in pd.date_range(end-pd.Timedelta(days=14),end)))
    for num,day in enumerate(dates,1):
        for hour in range(24):
            end=day+pd.Timedelta(hours=hour+1)
            path=decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            before=[path.stat().st_size,path.stat().st_mtime_ns]
            values=decoder.decode(gzip.decompress(path.read_bytes()))[ids]
            assert before==[path.stat().st_size,path.stat().st_mtime_ns]
            source_files.append(dict(path=str(path),stamp=before))
            cache[(day,hour)]=values
        if num%5==0 or num==len(dates):print(f'GIS 예시 강우 읽기 {num}/{len(dates)}일',flush=True)
    examples=[]
    for r in samples.itertuples():
        chosen=grid[grid.GRID_ID.isin(links.loc[links.SPATIAL_UNIT_ID.eq(r.SPATIAL_UNIT_ID),'GRID_ID'])].sort_values('GRID_ID')
        polygon=registry[registry.SPATIAL_UNIT_ID.eq(r.SPATIAL_UNIT_ID)].copy()
        assert len(polygon)==1
        points=gpd.GeoSeries(gpd.points_from_xy(chosen.LON,chosen.LAT),crs=4326).to_crs(5186)
        assert points.within(polygon.geometry.iloc[0]).all()
        bounds=polygon.total_bounds;surround=box(bounds[0]-1500,bounds[1]-1500,bounds[2]+1500,bounds[3]+1500)
        nearby=registry[registry.geometry.intersects(surround)&~registry.SPATIAL_UNIT_ID.eq(r.SPATIAL_UNIT_ID)].copy()
        nearby=nearby[['SPATIAL_UNIT_ID','ADMIN_NAME','geometry']]
        nearby.geometry=nearby.geometry.simplify(10,preserve_topology=True)
        context=json.loads(nearby.to_crs(4326).to_json(drop_id=True))
        boundary=json.loads(polygon[['SPATIAL_UNIT_ID','ADMIN_NAME','geometry']].to_crs(4326).to_json(drop_id=True))
        indexes=np.searchsorted(ids,chosen.GRID_ID.to_numpy());hourly=[]
        days=list(pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE))
        for day in days:
            for hour in range(24):
                vals=cache[(day,hour)][indexes]
                hourly.append([round(float(v),4) if np.isfinite(v) and v>=0 else None for v in vals])
        if not any(v is None for row in hourly[-24:] for v in row):
            daily=sum(sum(row)/len(row) for row in hourly[-24:])
            assert abs(daily-r.FUSION_01D_MM)<1e-3
        examples.append(dict(type=r.SAMPLE_TYPE,unit=r.SPATIAL_UNIT_ID,admin=r.ADMIN_NAME,end=str(r.END_DATE.date()),
            days=[str(d.date()) for d in days],grids=chosen[['GRID_ID','LON','LAT']].to_dict('records'),
            boundary=boundary,context=context,hourly=hourly))
    payload=json.dumps(examples,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (OUT/'gis_example.html').write_text(PAGE.replace('GIS_DATA',payload),encoding='utf-8')
    (LOG/'gis_example.json').write_text(json.dumps(dict(status='COMPLETE',group=GROUP,
        examples=[dict(type=e['type'],unit=e['unit'],grid_count=len(e['grids'])) for e in examples],
        source_files=source_files,unit_rule='RAW_NETCDF_DATA_SCALE_APPLIED',
        boundary_crs_display='EPSG4326',center_selection_crs='EPSG5186',zero_imputation=False,
        background='OpenStreetMap online tiles plus embedded neighboring admin boundaries',
        elapsed_seconds=time.time()-start),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved GIS example with actual boundaries, grid coordinates and 360 hourly values per example',flush=True)

PAGE='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>행정구역·융합강우 GIS 예시</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{font:15px/1.6 "Malgun Gothic",sans-serif;margin:20px;color:#21344a;background:#f5f8fb}h1{font-size:23px;margin:0 0 12px}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}select,input,button{font:inherit;padding:5px}#map{height:510px;min-height:330px;margin:12px 0;background:#eef2f6}#coord{font-size:13px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:7px;border-bottom:1px solid #dbe3ed;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}#status{font-weight:600}.note{font-size:13px;color:#617186}.point-label{font-size:12px;background:#ffffffdd;border:0;box-shadow:none}.selected{background:#e1eefb}#trend{width:100%;height:150px;display:block}#map-error{color:#ac3434}@media(max-width:500px){body{margin:10px}#map{height:420px}th,td{padding:5px}}</style></head><body>
<h1>행정구역 경계 + 실제 융합강우 격자</h1>
<div class="controls"><label>표본 <select id="sample"></select></label><label>날짜 <select id="day"></select></label><label>시간 <input id="hour" type="range" min="0" max="23" step="1" value="0"></label><span id="clock"></span></div>
<div id="status" aria-live="polite"></div><div id="query" class="note"></div><div id="map-error"></div>
<div id="map" aria-label="실제 행정구역과 강우 격자 중심 지도"></div>
<div class="note">굵은 경계: 선택 구역 · 얇은 경계: 인접 구역 · G1~G번호: 계산에 사용한 격자 중심점 · 점의 색과 크기: 시간강우량</div>
<div id="coord" aria-live="polite">격자를 클릭하면 경도·위도와 시간강우를 확인할 수 있습니다.</div>
<div class="scroll"><table><thead><tr><th>지도 번호</th><th>격자 ID</th><th>경도 (°E)</th><th>위도 (°N)</th><th>1시간 강우(mm)</th></tr></thead><tbody id="grid-table"></tbody></table></div>
<h2 style="font-size:18px">선택 날짜의 시간별 행정구역 평균</h2><svg id="trend" role="img" aria-label="24시간 강우 평균 그래프"></svg>
<p class="note">중심점이 구역 안에 있는 격자를 모두 동일 가중 평균했습니다. 면적 가중이나 STN 강우가 아닙니다. 시간강우는 원응답 data_scale=10을 적용했습니다. 행정구역 경계의 사건 당시 유효성은 별도 검토 대상입니다.</p>
<p class="note">배경지도는 온라인 연결이 필요합니다. 격자·행정구역·강우 자료는 이 파일에 포함되어 있습니다. <a href="summary.html" target="_top">ABC 전체 요약</a> · <a href="https://leafletjs.com/examples/quick-start/">지도 표시 도구 설명</a></p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const examples=GIS_DATA;const sample=document.getElementById('sample'),day=document.getElementById('day'),hour=document.getElementById('hour');let current,outline,context,markers=[],selected=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let map=null;if(typeof L==='undefined'){document.getElementById('map-error').textContent='지도 라이브러리를 불러오지 못했습니다. 인터넷 연결 후 새로고침하세요. 좌표·강우 표는 아래에서 확인할 수 있습니다.';}else{map=L.map('map');L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);L.control.scale({imperial:false}).addTo(map);}
for(let i=0;i<examples.length;i++){const option=document.createElement('option');option.value=i;option.textContent=examples[i].type+' · '+examples[i].admin+' · 종료 '+examples[i].end;sample.appendChild(option)}
function selectSample(){current=examples[Number(sample.value)];selected=null;day.replaceChildren();current.days.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d;day.appendChild(o)});day.value=14;
if(map){if(outline)map.removeLayer(outline);if(context)map.removeLayer(context);context=L.geoJSON(current.context,{style:{color:'#9aa9b7',weight:1,fillColor:'#dfe6ee',fillOpacity:.12},onEachFeature:(f,l)=>l.bindTooltip(esc(f.properties.ADMIN_NAME))}).addTo(map);outline=L.geoJSON(current.boundary,{style:{color:'#263e5b',weight:3,fillColor:'#2196f3',fillOpacity:.06}}).addTo(map);map.fitBounds(outline.getBounds().pad(.35));}
const last=current.hourly.slice(336);const means=last.map(row=>row.includes(null)?-1:row.reduce((a,b)=>a+b,0)/row.length);hour.value=means.indexOf(Math.max(...means));draw();}
function draw(){const d=Number(day.value),h=Number(hour.value),rain=current.hourly[d*24+h],values=rain.filter(v=>v!==null),mean=values.length===rain.length?values.reduce((a,b)=>a+b,0)/values.length:null,max=Math.max(1,...values);document.getElementById('clock').textContent=String(h).padStart(2,'0')+':00–'+String(h+1).padStart(2,'0')+':00';document.getElementById('status').textContent=current.type+' · '+current.admin+' · '+current.grids.length+'개 격자 평균 '+(mean===null?'결측':mean.toFixed(3)+' mm');document.getElementById('query').textContent=current.days[d]+' '+String(h+1).padStart(2,'0')+':00의 rn_60m 원응답 (24:00은 다음 날 00:00 요청)';
if(map){markers.forEach(m=>map.removeLayer(m));markers=[];}
const table=document.getElementById('grid-table');table.replaceChildren();current.grids.forEach((g,i)=>{const v=rain[i],row=document.createElement('tr');row.dataset.index=i;for(const t of ['G'+(i+1),g.GRID_ID,g.LON.toFixed(6),g.LAT.toFixed(6),v===null?'결측':v.toFixed(3)]){const td=document.createElement('td');td.textContent=t;row.appendChild(td)}table.appendChild(row);
if(map){const marker=L.circleMarker([g.LAT,g.LON],{radius:v===null?6:6+8*Math.sqrt(v/max),color:v===null?'#777':v===0?'#768b9b':'#176abb',fillColor:v===null?'#ccc':v===0?'#edf2f7':'#2784d7',fillOpacity:v===0?.7:.4+.5*v/max,weight:2}).addTo(map);marker.bindTooltip('G'+(i+1),{permanent:true,direction:'right',className:'point-label'});marker.bindPopup('<b>G'+(i+1)+' · GRID '+g.GRID_ID+'</b><br>경도 '+g.LON.toFixed(6)+'°E<br>위도 '+g.LAT.toFixed(6)+'°N<br>'+esc(current.days[d])+' '+h+':00–'+(h+1)+':00<br>시간강우 '+(v===null?'결측':v.toFixed(3)+' mm'));marker.on('click',()=>choose(i));markers.push(marker);row.onclick=()=>{choose(i);marker.openPopup()};}
});if(selected!==null)choose(selected);trend(d,h);}
function choose(i){selected=i;const g=current.grids[i],v=current.hourly[Number(day.value)*24+Number(hour.value)][i];document.getElementById('coord').textContent='G'+(i+1)+' · 격자 '+g.GRID_ID+' · 경도 '+g.LON.toFixed(6)+'°E / 위도 '+g.LAT.toFixed(6)+'°N · '+(v===null?'결측':v.toFixed(3)+' mm');for(const row of document.getElementById('grid-table').children)row.classList.toggle('selected',Number(row.dataset.index)===i);}
function trend(d,h){const svg=document.getElementById('trend'),w=Math.max(300,svg.clientWidth),height=150,pad=46,max=Math.max(1,...current.hourly.slice(d*24,d*24+24).map(row=>row.includes(null)?0:row.reduce((a,b)=>a+b,0)/row.length)),x=i=>pad+i*(w-pad-16)/24,y=v=>120-v/max*95;svg.setAttribute('viewBox',`0 0 ${w} ${height}`);let out='<text x="0" y="15" font-size="12">mm/시간</text>';for(let j=0;j<=3;j++){const v=max*j/3;out+=`<line x1="${pad}" x2="${w-8}" y1="${y(v)}" y2="${y(v)}" stroke="#dce4ed"/><text x="${pad-5}" y="${y(v)+4}" text-anchor="end" font-size="12">${v.toFixed(1)}</text>`}for(let i=0;i<24;i++){const row=current.hourly[d*24+i],v=row.includes(null)?null:row.reduce((a,b)=>a+b,0)/row.length;if(v!==null)out+=`<rect x="${x(i)+1}" y="${y(v)}" width="${Math.max(2,(w-pad-16)/24-2)}" height="${120-y(v)}" fill="${i===h?'#ef8738':'#2580c7'}"/>`;if(i%6===0)out+=`<text x="${x(i)}" y="141" font-size="12">${i}시</text>`}svg.innerHTML=out;}
sample.onchange=selectSample;day.onchange=draw;hour.oninput=draw;window.addEventListener('resize',()=>{if(map)map.invalidateSize();draw()});selectSample();
</script></body></html>'''

if __name__=='__main__':main()
