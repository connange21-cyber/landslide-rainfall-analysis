"""Damage area vs period rainfall / maximum hourly administrative mean, A samples only."""
from pathlib import Path
import importlib.util
import gzip
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from plotly.offline import get_plotlyjs

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/12/full/03_damage_rainfall'
LOG=V3/'log/12/full/03_damage_rainfall'
PERIODS=[1,3,5,7,10,15]

def signature(path):
    st=path.stat();return [st.st_size,st.st_mtime_ns]

def area_qc(row):
    area=pd.to_numeric(row.DAMAGE_HA,errors='coerce')
    if pd.notna(area) and np.isfinite(area) and area>0:return float(area)*10000,'VALID_AREA_HA_TO_M2'
    volume=pd.to_numeric(row.DAMAGE_VOLUME_M3,errors='coerce')
    if pd.notna(volume) and np.isfinite(volume) and volume>0:return np.nan,'EXCLUDED_VOLUME_ONLY'
    if pd.notna(area) and area==0:return np.nan,'EXCLUDED_ZERO_AREA'
    if pd.notna(area) and (not np.isfinite(area) or area<0):return np.nan,'EXCLUDED_INVALID_AREA'
    return np.nan,'EXCLUDED_NO_AREA'

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    paths=[V3/'output/11/full/01_rainfall_summary/abc_fusion_rainfall.parquet',
        V3/'output/08/full/15_abc_selection/occurrence_selected.parquet',
        V3/'output/01/full/landslide_station_links.parquet',
        V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet',
        V3/'codes/10_융합강우_수집/01_collect_fusion_hourly_fixed.py']
    inputs={str(p):signature(p) for p in paths}
    rain=pd.read_parquet(paths[0]);rain=rain[rain.SAMPLE_TYPE.eq('A')].copy();assert len(rain)==5673
    assert rain.VALUE_PROVENANCE.str.startswith('RAW_NETCDF').all()
    a=pd.read_parquet(paths[1],columns=['A_REFERENCE_ID','LANDSLIDE_EVENT_IDS'])
    events=pd.read_parquet(paths[2],columns=['EVENT_ID','SOURCE','DAMAGE_HA','DAMAGE_VOLUME_M3','DAMAGE_RAW'])
    assert events.EVENT_ID.is_unique
    normalized=[area_qc(r) for r in events.itertuples(index=False)]
    events['AREA_M2']=[x[0] for x in normalized];events['AREA_STATUS']=[x[1] for x in normalized]
    membership=a.explode('LANDSLIDE_EVENT_IDS').rename(columns={'LANDSLIDE_EVENT_IDS':'EVENT_ID'})
    membership=membership.drop_duplicates(['A_REFERENCE_ID','EVENT_ID']).merge(events,on='EVENT_ID',how='left',validate='many_to_one')
    assert membership.SOURCE.notna().all()
    records=[]
    for group,g in membership.groupby('A_REFERENCE_ID',sort=False):
        valid=g.AREA_M2.notna();total=int(valid.sum())
        records.append(dict(A_REFERENCE_ID=group,DAMAGE_AREA_M2=float(g.loc[valid,'AREA_M2'].sum()) if total else np.nan,
            DAMAGE_RECORDS=len(g),AREA_VALID_RECORDS=total,AREA_EXCLUDED_RECORDS=int((~valid).sum()),
            DAMAGE_AREA_STATUS='NO_VALID_AREA' if total==0 else 'PARTIAL_RECORDED_AREA' if total<len(g) else 'COMPLETE_RECORDED_AREA',
            AREA_EVENT_IDS=g.loc[valid,'EVENT_ID'].tolist(),EXCLUDED_AREA_EVENT_IDS=g.loc[~valid,'EVENT_ID'].tolist()))
    table=rain.merge(pd.DataFrame(records),on='A_REFERENCE_ID',validate='one_to_one')
    used=table[table.DAMAGE_AREA_M2.notna()].copy()
    units=sorted(used.SPATIAL_UNIT_ID.unique());unit_index={u:i for i,u in enumerate(units)}
    links=pd.read_parquet(paths[3]);links=links[links.SPATIAL_UNIT_ID.isin(units)].drop_duplicates()
    grids=np.sort(links.GRID_ID.unique())
    matrix=csr_matrix((np.ones(len(links)),(links.SPATIAL_UNIT_ID.map(unit_index).to_numpy(),np.searchsorted(grids,links.GRID_ID))),shape=(len(units),len(grids)))
    counts=np.asarray(matrix.sum(axis=1)).ravel()
    required={}
    for r in used[['SPATIAL_UNIT_ID','END_DATE']].drop_duplicates().itertuples(index=False):
        for day in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE):required.setdefault(day,set()).add(unit_index[r.SPATIAL_UNIT_ID])
    spec=importlib.util.spec_from_file_location('plot_decoder',paths[4]);decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    daily={};sources=[]
    for num,(day,needed) in enumerate(sorted(required.items()),1):
        values=[];valids=[]
        for hour in range(24):
            end=day+pd.Timedelta(hours=hour+1);path=decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            before=signature(path);v=decoder.decode(gzip.decompress(path.read_bytes()))[grids]
            valid=np.isfinite(v)&(v>=0);values.append(np.where(valid,v,0));valids.append(valid)
            assert before==signature(path);sources.append(dict(path=str(path),stamp=before))
        sums=matrix@np.stack(values,axis=1);numbers=matrix@np.stack(valids,axis=1).astype(float)
        for i in needed:
            avg=np.divide(sums[i],numbers[i],out=np.full(24,np.nan),where=numbers[i]>0)
            avg[(numbers[i]!=counts[i])|(counts[i]==0)]=np.nan
            daily[(i,day)]=avg
        if num%25==0 or num==len(required):
            print(f'12 피해·강우 점검 {num}/{len(required)}일',flush=True)
            (LOG/'progress.json').write_text(json.dumps(dict(status='CALCULATING_HOURLY_MAX',done=num,total=len(required))),encoding='utf-8')
    maximum=[];cache={}
    for r in used.itertuples(index=False):
        key=(unit_index[r.SPATIAL_UNIT_ID],r.END_DATE)
        if key not in cache:
            sequence=np.concatenate([daily[(key[0],day)] for day in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE)])
            f={}
            for n in PERIODS:
                x=sequence[-n*24:];nv=int(np.isfinite(x).sum())
                f[f'MAX_HOURLY_{n:02d}D_MM_H']=float(x.max()) if nv==n*24 else np.nan
                f[f'CHECK_SUM_{n:02d}D_MM']=float(x.sum()) if nv==n*24 else np.nan
            cache[key]=f
        maximum.append(cache[key])
    features=pd.DataFrame(maximum,index=used.index)
    for n in PERIODS:
        assert np.allclose(features[f'CHECK_SUM_{n:02d}D_MM'],used[f'FUSION_{n:02d}D_MM'],equal_nan=True,rtol=1e-6,atol=.001)
    used=pd.concat([used,features.drop(columns=[c for c in features if c.startswith('CHECK_SUM_')])],axis=1)
    table=table.merge(used[['A_REFERENCE_ID']+[f'MAX_HOURLY_{n:02d}D_MM_H' for n in PERIODS]],on='A_REFERENCE_ID',how='left',validate='one_to_one')
    assert inputs=={str(p):signature(p) for p in paths}
    assert all(s['stamp']==signature(Path(s['path'])) for s in sources)
    table.to_parquet(OUT/'damage_rainfall.parquet',index=False)
    event_used=events[events.EVENT_ID.isin(membership.EVENT_ID)]
    event_used.to_parquet(LOG/'event_area_qc.parquet',index=False)
    report=dict(status='COMPLETE',a_samples=len(table),area_valid_samples=len(used),area_excluded_samples=int(table.DAMAGE_AREA_M2.isna().sum()),
        partial_area_samples=int(used.DAMAGE_AREA_STATUS.eq('PARTIAL_RECORDED_AREA').sum()),
        unique_area_event_status_counts=event_used.AREA_STATUS.value_counts().to_dict(),
        periods=PERIODS,point_unit='A_ADMIN_RAIN_GROUP;SUM_OF_KNOWN_EVENT_AREA_M2',
        hourly_max_rule='MAX_OF_HOURLY_EQUAL_WEIGHT_ADMIN_GRID_MEANS_IN_SELECTED_CALENDAR_WINDOW',
        sources=sources,inputs=inputs,original_data_unchanged=True,elapsed_seconds=time.time()-started)
    render(used,report)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps(dict(status='COMPLETE',samples=len(used))),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','inputs']},ensure_ascii=False),flush=True)

def render(table,report):
    data=[]
    for r in table.itertuples(index=False):
        data.append(dict(id=r.A_REFERENCE_ID,admin=r.ADMIN_NAME,date=str(pd.Timestamp(r.END_DATE).date()),
            area=round(float(r.DAMAGE_AREA_M2),3),partial=r.DAMAGE_AREA_STATUS=='PARTIAL_RECORDED_AREA',
            events=int(r.AREA_VALID_RECORDS),excluded=int(r.AREA_EXCLUDED_RECORDS),
            cumulative={str(n):None if pd.isna(getattr(r,f'FUSION_{n:02d}D_MM')) else float(getattr(r,f'FUSION_{n:02d}D_MM')) for n in PERIODS},
            maximum={str(n):None if pd.isna(getattr(r,f'MAX_HOURLY_{n:02d}D_MM_H')) else float(getattr(r,f'MAX_HOURLY_{n:02d}D_MM_H')) for n in PERIODS}))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>12 피해면적·강우 분포</title>
<style>body{font:15px/1.7 "Malgun Gothic",sans-serif;background:#f5f8fb;color:#21344a;margin:28px}.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;position:sticky;top:0;background:#f5f8fb;padding:12px 0;z-index:5}select,input{font:inherit}.chart{height:520px;background:white;margin:16px 0}.note{color:#607083;font-size:13px}details{margin:14px 0}h1{font-size:24px}</style>
<h1>피해면적과 강우 분포 · A 발생 표본</h1><p id="counts"></p>
<div class="controls"><label>기간 <select id="period"><option value="1">1일</option><option value="3">3일</option><option value="5">5일</option><option value="7">7일</option><option value="10">10일</option><option value="15" selected>15일</option></select></label>
<label><input type="checkbox" id="logx"> 피해면적 로그축</label><label><input type="checkbox" id="complete-area"> 면적 기록이 모두 있는 묶음만</label><span id="plotted"></span></div>
<div id="cumulative" class="chart"></div><div id="maximum" class="chart"></div>
<p class="note">점에 마우스를 올리면 행정구역·종료일·면적·강우를 확인할 수 있습니다. 드래그로 확대하고 더블클릭으로 전체 범위로 돌아갑니다.</p>
<details><summary>자료·단위·제외 기준</summary><p>한 점은 현재 A의 행정구역–강우 이벤트 묶음입니다. 포함된 사건 기록의 피해면적 중 확인된 값을 합산합니다. EVENT_ID는 묶음 안에서 중복 제거합니다. 일부 기록의 면적이 없으면 ‘부분 면적’으로 구분합니다. 서로 다른 묶음에 같은 사건이 연결될 수 있으므로 점들의 면적 합계를 전국 피해면적으로 해석하지 않습니다.</p><p>사건 마스터 DAMAGE_HA(ha)를 ×10,000으로 m²로 변환했습니다. 면적 없음·0·음수·부피(m³)만 있는 기록은 제외하며 부피를 면적으로 환산하지 않습니다. 같은 기록에 확인된 면적과 부피가 모두 있으면 면적만 사용합니다.</p><p>기간은 이벤트 종료일 전체를 포함한 이전 n일입니다. 강우는 원응답 NetCDF data_scale를 적용한 행정구역 격자 평균입니다. 최대 시우량은 각 시간의 행정구역 평균 중 선택 기간 내 최댓값(mm/h)입니다. 격자 하나의 최대값이나 STN 관측 최대값이 아닙니다. 결측이 있으면 완전 누적/최대값으로 표시하지 않습니다.</p><p>B/C는 실제 피해면적이 없는 비교 표본이므로 표시하지 않았습니다. RAIN_DATAS 원본과 이전 단계 출력은 변경하지 않았습니다.</p></details>
<p><a href="damage_rainfall.parquet">면적·강우·QC 데이터</a></p><script>PLOTLY_JS</script><script>
const rows=DATA,report=REPORT;
document.getElementById('counts').textContent=`A 전체 ${report.a_samples.toLocaleString()}개 중 면적 연결 ${report.area_valid_samples.toLocaleString()}개 · 면적 없음 ${report.area_excluded_samples}개 제외 · 부분 면적 ${report.partial_area_samples}개`;
function draw(){const n=document.getElementById('period').value,log=document.getElementById('logx').checked,full=document.getElementById('complete-area').checked;
const chosen=rows.filter(r=>!full||!r.partial),colors={false:'#2376b9',true:'#dc9134'};let counts=[];
for(const [target,key,title,unit] of [['cumulative','cumulative','누적강우량','mm'],['maximum','maximum','최대 시우량','mm/h']]){
 const traces=[];let validCount=0;
 for(const partial of [false,true]){const items=chosen.filter(r=>r.partial===partial&&r[key][n]!==null);validCount+=items.length;
 traces.push({type:'scattergl',mode:'markers',name:partial?'부분 면적':'면적 기록 완전',x:items.map(r=>r.area),y:items.map(r=>r[key][n]),customdata:items.map(r=>[r.admin,r.date,r.id,r.events,r.excluded]),marker:{color:colors[partial],size:6,opacity:.5},hovertemplate:'%{customdata[0]}<br>종료일 %{customdata[1]}<br>피해면적 %{x:,.1f} m²<br>'+title+' %{y:.3f} '+unit+'<br>면적 기록 %{customdata[3]}건 · 제외 %{customdata[4]}건<br>%{customdata[2]}<extra></extra>'});}
 const layout={title:{text:'피해면적 × '+n+'일 '+title},xaxis:{title:{text:'기록된 피해면적 합계 (m²)'},type:log?'log':'linear',rangemode:log?undefined:'tozero'},yaxis:{title:{text:n+'일 '+title+' ('+unit+')'},rangemode:'tozero'},margin:{l:85,r:25,t:65,b:75},legend:{orientation:'h',y:1.12},hovermode:'closest',paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff'};
 Plotly.react(target,traces,layout,{responsive:true,displaylogo:false});counts.push(validCount);
}
document.getElementById('plotted').textContent=`누적 ${counts[0].toLocaleString()}점 · 최대 시우량 ${counts[1].toLocaleString()}점`;
}
for(const id of ['period','logx','complete-area'])document.getElementById(id).onchange=draw;draw();
</script></html>'''
    # The local bundled Plotly code keeps the report usable without network access.
    page=page.replace('PLOTLY_JS',get_plotlyjs()).replace('const rows=DATA,report=REPORT;',
        'const rows='+json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')+',report='+json.dumps({k:v for k,v in report.items() if k not in ['sources','inputs']},ensure_ascii=False)+';')
    (OUT/'summary.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
