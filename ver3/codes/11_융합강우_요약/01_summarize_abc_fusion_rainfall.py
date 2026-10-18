"""Summarize ABC from unit-verified cached fusion responses; originals remain unchanged."""
from pathlib import Path
import json
import time
import html
import gzip
import importlib.util
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/11/full/01_rainfall_summary'
LOG=V3/'log/11/full/01_rainfall_summary'
INT=Path('/data/RAIN_DATAS/융합기상/INT')

RAW=Path('/data/RAIN_DATAS/원본/융합기상/정시_1시간강우/ver3_10')

def stamp(path):
    st=path.stat();return [st.st_size,st.st_mtime_ns]

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    selected=V3/'output/08/full/15_abc_selection'
    paths=[selected/n for n in ['occurrence_selected.parquet','b_event_selected.parquet','c_event_selected.parquet']]
    paths.append(V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet')
    inputs={str(p):stamp(p) for p in paths}
    decoder_path=V3/'codes/10_융합강우_수집/01_collect_fusion_hourly_fixed.py'
    inputs[str(decoder_path)]=stamp(decoder_path)
    spec=importlib.util.spec_from_file_location('verified_fusion_decoder',decoder_path)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    frames=[]
    for kind,path in zip('ABC',paths):
        f=pd.read_parquet(path,columns=['A_REFERENCE_ID','SPATIAL_UNIT_ID','ADMIN_NAME','END_DATE','STN'])
        f['SAMPLE_TYPE']=kind;f['END_DATE']=pd.to_datetime(f.END_DATE).dt.normalize();frames.append(f)
    samples=pd.concat(frames,ignore_index=True)
    assert samples.groupby('SAMPLE_TYPE').size().eq(5673).all()
    assert not samples.duplicated(['SAMPLE_TYPE','A_REFERENCE_ID']).any()
    assert samples.groupby('A_REFERENCE_ID').SAMPLE_TYPE.nunique().eq(3).all()
    a=frames[0].set_index('A_REFERENCE_ID');b=frames[1].set_index('A_REFERENCE_ID');c=frames[2].set_index('A_REFERENCE_ID')
    assert a.END_DATE.sort_index().equals(c.END_DATE.sort_index())
    assert a.SPATIAL_UNIT_ID.sort_index().equals(b.SPATIAL_UNIT_ID.sort_index())
    units=sorted(samples.SPATIAL_UNIT_ID.unique());unit_index={u:i for i,u in enumerate(units)}
    links=pd.read_parquet(paths[-1]).drop_duplicates()
    links=links[links.SPATIAL_UNIT_ID.isin(units)]
    grids=np.sort(links.GRID_ID.unique());grid_index=np.searchsorted(grids,links.GRID_ID)
    membership=csr_matrix((np.ones(len(links),dtype=np.float64),
        (links.SPATIAL_UNIT_ID.map(unit_index).to_numpy(),grid_index)),shape=(len(units),len(grids)))
    counts=np.asarray(membership.sum(axis=1)).ravel().astype(int)
    required={}
    for r in samples[['SPATIAL_UNIT_ID','END_DATE']].drop_duplicates().itertuples(index=False):
        for day in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE):
            required.setdefault(day,set()).add(unit_index[r.SPATIAL_UNIT_ID])
    daily={};source_files=[];missing_files=[]
    for num,(day,needed) in enumerate(sorted(required.items()),1):
        path=INT/'hourly'/day.strftime('%Y/%m/%d')/f'data_{day:%Y%m%d}.parquet'
        values=np.zeros((len(grids),24),dtype=np.float32)
        present=np.zeros((len(grids),24),dtype=bool);valid=present.copy()
        # Re-derive from immutable raw responses; canonical Parquet unit repair is pending.
        for hour in range(24):
            end=day+pd.Timedelta(hours=hour+1)
            raw=RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            if not raw.exists():raise FileNotFoundError(f'Unit-verified raw response missing: {raw}')
            signature=stamp(raw);source_files.append(dict(path=str(raw),stamp=signature))
            rain=decoder.decode(gzip.decompress(raw.read_bytes()))[grids]
            ok=np.isfinite(rain)&(rain>=0)
            present[:,hour]=True;valid[:,hour]=ok;values[:,hour]=np.where(ok,rain,0)
            assert signature==stamp(raw), 'Raw response changed while reading'
        sums=membership@values;nv=membership@valid.astype(np.float64);npresent=membership@present.astype(np.float64)
        for i in needed:
            n=int(counts[i]);partial=np.divide(sums[i],nv[i],out=np.full(24,np.nan),where=nv[i]>0)
            complete=(nv[i]==n)&(n>0);strict=np.where(complete,partial,np.nan)
            daily[(i,day)]=dict(strict=strict,partial=partial,valid_hours=int(complete.sum()),
                partial_hours=int((nv[i]>0).sum()),absent=int((n-npresent[i]).sum()),invalid=int((npresent[i]-nv[i]).sum()))
        if num%25==0 or num==len(required):
            progress=dict(status='READING_FUSION',done=num,total=len(required),elapsed_seconds=time.time()-started)
            (LOG/'progress.json').write_text(json.dumps(progress),encoding='utf-8')
            print(f'융합강우 ABC 점검 {num}/{len(required)}일',flush=True)
    rows=[];viewer=[];cache={}
    for r in samples.itertuples(index=False):
        i=unit_index[r.SPATIAL_UNIT_ID];key=(i,r.END_DATE)
        if key not in cache:
            days=[daily[(i,d)] for d in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE)]
            strict=np.concatenate([d['strict'] for d in days]);partial=np.concatenate([d['partial'] for d in days])
            features=dict(GRID_COUNT=int(counts[i]),ABSENT_GRID_HOURS=sum(d['absent'] for d in days),
                INVALID_GRID_HOURS=sum(d['invalid'] for d in days),NO_GRID=bool(counts[i]==0))
            for n in range(1,16):
                x=strict[-n*24:];y=partial[-n*24:];nv=int(np.isfinite(x).sum())
                features.update({f'FUSION_{n:02d}D_MM':float(x.sum()) if nv==n*24 else np.nan,
                    f'FUSION_{n:02d}D_OBSERVED_SUM_MM':float(np.nansum(x)) if nv else np.nan,
                    f'FUSION_{n:02d}D_VALID_HOURS':nv,f'FUSION_{n:02d}D_MISSING_HOURS':n*24-nv,
                    f'FUSION_{n:02d}D_PARTIAL_GRID_OBSERVED_SUM_MM':float(np.nansum(y)) if np.isfinite(y).any() else np.nan})
            features['ANY_GRID_VALID_HOURS']=int(np.isfinite(partial).sum())
            graph=[float(d['strict'].sum()) if d['valid_hours']==24 else None for d in days]
            cache[key]=(features,graph)
        f,graph=cache[key];rows.append(f)
        viewer.append(dict(group=r.A_REFERENCE_ID,type=r.SAMPLE_TYPE,admin=r.ADMIN_NAME,
            end=str(r.END_DATE.date()),grid=int(counts[i]),daily=graph,valid=f['FUSION_15D_VALID_HOURS']))
    result=pd.concat([samples.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    result['RAIN_SOURCE']='FUSION_RN_60M_ADMIN_GRID_CENTER_EQUAL_WEIGHT_MEAN'
    result['VALUE_PROVENANCE']='RAW_NETCDF_DATA_SCALE_APPLIED;CANONICAL_PARQUET_UNIT_REPAIR_PENDING'
    result['SPATIAL_RULE']='EXISTING_GRID_CENTERS_WITHIN_ADMIN;STRICT_ALL_LINKED_GRIDS_VALID_EACH_HOUR'
    result['TIME_RULE']='END_DATE_INCLUDED_PREVIOUS_1_TO_15_CALENDAR_DAYS_KST'
    assert result.FUSION_15D_VALID_HOURS.between(0,360).all()
    assert (result.FUSION_15D_VALID_HOURS+result.FUSION_15D_MISSING_HOURS).eq(360).all()
    assert result.loc[result.FUSION_15D_MISSING_HOURS.gt(0),'FUSION_15D_MM'].isna().all()
    assert all(sig==stamp(Path(p)) for p,sig in inputs.items())
    assert all(s['stamp']==stamp(Path(s['path'])) for s in source_files)
    result.to_parquet(OUT/'abc_fusion_rainfall.parquet',index=False)
    summary=[];curve={}
    maxima={k:len(pd.read_parquet(V3/p,columns=['A_REFERENCE_ID'])) for k,p in {
        'A':'output/06/full/occurrence_reference.parquet','B':'output/06/full/b_event_samples.parquet','C':'output/07/full/c_event_samples.parquet'}.items()}
    for kind,g in result.groupby('SAMPLE_TYPE'):
        summary.append(dict(표본=kind,최초전체=maxima[kind],현재선택=len(g),
            누적15일완전=int(g.FUSION_15D_MISSING_HOURS.eq(0).sum()),
            일부자료있음=int((g.ANY_GRID_VALID_HOURS.gt(0)&g.FUSION_15D_MISSING_HOURS.gt(0)).sum()),
            유효강우없음=int(g.ANY_GRID_VALID_HOURS.eq(0).sum()),격자연결없음=int(g.NO_GRID.sum()),
            저장행없음_격자시간=int(g.ABSENT_GRID_HOURS.sum()),값결측_격자시간=int(g.INVALID_GRID_HOURS.sum())))
        curve[kind]=[dict(days=n,count=int(g[f'FUSION_{n:02d}D_MM'].notna().sum()),
            mean=None if g[f'FUSION_{n:02d}D_MM'].notna().sum()==0 else float(g[f'FUSION_{n:02d}D_MM'].mean())) for n in range(1,16)]
    triples=result.assign(PASS=result.FUSION_15D_MISSING_HOURS.eq(0)).groupby('A_REFERENCE_ID').PASS.all()
    render(summary,curve,viewer,int(triples.sum()),len(triples),len(missing_files))
    report=dict(status='COMPLETE_WITH_REVIEW',samples=summary,complete_triplets=int(triples.sum()),
        selected_triplets=len(triples),missing_daily_files=missing_files,source_files=source_files,inputs=inputs,
        spatial_aggregation='equal-weight mean of existing linked grid centers; complete requires every linked grid',
        value_source='RAW_NETCDF_DATA_SCALE_APPLIED',canonical_parquet_unit_repair='PENDING',
        original_observations_unchanged=True,zero_imputation=False,elapsed_seconds=time.time()-started)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps(dict(status=report['status'],rows=len(result))),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

def render(summary,curve,viewer,complete,total,missing_files):
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>11 ABC 융합강우 요약</title>
<style>body{font:15px/1.7 "Malgun Gothic",sans-serif;background:#f4f7fb;color:#21344a;margin:28px}section{background:white;padding:22px;margin:16px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #dbe3ed;text-align:right}th:first-child,td:first-child{text-align:left}.cards{display:flex;gap:16px;flex-wrap:wrap}.cards section{flex:1;min-width:210px}.big{font-size:30px;font-weight:700}input,select,button{padding:8px}svg{width:100%;max-width:1000px}small{color:#617186}.scroll{overflow:auto}</style>
<h1>ABC · 융합 시간강우</h1><p>행정구역별 격자 평균을 시간강우로 연결하고, 이벤트 종료일 포함 이전 1~15일을 계산했습니다.</p><div class="cards">'''
    page=page.replace('<style>', '<style>:root{--viz-series-1:#1976c9;--muted:#eef2f7;--foreground:#21344a}',1)
    process=OUT/'rainfall-process.html'
    if process.exists():
        page=page.replace('<div class="cards">', '<section><details><summary>수집·집계 계산 과정</summary>'+process.read_text(encoding='utf-8')+'</details></section><div class="cards">',1)
    if (OUT/'gis_example.html').exists():
        page=page.replace('<div class="cards">', '<section><h2>행정구역·격자 경위도·시간강우 GIS 예시</h2><p><a href="gis_example.html" target="_blank">지도 크게 열기</a> · A/B 같은 공간, C 다른 공간. 날짜와 시간을 바꿔 실제 강우를 비교합니다.</p><iframe src="gis_example.html" title="실제 행정구역과 강우 격자 지도" style="width:100%;height:1000px;border:0" loading="lazy"></iframe></section><div class="cards">',1)
    page=page.replace('<h1>ABC · 융합 시간강우</h1>', '<h1>ABC · 융합 시간강우</h1><p><strong>단위 보정 완료:</strong> 원응답 data_scale=10을 적용해 재계산했습니다. RAIN_DATAS의 기존 hourly Parquet 단위 복구는 아직 미수행이며, 이 보고서는 원응답에서 재산출한 값입니다.</p>')
    for r in summary:
        page+=f'<section><h2>{r["표본"]} · '+{'A':'산사태 발생 기록','B':'같은 공간·다른 강우 이벤트','C':'다른 공간·A와 같은 날짜'}[r['표본']]+f'</h2><div class="big">{r["누적15일완전"]:,} / {r["현재선택"]:,}</div><p>15일 완전 · 최초 전체 {r["최초전체"]:,}개</p><small>부분 자료 {r["일부자료있음"]:,} · 유효 자료 없음 {r["유효강우없음"]:,}<br>격자 연결 없음 {r["격자연결없음"]:,} (유효 자료 없음에 포함)</small></section>'
    page+=f'</div><section><strong>ABC 모두 15일 완전: {complete:,} / {total:,}묶음</strong><p>읽을 일별 파일 없음: {missing_files}일. 완전 = 360시간 모두 해당 구역에 연결된 모든 격자가 유효. 부분 평균과 결측은 완전 누적량에 섞지 않았습니다.</p><p>격자는 기존 INT 격자 중심점 기준의 동일 가중 평균입니다. 행정구역 경계 픽셀의 면적 가중이나 공간 전체 확보율을 뜻하지 않습니다.</p></section>'
    page+='<section><details><summary>수집·저장·집계 기준</summary><p>rn_60m 정시 한 시각의 전국 격자를 한 번 요청합니다. 서로 다른 구역이 같은 응답을 공유하고, 이미 있는 격자·시간은 수집 때 건너뛰었습니다. 요청 시각 13:00의 값은 직전 12:00~13:00 강우입니다.</p><p>행정구역 안에 중심점이 있는 기존 INT 격자를 동일 가중 평균합니다. STN 한 곳의 강우로 대체하거나 격자 강우를 공간적으로 합산하지 않습니다. 면적 가중 평균도 아닙니다.</p><p>그 시간 평균을 24시간 더하면 일강우, 종료일 포함 이전 n일을 더하면 1~15일 누적량입니다. 종료일의 24시간을 모두 포함하므로 이벤트 종료 이후 같은 날의 강우가 포함될 수 있습니다.</p><p>원응답: RAIN_DATAS/원본/융합기상/정시_1시간강우/ver3_10/YYYY/MM/*.nc.gz<br>일별 저장소: RAIN_DATAS/융합기상/INT/hourly/YYYY/MM/DD/data_YYYYMMDD.parquet<br>현재 이 보고서는 원응답 배율을 적용해 별도로 재산출했습니다. 일별 저장소의 배율 복구는 아직 미수행입니다.</p><p><a href="https://apihub.kma.go.kr/apiList.do?seqApi=971">기상청 융합기상 공식 API 명세</a></p></details></section>'
    page+='<section><h2>그룹별 자료 점검</h2><p>격자·시간 합계는 표본별 합계입니다. 같은 자료를 공유하는 표본이 있어 고유 결측량과는 다릅니다.</p><div class="scroll">'+pd.DataFrame(summary).to_html(index=False,border=0)+'</div></section>'
    page+='<section><h2>그룹별 누적강우 평균</h2><p>각 누적기간이 완전한 표본만 포함합니다. 기간마다 포함 표본 수가 다를 수 있습니다.</p><div class="scroll">'
    rows=[]
    for n in range(15):
        row={'누적기간':f'{n+1}일'}
        for k in 'ABC':
            item=curve[k][n];row[k+' 평균(mm)']='—' if item['mean'] is None else f"{item['mean']:.2f}";row[k+' 표본수']=item['count']
        rows.append(row)
    page+=pd.DataFrame(rows).to_html(index=False,border=0)+'</div></section>'
    page+='''<section><h2>같은 ABC 묶음의 일강우 비교</h2><p>날짜는 각 표본의 종료일을 0일로 맞춥니다. B는 B 자체 종료일입니다. 빈칸은 결측이며 0mm와 구분합니다.</p><input id="search" placeholder="행정구역·날짜·묶음 ID 검색"><button id="find">검색</button> <select id="pick"></select><div id="detail"></div><svg id="chart" viewBox="0 0 960 300"></svg><div class="scroll" id="days"></div></section>
<section><a href="abc_fusion_rainfall.parquet">표본별 1~15일 누적강우·QC Parquet</a><p>B·C는 산사태 발생기록 미포착이며 확정 미발생이 아닙니다. 융합강우는 분석값으로, AWS·ASOS 원본을 덮어쓰지 않았습니다. 이 표는 융합강우만 점검하며 최종 학습 가능 수를 뜻하지 않습니다.</p></section>
<script>const rows=DATA;const groups=new Map();for(const r of rows){if(!groups.has(r.group))groups.set(r.group,[]);groups.get(r.group).push(r)}
const pick=document.getElementById('pick'),colors={A:'#dc4d4d',B:'#1976c9',C:'#279366'};
function options(q=''){pick.replaceChildren();for(const [id,rs] of groups){if(q&&!JSON.stringify(rs).toLowerCase().includes(q.toLowerCase()))continue;const a=rs.find(r=>r.type==='A');const o=document.createElement('option');o.value=id;o.textContent=a.admin+' · '+a.end+' · '+id;pick.appendChild(o)}show()}
function show(){const rs=groups.get(pick.value)||[];const detail=document.getElementById('detail');detail.replaceChildren();for(const r of rs){const p=document.createElement('p');p.style.color=colors[r.type];p.textContent=`${r.type} · ${r.admin} · 종료일 ${r.end} · ${r.grid}격자 · 완전 ${r.valid}/360시간`;detail.appendChild(p)}
const nums=rs.flatMap(r=>r.daily).filter(v=>v!==null),max=Math.max(1,...nums)*1.1;let svg='<path d="M60 20V260H930" fill="none" stroke="#9aa9b9"/>';
for(let j=0;j<=4;j++){let v=max*j/4,y=260-v/max*230;svg+=`<line x1="60" y1="${y}" x2="930" y2="${y}" stroke="#e4eaf1"/><text x="52" y="${y+4}" font-size="12" text-anchor="end">${v.toFixed(1)}</text>`}
for(let j=0;j<15;j++){let x=60+j*60;svg+=`<text x="${x}" y="284" font-size="13" text-anchor="middle">${j-14}</text>`}svg+='<text x="6" y="18" font-size="13">mm/일</text>';
for(const r of rs){let prev=null;for(let j=0;j<15;j++){const v=r.daily[j];if(v===null){prev=null;continue}let x=60+j*60,y=260-v/max*230;if(prev)svg+=`<line x1="${prev[0]}" y1="${prev[1]}" x2="${x}" y2="${y}" stroke="${colors[r.type]}" stroke-width="2"/>`;svg+=`<circle cx="${x}" cy="${y}" r="3" fill="${colors[r.type]}"/>`;prev=[x,y]}}document.getElementById('chart').innerHTML=svg;
let table='<table><tr><th>종료일 대비</th><th>A mm</th><th>B mm</th><th>C mm</th></tr>';for(let j=0;j<15;j++){table+=`<tr><td>${j-14}일</td>`;for(const k of ['A','B','C']){let r=rs.find(r=>r.type===k),v=r?.daily[j];table+=`<td>${v==null?'결측':v.toFixed(2)}</td>`}table+='</tr>'}document.getElementById('days').innerHTML=table+'</table>'}
pick.onchange=show;document.getElementById('find').onclick=()=>options(document.getElementById('search').value.trim());options();</script></html>'''
    payload=json.dumps(viewer,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page=page.replace('const rows=DATA;',f'const rows={payload};')
    (OUT/'summary.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
