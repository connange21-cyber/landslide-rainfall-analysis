"""Audit B/C 15-day rainfall gaps: absent rows, stored empty values, and location exclusions."""
import sys
import re
import json
import gzip
import base64
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def classify(q,aws_present,asos_present,aws,asos):
    reason=np.full(len(q),'VALID_OBSERVATION',dtype=object)
    missing=q.eq(2).to_numpy()
    present=aws_present|asos_present
    negative=aws.lt(0)|asos.lt(0)
    nonfinite=(aws.notna()&~np.isfinite(aws))|(asos.notna()&~np.isfinite(asos))
    reason[missing&~present]='NO_OBSERVATION_ROW'
    reason[missing&present]='STORED_NULL_VALUE'
    reason[missing&present&nonfinite]='STORED_NONFINITE_VALUE'
    reason[missing&present&negative]='STORED_NEGATIVE_VALUE'
    reason[q.eq(3)]='LOCATION_HISTORY_UNUSABLE'
    reason[q.eq(4)]='DIFFERENT_STATION_COORDINATES'
    return reason


def main():
    out=V3/'output/07/full/04_bc_missing';log=V3/'log/07/full/04_bc_missing'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing audit; no automatic overwrite')
    source=V3/'output/07/full/03_abc_comparison/summary.html'
    configpath=V3/'config/rainfall.json';oldlog=V3/'log/07/full/03_abc_comparison/run.json'
    before=[fingerprint(p) for p in [source,configpath,oldlog]]
    text=source.read_text(encoding='utf-8')
    js=re.findall(r'<script>(.*?)</script>',text,re.S)[-1]
    payload=json.loads(gzip.decompress(base64.b64decode(re.search(r"atob\('([^']+)'\)",js).group(1))))
    old=json.loads(oldlog.read_text(encoding='utf-8'))
    for f in old['source_files']:
        stamp=Path(f['path']).stat()
        assert (stamp.st_size,stamp.st_mtime_ns)==(f['bytes'],f['mtime_ns']), 'Rain source changed: rebuild comparison first'
    requirements=[];sample_rows=[]
    for typ in ['b','c']:
        windows={}
        for p in payload['pairs']:
            e=payload['series'][p[typ]];end=pd.Timestamp(e['end'])
            # Coordinates come from approved matching records, not inferred from raw observations.
            sample_rows.append(dict(SAMPLE_TYPE=typ.upper(),A_REFERENCE_ID=p['id'],SERIES_ID=p[typ],
                STN=e['stn'],END_DATE=end-pd.Timedelta(days=1),MISSING_HOURS=sum(q>=2 for q in e['q'])))
        path=V3/'output/06/full/05_selection/b_event_selected.parquet' if typ=='b' else V3/'output/07/full/c_event_samples.parquet'
        frame=pd.read_parquet(path).set_index('A_REFERENCE_ID')
        for p in payload['pairs']:
            r=frame.loc[p['id']];e=payload['series'][p[typ]]
            key=(int(r.STN),float(r.STATION_LAT),float(r.STATION_LON),pd.Timestamp(e['end']))
            if key not in windows:windows[key]=[e,0]
            windows[key][1]+=1
        pieces=[]
        for (stn,lat,lon,end),(e,uses) in windows.items():
            pieces.append(pd.DataFrame(dict(STN=stn,STATION_LAT=lat,STATION_LON=lon,
                time_start=pd.date_range(end-pd.Timedelta(days=15),end,freq='h',inclusive='left'),
                QC_CODE=e['q'],SAMPLE_USES=uses)))
        merged=pd.concat(pieces,ignore_index=True)
        keys=['STN','STATION_LAT','STATION_LON','time_start']
        assert not merged.groupby(keys).QC_CODE.nunique().gt(1).any()
        merged=merged.groupby(keys,as_index=False).agg(QC_CODE=('QC_CODE','first'),SAMPLE_USES=('SAMPLE_USES','sum'))
        merged['SAMPLE_TYPE']=typ.upper();requirements.append(merged)
    req=pd.concat(requirements,ignore_index=True)
    stations=sorted(req.STN.unique().astype(int).tolist());dates=set(req.time_start.dt.normalize())
    root=Path(json.loads(configpath.read_text(encoding='utf-8'))['rain_root'])
    raw={};catalog={};filedates={};daykeys={};stamps=[]
    for net in ['AWS','ASOS']:
        paths=sorted((root/net/'hourly').glob('*/*/*/*.parquet'))
        catalog[net]=[];filedates[net]=set();frames=[]
        for path in paths:
            date=pd.to_datetime(path.stem.removeprefix('data_'),format='%Y%m%d')
            filedates[net].add(date);catalog[net].append(dict(NETWORK=net,YEAR=date.year,DATE=date))
            if date not in dates:continue
            stamp=path.stat();f=pq.read_table(path,filters=[('STN','in',stations)]).to_pandas()
            assert not f.duplicated(['STN','time_start']).any()
            frames.append(f[['STN','time_start','RN_1H']]);stamps.append((path,stamp.st_size,stamp.st_mtime_ns))
        f=pd.concat(frames,ignore_index=True)
        assert not f.duplicated(['STN','time_start']).any()
        daykeys[net]=pd.MultiIndex.from_frame(f.assign(DATE=f.time_start.dt.normalize())[['STN','DATE']].drop_duplicates())
        raw[net]=f.set_index(['STN','time_start']).RN_1H
        print(f'BC raw audit: {net} {len(f):,} stored rows',flush=True)
    idx=pd.MultiIndex.from_frame(req[['STN','time_start']]);dayidx=pd.MultiIndex.from_arrays([req.STN,req.time_start.dt.normalize()])
    for net in ['AWS','ASOS']:
        req[net+'_ROW_PRESENT']=idx.isin(raw[net].index)
        req[net+'_RAW_RN_1H']=raw[net].reindex(idx).to_numpy()
        req[net+'_DAY_FILE_PRESENT']=req.time_start.dt.normalize().isin(filedates[net])
        req[net+'_STATION_DAY_PRESENT']=dayidx.isin(daykeys[net])
    req['REASON']=classify(req.QC_CODE,req.AWS_ROW_PRESENT,req.ASOS_ROW_PRESENT,req.AWS_RAW_RN_1H,req.ASOS_RAW_RN_1H)
    no_row=req.REASON.eq('NO_OBSERVATION_ROW')
    req['ABSENCE_DETAIL']='NOT_APPLICABLE'
    req.loc[no_row,'ABSENCE_DETAIL']='HOUR_ABSENT_IN_STATION_DAY'
    req.loc[no_row&~(req.AWS_STATION_DAY_PRESENT|req.ASOS_STATION_DAY_PRESENT),'ABSENCE_DETAIL']='STATION_ABSENT_IN_DAY'
    req.loc[no_row&~(req.AWS_DAY_FILE_PRESENT|req.ASOS_DAY_FILE_PRESENT),'ABSENCE_DETAIL']='DAY_FILES_ABSENT'
    # Audit counts must exactly reproduce the displayed sample-weighted missing hours.
    result={}
    for typ in ['B','C']:
        f=req[req.SAMPLE_TYPE.eq(typ)];missing=f[~f.REASON.eq('VALID_OBSERVATION')]
        assert int(f.SAMPLE_USES.sum())==5866*360
        expected=old['summary']['360'][typ.lower()]
        assert int(missing.SAMPLE_USES.sum())==expected['requiredHours']-expected['validHours']
        count=f.groupby('REASON').agg(UNIQUE_HOURS=('SAMPLE_USES','size'),SAMPLE_WEIGHTED_HOURS=('SAMPLE_USES','sum'))
        early=f[f.time_start.dt.strftime('%m%d').lt('0620')]
        result[typ]=dict(total_unique_hours=len(f),total_sample_hours=int(f.SAMPLE_USES.sum()),
            missing_unique_hours=len(missing),missing_sample_hours=int(missing.SAMPLE_USES.sum()),
            reasons=count.reset_index().to_dict('records'),
            no_row_details=f[no_row.loc[f.index]].ABSENCE_DETAIL.value_counts().to_dict(),
            before_june20_unique_hours=len(early),
            before_june20_missing_unique_hours=int((~early.REASON.eq('VALID_OBSERVATION')).sum()),
            before_june20_no_row_unique_hours=int(early.REASON.eq('NO_OBSERVATION_ROW').sum()),
            before_june20_no_files_unique_hours=int(early.ABSENCE_DETAIL.eq('DAY_FILES_ABSENT').sum()))
    catalog_table=pd.concat([pd.DataFrame(v) for v in catalog.values()]).groupby(['NETWORK','YEAR']).agg(
        FIRST_PARTITION=('DATE','min'),LAST_PARTITION=('DATE','max'),PARTITION_DAYS=('DATE','nunique')).reset_index()
    daily=req.assign(DATE=req.time_start.dt.normalize()).groupby(['SAMPLE_TYPE','DATE','REASON','ABSENCE_DETAIL']).agg(
        UNIQUE_HOURS=('SAMPLE_USES','size'),SAMPLE_WEIGHTED_HOURS=('SAMPLE_USES','sum')).reset_index()
    samples=pd.DataFrame(sample_rows)
    assert before==[fingerprint(p) for p in [source,configpath,oldlog]]
    for path,size,mtime in stamps:
        stamp=path.stat();assert (stamp.st_size,stamp.st_mtime_ns)==(size,mtime)
    out.mkdir(parents=True)
    req.to_parquet(out/'bc_hourly_availability.parquet',index=False)
    daily.to_parquet(out/'bc_daily_missing_summary.parquet',index=False)
    samples.to_parquet(out/'bc_sample_missing_summary.parquet',index=False)
    write_json(log/'run.json',dict(status='COMPLETE_AUDIT',inputs=before,results=result,
        api_requests=0,upstream_unchanged=True,stored_null_not_original_api_sentinel=True,
        source_file_stamps=[dict(path=str(p),bytes=s,mtime_ns=m) for p,s,m in stamps]))
    names={'VALID_OBSERVATION':'유효 관측','NO_OBSERVATION_ROW':'관측행 없음',
        'STORED_NULL_VALUE':'관측행 있음 · 강우값 null', 'STORED_NEGATIVE_VALUE':'관측행 있음 · 음수',
        'STORED_NONFINITE_VALUE':'관측행 있음 · 무한값', 'LOCATION_HISTORY_UNUSABLE':'위치 이력 사용 불가',
        'DIFFERENT_STATION_COORDINATES':'선택 좌표와 다른 관측'}
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>B·C 강우 부족 분석</title><style>body{font:15px/1.7 Malgun Gothic;margin:35px;color:#24364a}section{margin:25px 0}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}details{margin-top:20px}a{color:#246b9a}</style><h1>B·C 이전 15일 강우 부족 분석</h1><p>각 5,866개 표본, 종료일 포함 360시간. 저장소에 관측행이 없는 경우와 관측행의 강우값이 비어 있는 경우를 구분했습니다.</p>'''
    for typ in ['B','C']:
        r=result[typ];table=pd.DataFrame(r['reasons']);table['사유']=table.REASON.map(names)
        table['고유 시간 비율(%)']=100*table.UNIQUE_HOURS/r['total_unique_hours']
        table['표본 시간 비율(%)']=100*table.SAMPLE_WEIGHTED_HOURS/r['total_sample_hours']
        page+=f'<section><h2>{typ}</h2><p>고유 필요 시간 {r["total_unique_hours"]:,} 중 부족 {r["missing_unique_hours"]:,}. 표본별 반복 사용을 포함하면 {r["missing_sample_hours"]:,} / {r["total_sample_hours"]:,}시간.</p>'
        page+=table[['사유','UNIQUE_HOURS','고유 시간 비율(%)','SAMPLE_WEIGHTED_HOURS','표본 시간 비율(%)']].rename(columns={'UNIQUE_HOURS':'고유 관측 시간','SAMPLE_WEIGHTED_HOURS':'표본별 사용 시간'}).to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}')
        page+=f'<p>각 연도 6월 20일 이전: 필요 {r["before_june20_unique_hours"]:,}시간, 관측행 없음 {r["before_june20_no_row_unique_hours"]:,}시간(그중 날짜 파일 자체 없음 {r["before_june20_no_files_unique_hours"]:,}시간).</p></section>'
    page+='<h2>공유 저장소의 연도별 날짜 파일 범위</h2>'+catalog_table.to_html(index=False,border=0)
    page+='<p>첫·마지막 파일 사이 모든 날짜/관측소가 완전하다는 뜻은 아닙니다. 관측행 없음도 실제 기상청 자료 부재를 뜻하지 않으며 아직 수집하지 않았거나 저장 과정에서 빠졌을 수 있습니다. null은 현재 Parquet 상태이고 원래 API 응답이 -99였는지 빈칸이었는지는 이 표만으로 확정하지 않습니다.</p>'
    page+='<p>고유 시간은 유형별 STN·좌표·시간 중복 제거, 표본별 사용 시간은 같은 관측이 여러 표본에 쓰인 횟수를 반영합니다. 위치 문제는 단순 다운로드로 보완된다고 가정하지 않습니다.</p>'
    page+='<p><a href="bc_hourly_availability.parquet">시간별 원본 행·값 상태</a> · <a href="bc_daily_missing_summary.parquet">날짜별 집계</a> · <a href="bc_sample_missing_summary.parquet">표본별 부족 시간</a></p></html>'
    (out/'summary.html').write_text(page,encoding='utf-8')
    print(json.dumps(result,default=str))


if __name__=='__main__':main()
