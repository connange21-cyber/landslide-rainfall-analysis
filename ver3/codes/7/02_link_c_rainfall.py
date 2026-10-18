"""Link C controls to current shared rainfall with explicit missingness and 1–15-day features."""
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def stats(values,prefix):
    v=np.array([np.nan if x is None else x for x in values],dtype=float)
    count=int(np.isfinite(v).sum());total=len(v)
    return {prefix+'_OBSERVED_SUM_MM':float(np.nansum(v)) if count else np.nan,
            prefix+'_COMPLETE_SUM_MM':float(v.sum()) if count==total else np.nan,
            prefix+'_VALID_HOURS':count,prefix+'_MISSING_HOURS':total-count,
            prefix+'_COVERAGE':count/total if total else np.nan}


def main():
    folder=V3/'output/07/full';out=folder/'02_rainfall';log=V3/'log/07/full/02_rainfall'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Rainfall outputs already exist; no automatic overwrite')
    inputs=[folder/'c_event_samples.parquet',V3/'config/rainfall.json',V3/'base/stations/station_location_history.xlsx']
    before=[fingerprint(p) for p in inputs]
    c=pd.read_parquet(inputs[0]);config=json.loads(inputs[1].read_text(encoding='utf-8'))
    history=pd.read_excel(inputs[2],sheet_name='location_history')
    spec=importlib.util.spec_from_file_location('shared_rain_qc',V3/'codes/6/04_build_hourly_comparison.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    dates=set()
    for r in c.itertuples():
        start=min(r.RAIN_START,r.CUMULATIVE_START_DATE)
        end=max(r.RAIN_END_EXCLUSIVE,r.END_DATE+pd.Timedelta(days=1))
        dates.update(pd.date_range(start.normalize(),(end-pd.Timedelta(nanoseconds=1)).normalize(),freq='D'))
    rain,sources=loader.current_rain(config,sorted(c.STN.unique().astype(int).tolist()),dates,history)
    empty=pd.DataFrame({'VALUE':pd.Series(dtype='float64'),'USABLE':pd.Series(dtype='bool'),
        'LAT':pd.Series(dtype='float64'),'LON':pd.Series(dtype='float64'),'SOURCE':pd.Series(dtype='str')},index=pd.DatetimeIndex([]))
    cache={};hourly=[];features=[]
    for i,r in enumerate(c.itertuples()):
        start=min(r.RAIN_START,r.CUMULATIVE_START_DATE)
        end=max(r.RAIN_END_EXCLUSIVE,r.END_DATE+pd.Timedelta(days=1))
        key=(r.STN,r.STATION_LAT,r.STATION_LON,start,end)
        if key not in cache:
            values,q=loader.sequence(rain.get(int(r.STN),empty),start,end,r.STATION_LAT,r.STATION_LON)
            cache[key]=(values,q)
            times=pd.date_range(start,end,freq='h',inclusive='left')
            hourly.append(pd.DataFrame(dict(STN=int(r.STN),STATION_LAT=r.STATION_LAT,STATION_LON=r.STATION_LON,
                time_start=times,time_end=times+pd.Timedelta(hours=1),RAIN_1H_ANALYSIS=values,QC_CODE=q)))
        values,q=cache[key]
        def window(left,right):
            lo=int((left-start)/pd.Timedelta(hours=1));hi=int((right-start)/pd.Timedelta(hours=1))
            assert 0<=lo<hi<=len(values)
            return values[lo:hi]
        f=r._asdict();f.update(stats(window(r.RAIN_START,r.RAIN_END_EXCLUSIVE),'EVENT'))
        for days in range(1,16):
            span=window(r.END_DATE-pd.Timedelta(days=days-1),r.END_DATE+pd.Timedelta(days=1))
            assert len(span)==24*days
            f.update(stats(span,f'RAIN_{days:02d}D'))
        f['RAIN_STATUS']='COMPLETE' if f['EVENT_MISSING_HOURS']==0 and f['RAIN_15D_MISSING_HOURS']==0 else 'LINKED_WITH_MISSING'
        features.append(f)
        if i%500==0 or i+1==len(c):
            print(f'07 rainfall features: {i+1:,}/{len(c):,}',flush=True)
            write_json(log/'progress.json',dict(done=i+1,total=len(c)))
    table=pd.DataFrame(features);hours=pd.concat(hourly,ignore_index=True)
    hkeys=['STN','STATION_LAT','STATION_LON','time_start']
    # Overlapping sample windows refer to the same observation, never duplicate publication.
    assert not hours.groupby(hkeys,dropna=False).RAIN_1H_ANALYSIS.nunique(dropna=False).gt(1).any()
    hours=hours.drop_duplicates(hkeys).sort_values(['STN','time_start','STATION_LAT','STATION_LON'])
    missing=hours[hours.RAIN_1H_ANALYSIS.isna()].copy()
    reasons={2:'NO_VALID_OBSERVATION',3:'LOCATION_HISTORY_UNUSABLE',4:'DIFFERENT_STATION_COORDINATES'}
    missing['MISSING_REASON']=missing.QC_CODE.map(reasons)
    assert missing.MISSING_REASON.notna().all()
    assert len(table)==len(c) and table.C_SAMPLE_ID.is_unique
    assert table.RAIN_STATUS.isin(['COMPLETE','LINKED_WITH_MISSING']).all()
    for days in range(1,16):
        assert table[f'RAIN_{days:02d}D_VALID_HOURS'].add(table[f'RAIN_{days:02d}D_MISSING_HOURS']).eq(days*24).all()
        assert table.loc[table[f'RAIN_{days:02d}D_MISSING_HOURS'].gt(0),f'RAIN_{days:02d}D_COMPLETE_SUM_MM'].isna().all()
    assert before==[fingerprint(p) for p in inputs]
    for s in sources:
        stamp=Path(s['path']).stat();assert (stamp.st_size,stamp.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
    out.mkdir(parents=True)
    table.to_parquet(out/'c_rainfall_features.parquet',index=False)
    hours.to_parquet(out/'c_rainfall_hourly.parquet',index=False)
    missing.to_parquet(out/'c_missing_hours.parquet',index=False)
    result=dict(status='COMPLETE_LINKAGE' if missing.empty else 'COMPLETE_LINKAGE_WITH_MISSING',
        c_samples=len(c),complete_samples=int(table.RAIN_STATUS.eq('COMPLETE').sum()),
        event_complete_samples=int(table.EVENT_MISSING_HOURS.eq(0).sum()),
        cumulative15_complete_samples=int(table.RAIN_15D_MISSING_HOURS.eq(0).sum()),
        required_unique_hour_slots=len(hours),missing_unique_hour_slots=len(missing),
        missing_reasons=missing.MISSING_REASON.value_counts().to_dict(),
        inputs=before,sources=sources,upstream_unchanged=True,api_requests=0,
        rainfall_source='CURRENT_RAIN_DATAS_PARQUET',zero_imputation=False,
        partial_sum_not_complete_sum=True,location_history_applied=True)
    write_json(log/'run.json',result)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>C 강우 연결</title><style>body{font:15px/1.7 Malgun Gothic;margin:35px;color:#24364a}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>07 · 02 C 강우량 연결</h1>'''
    page+=f'<p>C {len(c):,}개 모두 시간·STN 기준으로 연결. 이벤트와 15일 누적 창 모두 완전한 표본 {result["complete_samples"]:,}개.</p>'
    page+=f'<p>이벤트 구간 완전 {result["event_complete_samples"]:,}개 · 이전 15일 완전 {result["cumulative15_complete_samples"]:,}개. 필요한 고유 관측 시간 {len(hours):,}개 중 결측·위치 제외 {len(missing):,}개.</p>'
    page+='<p><a href="c_rainfall_features.parquet">C 이벤트·1~15일 강우 피처</a> · <a href="c_rainfall_hourly.parquet">시간별 분석 강우</a> · <a href="c_missing_hours.parquet">부족 시간 목록</a></p>'
    page+='<p>OBSERVED_SUM_MM는 관측 부분합, COMPLETE_SUM_MM는 전 시간 확보 시에만 값이 있습니다. 관측이 하나도 없으면 부분합도 결측입니다. 0mm와 결측을 구분하며 VALID_HOURS·MISSING_HOURS·COVERAGE를 제공합니다.</p>'
    page+='<p>AWS 우선·ASOS 보완 후 관측망별 위치 이력과 C 좌표를 확인했습니다. 종료일 포함 이전 1~15일은 KST 달력일 기준입니다. 추가 API 수집·원본 수정은 하지 않았습니다.</p>'
    page+=missing.MISSING_REASON.value_counts().rename_axis('사유').rename('고유 시간').to_frame().to_html(border=0)
    page+='<h2>연결 결과 예시 20개</h2>'+table[['ADMIN_NAME','STN','END_DATE','EVENT_OBSERVED_SUM_MM','EVENT_MISSING_HOURS','RAIN_15D_OBSERVED_SUM_MM','RAIN_15D_MISSING_HOURS','RAIN_STATUS']].head(20).to_html(index=False,border=0)
    (out/'summary.html').write_text(page+'</html>',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','sources']}))


if __name__=='__main__':main()
