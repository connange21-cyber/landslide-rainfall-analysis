"""Offline A/B/C rainfall comparison anchored at rainfall-event end calendar days."""
import sys
import json
import gzip
import base64
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    out=V3/'output/07/full/03_abc_comparison';log=V3/'log/07/full/03_abc_comparison'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing comparison; no automatic overwrite')
    paths=[V3/'output/06/full/05_selection/occurrence_selected.parquet',
           V3/'output/06/full/05_selection/b_event_selected.parquet',V3/'output/07/full/c_event_samples.parquet',
           V3/'config/rainfall.json',V3/'base/stations/station_location_history.xlsx',Path(__file__).with_suffix('.html')]
    before=[fingerprint(p) for p in paths]
    a,b,c=[pd.read_parquet(p).set_index('A_REFERENCE_ID') for p in paths[:3]]
    assert a.index.is_unique and b.index.is_unique and c.index.is_unique
    assert set(a.index)==set(b.index)==set(c.index)
    config=json.loads(paths[3].read_text(encoding='utf-8'));history=pd.read_excel(paths[4],sheet_name='location_history')
    spec=importlib.util.spec_from_file_location('rain_view_source',V3/'scripts/6_시간비교표본_B/04_build_hourly_comparison.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    pairs=[];series={};obs={};dates=set();stations=set()
    for aid,r in a.sort_index().iterrows():
        br,cr=b.loc[aid],c.loc[aid]
        assert r.END_DATE==cr.END_DATE and int(r.STN)==int(br.STN)
        p=dict(id=aid,unitId=r.SPATIAL_UNIT_ID,unit=r.ADMIN_NAME,stn=int(r.STN),
            aDate=r.END_DATE.strftime('%Y-%m-%d'),bDate=br.END_DATE.strftime('%Y-%m-%d'),cUnit=cr.ADMIN_NAME)
        for typ,row in [('a',r),('b',br),('c',cr)]:
            # The end day is included, exactly as in existing 1–15 calendar-day cumulative features.
            end=row.END_DATE.normalize()+pd.Timedelta(days=1)
            lat,lon=(br.STATION_LAT,br.STATION_LON) if typ=='a' else (row.STATION_LAT,row.STATION_LON)
            key=(int(row.STN),float(lat),float(lon),end)
            obs.setdefault(key,None)
            sid=f'{typ}:{row.SPATIAL_UNIT_ID}:{row.STN}:{row.END_DATE:%Y%m%d}'
            if sid not in series:
                series[sid]=dict(stn=int(row.STN),unit=row.ADMIN_NAME,end=end.isoformat(),key=key)
            p[typ]=sid;stations.add(int(row.STN))
            dates.update(pd.date_range(end-pd.Timedelta(days=15),end-pd.Timedelta(days=1),freq='D'))
        pairs.append(p)
    rain,sources=loader.current_rain(config,sorted(stations),dates,history)
    for i,key in enumerate(obs):
        stn,lat,lon,end=key
        assert stn in rain, f'No rainfall rows for STN {stn}'
        obs[key]=loader.sequence(rain[stn],end-pd.Timedelta(days=15),end,lat,lon)
        assert len(obs[key][0])==360
        if i%1000==0:print(f'ABC rainfall: {i+1:,}/{len(obs):,} unique station/date/coordinate windows',flush=True)
    for e in series.values():e['v'],e['q']=obs[e.pop('key')]
    summary={}
    for hours in [15,360]:
        summary[str(hours)]={}
        for typ in ['a','b','c']:
            matrix=np.array([[np.nan if v is None else v for v in series[p[typ]]['v'][-hours:]] for p in pairs],dtype=float)
            valid=np.isfinite(matrix);counts=valid.sum(axis=0)
            mean=np.divide(np.nansum(matrix,axis=0),counts,out=np.full(hours,np.nan),where=counts>0)
            rows=valid.sum(axis=1);partial=np.nansum(matrix,axis=1)
            summary[str(hours)][typ]=dict(mean=[float(v) if np.isfinite(v) else None for v in mean],
                valid=counts.astype(int).tolist(),samples=len(pairs),complete=int((rows==hours).sum()),
                validHours=int(valid.sum()),requiredHours=matrix.size,
                meanPartial=float(partial[rows>0].mean()) if (rows>0).any() else None)
    payload=dict(pairs=pairs,series=series,summary=summary,hours=15,
        asOf=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
        timeNote='0은 강우 이벤트 종료일 다음 날 00:00(KST), 즉 종료일 24:00입니다. A·C는 A 종료일, B는 B 자체 종료일을 사용합니다. -1시간 막대는 종료일 23~24시입니다. 종료일 포함 이전 15일(360시간)도 전환할 수 있습니다. 실제 이벤트 종료 후 같은 날의 나머지 시간도 포함되며, 산사태 발생 시각 기준이 아닙니다.')
    encoded=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))).decode()
    html=paths[-1].read_text(encoding='utf-8').replace('__PLOTLY_JS__',get_plotlyjs()).replace('__PAYLOAD__',encoded)
    assert before==[fingerprint(p) for p in paths]
    for s in sources:
        st=Path(s['path']).stat();assert (st.st_size,st.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
    out.mkdir(parents=True)
    (out/'summary.html').write_text(html,encoding='utf-8')
    write_json(log/'run.json',dict(status='COMPLETE',triplets=len(pairs),station_coordinate_day_windows=len(obs),
        source_files=sources,inputs=before,upstream_unchanged=True,api_requests=0,
        anchor='END_DATE_PLUS_ONE_DAY_00_KST',windows_hours=[15,360],
        sample_weighted_statistics=True,missing_not_zero=True,summary=summary,
        output_bytes=(out/'summary.html').stat().st_size))
    print(f'ABC comparison COMPLETE: {len(pairs):,} triplets, {(out/"summary.html").stat().st_size:,} bytes')


if __name__=='__main__':main()
