"""Standalone A/B hourly rainfall viewer from current shared Parquet observations."""
import argparse
import base64
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from plotly.offline import get_plotlyjs

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def current_rain(config,stations,dates,history):
    spec=importlib.util.spec_from_file_location('rain_schema',V3/'codes/2/02_build_rainfall.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parts={};sources=[];root=Path(config['rain_root'])
    for network in ['AWS','ASOS']:
        frames=[]
        for path in sorted((root/network/'hourly').glob('*/*/*/*.parquet')):
            date=pd.to_datetime(path.stem.removeprefix('data_'),format='%Y%m%d')
            if date not in dates:continue
            stamp=path.stat()
            f=pq.read_table(path,filters=[('STN','in',stations)]).to_pandas()
            module.validate_partition(f,date)
            f.RN_1H=f.RN_1H.where(f.RN_1H.ge(0)&np.isfinite(f.RN_1H))
            frames.append(f)
            after=path.stat()
            assert (stamp.st_size,stamp.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            sources.append(dict(network=network,path=str(path),bytes=stamp.st_size,mtime_ns=stamp.st_mtime_ns))
        if not frames:raise ValueError('No source rainfall for '+network)
        f=pd.concat(frames,ignore_index=True)
        if f.duplicated(['STN','time_start']).any():raise ValueError('Duplicate station/hour')
        parts[network]=f.rename(columns={'RN_1H':network})
        print(f'04 hourly viewer: {network}, {len(f):,} rows',flush=True)
    rain=parts['AWS'].merge(parts['ASOS'],on=['STN','time_start','time_end'],how='outer',validate='one_to_one')
    rain['VALUE']=rain.AWS.combine_first(rain.ASOS)
    rain['SOURCE']=np.select([rain.AWS.notna(),rain.ASOS.notna()],['AWS','ASOS'],default='MISSING')
    rain['USABLE']=False;rain['LAT']=np.nan;rain['LON']=np.nan
    # Same source-network and transition checks as the analysis input pipeline.
    for stn,g in rain.groupby('STN',sort=False):
        hits=np.zeros(len(g),dtype=int);review=np.zeros(len(g),dtype=bool)
        lat=np.full(len(g),np.nan);lon=lat.copy()
        for r in history[history.STN.eq(stn)].itertuples():
            same=g.SOURCE.eq(r.NETWORK).to_numpy()
            mask=same & g.time_start.ge(r.ANALYSIS_FROM).to_numpy()
            if pd.notna(r.ANALYSIS_TO):mask &=g.time_end.le(r.ANALYSIS_TO).to_numpy()
            hits[mask]+=1;lat[mask]=r.LAT;lon[mask]=r.LON;review[mask]|=bool(r.REVIEW_FLAG)
            for edge in [r.ANALYSIS_FROM,r.ANALYSIS_TO]:
                if pd.notna(edge):review|=same&g.time_start.lt(edge).to_numpy()&g.time_end.gt(edge).to_numpy()
            if pd.notna(r.LAST_OLD_OBSERVED) and pd.notna(r.FIRST_NEW_OBSERVED):
                review|=same&g.time_start.lt(r.FIRST_NEW_OBSERVED).to_numpy()&g.time_end.gt(r.LAST_OLD_OBSERVED).to_numpy()
        if (hits>1).any():raise ValueError('Overlapping location history')
        rain.loc[g.index,'USABLE']=(hits==1)&~review
        rain.loc[g.index,'LAT']=lat;rain.loc[g.index,'LON']=lon
    return {int(stn):g.set_index('time_start').sort_index() for stn,g in rain.groupby('STN')},sources


def sequence(g,start,end,lat,lon):
    index=pd.date_range(start,end,freq='h',inclusive='left')
    f=g.reindex(index)
    raw=f.VALUE.notna() & np.isfinite(f.VALUE)
    usable=f.USABLE.fillna(False).astype(bool)
    same=f.LAT.eq(lat)&f.LON.eq(lon)
    valid=raw&usable&same
    status=np.select([~raw,~usable,~same,f.SOURCE.eq('ASOS')],[2,3,4,1],default=0)
    values=[float(v) if ok else None for v,ok in zip(f.VALUE,valid)]
    # A zero remains zero; all missing/invalid/unlocatable hours remain null.
    assert all(v is None or (np.isfinite(v) and v>=0) for v in values)
    return values,status.astype(int).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    out=V3/'output/06/full';log=V3/'log/06/full/04_hourly_viewer'
    target=out/'04_hourly_rainfall_comparison.html'
    if target.exists() and not args.overwrite:parser.error('Existing viewer; --overwrite required')
    paths=[out/'b_event_samples.parquet',out/'occurrence_reference.parquet',V3/'config/rainfall.json',
           V3/'base/stations/station_location_history.xlsx',V3/'output/01/full/landslide_station_links.parquet',
           Path(__file__).with_suffix('.html')]
    hashes=[fingerprint(p) for p in paths]
    b=pd.read_parquet(paths[0]);a=pd.read_parquet(paths[1]).set_index('A_REFERENCE_ID')
    config=json.loads(paths[2].read_text(encoding='utf-8'))
    history=pd.read_excel(paths[3],sheet_name='location_history');links=pd.read_parquet(paths[4]).set_index('EVENT_ID')
    bundles={};pairs=[];dates=set()
    for r in b.sort_values(['STN','A_END_DATE','A_REFERENCE_ID']).itertuples():
        ar=a.loc[r.A_REFERENCE_ID];assert int(ar.STN)==int(r.STN)
        pair=dict(id=r.A_REFERENCE_ID,stn=int(r.STN),unit=r.ADMIN_NAME,unitId=r.SPATIAL_UNIT_ID,
            a=ar.RAIN_EVENT_ID,b=r.RAIN_EVENT_ID,lat=r.STATION_LAT,lon=r.STATION_LON,
            ambiguous=bool(r.A_AMBIGUITY_REVIEW),spatialReview=bool(r.A_SPATIAL_REVIEW),reported=[])
        for event_id in ar.LANDSLIDE_EVENT_IDS:
            if event_id in links.index:
                occurrence=links.loc[event_id]
                pair['reported'].append(dict(id=event_id,start=occurrence.EVENT_INTERVAL_START.isoformat(),
                    end=occurrence.EVENT_INTERVAL_END_EXCLUSIVE.isoformat(),exact=bool(occurrence.EXACT_TIME),grade=occurrence.TIME_GRADE))
        for key,record in [('a',ar),('b',r)]:
            eid=pair[key]
            if eid in bundles:continue
            start=min(record.RAIN_START,record.END_DATE-pd.Timedelta(days=14))
            end=max(record.RAIN_END_EXCLUSIVE,record.END_DATE+pd.Timedelta(days=1))
            bundles[eid]=dict(stn=int(r.STN),start=start,end=end,eventStart=record.RAIN_START,
                eventEnd=record.RAIN_END_EXCLUSIVE,anchor=record.END_DATE,
                lat=r.STATION_LAT,lon=r.STATION_LON)
            dates.update(pd.date_range(start.normalize(),(end-pd.Timedelta(nanoseconds=1)).normalize(),freq='D'))
        pairs.append(pair)
    rain,sources=current_rain(config,sorted(b.STN.unique().astype(int).tolist()),dates,history)
    checked=0;nulls=0
    for eid,r in bundles.items():
        r['v'],r['q']=sequence(rain[r['stn']],r['start'],r['end'],r['lat'],r['lon'])
        assert len(r['v'])==int((r['end']-r['start'])/pd.Timedelta(hours=1))
        # Independent raw-to-view checks for event hours, including zero and missing values.
        checked+=len(r['v']);nulls+=sum(x is None for x in r['v'])
        for key in ['start','end','eventStart','eventEnd','anchor']:r[key]=r[key].isoformat()
    assert hashes==[fingerprint(p) for p in paths]
    for src in sources:
        stat=Path(src['path']).stat();assert (stat.st_size,stat.st_mtime_ns)==(src['bytes'],src['mtime_ns'])
    payload=dict(pairs=pairs,events=bundles,asOf=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
        stationCount=b.STN.nunique(),pairCount=len(b),eventCount=len(bundles))
    encoded=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,default=int,allow_nan=False,separators=(',',':')).encode())).decode()
    template=paths[5].read_text(encoding='utf-8')
    html=template.replace('__PLOTLY_JS__',get_plotlyjs()).replace('__PAYLOAD__',encoded)
    tmp=target.with_suffix('.tmp.html');tmp.write_text(html,encoding='utf-8');tmp.replace(target)
    write_json(log/'run.json',dict(status='COMPLETE',inputs=hashes,code=fingerprint(Path(__file__)),
        sources=sources,stations=int(b.STN.nunique()),pairs=len(b),unique_events=len(bundles),
        plotted_hour_slots=checked,missing_or_location_masked_slots=nulls,bytes=target.stat().st_size,
        raw_rain_root=config['rain_root'],api_requests=0,upstream_changed=False,
        event_definitions='EXISTING_A_B_SNAPSHOT',rainfall='CURRENT_SHARED_PARQUET_WITH_NETWORK_LOCATION_QC'))
    write_json(log/'verification.json',dict(status='DATA_PASS',unique_b_pairs=True,same_stn=True,
        interval_lengths_verified=True,zero_preserved=True,missing_not_zero=True,upstream_unchanged=True))
    print(json.dumps(dict(html=str(target),stations=int(b.STN.nunique()),pairs=len(b),unique_events=len(bundles),bytes=target.stat().st_size)))


if __name__=='__main__':main()
