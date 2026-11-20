"""Plan/collect hourly fusion rainfall once per shared time; default makes no API calls.

Official specification: https://apihub.kma.go.kr/apiList.do?seqApi=971
Entire-area rn_60m NetCDF4 (binary alternative). HH:00 represents the preceding hour.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import io
import json
import os
from pathlib import Path
import threading
import time
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import openpyxl
import h5py

V3=Path(__file__).resolve().parents[2]
ROOT=Path('/data/RAIN_DATAS')
INT=ROOT/'융합기상/INT'
OUT=V3/'output/10/full/01_fusion_hourly'
LOG=V3/'log/10/full/01_fusion_hourly'
RAW=ROOT/'원본/융합기상/정시_1시간강우/ver3_10'
URL='https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_nc_api'
NC_URL='https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php'
COLS=['GRID_ID','time_start','time_end','RN_1H']
LOCAL=threading.local();RATE_LOCK=threading.Lock();NEXT_REQUEST=0.

def partition(date):
    return INT/'hourly'/date.strftime('%Y/%m/%d')/f'data_{date:%Y%m%d}.parquet'

def decode(body):
    if body.startswith(b'\x89HDF\r\n\x1a\n'):
        with h5py.File(io.BytesIO(body),'r') as file:
            candidates=[name for name in file if isinstance(file[name],h5py.Dataset)
                        and file[name].shape==(2049,2049) and name not in ['lat','lon']]
            if len(candidates)!=1:raise ValueError('Ambiguous NetCDF rain variable; inspect pilot response before proceeding')
            data=file[candidates[0]]
            axes=[]
            for i in range(2):
                scales=list(data.dims[i].values())
                axes.append(Path(scales[0].name).name if len(scales)==1 else '')
            if axes!=['ny','nx']:raise ValueError('Unverified NetCDF dimension order; expected ny,nx')
            units=data.attrs.get('units',data.attrs.get('unit',b'mm'))
            if isinstance(units,bytes):units=units.decode()
            if str(units).strip().lower() not in ['mm','mm/h','millimeter','millimeters']:
                raise ValueError('Unexpected NetCDF rainfall units')
            values=data[...].astype('float32')
            for attr in ['_FillValue','missing_value']:
                if attr in data.attrs:
                    for missing in np.asarray(data.attrs[attr]).ravel():values[values==missing]=np.nan
            if 'data_scale' in data.attrs:
                if 'scale_factor' in data.attrs or 'add_offset' in data.attrs:raise ValueError('Ambiguous scale conventions')
                divisor=float(np.asarray(data.attrs['data_scale']).reshape(-1)[0])
                if not np.isfinite(divisor) or divisor<=0:raise ValueError('Invalid data_scale')
                values=values/divisor
            else:
                values=values*float(data.attrs.get('scale_factor',1.))+float(data.attrs.get('add_offset',0.))
            return values.reshape(-1)
    if len(body)<4:raise ValueError('Empty/truncated binary response')
    nx,ny=np.frombuffer(body[:4],dtype='<u2').astype(int)
    if (nx,ny)!=(2049,2049) or len(body)!=4+int(nx)*int(ny)*4:
        raise ValueError('Unexpected grid/header/response length; stop and verify API response')
    values=np.frombuffer(body,dtype='<f4',offset=4)
    return values

def prepare():
    source=V3/'output/08/full/15_abc_selection'
    frames=[pd.read_parquet(source/n,columns=['SPATIAL_UNIT_ID','END_DATE']) for n in
            ['occurrence_selected.parquet','b_event_selected.parquet','c_event_selected.parquet']]
    samples=pd.concat(frames).drop_duplicates()
    registry=gpd.read_parquet(V3/'output/04/full/spatial_units_5186.parquet')
    units=registry[registry.SPATIAL_UNIT_ID.isin(samples.SPATIAL_UNIT_ID)][['SPATIAL_UNIT_ID','geometry']].to_crs(5186)
    assert units.SPATIAL_UNIT_ID.is_unique and set(samples.SPATIAL_UNIT_ID)==set(units.SPATIAL_UNIT_ID)
    grid=pd.read_parquet(INT/'grid.parquet')
    assert grid.GRID_ID.is_unique and (grid.GRID_ID==grid.GRID_Y*2049+grid.GRID_X).all()
    mapping_sources=[INT/'grid.parquet',V3/'output/04/full/spatial_units_5186.parquet']
    mapping_sources += [source/n for n in ['occurrence_selected.parquet','b_event_selected.parquet','c_event_selected.parquet']]
    signature={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in mapping_sources}
    cache=OUT/'admin_grid_links.parquet';cache_stamp=LOG/'mapping_inputs.json'
    if cache.exists() and cache_stamp.exists() and json.loads(cache_stamp.read_text(encoding='utf-8'))==signature:
        print('행정구역–격자 연결 캐시 재사용',flush=True)
        links=pd.read_parquet(cache)
    else:
        points=gpd.GeoDataFrame(grid[['GRID_ID']],geometry=gpd.points_from_xy(grid.LON,grid.LAT),crs=4326).to_crs(5186)
        print('기존 INT 격자와 ABC 행정구역 연결 중',flush=True)
        links=gpd.sjoin(points,units,how='inner',predicate='within')[['GRID_ID','SPATIAL_UNIT_ID']].drop_duplicates()
    mapping={unit:np.unique(part.GRID_ID.to_numpy(dtype=np.int32)) for unit,part in links.groupby('SPATIAL_UNIT_ID')}
    daily={}
    for row in samples.itertuples():
        ids=mapping.get(row.SPATIAL_UNIT_ID,np.array([],dtype=np.int32))
        if not len(ids):continue
        end=pd.Timestamp(row.END_DATE).normalize()
        for day in pd.date_range(end-pd.Timedelta(days=14),end):daily.setdefault(day,[]).append(ids)
    rows=[];total_slots=0;existing_slots=0
    for day,groups in sorted(daily.items()):
        ids=np.unique(np.concatenate(groups));total_slots+=len(ids)*24
        path=partition(day)
        existing=pq.read_table(path,columns=['GRID_ID','time_start']).to_pandas() if path.exists() else pd.DataFrame(columns=['GRID_ID','time_start'])
        for hour in range(24):
            start=day+pd.Timedelta(hours=hour)
            present=existing.loc[existing.time_start.eq(start),'GRID_ID'].to_numpy(dtype=np.int32)
            missing=np.setdiff1d(ids,present,assume_unique=False)
            existing_slots+=len(ids)-len(missing)
            rows.append(dict(DATE=day,TIME_START=start,TIME_END=start+pd.Timedelta(hours=1),
                             REQUIRED_GRID_COUNT=len(ids),MISSING_GRID_COUNT=len(missing)))
        # Only one grid list per day, rather than duplicating it 24 times.
        daily[day]=ids
    plan=pd.DataFrame(rows)
    review=pd.DataFrame(dict(SPATIAL_UNIT_ID=units.SPATIAL_UNIT_ID,
                             GRID_COUNT=units.SPATIAL_UNIT_ID.map(lambda u:len(mapping.get(u,[])))))
    report=dict(status='PLAN_READY',required_hours=len(plan),api_calls_upper_bound=int(plan.MISSING_GRID_COUNT.gt(0).sum()),
        selected_unique_grids=int(links.GRID_ID.nunique()),required_grid_hours=total_slots,existing_grid_hours=existing_slots,
        no_grid_units=int(review.GRID_COUNT.eq(0).sum()),
        nationwide_uncompressed_bytes=(4+2049*2049*4)*len(plan),
        selected_value_bytes_uncompressed=total_slots*4,
        storage_estimate='Value bytes only; Parquet and retained compressed raw size require a pilot measurement',
        source_grid_scope='EXISTING_INT_GRID_REFERENCE_676081_CELLS',
        api_calls_pointwise_upper_bound=total_slots,source='OFFICIAL_FULL_AREA_RN_60M',
        time_rule='KST_TIME_END_PRECEDING_60_MINUTES',missing_values_not_retried=True)
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    plan.to_parquet(OUT/'request_plan.parquet',index=False)
    links.to_parquet(OUT/'admin_grid_links.parquet',index=False)
    cache_stamp.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    review.to_parquet(OUT/'admin_grid_coverage_review.parquet',index=False)
    (LOG/'plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>10 융합기상 수집 계획</title><style>body{font:15px/1.8 Malgun Gothic;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>10 · 시간강우 융합기상</h1>'
    page+=f'<p>ABC 전체 15일 창의 겹침을 합친 {len(plan):,}시간. 기존 격자·시간은 건너뛰며 남은 최대 요청 {report["api_calls_upper_bound"]:,}회.</p><p>선택 격자 {report["selected_unique_grids"]:,}개, 필요한 격자·시간 {total_slots:,}개 중 기존 {existing_slots:,}개.</p>'
    page+=f'<p>전국 모든 격자 보관 시 값 배열만 {report["nationwide_uncompressed_bytes"]/1e9:.1f}GB. 선택 구역은 값 배열만 {total_slots*4/1e9:.2f}GB이며 압축·ID·시각·원응답 보관을 포함한 실제 용량은 시험 수집 후 판단합니다.</p>'
    page+=f'<p>기존 INT 격자 중심점 포함 기준. 연결 격자가 없는 {report["no_grid_units"]:,}구역은 REVIEW입니다. 실제 행정구역 면적 확보율은 아직 검증하지 않았습니다.</p>'
    page+='<p>전국 한 시각을 한 번 받아 여러 구역이 공유합니다. 공식 지점 시계열의 최대 기간은 60분이므로 여러 날짜·구역을 한 요청으로 받을 수 있다고 가정하지 않습니다. 일강우나 5분 중첩 60분 강우를 시간강우 대신 합산하지 않습니다.</p><p>관측 AWS·ASOS 원본은 유지합니다. 융합기상은 분석값이며 이후 별도 강우원으로 연결해야 합니다.</p><p><a href="request_plan.parquet">요청 계획</a> · <a href="admin_grid_coverage_review.parquet">격자 연결 점검</a> · <a href="https://apihub.kma.go.kr/apiList.do?seqApi=971">공식 명세</a></p></html>'
    page=page.replace('</html>','<p>기본 응답은 공식 NetCDF4 다운로드 API입니다. 차원 순서(ny,nx), 격자 크기, 결측·배율·단위를 검사하고 예상과 다르면 중단합니다. 실제 API 응답과 압축률은 아직 확인 전입니다. 먼저 --execute --max-requests 3으로 시험하세요. 인증키는 KMA_API_KEY 환경변수 또는 숨김 입력을 사용합니다. 바이너리 대안은 --response-format binary이며 자동 대체 요청은 하지 않습니다.</p></html>')
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    return plan,daily,report

def fetch(task,key,interval,response_format):
    global NEXT_REQUEST
    end=task['TIME_END'];extension='nc' if response_format=='netcdf' else 'bin'
    path=RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.{extension}.gz'
    if path.exists():
        body=gzip.decompress(path.read_bytes());decode(body)
        return task,path,body,'CACHE'
    with RATE_LOCK:
        wait=max(0.,NEXT_REQUEST-time.monotonic())
        if wait:time.sleep(wait)
        NEXT_REQUEST=time.monotonic()+interval
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    try:
        params=dict(obs='rn_60m',tm=end.strftime('%Y%m%d%H%M'),authKey=key)
        if response_format=='binary':params['disp']='B'
        response=LOCAL.session.get(NC_URL if response_format=='netcdf' else URL,params=params,
                                   timeout=(15,120),allow_redirects=False)
    except requests.RequestException:
        raise RuntimeError('Network failure; raw cache and published Parquet retained. No automatic retry.') from None
    if response.status_code in [401,403,429]:raise RuntimeError(f'API access/quota HTTP {response.status_code}; stopped without retry')
    if response.status_code!=200:raise RuntimeError(f'API HTTP {response.status_code}; stopped without retry')
    body=response.content;decode(body)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.part')
    temporary.write_bytes(gzip.compress(body,compresslevel=3));temporary.replace(path)
    return task,path,body,'API'

def append_sources(records):
    workbook=INT/'index.xlsx';wb=openpyxl.load_workbook(workbook)
    sheet=wb['source'];known={str(r[1]):int(r[0]) for r in sheet.iter_rows(min_row=2,values_only=True) if r[0] is not None and r[1] is not None}
    next_id=max(known.values(),default=0)+1
    for path in records:
        key=str(path)
        if key not in known:
            sheet.append([next_id,key,'hourly_interval_end','Preprocessing/10 rn_60m full-area NetCDF/binary; KST preceding 60min; selected grids'])
            known[key]=next_id;next_id+=1
    tmp=workbook.with_name('index.ver3_10.tmp.xlsx');wb.save(tmp);tmp.replace(workbook)
    return [known[str(p)] for p in records]

def publish(day,frames,raw_paths):
    if not frames:return
    path=partition(day);path.parent.mkdir(parents=True,exist_ok=True)
    new=pd.concat(frames,ignore_index=True);keys=['GRID_ID','time_start']
    old_table=pq.read_table(path) if path.exists() else None
    old=old_table.to_pandas() if old_table is not None else pd.DataFrame(columns=COLS)
    assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
    new=new.merge(old[keys].assign(_EXISTS=True),on=keys,how='left')
    new=new.loc[new._EXISTS.isna(),COLS]
    if new.empty:return
    types={'GRID_ID':old.GRID_ID.dtype if len(old) else 'int32',
           'RN_1H':old.RN_1H.dtype if len(old) else 'float32'}
    merged=pd.concat([old,new],ignore_index=True).astype(types)
    assert not merged.duplicated(keys).any()
    assert merged.time_start.dt.normalize().eq(day).all()
    assert (merged.time_end-merged.time_start).eq(pd.Timedelta(hours=1)).all()
    source_ids=append_sources(raw_paths)
    metadata=dict(old_table.schema.metadata or {}) if old_table is not None else {}
    existing_ids=json.loads(metadata.get(b'source_ids',b'[]'))
    metadata.update({b'timezone':b'Asia/Seoul',b'rain_unit':b'mm',
                     b'source_ids':json.dumps(sorted(set(existing_ids+source_ids))).encode(),
                     b'interval':b'[time_start,time_end)',b'partition_by':b'time_start_date'})
    table=pa.Table.from_pandas(merged[COLS],preserve_index=False).replace_schema_metadata(metadata)
    tmp=path.with_suffix('.ver3_10.tmp.parquet');pq.write_table(table,tmp,compression='zstd')
    check=pq.read_table(tmp).to_pandas()
    pd.testing.assert_frame_equal(check[COLS],merged[COLS],check_dtype=False)
    tmp.replace(path)
    index_path=INT/'index.parquet';idx=pd.read_parquet(index_path)
    relative=path.relative_to(INT).as_posix()
    idx=idx[~idx.file.eq(relative)]
    item=dict(kind='hourly',date=day.strftime('%Y/%m/%d'),rows=len(merged),locations=merged.GRID_ID.nunique(),
              missing=int(merged.RN_1H.isna().sum()),file=relative)
    idx=pd.concat([idx,pd.DataFrame([item])],ignore_index=True)
    temp=index_path.with_suffix('.ver3_10.tmp.parquet');idx.to_parquet(temp,index=False);temp.replace(index_path)
    wb=openpyxl.load_workbook(INT/'index.xlsx');sheet=wb['data'];sheet.delete_rows(1,sheet.max_row)
    sheet.append(idx.columns.tolist())
    for row in idx.itertuples(index=False,name=None):sheet.append([None if pd.isna(x) else x for x in row])
    tmp=INT/'index.ver3_10.tmp.xlsx';wb.save(tmp);tmp.replace(INT/'index.xlsx')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-requests',type=int,default=0,help='Explicit per-run submitted request budget, including cache checks')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--min-interval',type=float,default=1.)
    parser.add_argument('--response-format',choices=['netcdf','binary'],default='netcdf',
                        help='NetCDF4 reduces transfer size when server compression is enabled; pilot first. No automatic fallback.')
    args=parser.parse_args()
    if not 1<=args.workers<=4 or args.min_interval<.3:parser.error('workers 1–4; interval >=0.3 sec')
    if args.execute and args.max_requests<1:parser.error('--execute requires explicit --max-requests within your remaining API quota')
    plan,daily,report=prepare()
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not args.execute:return
    key=os.environ.get('KMA_API_KEY') or getpass.getpass('KMA API key (hidden): ')
    if not key:raise ValueError('KMA_API_KEY required')
    lock=LOG/'collection.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    current_day=None;buffer=[];raw_paths=[];submitted=0;received=0;raw_bytes=0;status='RUNNING';started=time.time()
    log_handle=(LOG/'requests.jsonl').open('a',encoding='utf-8')
    try:
        tasks=plan.loc[plan.MISSING_GRID_COUNT.gt(0)].head(args.max_requests).to_dict('records')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Bounded submission: at most workers requests in flight, no giant future queue.
            for offset in range(0,len(tasks),args.workers):
                batch=tasks[offset:offset+args.workers]
                futures=[pool.submit(fetch,t,key,args.min_interval,args.response_format) for t in batch];submitted+=len(batch)
                for future in futures:
                    task,path,body,origin=future.result();received+=1
                    day=task['DATE'];start=task['TIME_START'];end=task['TIME_END']
                    if current_day is not None and day!=current_day:
                        publish(current_day,buffer,raw_paths);buffer=[];raw_paths=[]
                    current_day=day;ids=daily[day];values=decode(body)[ids].copy()
                    values[~np.isfinite(values)|(values<0)]=np.nan
                    frame=pd.DataFrame(dict(GRID_ID=ids,time_start=start,time_end=end,RN_1H=values))
                    buffer.append(frame);raw_paths.append(path)
                    raw_bytes+=path.stat().st_size
                    item=dict(time_end=end.isoformat(),origin=origin,grid_count=len(ids),null_count=int(np.isnan(values).sum()),
                              raw_compressed_bytes=path.stat().st_size,elapsed_seconds=time.time()-started)
                    log_handle.write(json.dumps(item)+'\n');log_handle.flush()
                    progress=dict(status='RUNNING',received=received,run_budget=args.max_requests,submitted=submitted,
                                  remaining_plan_times=max(0,report['api_calls_upper_bound']-received),current=end.isoformat())
                    (LOG/'progress.json').write_text(json.dumps(progress),encoding='utf-8')
                    print(f'융합 시간강우 {received:,}/{len(tasks):,} | {end} | {len(ids):,}격자 | {origin}',flush=True)
        status='BUDGET_REACHED' if len(tasks)<report['api_calls_upper_bound'] else 'COMPLETE_REQUESTED_TIMES'
    except KeyboardInterrupt:
        status='CANCELLED';raise
    except Exception:
        status='STOPPED_ERROR';raise
    finally:
        try:publish(current_day,buffer,raw_paths)
        finally:
            log_handle.close();lock.unlink(missing_ok=True)
            result=dict(status=status,submitted=submitted,received=received,elapsed_seconds=time.time()-started,
                        api_key_logged=False,original_observations_unchanged=True,
                        response_format=args.response_format,
                        retained_raw_bytes_in_run=raw_bytes,
                        estimated_raw_total_gb_from_run_sample=(raw_bytes/received*report['api_calls_upper_bound']/1e9) if received else None,
                        size_estimate_caution='Run sample may not represent heavy-rain periods; not a storage guarantee',
                        saved_to=str(INT/'hourly'),null_is_not_zero=True)
            (LOG/'run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            (LOG/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
