"""Export five matched ABC examples to FileGDB with 360 hourly records per grid."""
from pathlib import Path
import importlib.util
import gzip
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/12/full/01_time_slider'
LOG=V3/'log/12/full/01_time_slider'
SEED=20261014
FIRST='ARG_002370d5ed8a54f45fc8'

def signature(path):
    st=path.stat();return [st.st_size,st.st_mtime_ns]

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    gdb=OUT/'abc_rainfall_examples.gdb'
    if gdb.exists():raise FileExistsError('Existing GDB is preserved; choose a different output path for another export')
    paths=[V3/'output/11/full/01_rainfall_summary/abc_fusion_rainfall.parquet',
        V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet',
        V3/'output/04/full/spatial_units_5186.parquet',
        Path('/data/RAIN_DATAS/융합기상/INT/grid.parquet'),
        V3/'scripts/10_융합강우_수집/01_collect_fusion_hourly_fixed.py']
    inputs={str(p):signature(p) for p in paths}
    all_samples=pd.read_parquet(paths[0]);assert all_samples.VALUE_PROVENANCE.str.startswith('RAW_NETCDF').all()
    counts=all_samples.pivot(index='A_REFERENCE_ID',columns='SAMPLE_TYPE',values='GRID_COUNT')
    complete=all_samples.assign(OK=all_samples.FUSION_15D_MISSING_HOURS.eq(0)).groupby('A_REFERENCE_ID').OK.all()
    eligible=counts[(counts.min(axis=1)>0)&(counts.max(axis=1)<=30)&complete].index.tolist()
    assert FIRST in eligible
    random=np.random.default_rng(SEED);order=[FIRST]+list(random.permutation([x for x in eligible if x!=FIRST]))
    chosen=[];seen_a=set();seen_c=set()
    for group in order:
        part=all_samples[all_samples.A_REFERENCE_ID.eq(group)]
        au=part.loc[part.SAMPLE_TYPE.eq('A'),'SPATIAL_UNIT_ID'].iloc[0]
        cu=part.loc[part.SAMPLE_TYPE.eq('C'),'SPATIAL_UNIT_ID'].iloc[0]
        if au in seen_a or cu in seen_c:continue
        chosen.append(group);seen_a.add(au);seen_c.add(cu)
        if len(chosen)==5:break
    assert len(chosen)==5
    samples=all_samples[all_samples.A_REFERENCE_ID.isin(chosen)].copy()
    case_numbers={group:i+1 for i,group in enumerate(chosen)}
    samples['CASE_NO']=samples.A_REFERENCE_ID.map(case_numbers).astype('int32')
    samples=samples.sort_values(['CASE_NO','SAMPLE_TYPE']);assert len(samples)==15
    links=pd.read_parquet(paths[1]);grid=pd.read_parquet(paths[3])
    used_links=links[links.SPATIAL_UNIT_ID.isin(samples.SPATIAL_UNIT_ID)]
    ids=np.sort(used_links.GRID_ID.unique())
    grid=grid[grid.GRID_ID.isin(ids)].sort_values('GRID_ID')
    assert np.array_equal(grid.GRID_ID.to_numpy(),ids)
    spec=importlib.util.spec_from_file_location('export_decoder',paths[4]);decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    dates=sorted(set(day for end in samples.END_DATE for day in pd.date_range(end-pd.Timedelta(days=14),end)))
    cache={};raw_sources=[]
    for num,day in enumerate(dates,1):
        values=[]
        for h in range(24):
            end=day+pd.Timedelta(hours=h+1)
            path=decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            before=signature(path)
            rain=decoder.decode(gzip.decompress(path.read_bytes()))[ids].copy()
            rain[(~np.isfinite(rain))|(rain<0)]=np.nan
            values.append(rain);assert before==signature(path)
            raw_sources.append(dict(path=str(path),stamp=before))
        cache[day]=np.stack(values)
        if num%10==0 or num==len(dates):
            print(f'12 원응답 읽기 {num}/{len(dates)}일',flush=True)
            (LOG/'progress.json').write_text(json.dumps(dict(status='READING_RAW',done=num,total=len(dates))),encoding='utf-8')
    registry=gpd.read_parquet(paths[2]).to_crs(5186)
    boundary_rows=[];layers={k:[] for k in 'ABC'};selection=[]
    for r in samples.itertuples(index=False):
        unit=r.SPATIAL_UNIT_ID
        gg=grid[grid.GRID_ID.isin(used_links.loc[used_links.SPATIAL_UNIT_ID.eq(unit),'GRID_ID'])].copy()
        indexes=np.searchsorted(ids,gg.GRID_ID.to_numpy());n=len(gg)
        times=pd.date_range(pd.Timestamp(r.END_DATE)-pd.Timedelta(days=14),periods=360,freq='h')
        rain=np.concatenate([cache[day][:,indexes] for day in pd.date_range(times[0].normalize(),times[-1].normalize())]).reshape(-1)
        f=pd.DataFrame(dict(CASE_NO=np.int32(r.CASE_NO),SAMPLE_TYPE=r.SAMPLE_TYPE,
            EXAMPLE_ID=f'E{r.CASE_NO:02d}_{r.SAMPLE_TYPE}',TRIPLET_ID=r.A_REFERENCE_ID,
            SPATIAL_UNIT_ID=unit,ADMIN_NAME=r.ADMIN_NAME,
            GRID_ID=np.tile(gg.GRID_ID.to_numpy(dtype='int32'),360),
            LON=np.tile(gg.LON.to_numpy(),360),LAT=np.tile(gg.LAT.to_numpy(),360),
            START_TIME=np.repeat(times.to_numpy(),n),END_TIME=np.repeat((times+pd.Timedelta(hours=1)).to_numpy(),n),
            REFERENCE_END_DATE=pd.Timestamp(r.END_DATE),RN_1H=rain.astype('float32'),
            RAIN_QC=np.where(np.isfinite(rain),'VALID','MISSING'),TIME_ZONE='Asia/Seoul',
            RAIN_SOURCE='FUSION_RN_60M_RAW_SCALE_APPLIED',REFERENCE_STN=np.int32(r.STN)))
        g=gpd.GeoDataFrame(f,geometry=gpd.points_from_xy(f.LON,f.LAT),crs=4326).to_crs(5186)
        poly=registry[registry.SPATIAL_UNIT_ID.eq(unit)]
        assert len(poly)==1 and g.geometry.within(poly.geometry.iloc[0]).all()
        assert not f.duplicated(['EXAMPLE_ID','GRID_ID','START_TIME']).any()
        assert f.groupby('GRID_ID').size().eq(360).all()
        assert np.isfinite(rain).all(), 'Complete example unexpectedly missing'
        cumulative=rain.reshape(360,n).mean(axis=1).sum()
        assert abs(cumulative-r.FUSION_15D_MM)<.002
        layers[r.SAMPLE_TYPE].append(g)
        item=dict(CASE_NO=int(r.CASE_NO),SAMPLE_TYPE=r.SAMPLE_TYPE,TRIPLET_ID=r.A_REFERENCE_ID,
            SPATIAL_UNIT_ID=unit,ADMIN_NAME=r.ADMIN_NAME,GRID_COUNT=n,
            WINDOW_START=str(times[0]),WINDOW_END=str(times[-1]+pd.Timedelta(hours=1)),
            END_DATE=str(pd.Timestamp(r.END_DATE).date()),ROWS=len(f),RAIN_15D_MM=float(cumulative))
        selection.append(item)
        boundary_rows.append(dict(CASE_NO=np.int32(r.CASE_NO),SAMPLE_TYPE=r.SAMPLE_TYPE,
            EXAMPLE_ID=f'E{r.CASE_NO:02d}_{r.SAMPLE_TYPE}',TRIPLET_ID=r.A_REFERENCE_ID,
            SPATIAL_UNIT_ID=unit,ADMIN_NAME=r.ADMIN_NAME,GRID_COUNT=np.int32(n),geometry=poly.geometry.iloc[0]))
    counts_written={}
    for kind,parts in layers.items():
        frame=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),geometry='geometry',crs=5186)
        name=f'{kind}_hourly'
        pyogrio.write_dataframe(frame,gdb,layer=name,driver='OpenFileGDB')
        check=pyogrio.read_dataframe(gdb,layer=name)
        assert len(check)==len(frame) and check.crs.to_epsg()==5186
        fields=['EXAMPLE_ID','GRID_ID','START_TIME','END_TIME','RN_1H','RAIN_QC','LON','LAT']
        pd.testing.assert_frame_equal(check[fields].reset_index(drop=True),frame[fields].reset_index(drop=True),check_dtype=False,rtol=1e-6)
        counts_written[name]=len(frame)
        print(f'12 GDB 저장 {name}: {len(frame):,}행',flush=True)
    boundaries=gpd.GeoDataFrame(boundary_rows,geometry='geometry',crs=5186)
    pyogrio.write_dataframe(boundaries,gdb,layer='admin_boundaries',driver='OpenFileGDB')
    assert len(pyogrio.read_dataframe(gdb,layer='admin_boundaries'))==15
    assert inputs=={str(p):signature(p) for p in paths}
    assert all(s['stamp']==signature(Path(s['path'])) for s in raw_sources)
    summary=pd.DataFrame(selection)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>12 ABC 시간 슬라이더 GDB</title><style>body{font:15px/1.7 Malgun Gothic;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}code{overflow-wrap:anywhere}</style><h1>12 · ABC 시간강우 GDB</h1>'
    page+='<p>5묶음 × ABC = 15표본. 각 격자별 360시간을 실제 시각으로 저장했습니다. EPSG:5186, 경위도 LON/LAT는 EPSG:4326입니다.</p>'
    page+='<p><code>abc_rainfall_examples.gdb</code> 안의 A_hourly / B_hourly / C_hourly / admin_boundaries를 ArcGIS Pro에 추가하세요.</p>'
    page+='<ol><li>레이어 속성 → 시간 → 단일 시간 필드 START_TIME, 한국 시간, 1시간 간격.</li><li>RN_1H(mm)로 점 색상/크기를 설정합니다. END_TIME은 원래 구간의 배타적 종료 시각입니다.</li><li>CASE_NO = 1~5로 한 묶음을 선택하면 공간·시간 비교가 쉽습니다. 실제 날짜이므로 B는 A/C와 다른 날 표시됩니다.</li></ol>'
    page+='<p>ArcGIS Pro Python 창에서 다음을 실행하면 레이어 시간 설정과 .lyrx 저장을 자동 적용합니다.</p><pre><code>exec(open(r"'+str(V3/'scripts/12_시각화_A재연결/02_enable_time_in_arcgis.py')+'", encoding="utf-8").read())</code></pre>'
    page+='<p>표본은 자료가 완전하고 각 구역의 격자가 30개 이하인 예시 후보에서 기존 하탄방리 묶음과 고정 시드로 선택했습니다. 학습용 대표 추출은 아닙니다. B/C는 발생기록 미포착입니다.</p>'
    page+=summary.to_html(index=False,border=0)+'<p>단위 보정된 원응답에서 재산출했습니다. RAIN_DATAS의 기존 hourly Parquet은 변경하지 않았습니다. ArcPy 별도 실행 라이선스 오류로 .lyrx 자동 설정은 Pro 안에서 실행해야 합니다.</p></html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    report=dict(status='COMPLETE_GDB_TIME_CONFIGURATION_IN_PRO_REQUIRED',gdb=str(gdb),rows=counts_written,
        selected_samples=selection,seed=SEED,crs=5186,time_field='START_TIME',time_zone='Asia/Seoul',
        intervals='[START_TIME,END_TIME)',inputs=inputs,raw_sources=raw_sources,
        original_data_unchanged=True,elapsed_seconds=time.time()-started)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps(dict(status=report['status'],rows=counts_written)),encoding='utf-8')

if __name__=='__main__':main()
