"""Aggregate the slope raster by 04's RI-first units; ratios use valid slope area."""
import argparse
import json
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask,geometry_window
from rasterio.windows import Window
from common import V3,write_json,fingerprint
from dem import require_vertical,summarize,report,Progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    mode='smoke' if args.smoke else 'full'
    out,log=V3/'output/05'/mode,V3/'log/05'/mode
    progress=Progress(log,'02 행정구역별 경사 집계')
    progress.update('입력 확인',force=True)
    config_path=V3/'config/slope.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    scale=require_vertical(config)
    extract=json.loads((log/'01_run.json').read_text(encoding='utf-8'))
    if extract['status']!='COMPLETE':
        detail=extract.get('error',extract.get('reason','')).strip().splitlines()
        message=(f"경사각 추출 미완료 (상태: {extract['status']}). "
                 "01_extract_slope.py를 먼저 완료해야 합니다. "
                 + (f"이전 원인: {detail[-1]}" if detail else ''))
        progress.finish('FAILED',message)
        raise ValueError(message)
    slope_path=out/'slope_degrees.tif'
    stamp=extract['slope_file']
    if slope_path.stat().st_size!=stamp['bytes'] or slope_path.stat().st_mtime_ns!=stamp['mtime_ns']:
        raise ValueError('Slope raster changed after extraction')
    if (out/'admin_slope_features.parquet').exists() and not args.overwrite:
        raise FileExistsError('Explicit --overwrite required')
    ids=pd.read_parquet(out/'processed_units.parquet').SPATIAL_UNIT_ID
    registry=gpd.read_parquet(V3/'output/04/full/spatial_units_5186.parquet')
    thresholds=config['thresholds_deg']
    if any(not 0<=x<=90 for x in thresholds):raise ValueError('Invalid angle threshold')
    run=dict(status='RUNNING',mode=mode,config=config,inputs=[fingerprint(Path(__file__)),fingerprint(config_path)],
             started=pd.Timestamp.now(tz='Asia/Seoul').isoformat())
    features,histograms=[],[]
    try:
        with rasterio.open(slope_path) as ds:
            if float(ds.tags()['vertical_scale_to_m'])!=scale:raise ValueError('Vertical scale changed')
            units=registry[registry.SPATIAL_UNIT_ID.isin(ids)].to_crs(ds.crs)
            cell_area=abs(ds.transform.a*ds.transform.e)
            for i,unit in enumerate(units.itertuples()):
                progress.update('행정구역 집계',i,len(units),f'{unit.ADMIN_NAME} ({unit.SPATIAL_UNIT_ID})')
                geometry=unit.geometry
                win=geometry_window(ds,[geometry])
                window_total=int(np.ceil(win.height/512)*np.ceil(win.width/512))
                window_done=0
                count,total=0,0.
                steep={t:0 for t in thresholds}
                hist=np.zeros(90,dtype='int64')
                for row in range(int(win.row_off),int(win.row_off+win.height),512):
                    for col in range(int(win.col_off),int(win.col_off+win.width),512):
                        window=Window(col,row,min(512,win.col_off+win.width-col),min(512,win.row_off+win.height-row))
                        z=ds.read(1,window=window,masked=True).filled(np.nan)
                        inside=geometry_mask([geometry],z.shape,ds.window_transform(window),invert=True,all_touched=False)
                        values=z[inside & np.isfinite(z)]
                        part=summarize(values,cell_area,thresholds)
                        count+=part['VALID_CELLS'];total+=float(values.sum(dtype='float64'))
                        for t in thresholds:steep[t]+=part[f'GE_{t}DEG_CELLS']
                        hist+=np.histogram(values,bins=np.arange(91))[0]
                        window_done+=1
                        progress.update('행정구역 집계',i,len(units),
                            f'{unit.ADMIN_NAME} ({unit.SPATIAL_UNIT_ID}) · 내부 블록 {window_done}/{window_total}')
                area=float(geometry.area)
                record=dict(SPATIAL_UNIT_ID=unit.SPATIAL_UNIT_ID,ADMIN_LEVEL=unit.ADMIN_LEVEL,ADMIN_CODE=unit.ADMIN_CODE,
                    ADMIN_NAME=unit.ADMIN_NAME,VALID_CELLS=count,VALID_AREA_M2=count*cell_area,ADMIN_AREA_NATIVE_M2=area,
                    AREA_COVERAGE_FRACTION=count*cell_area/area if area else np.nan,
                    MEAN_SLOPE_DEG=total/count if count else np.nan,STATUS='COMPUTED' if count else 'NO_VALID_SLOPE',
                    BOUNDARY_TIME_VALIDITY='NOT_VERIFIED',PARENT_BOUNDARY_PRESENT=unit.PARENT_BOUNDARY_PRESENT)
                for t in thresholds:
                    record[f'GE_{t}DEG_CELLS']=steep[t]
                    record[f'GE_{t}DEG_AREA_M2']=steep[t]*cell_area
                    record[f'GE_{t}DEG_FRACTION']=steep[t]/count if count else np.nan
                    record[f'GE_{t}DEG_PERCENT']=100*steep[t]/count if count else np.nan
                features.append(record)
                histograms.extend(dict(SPATIAL_UNIT_ID=unit.SPATIAL_UNIT_ID,LOWER_DEG=k,UPPER_DEG=k+1,CELLS=int(n),AREA_M2=int(n)*cell_area)
                                  for k,n in enumerate(hist))
                progress.update('행정구역 집계',i+1,len(units),unit.SPATIAL_UNIT_ID)
        progress.update('집계 파일 저장·검증',force=True)
        table=pd.DataFrame(features);hist=pd.DataFrame(histograms)
        for frame,name in [(table,'admin_slope_features'),(hist,'slope_histogram_1deg')]:
            temp=out/(name+'.tmp.parquet');frame.to_parquet(temp,index=False);temp.replace(out/(name+'.parquet'))
        if table.SPATIAL_UNIT_ID.duplicated().any() or not int(hist.CELLS.sum())==int(table.VALID_CELLS.sum()):
            raise ValueError('Unit uniqueness/histogram counts failed')
        for t in thresholds:
            valid=table.VALID_CELLS.gt(0)
            if not table.loc[valid,f'GE_{t}DEG_FRACTION'].between(0,1).all():raise ValueError('Invalid steep fraction')
        run.update(status='COMPLETE',units=len(table),no_valid_slope=int(table.VALID_CELLS.eq(0).sum()),
                   low_coverage_lt_90pct=int(table.AREA_COVERAGE_FRACTION.lt(.9).sum()),
                   boundary_sampling_coverage_over_100pct=int(table.AREA_COVERAGE_FRACTION.gt(1).sum()),
                   ratios_denominator='VALID_SLOPE_CELL_AREA',histogram_degree_width=1,dem_original_unchanged=True)
        report(out,'집계 완료',[('처리 행정구역',len(table)),('유효 경사 없음',run['no_valid_slope']),
                    ('분석 면적 확보율 90% 미만',run['low_coverage_lt_90pct'])],
               '20°·30°·40° 이상 비율의 분모는 유효 경사 면적이다. 전체 구역 대비 확보율은 별도로 저장했다. 경계 격자는 중심점 기준으로 포함하며 면적은 5m 격자 기반 추정치다.')
        progress.finish('COMPLETE',f'{len(table):,}개 행정구역 집계 완료')
    except KeyboardInterrupt:
        run.update(status='CANCELLED');progress.finish('CANCELLED','사용자 중단');raise
    except Exception:
        run.update(status='FAILED',error=traceback.format_exc());progress.finish('FAILED',run['error'].splitlines()[-1]);raise
    finally:
        run['finished']=pd.Timestamp.now(tz='Asia/Seoul').isoformat()
        write_json(log/'02_run.json',run)


if __name__=='__main__':main()
