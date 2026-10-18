"""Extract slope for resolved occurrence units; one sparse native-grid GeoTIFF."""
import argparse
import json
import math
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, bounds as window_bounds
from rasterio.features import geometry_mask
from shapely.geometry import box
from common import V3,write_json,fingerprint
from dem import require_vertical,build_index,Mosaic,horn,report,Progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--unit-id',action='append',help='Explicit additional/selected registry unit, repeatable')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--header-workers',type=int,default=None,help='Concurrent header readers (default: config, 4)')
    parser.add_argument('--header-executor',choices=['process','thread'],default=None)
    parser.add_argument('--index-only',action='store_true',help='Build headers-only tile index; no elevation arrays')
    args = parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    mode = 'smoke' if args.limit else 'full'
    out,log = V3/'output/05'/mode,V3/'log/05'/mode
    out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
    progress=Progress(log,'01 경사각 추출')
    progress.update('설정 확인',force=True)
    config_path = V3/'config/slope.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run = dict(status='RUNNING',config=config,mode=mode,started=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
               inputs=[fingerprint(config_path),fingerprint(Path(__file__)),fingerprint(Path(__file__).with_name('dem.py'))])
    try:
        if not args.index_only:
            try:
                scale = require_vertical(config)
            except ValueError as exc:
                run.update(status='WAITING_HEIGHT_UNIT_CONFIRMATION',reason=str(exc),dem_pixel_arrays_read=False)
                report(out,'높이 단위 확인 대기',
                    [('경사 계산 코드','작성 완료'),('20°·30°·40° 집계 코드','작성 완료'),('실제 DEM 계산','미실행')],
                    'DEM에 높이 단위가 기록되지 않아 실제 경사각을 확정하지 않았다. 단위 확인 후 실행한다.')
                print('WAITING_HEIGHT_UNIT_CONFIRMATION: no DEM pixels read',flush=True)
                return
        target = out/'slope_degrees.tif'
        if target.exists() and not args.overwrite and not args.index_only:
            raise FileExistsError('Existing slope raster; explicit --overwrite required')
        index,crs = build_index(config['dem_root'],out,log,progress,
            workers=args.header_workers if args.header_workers is not None else config.get('header_workers',4),
            executor=args.header_executor or config.get('header_executor','process'),target_crs=config.get('target_crs','EPSG:5186'))
        if args.index_only:
            run['status']='INDEX_COMPLETE';progress.finish('COMPLETE','DEM 헤더 인덱스 완료');return
        progress.update('분석 공간 경계 준비',force=True)
        upstream = json.loads((V3/'log/04/full/run.json').read_text(encoding='utf-8'))
        if upstream['status']!='COMPLETE':raise ValueError('Stage 04 incomplete')
        groups = pd.read_parquet(V3/'output/04/full/admin_rain_event_features.parquet',columns=['SPATIAL_UNIT_ID','ADMIN_GROUP_RESOLVED'])
        requested = sorted(set(args.unit_id or groups.loc[groups.ADMIN_GROUP_RESOLVED,'SPATIAL_UNIT_ID']))
        if args.limit:requested=requested[:args.limit]
        registry = gpd.read_parquet(V3/'output/04/full/spatial_units_5186.parquet')
        if not set(requested).issubset(set(registry.SPATIAL_UNIT_ID)):
            raise ValueError('Unknown requested spatial unit ID')
        units = registry[registry.SPATIAL_UNIT_ID.isin(requested)].to_crs(crs)
        if units.empty:raise ValueError('No selected units')
        units[['SPATIAL_UNIT_ID','ADMIN_LEVEL','ADMIN_CODE']].to_parquet(out/'processed_units.parquet',index=False)
        minx,miny,maxx,maxy = units.total_bounds
        left,top=math.floor(minx/5)*5,math.ceil(maxy/5)*5
        width,height=math.ceil((maxx-left)/5),math.ceil((top-miny)/5)
        transform=from_origin(left,top,5,5)
        block=int(config['block_size'])
        if block<16 or block%16:raise ValueError('Block size must be a positive multiple of 16')
        needed={}
        for unit_number,geom in enumerate(units.geometry):
            progress.update('처리 블록 선정',unit_number,len(units),units.iloc[unit_number].SPATIAL_UNIT_ID)
            a,b,c,d=geom.bounds
            c0=max(0,math.floor((a-left)/(5*block)));c1=math.floor((c-left)/(5*block))
            r0=max(0,math.floor((top-d)/(5*block)));r1=math.floor((top-b)/(5*block))
            for row in range(r0,r1+1):
                for col in range(c0,c1+1):
                    win=Window(col*block,row*block,min(block,width-col*block),min(block,height-row*block))
                    if win.width>0 and win.height>0 and geom.intersects(box(*window_bounds(win,transform))):
                        needed.setdefault((row,col),[]).append(geom)
        mosaic=Mosaic(index,crs)
        profile=dict(driver='GTiff',width=width,height=height,count=1,dtype='float32',crs=crs,transform=transform,
                     nodata=-9999.,tiled=True,blockxsize=block,blockysize=block,compress='deflate',BIGTIFF='YES',SPARSE_OK='TRUE')
        temporary=out/'slope_degrees.tmp.tif'
        valid_cells=conflicts=0
        progress.update('경사각 계산',0,len(needed),force=True)
        with rasterio.open(temporary,'w',**profile) as dst:
            dst.update_tags(method='HORN_3X3',slope_unit='degree',vertical_scale_to_m=str(scale),
                            vertical_unit_confirmation=config['vertical_unit_confirmation'],neighborhood='ALL_9_VALID',pixel_rule='CENTER',
                            overlap_policy='SMALLER_SHEET_FIRST_VALID',shifted_grid='BILINEAR_ESTIMATE')
            for i,((row,col),geoms) in enumerate(sorted(needed.items())):
                win=Window(col*block,row*block,min(block,width-col*block),min(block,height-row*block))
                a,b,c,d=window_bounds(win,transform)
                z,bad=mosaic.read((a-5,b-5,c+5,d+5),(int(win.height)+2,int(win.width)+2))
                slope=horn(z,5,5,scale)
                inside=geometry_mask(geoms,slope.shape,rasterio.windows.transform(win,transform),invert=True,all_touched=False)
                slope[~inside]=np.nan
                valid_cells+=int(np.isfinite(slope).sum());conflicts+=bad
                dst.write(np.where(np.isfinite(slope),slope,-9999).astype('float32'),1,window=win)
                progress.update('경사각 계산',i+1,len(needed),f'블록 행={row}, 열={col}; 유효 격자={valid_cells:,}')
        progress.update('경사 파일 확정',force=True)
        temporary.replace(target)
        run.update(status='COMPLETE',processed_units=len(units),blocks=len(needed),valid_slope_cells=valid_cells,
                   conflicting_overlap_pixel_occurrences=conflicts,overlap_policy='SMALLER_SHEET_FIRST_VALID',
                   shifted_grid_policy='BILINEAR_ESTIMATE',resampled_source_paths=sorted(mosaic.resampled_paths),
                   native_crs=crs.to_wkt(),dem_source_files=len(index),
                   slope_file=dict(path=str(target),bytes=target.stat().st_size,mtime_ns=target.stat().st_mtime_ns),
                   dem_pixel_arrays_read=True)
        report(out,'경사각 추출 완료 · 행정구역 집계 대기',[('처리 공간 단위',len(units)),('유효 경사 격자',valid_cells)],
               '작은 도엽번호의 유효값 우선. 어긋난 격자는 쌍선형 보간한 추정값이다. 이웃 1격자를 함께 읽으며 결측 주변 3×3은 경사를 계산하지 않았다.')
        progress.finish('COMPLETE','다음: 02_summarize_slope.py 실행')
    except KeyboardInterrupt:
        run.update(status='CANCELLED');progress.finish('CANCELLED','사용자 중단');raise
    except Exception:
        run.update(status='FAILED',error=traceback.format_exc());progress.finish('FAILED',run['error'].splitlines()[-1]);raise
    finally:
        run['finished']=pd.Timestamp.now(tz='Asia/Seoul').isoformat()
        write_json(log/'01_run.json',run)


if __name__=='__main__':main()
