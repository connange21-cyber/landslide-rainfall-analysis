"""Supplement missing C terrain features from indexed DEM windows, without modifying 05."""
import argparse
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from shapely.geometry import box

V3=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(V3/'codes/5'))
from dem import Mosaic,horn,require_vertical,Progress
MOSAIC=None
CONFIG=None

def initialize():
    global MOSAIC,CONFIG
    CONFIG=json.loads((V3/'config/slope.json').read_text(encoding='utf-8'))
    require_vertical(CONFIG)
    MOSAIC=Mosaic(pd.read_parquet(V3/'output/05/full/dem_tile_index.parquet'),'EPSG:5186')

def calculate(unit):
    started=time.time();geom=unit['geometry'];thresholds=CONFIG['thresholds_deg']
    try:
        minx,miny,maxx,maxy=geom.bounds
        left,top=math.floor(minx/5)*5,math.ceil(maxy/5)*5
        width,height=math.ceil((maxx-left)/5),math.ceil((top-miny)/5)
        transform=from_origin(left,top,5,5)
        count=0;total=0.;steep={t:0 for t in thresholds};blocks=0
        for row in range(0,height,512):
            for col in range(0,width,512):
                win=Window(col,row,min(512,width-col),min(512,height-row))
                a,b,c,d=window_bounds(win,transform)
                if not geom.intersects(box(a,b,c,d)):continue
                z,_=MOSAIC.read((a-5,b-5,c+5,d+5),(int(win.height)+2,int(win.width)+2))
                slope=horn(z,5,5,require_vertical(CONFIG))
                inside=geometry_mask([geom],slope.shape,window_transform(win,transform),invert=True,all_touched=False)
                values=slope[inside & np.isfinite(slope)]
                assert np.all((values>=0)&(values<=90))
                count+=len(values);total+=float(values.sum(dtype='float64'));blocks+=1
                for t in thresholds:steep[t]+=int((values>=t).sum())
        area=geom.area
        result={k:unit[k] for k in ['SPATIAL_UNIT_ID','ADMIN_NAME','ADMIN_LEVEL','ADMIN_CODE']}
        result.update(VALID_CELLS=count,VALID_AREA_M2=count*25.,ADMIN_AREA_NATIVE_M2=area,
            AREA_COVERAGE_FRACTION=count*25/area if area else np.nan,
            MEAN_SLOPE_DEG=total/count if count else np.nan,
            STATUS='COMPUTED' if count else 'NO_VALID_SLOPE',BOUNDARY_TIME_VALIDITY='NOT_VERIFIED',
            BLOCKS=blocks,ELAPSED_SECONDS=time.time()-started,ERROR='')
        for t in thresholds:
            result[f'GE_{t}DEG_CELLS']=steep[t];result[f'GE_{t}DEG_AREA_M2']=steep[t]*25.
            result[f'GE_{t}DEG_FRACTION']=steep[t]/count if count else np.nan
            result[f'GE_{t}DEG_PERCENT']=100*steep[t]/count if count else np.nan
        return result
    except Exception as error:
        return dict(SPATIAL_UNIT_ID=unit['SPATIAL_UNIT_ID'],ADMIN_NAME=unit['ADMIN_NAME'],
                    ADMIN_LEVEL=unit['ADMIN_LEVEL'],ADMIN_CODE=unit['ADMIN_CODE'],STATUS='REVIEW_ERROR',ERROR=repr(error))

def attach(features,out):
    source=V3/'output/09/full/01_integration/abc_integrated.parquet'
    table=pd.read_parquet(source)
    names=['VALID_AREA_M2','AREA_COVERAGE_FRACTION','MEAN_SLOPE_DEG','STATUS']
    names += [f'GE_{n}DEG_{s}' for n in [20,30,40] for s in ['AREA_M2','FRACTION','PERCENT']]
    available=features.set_index('SPATIAL_UNIT_ID')
    mask=table.SAMPLE_TYPE.eq('C') & table.SLOPE_STATUS.eq('PENDING_NO_ADMIN_FEATURES')
    for col in names:
        table.loc[mask,'SLOPE_'+col]=table.loc[mask,'SPATIAL_UNIT_ID'].map(available[col])
    table['SLOPE_STATUS']=table.SLOPE_STATUS.fillna('PENDING_NO_ADMIN_FEATURES')
    table.to_parquet(out/'abc_integrated.parquet',index=False)
    failed=table.loc[~table.SLOPE_STATUS.eq('COMPUTED')|table.SLOPE_AREA_COVERAGE_FRACTION.lt(.9),
                     ['SPATIAL_UNIT_ID','ADMIN_NAME','ADMIN_LEVEL','SLOPE_STATUS','SLOPE_AREA_COVERAGE_FRACTION']].drop_duplicates()
    failed.to_parquet(out/'pending_slope_units.parquet',index=False)
    spec=importlib.util.spec_from_file_location('integration_summary',Path(__file__).with_name('01_integrate_abc.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=out
    return module.render_summary(table)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=4 or args.limit<0:parser.error('workers 1–4; limit >=0')
    mode='smoke' if args.limit else 'full'
    out=V3/f'output/09/{mode}/02_c_slope';log=V3/f'log/09/{mode}/02_c_slope'
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Use --resume; existing results preserved')
    out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
    source_paths=[V3/'output/09/full/01_integration/pending_slope_units.parquet',
                  V3/'output/09/full/01_integration/abc_integrated.parquet',
                  V3/'output/07/full/c_spatial_units_5186.parquet',
                  V3/'output/05/full/dem_tile_index.parquet',V3/'config/slope.json']
    signature={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in source_paths}
    metadata=log/'inputs.json'
    if metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8'))!=signature:raise ValueError('Inputs changed; cannot resume existing checkpoints')
    else:metadata.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    progress_view=Progress(log,'C 행정구역 경사 보강')
    pending=pd.read_parquet(V3/'output/09/full/01_integration/pending_slope_units.parquet')
    registry=gpd.read_parquet(V3/'output/07/full/c_spatial_units_5186.parquet').to_crs(5186)
    units=registry[registry.SPATIAL_UNIT_ID.isin(pending.SPATIAL_UNIT_ID)].copy()
    assert len(units)==len(pending) and units.SPATIAL_UNIT_ID.is_unique
    units['_AREA']=units.geometry.area
    units=units.sort_values(['_AREA','SPATIAL_UNIT_ID'])
    if args.limit:units=units.head(args.limit)
    ledger=log/'units.jsonl';results=[]
    if ledger.exists():
        lines=ledger.read_text(encoding='utf-8').splitlines()
        for i,line in enumerate(lines):
            try:results.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                # Recover only the torn final checkpoint record, in this step's own log.
                ledger.write_text('\n'.join(lines[:i])+'\n',encoding='utf-8')
    done={r['SPATIAL_UNIT_ID'] for r in results}
    tasks=[r for r in units.to_dict('records') if r['SPATIAL_UNIT_ID'] not in done]
    started=time.time()
    with ledger.open('a',encoding='utf-8') as handle,ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for result in pool.map(calculate,tasks,chunksize=1):
            results.append(result);handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush()
            progress=dict(status='RUNNING',done=len(results),total=len(units),current=result['SPATIAL_UNIT_ID'],
                          elapsed_seconds=time.time()-started,workers=args.workers)
            (log/'progress.json').write_text(json.dumps(progress,ensure_ascii=False),encoding='utf-8')
            progress_view.update('DEM 구역별 계산',len(results),len(units),result['ADMIN_NAME'])
            print(f'C 경사 보강 {len(results):,}/{len(units):,} | {result["STATUS"]} | {result["ADMIN_NAME"]}',flush=True)
    features=pd.DataFrame(results)
    assert signature=={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in source_paths}
    assert features.SPATIAL_UNIT_ID.is_unique and len(features)==len(units)
    good=features.STATUS.eq('COMPUTED')
    for t in [20,30,40]:
        assert features.loc[good,f'GE_{t}DEG_FRACTION'].between(0,1).all()
    assert features.loc[good,'GE_20DEG_CELLS'].ge(features.loc[good,'GE_30DEG_CELLS']).all()
    assert features.loc[good,'GE_30DEG_CELLS'].ge(features.loc[good,'GE_40DEG_CELLS']).all()
    features.to_parquet(out/'c_admin_slope_features.parquet',index=False)
    summary=attach(features,out) if not args.limit else None
    report=dict(status='COMPLETE_WITH_REVIEW' if features.STATUS.ne('COMPUTED').any() or features.AREA_COVERAGE_FRACTION.lt(.9).any() else 'COMPLETE',
        units=len(features),status_counts=features.STATUS.value_counts().to_dict(),
        low_coverage_units=int(features.AREA_COVERAGE_FRACTION.lt(.9).sum()),summary=summary,
        elapsed_seconds=time.time()-started,prior_outputs_unchanged=True,
        method='HORN_3X3_ALL_9_VALID',overlap_policy='SMALLER_SHEET_FIRST_VALID',
        shifted_grid='BILINEAR_ESTIMATE',output='FEATURES_ONLY_NO_NEW_RASTER')
    (log/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (log/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    progress_view.finish(report['status'],f'{len(features):,}구역 완료')
    print(json.dumps({k:v for k,v in report.items() if k!='summary'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
