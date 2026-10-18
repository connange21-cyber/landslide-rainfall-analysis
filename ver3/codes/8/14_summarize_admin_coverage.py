"""Measure actual downloaded land-cover polygon coverage; earlier outputs are read-only."""
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

V3 = Path(__file__).resolve().parents[2]
OUT = V3/'output/08/full/14_admin_coverage'
LOG = V3/'log/08/full/14_admin_coverage'

def footprint(task):
    year, tile, folder = task
    try:
        paths = list(folder.glob('*.shp'))
        if not paths:
            raise ValueError('No SHP')
        geometries = []
        for path in paths:
            frame = pyogrio.read_dataframe(path, columns=[])
            if frame.crs is None:
                raise ValueError('CRS missing')
            frame = frame.to_crs(5186)
            geometries.extend(shapely.make_valid(frame.geometry.array))
        shape = shapely.union_all(geometries)
        if shape.is_empty:
            raise ValueError('Empty footprint')
        return dict(REFERENCE_YEAR=year,TILE_ID=tile,STATUS='OK',ERROR='',geometry=shape)
    except Exception as error:
        return dict(REFERENCE_YEAR=year,TILE_ID=tile,STATUS='REVIEW',ERROR=str(error),geometry=None)

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    started=time.time()
    audit=pd.read_parquet(V3/'output/08/full/13_file_audit/archive_inventory.parquet')
    tasks=[]
    for row in audit.itertuples():
        if row.STATUS=='OK':
            tasks.append((int(row.YEAR),str(row.TILE_ID),V3/'output/08/full/12_extracted'/str(row.YEAR)/Path(row.ZIP).stem))
    cache=OUT/'tile_footprints.parquet'
    rows=gpd.read_parquet(cache).to_dict('records') if cache.exists() else []
    known={(r['REFERENCE_YEAR'],r['TILE_ID']) for r in rows}
    remaining=[t for t in tasks if t[:2] not in known]
    def save():
        temp=cache.with_suffix('.tmp.parquet')
        gpd.GeoDataFrame(rows,geometry='geometry',crs=5186).to_parquet(temp,index=False)
        temp.replace(cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(footprint,remaining),1):
            rows.append(row)
            if i%100==0 or i==len(remaining):
                elapsed=time.time()-started
                progress=dict(status='READING_FOOTPRINTS',done=len(rows),total=len(tasks),elapsed_seconds=round(elapsed,1))
                (LOG/'progress.json').write_text(json.dumps(progress),encoding='utf-8')
                print(f'피복 경계 {len(rows):,}/{len(tasks):,} | 경과 {elapsed/60:.1f}분',flush=True)
            if i%500==0 or i==len(remaining):save()
    footprints=gpd.GeoDataFrame(rows,geometry='geometry',crs=5186)
    results=[]
    for file in sorted((V3/'output/08/full/02_yearly_requirements').glob('*/required_admin_units.parquet')):
        units=gpd.read_parquet(file).to_crs(5186)
        year=int(file.parent.name)
        tiles=footprints[footprints.REFERENCE_YEAR.eq(year)&footprints.STATUS.eq('OK')].reset_index(drop=True)
        for admin in units.itertuples():
            geom=shapely.make_valid(admin.geometry)
            candidates=tiles.iloc[tiles.sindex.query(geom,predicate='intersects')]
            clipped=shapely.intersection(candidates.geometry.array,geom)
            covered=shapely.union_all(clipped).area if len(clipped) else 0.0
            total=geom.area
            ratio=min(1.0,covered/total) if total else float('nan')
            results.append(dict(REFERENCE_YEAR=year,SPATIAL_UNIT_ID=admin.SPATIAL_UNIT_ID,
                ADMIN_NAME=admin.ADMIN_NAME,ADMIN_CODE=admin.ADMIN_CODE,ADMIN_LEVEL=admin.ADMIN_LEVEL,
                A_SAMPLE_COUNT=admin.A_SAMPLE_COUNT,B_SAMPLE_COUNT=admin.B_SAMPLE_COUNT,C_SAMPLE_COUNT=admin.C_SAMPLE_COUNT,
                ADMIN_AREA_M2=total,COVERED_AREA_M2=covered,UNCOVERED_AREA_M2=max(0,total-covered),
                COVERAGE_FRACTION=ratio,COVERAGE_PERCENT=ratio*100,AVAILABLE_TILE_COUNT=len(candidates),
                STATUS='NO_COVERAGE' if covered==0 else 'COVERED_WITHIN_1M2' if total-covered<=1 else 'PARTIAL_COVERAGE'))
        print(f'{year} 행정구역 {len(units):,}개 집계 완료',flush=True)
    result=pd.DataFrame(results)
    result.to_parquet(OUT/'admin_landcover_coverage.parquet',index=False)
    bands=pd.cut(result.COVERAGE_PERCENT,[-1,0,90,99,99.9,100.00001],labels=['0%','0–90%','90–99%','99–99.9%','99.9–100%'])
    table=pd.crosstab(result.REFERENCE_YEAR,bands,dropna=False)
    review=result[result.UNCOVERED_AREA_M2.gt(1)].sort_values('COVERAGE_PERCENT')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>14 행정구역 피복 확보율</title><style>body{font:14px/1.6 Malgun Gothic;margin:30px}table{border-collapse:collapse}th,td{padding:6px;border-bottom:1px solid #ddd}</style><h1>행정구역별 실제 피복 확보율</h1>'
    page+=f'<p>ABC 사용 행정구역–연도 {len(result):,}건. 다운로드한 정상 SHP 폴리곤 합집합 ∩ 행정구역 면적 / 전체 행정구역 면적. EPSG:5186.</p>'
    page+='<p>확보율은 토지피복 분류 비율이 아닙니다. 미제공 원인·수면 여부는 미확정이며 표본을 자동 제외하지 않았습니다. 미완료 압축 해제 2건은 미확보로 계산합니다. 촬영 시점 적합성은 별도 점검 대상입니다.</p>'
    page+=table.to_html()+f'<h2>미확보 면적 1㎡ 초과: {len(review):,}건</h2>'
    page+='<p>전체 행정구역과 ABC 표본 수는 <a href="admin_landcover_coverage.parquet">Parquet</a>에 저장했습니다. 아래 표는 확보율이 낮은 순서이며 브라우저 검색으로 구역명을 찾을 수 있습니다.</p>'
    page+=review[['REFERENCE_YEAR','ADMIN_NAME','SPATIAL_UNIT_ID','A_SAMPLE_COUNT','B_SAMPLE_COUNT','C_SAMPLE_COUNT','COVERAGE_PERCENT','UNCOVERED_AREA_M2']].to_html(index=False,float_format=lambda x:f'{x:,.3f}')+'</html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    run=dict(status='COMPLETE',admin_year_count=len(result),unique_admin_count=result.SPATIAL_UNIT_ID.nunique(),
        tile_count=len(footprints),tile_read_errors=int(footprints.STATUS.ne('OK').sum()),
        uncovered_over_1m2_count=len(review),coverage_bands=table.to_dict(),elapsed_seconds=time.time()-started)
    (LOG/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table.to_string(),flush=True)

if __name__=='__main__':main()
