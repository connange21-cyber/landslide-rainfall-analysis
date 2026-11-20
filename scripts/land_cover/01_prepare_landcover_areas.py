"""Prepare only the selected A/B spatial coverage for land-cover acquisition."""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    out=V3/'output/08/full'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Existing 08 preparation; no automatic overwrite')
    paths=[V3/'output/06/full/05_selection/occurrence_selected.parquet',
           V3/'output/06/full/05_selection/b_event_selected.parquet',
           V3/'output/04/full/spatial_units_5186.parquet']
    before=[fingerprint(p) for p in paths]
    a,b=[pd.read_parquet(p) for p in paths[:2]]
    registry=gpd.read_parquet(paths[2])
    assert set(a.SPATIAL_UNIT_ID)==set(b.SPATIAL_UNIT_ID)
    units=registry[registry.SPATIAL_UNIT_ID.isin(a.SPATIAL_UNIT_ID)].copy()
    assert len(units)==a.SPATIAL_UNIT_ID.nunique() and units.SPATIAL_UNIT_ID.is_unique
    counts=a.groupby('SPATIAL_UNIT_ID').size().rename('A_SAMPLE_COUNT')
    units=units.merge(counts,on='SPATIAL_UNIT_ID',validate='one_to_one')
    units['B_SAMPLE_COUNT']=units.SPATIAL_UNIT_ID.map(b.groupby('SPATIAL_UNIT_ID').size())
    assert units.A_SAMPLE_COUNT.eq(units.B_SAMPLE_COUNT).all()
    geographic=units.to_crs(4326)
    for column in ['minx','miny','maxx','maxy']:
        units['WGS84_'+column.upper()]=geographic.bounds[column].to_numpy()
    units['LANDCOVER_STATUS']='WAITING_SOURCE_DATA'
    assert before==[fingerprint(p) for p in paths]
    out.mkdir(parents=True)
    units.to_parquet(out/'required_spatial_units.parquet',index=False)
    unitcols=['ADMIN_NAME','SPATIAL_UNIT_ID','A_SAMPLE_COUNT','B_SAMPLE_COUNT',
              'WGS84_MINX','WGS84_MINY','WGS84_MAXX','WGS84_MAXY','LANDCOVER_STATUS']
    table=units[unitcols].sort_values('ADMIN_NAME').to_html(index=False,border=0,escape=True)
    (out/'summary.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>08 토지피복 준비</title>
<style>body{font:14px/1.6 Malgun Gothic;margin:30px;color:#24364a}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}section{max-height:600px;overflow:auto}</style>
<h1>08 토지피복 자료 확보 준비</h1>'''+f'<p>A·B 각 {len(a):,}개 표본에 필요한 행정구역 {len(units):,}개를 추렸습니다. 행정구역 경계는 EPSG:5186, 검색용 범위는 WGS84 경위도입니다.</p>'+
        '<p>자료 저장 위치: base/landcover. 아직 토지피복 자료·도엽 인덱스가 없어 필요한 도엽번호와 면적 비율은 계산하지 않았습니다.</p>'+
        '<p>다음: 세분류 자료의 기준연도·분류코드·좌표계 확인 → 이 경계와 겹치는 도엽 확보 → 구역별 분류 면적·확보율 계산. 이벤트 이후 자료는 별도 표시합니다.</p><section>'+table+'</section></html>',encoding='utf-8')
    write_json(V3/'log/08/full/01_preparation/run.json',dict(status='COMPLETE_PREPARATION_ONLY',
        inputs=before,spatial_units=len(units),a_samples=len(a),b_samples=len(b),
        upstream_unchanged=True,source_landcover='NOT_PROVIDED',tile_numbers='NOT_DETERMINED',
        area_ratios_calculated=False))
    print(f'08 preparation: {len(units)} units; no land-cover classification or download performed')


if __name__=='__main__':
    main()
