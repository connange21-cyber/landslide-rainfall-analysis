"""Check supplied 5K index and prepare year/tile requirements with 1m QC tolerance."""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    source=V3/'base/국가기본도_도엽인덱스5K/TN_MAPINDX_5K.shp'
    sample=V3/'output/08/test.zip'
    files=sorted((V3/'output/08/full/02_yearly_requirements').glob('*/required_admin_units.parquet'))
    out=V3/'output/08/full/05_tile_index'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing index check; no overwrite')
    inputs=[fingerprint(p) for p in [*source.parent.glob('TN_MAPINDX_5K.*'),sample,*files]]
    raw=gpd.read_file(source)
    assert raw.crs is not None and raw.geometry.is_valid.all() and not raw.geometry.is_empty.any()
    tiles=raw.rename(columns={'MAPIDCD_NO':'TILE_ID','MAPID_NM':'TILE_NAME'})[['TILE_ID','TILE_NAME','geometry']].to_crs(5186)
    tiles['TILE_ID']=tiles.TILE_ID.astype(str)
    assert tiles.TILE_ID.is_unique and tiles.TILE_ID.str.fullmatch(r'\d{8}').all()
    cover=gpd.read_file('zip://'+sample.as_posix()+'!37705061.shp').to_crs(5186)
    shape=cover.geometry.union_all();tile=tiles[tiles.TILE_ID.eq('37705061')].geometry.iloc[0]
    comparison=dict(tile_id='37705061',tile_name='성동061',sample_features=len(cover),
        index_coverage_fraction=shape.intersection(tile).area/tile.area,
        boundary_hausdorff_m=shape.boundary.hausdorff_distance(tile.boundary),
        data_outside_index_m2=shape.difference(tile).area,
        data_outside_1m_buffer_m2=shape.difference(tile.buffer(1)).area,
        tolerance_m=1,scope='ONE_DOWNLOADED_TILE_ONLY')
    if comparison['data_outside_1m_buffer_m2']>1:comparison['status']='REVIEW'
    else:comparison['status']='MATCH_WITHIN_1M'
    units=gpd.GeoDataFrame(pd.concat([gpd.read_parquet(p) for p in files],ignore_index=True),crs=5186)
    links=[];qc=[]
    for i,r in enumerate(units.itertuples(),1):
        candidates=tiles.iloc[tiles.sindex.query(r.geometry,predicate='intersects')]
        matched=[]
        core=r.geometry.buffer(-1)
        for t in candidates.itertuples():
            area=r.geometry.intersection(t.geometry).area
            if area<=0:continue
            matched.append(t.geometry)
            links.append(dict(REFERENCE_YEAR=int(r.REFERENCE_YEAR),SPATIAL_UNIT_ID=r.SPATIAL_UNIT_ID,
                ADMIN_NAME=r.ADMIN_NAME,ADMIN_LEVEL=r.ADMIN_LEVEL,ADMIN_CODE=r.ADMIN_CODE,
                TILE_ID=t.TILE_ID,TILE_NAME=t.TILE_NAME,OVERLAP_M2=area,
                OVERLAP_STATUS='WITHIN_1M_EDGE_STRIP' if core.is_empty or not core.intersects(t.geometry.buffer(-1)) else 'INTERIOR_OVERLAP'))
        merged=gpd.GeoSeries(matched,crs=5186).union_all() if matched else r.geometry.difference(r.geometry)
        missing=r.geometry.difference(merged)
        residual=r.geometry.difference(merged.buffer(1))
        status='COVERED' if missing.area<.01 else 'WITHIN_1M_TOLERANCE' if residual.area<.01 else 'REVIEW_INDEX_COVERAGE'
        qc.append(dict(REFERENCE_YEAR=int(r.REFERENCE_YEAR),SPATIAL_UNIT_ID=r.SPATIAL_UNIT_ID,ADMIN_NAME=r.ADMIN_NAME,
            TILE_COUNT=len(matched),COVERAGE_FRACTION=1-missing.area/r.geometry.area,
            UNCOVERED_M2=missing.area,UNCOVERED_AFTER_1M_M2=residual.area,STATUS=status))
        if i%1000==0:print(f'Index check {i:,}/{len(units):,}',flush=True)
    links=pd.DataFrame(links);qc=pd.DataFrame(qc)
    plan=links.groupby(['REFERENCE_YEAR','TILE_ID','TILE_NAME'],as_index=False).agg(
        ADMIN_UNIT_COUNT=('SPATIAL_UNIT_ID','nunique'),OVERLAP_M2=('OVERLAP_M2','sum'))
    plan['AVAILABILITY_STATUS']='REVIEW_YEAR_SPECIFIC_WEBSITE_AVAILABILITY'
    assert not plan.duplicated(['REFERENCE_YEAR','TILE_ID']).any()
    out.mkdir(parents=True)
    links.to_parquet(out/'admin_tile_links.parquet',index=False)
    plan.to_parquet(out/'required_tiles_by_year.parquet',index=False)
    qc.to_parquet(out/'admin_index_coverage.parquet',index=False)
    summary=plan.groupby('REFERENCE_YEAR').size().rename('TILE_COUNT').reset_index()
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>08 · 05 도엽 인덱스 점검</title><style>body{font:15px/1.7 Malgun Gothic;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>08 · 05 국가기본도 5K 인덱스</h1>'''
    page+=f'<p>인덱스 {len(tiles):,}개, 원좌표계 {raw.crs}, 비교 EPSG:5186. 1m 차이는 허용 오차로 구분하고 원본 경계는 변경하지 않았습니다.</p>'
    page+=f'<p>성동061 시험 피복자료: 면적 일치 {comparison["index_coverage_fraction"]:.6%}, 경계 최대 거리 {comparison["boundary_hausdorff_m"]:.4f}m, 1m 범위 밖 면적 {comparison["data_outside_1m_buffer_m2"]:.4f}㎡.</p>'
    page+=summary.to_html(index=False,border=0)+qc.groupby('STATUS').size().rename('YEAR_ADMIN_COUNT').reset_index().to_html(index=False,border=0)
    page+='<p>한 도엽의 실제 경계 비교 결과이며 전국 모든 피복 경계를 검증한 것은 아닙니다. 국가기본도 인덱스는 신청 후보 선별용입니다. 연도별 사이트 제공 여부는 별도 확인합니다. 1m 경계 띠에만 걸치는 도엽도 누락 방지를 위해 후보에 남겼습니다.</p><p><a href="required_tiles_by_year.parquet">연도별 필요 도엽</a> · <a href="admin_tile_links.parquet">행정구역–도엽 연결</a> · <a href="admin_index_coverage.parquet">인덱스 확보율</a></p></html>'
    (out/'summary.html').write_text(page,encoding='utf-8')
    write_json(V3/'log/08/full/05_tile_index/run.json',dict(status='COMPLETE_INDEX_CHECK',inputs=inputs,
        tiles=len(tiles),source_crs=str(raw.crs),comparison_crs='EPSG:5186',tolerance_m=1,
        sample_comparison=comparison,year_summary=summary.to_dict('records'),
        coverage_status=qc.STATUS.value_counts().to_dict(),unique_required_tiles=plan.TILE_ID.nunique(),
        boundary_modified=False,download_performed=False,website_availability='UNVERIFIED'))
    print(summary.to_string(index=False));print(qc.STATUS.value_counts().to_string())


if __name__=='__main__':main()
