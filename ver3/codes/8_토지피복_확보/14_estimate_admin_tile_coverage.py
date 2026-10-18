"""Quick index-based coverage estimate, explicitly separate from actual polygon coverage."""
from pathlib import Path
import json
import geopandas as gpd
import pandas as pd
import shapely

V3=Path(__file__).resolve().parents[2]
def main():
    out=V3/'output/08/full/14_admin_coverage';out.mkdir(parents=True,exist_ok=True)
    log=V3/'log/08/full/14_admin_coverage';log.mkdir(parents=True,exist_ok=True)
    inventory=pd.read_parquet(V3/'output/08/full/13_file_audit/archive_inventory.parquet')
    available=inventory[inventory.STATUS.eq('OK')]
    index=gpd.read_file(V3/'base/국가기본도_도엽인덱스5K/TN_MAPINDX_5K.shp').to_crs(5186)
    index['TILE_ID']=index.MAPIDCD_NO.astype(str)
    rows=[]
    for f in sorted((V3/'output/08/full/02_yearly_requirements').glob('*/required_admin_units.parquet')):
        year=int(f.parent.name);units=gpd.read_parquet(f).to_crs(5186)
        ids=set(available.loc[available.YEAR.eq(year),'TILE_ID'].astype(str))
        tiles=index[index.TILE_ID.isin(ids)].reset_index(drop=True)
        for a in units.itertuples():
            geom=shapely.make_valid(a.geometry)
            candidates=tiles.iloc[tiles.sindex.query(geom,predicate='intersects')]
            pieces=shapely.intersection(candidates.geometry.array,geom)
            area=shapely.union_all(pieces).area if len(pieces) else 0.
            total=geom.area;missing=max(0.,total-area)
            rows.append(dict(REFERENCE_YEAR=year,SPATIAL_UNIT_ID=a.SPATIAL_UNIT_ID,ADMIN_NAME=a.ADMIN_NAME,
                ADMIN_CODE=a.ADMIN_CODE,ADMIN_LEVEL=a.ADMIN_LEVEL,A_SAMPLE_COUNT=a.A_SAMPLE_COUNT,
                B_SAMPLE_COUNT=a.B_SAMPLE_COUNT,C_SAMPLE_COUNT=a.C_SAMPLE_COUNT,ADMIN_AREA_M2=total,
                AVAILABLE_INDEX_AREA_M2=area,UNCOVERED_INDEX_AREA_M2=missing,
                ESTIMATED_COVERAGE_PERCENT=min(100.,100*area/total),AVAILABLE_TILE_COUNT=len(candidates),
                VALUE_TYPE='INDEX_BASED_ESTIMATE',STATUS='INDEX_COVERED_WITHIN_1M2' if missing<=1 else 'REVIEW_INDEX_GAP'))
        print(f'{year}: {len(units):,} 구역',flush=True)
    result=pd.DataFrame(rows)
    result.to_parquet(out/'admin_tile_coverage_estimate.parquet',index=False)
    result.to_csv(out/'admin_tile_coverage_estimate.csv',index=False,encoding='utf-8-sig')
    bands=pd.cut(result.ESTIMATED_COVERAGE_PERCENT,[-1,0,90,99,99.9,100.000001],labels=['0%','0–90%','90–99%','99–99.9%','99.9–100%'])
    counts=pd.crosstab(result.REFERENCE_YEAR,bands,dropna=False)
    review=result[result.UNCOVERED_INDEX_AREA_M2.gt(1)].sort_values('ESTIMATED_COVERAGE_PERCENT')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>행정구역 도엽 확보율 · 잠정치</title><style>body{font:14px/1.6 Malgun Gothic;margin:30px}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>ABC 행정구역별 도엽 확보율</h1>'
    page+=f'<p>행정구역–연도 {len(result):,}건 / 고유 행정구역 {result.SPATIAL_UNIT_ID.nunique():,}개. A·B·C 각각 {int(result.A_SAMPLE_COUNT.sum()):,} / {int(result.B_SAMPLE_COUNT.sum()):,} / {int(result.C_SAMPLE_COUNT.sum()):,} 표본.</p>'
    page+='<p><strong>잠정치:</strong> 정상 압축 해제된 파일의 5K 도엽 인덱스 면적 ∩ 행정구역 면적 / 행정구역 전체 면적. 실제 SHP 내부의 빈 영역을 검증한 값은 아닙니다. 미완료 압축 해제 2건은 제외했습니다. 미확보 원인과 표본 제외 여부는 확정하지 않았습니다.</p>'
    page+=counts.to_html()+f'<p>미확보 면적 1㎡ 초과 {len(review):,}건. 경계 오차도 포함할 수 있습니다.</p>'
    page+='<p>전체 목록: <a href="admin_tile_coverage_estimate.csv">CSV</a> · <a href="admin_tile_coverage_estimate.parquet">Parquet</a>. 실제 폴리곤 계산 완료 후 별도 summary.html과 admin_landcover_coverage.parquet에 저장합니다.</p>'
    page+=review[['REFERENCE_YEAR','ADMIN_NAME','SPATIAL_UNIT_ID','A_SAMPLE_COUNT','B_SAMPLE_COUNT','C_SAMPLE_COUNT','ESTIMATED_COVERAGE_PERCENT','UNCOVERED_INDEX_AREA_M2']].to_html(index=False,float_format=lambda n:f'{n:,.3f}')+'</html>'
    (out/'tile_coverage_estimate.html').write_text(page,encoding='utf-8')
    report=dict(status='COMPLETE_INDEX_ESTIMATE',rows=len(result),unique_admins=result.SPATIAL_UNIT_ID.nunique(),
        review_rows=len(review),coverage_counts=counts.to_dict(),abc_counts={s:int(result[f'{s}_SAMPLE_COUNT'].sum()) for s in 'ABC'})
    (log/'index_estimate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(counts.to_string());print('review',len(review))
if __name__=='__main__':main()
