"""Provisional nearest WAMIS station per ABC admin, using the replaced C set."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod, Transformer

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/13/full/04_admin_wamis_candidates'
LOG=V3/'log/13/full/04_admin_wamis_candidates'


def stamp(p):
    s=p.stat();return [s.st_size,s.st_mtime_ns]


def main():
    paths=[V3/'output/13/full/01_abc_rebuild/occurrence_selected.parquet',
           V3/'output/13/full/01_abc_rebuild/b_event_selected.parquet',
           V3/'output/13/full/03_c_grid_replacement/c_event_selected.parquet',
           V3/'output/04/full/spatial_units_5186.parquet',
           V3/'output/13/full/02_wamis_station_audit/wamis_probability_map_stations.parquet']
    signatures={str(p):stamp(p) for p in paths}
    frames=[pd.read_parquet(p) for p in paths[:3]]
    assert all(f.A_REFERENCE_ID.is_unique for f in frames)
    assert set(frames[0].A_REFERENCE_ID)==set(frames[1].A_REFERENCE_ID)==set(frames[2].A_REFERENCE_ID)
    used=set().union(*(set(f.SPATIAL_UNIT_ID) for f in frames))
    registry=gpd.read_parquet(paths[3]).to_crs(5186)
    registry=registry[registry.SPATIAL_UNIT_ID.isin(used)].sort_values('SPATIAL_UNIT_ID')
    assert registry.SPATIAL_UNIT_ID.is_unique and set(registry.SPATIAL_UNIT_ID)==used
    assert registry.geometry.notna().all() and registry.geometry.is_valid.all() and not registry.geometry.is_empty.any()
    stations=pd.read_parquet(paths[4])
    # Exclude unnamed six northern map markers without standard eight-digit codes.
    eligible=stations[stations.OBSCD.str.fullmatch(r'\d{8}') & stations.OBSNM.fillna('').str.strip().ne('') & stations.LAT.notna() & stations.LON.notna()].sort_values('OBSCD').reset_index(drop=True)
    assert eligible.OBSCD.is_unique
    transform=Transformer.from_crs(5186,4326,always_xy=True)
    geod=Geod(ellps='GRS80')
    counts={k:f.SPATIAL_UNIT_ID.value_counts() for k,f in zip('ABC',frames)}
    rows=[]
    for u in registry.itertuples():
        center=u.geometry.centroid
        inside=u.geometry.covers(center)
        point=center if inside else u.geometry.representative_point()
        lon,lat=transform.transform(point.x,point.y)
        dist=geod.inv(np.full(len(eligible),lon),np.full(len(eligible),lat),eligible.LON.to_numpy(),eligible.LAT.to_numpy())[2]
        order=np.argsort(dist,kind='stable')
        i,j=int(order[0]),int(order[1]);first,second=eligible.iloc[i],eligible.iloc[j]
        rows.append(dict(SPATIAL_UNIT_ID=u.SPATIAL_UNIT_ID,ADMIN_LEVEL=u.ADMIN_LEVEL,ADMIN_CODE=u.ADMIN_CODE,
            ADMIN_NAME=u.FULL_NAME if pd.notna(u.FULL_NAME) else u.ADMIN_NAME,
            USES_A=int(counts['A'].get(u.SPATIAL_UNIT_ID,0)),USES_B=int(counts['B'].get(u.SPATIAL_UNIT_ID,0)),USES_C=int(counts['C'].get(u.SPATIAL_UNIT_ID,0)),
            REPRESENTATIVE_LON=lon,REPRESENTATIVE_LAT=lat,
            REPRESENTATIVE_POINT_METHOD='CENTROID_INSIDE_ADMIN' if inside else 'INTERIOR_POINT_CENTROID_OUTSIDE',
            WAMIS_CANDIDATE_CODE=first.OBSCD,WAMIS_CANDIDATE_NAME=first.OBSNM,WAMIS_AGCCD=first.AGCCD,
            WAMIS_LON=float(first.LON),WAMIS_LAT=float(first.LAT),DISTANCE_M=float(dist[i]),DISTANCE_KM=float(dist[i]/1000),
            SECOND_CANDIDATE_CODE=second.OBSCD,SECOND_CANDIDATE_NAME=second.OBSNM,SECOND_DISTANCE_M=float(dist[j]),
            NEAREST_GAP_M=float(dist[j]-dist[i]),NEAR_TIE_WITHIN_1M=bool(dist[j]-dist[i]<=1),
            MATCH_METHOD='NEAREST_GEODESIC_FROM_ADMIN_REPRESENTATIVE_POINT',
            REVIEW_REQUIRED=True,REVIEW_STATUS='REVIEW_NEAREST_CANDIDATE',MATCH_CONFIRMED=False,
            FINAL_WAMIS_CODE=None,REVIEW_NOTE='',FREQUENCY_TABLE_VERIFIED=False,
            WAMIS_SOURCE='WAMIS_MAP_SNAPSHOT_13_02_20261015'))
    result=pd.DataFrame(rows)
    assert result.SPATIAL_UNIT_ID.is_unique and len(result)==len(used)
    assert result.DISTANCE_M.ge(0).all() and result.WAMIS_CANDIDATE_CODE.isin(eligible.OBSCD).all()
    assert all(result['USES_'+k].sum()==len(f) for k,f in zip('ABC',frames))
    assert signatures=={str(p):stamp(p) for p in paths}
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    result.to_parquet(OUT/'admin_wamis_candidates.parquet',index=False)
    bins=pd.cut(result.DISTANCE_KM,[-np.inf,5,10,20,np.inf],labels=['5km 이하','5~10km','10~20km','20km 초과']).value_counts(sort=False)
    report=dict(status='COMPLETE_PROVISIONAL_ONLY',admin_units=len(result),wamis_candidates_used=int(result.WAMIS_CANDIDATE_CODE.nunique()),
                eligible_wamis_stations=len(eligible),excluded_map_markers=len(stations)-len(eligible),
                representative_point_methods=result.REPRESENTATIVE_POINT_METHOD.value_counts().to_dict(),
                distance_bins=bins.to_dict(),median_distance_km=float(result.DISTANCE_KM.median()),max_distance_km=float(result.DISTANCE_KM.max()),
                reviewed=0,confirmed=0,abc_rows={k:len(f) for k,f in zip('ABC',frames)},c_source='13/03_REPLACED_C',inputs=signatures,upstream_unchanged=True)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    display=result.sort_values('DISTANCE_M',ascending=False)[['SPATIAL_UNIT_ID','ADMIN_NAME','USES_A','USES_B','USES_C','WAMIS_CANDIDATE_CODE','WAMIS_CANDIDATE_NAME','DISTANCE_KM','REVIEW_STATUS']].copy()
    display.DISTANCE_KM=display.DISTANCE_KM.round(3)
    display=display.rename(columns={'ADMIN_NAME':'행정구역','USES_A':'A 사용','USES_B':'B 사용','USES_C':'C 사용','WAMIS_CANDIDATE_CODE':'WAMIS 후보 코드','WAMIS_CANDIDATE_NAME':'후보 관측소','DISTANCE_KM':'거리 km','REVIEW_STATUS':'검토 상태'})
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>ABC 행정구역 WAMIS 후보</title><style>body{font:14px/1.7 Malgun Gothic;margin:28px;color:#24364a}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}thead{position:sticky;top:0;background:#eef4fa}input{padding:8px;width:400px}</style><h1>ABC 행정구역별 WAMIS 최근접 후보</h1>'
    page+=f'<p>사용 행정구역 {len(result):,}개 → WAMIS 후보 지점 {report["wamis_candidates_used"]:,}개 · 전체 미확정 검토 대상. 교체한 C 67개를 반영했습니다.</p>'
    page+='<p>EPSG:5186 도형의 무게중심을 사용하고, 무게중심이 구역 밖이면 구역 내부 대표점으로 대체했습니다. 이 점과 WAMIS 관측소의 타원체상 직선거리로 최근접을 선택했습니다. 실제 산사태 위치나 기존 STN과의 거리가 아닙니다.</p>'
    page+='<p>기존 지도 목록 중 이름과 8자리 코드가 있는 615개 지점에서 선택했습니다. 고도·산맥·해안·확률강우량 산정기간 및 지점별 자료 존재 여부는 미검토입니다. 후보 코드와 FINAL_WAMIS_CODE를 분리했으며 확정 코드는 비워뒀습니다. 거리 구간은 설명용이며 자동 승인 기준이 아닙니다.</p>'
    page+=bins.rename_axis('거리').reset_index(name='구역 수').to_html(index=False,border=0)
    page+='<p><a href="admin_wamis_candidates.parquet">전체 매칭표 Parquet</a></p><label>검색 <input id="search" placeholder="행정구역명 또는 WAMIS 코드"></label><span id="count"></span>'
    page+=display.to_html(index=False,border=0,table_id='matches')
    page+='''<script>const input=document.getElementById('search'),rows=[...document.querySelectorAll('#matches tbody tr')];function filter(){const q=input.value.trim().toLowerCase();let n=0;for(const r of rows){const show=r.textContent.toLowerCase().includes(q);r.hidden=!show;if(show)n++;}document.getElementById('count').textContent=' '+n.toLocaleString()+'개';}input.addEventListener('input',filter);filter();</script></html>'''
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
