"""Record A/B/C land-cover acquisition needs by reference year and admin unit."""
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    out = V3/'output/08/full/02_yearly_requirements'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Existing yearly requirements; no automatic overwrite')
    paths = [V3/'output/06/full/05_selection/occurrence_selected.parquet',
             V3/'output/06/full/05_selection/b_event_selected.parquet',
             V3/'output/07/full/c_event_samples.parquet',
             V3/'output/04/full/spatial_units_5186.parquet']
    before = [fingerprint(p) for p in paths]
    samples = []
    for typ, path in zip('ABC', paths[:3]):
        f = pd.read_parquet(path)
        idcol = {'A':'A_REFERENCE_ID', 'B':'B_SAMPLE_ID', 'C':'C_SAMPLE_ID'}[typ]
        r = f[['A_REFERENCE_ID', 'SPATIAL_UNIT_ID', 'END_DATE']].copy()
        r['SAMPLE_ID'] = f[idcol]
        r['SAMPLE_TYPE'] = typ
        r['END_DATE'] = pd.to_datetime(r.END_DATE, errors='raise')
        assert r.END_DATE.notna().all() and r.SAMPLE_ID.is_unique
        r['REFERENCE_YEAR'] = r.END_DATE.dt.year
        samples.append(r)
    mapping = pd.concat(samples, ignore_index=True)
    a, b, c = samples
    ac = a.merge(c, on='A_REFERENCE_ID', validate='one_to_one', suffixes=('_A','_C'))
    assert len(ac) == len(a) == len(b) == len(c)
    assert ac.END_DATE_A.eq(ac.END_DATE_C).all()
    registry = gpd.read_parquet(paths[3])
    assert registry.SPATIAL_UNIT_ID.is_unique
    assert set(mapping.SPATIAL_UNIT_ID).issubset(set(registry.SPATIAL_UNIT_ID))
    counts = mapping.groupby(['REFERENCE_YEAR','SPATIAL_UNIT_ID','SAMPLE_TYPE']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list('ABC'), fill_value=0).rename(columns={t:f'{t}_SAMPLE_COUNT' for t in 'ABC'}).reset_index()
    cols = ['SPATIAL_UNIT_ID','ADMIN_LEVEL','ADMIN_CODE','ADMIN_NAME','EMD_CODE','geometry']
    units = registry[cols].merge(counts, on='SPATIAL_UNIT_ID', validate='one_to_many')
    bounds = units.to_crs(4326).bounds
    for name in bounds:
        units['WGS84_'+name.upper()] = bounds[name].to_numpy()
    units['LANDCOVER_CLASSIFICATION'] = 'LV3_41_CLASSES'
    units['NOMINAL_RESOLUTION_M'] = 1
    units['REQUEST_MAP_YEAR'] = units.REFERENCE_YEAR
    units['MAP_YEAR_STATUS'] = 'SAME_YEAR_ACQUISITION_TARGET_NOT_TEMPORAL_VALIDATION'
    units['SOURCE_IMAGE_DATE'] = pd.NaT
    units['TEMPORAL_VALIDITY'] = 'REVIEW_SOURCE_IMAGE_DATE'
    units['TILE_STATUS'] = 'NOT_DETERMINED_NO_TILE_INDEX'
    units['DOWNLOAD_STATUS'] = 'NOT_DOWNLOADED'
    units = units.sort_values(['REFERENCE_YEAR','SPATIAL_UNIT_ID'])
    assert not units.duplicated(['REFERENCE_YEAR','SPATIAL_UNIT_ID']).any()
    for typ in 'ABC':
        assert units[f'{typ}_SAMPLE_COUNT'].sum() == len(mapping[mapping.SAMPLE_TYPE.eq(typ)])
    assert before == [fingerprint(p) for p in paths]
    out.mkdir(parents=True)
    mapping.to_parquet(out/'sample_year_admin_mapping.parquet', index=False)
    summary = []
    sections = []
    for year, group in units.groupby('REFERENCE_YEAR', sort=True):
        folder = out/str(year)
        folder.mkdir()
        group.to_parquet(folder/'required_admin_units.parquet', index=False)
        reread = gpd.read_parquet(folder/'required_admin_units.parquet')
        assert len(reread) == len(group) and reread.crs == registry.crs
        row = dict(YEAR=int(year), UNIQUE_ADMIN_UNITS=len(group))
        for typ in 'ABC':
            row[f'{typ}_SAMPLES'] = int(group[f'{typ}_SAMPLE_COUNT'].sum())
            row[f'{typ}_ADMIN_UNITS'] = int(group[f'{typ}_SAMPLE_COUNT'].gt(0).sum())
        summary.append(row)
        display = group[['ADMIN_NAME','ADMIN_LEVEL','ADMIN_CODE','SPATIAL_UNIT_ID','A_SAMPLE_COUNT','B_SAMPLE_COUNT','C_SAMPLE_COUNT']]
        sections.append(f'<details><summary>{year}년 · {len(group):,}개 구역</summary><p><a href="{year}/required_admin_units.parquet">연도별 경계·신청 목록</a></p>'+display.to_html(index=False, border=0, escape=True)+'</details>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>08 · 02 연도별 ABC 토지피복 신청 범위</title>
<style>body{font:14px/1.6 Malgun Gothic;margin:30px;color:#24364a}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}details{margin:18px 0}input{padding:9px;width:400px}summary{cursor:pointer;font-weight:bold}</style>
<h1>08 · 02 연도별 A·B·C 토지피복 신청 범위</h1>
<p>목적: 세분류 1m급 자료 확보에 필요한 연도와 행정구역을 정리. A/C는 A 이벤트 종료일, B는 B 자체 이벤트 종료일의 연도. 리 우선·읍면동 보완, 경계 EPSG:5186.</p>
<p>동일 연도·행정구역은 A/B/C 사이에서 한 번만 신청합니다. 표의 구역 수는 유형 사이에 겹치므로 단순 합산하지 않습니다.</p>
<p>신청 지도연도는 표본과 같은 연도로 기록했습니다. 실제 촬영일·자료 가용성은 미확인으로, 학습 사용 시점 적합성이 확정된 것은 아닙니다. 도엽번호는 공식 인덱스 확보 후 연결합니다. 자료 다운로드·피복 비율 계산은 수행하지 않았습니다.</p>
<p><a href="sample_year_admin_mapping.parquet">전체 표본–연도–행정구역 연결표</a></p>
'''
    page += pd.DataFrame(summary).to_html(index=False, border=0)
    page += '<p>행정구역 검색: <input id="query" placeholder="행정구역 이름 또는 코드"></p>'+''.join(sections)
    page += '''<script>document.getElementById('query').addEventListener('input',function(){let q=this.value.trim().toLowerCase();document.querySelectorAll('details').forEach(d=>{let found=false;d.querySelectorAll('tbody tr').forEach(r=>{let ok=r.textContent.toLowerCase().includes(q);r.hidden=!ok;found=found||ok;});d.hidden=!found;if(q&&found)d.open=true;});});</script></html>'''
    (out/'summary.html').write_text(page, encoding='utf-8')
    write_json(V3/'log/08/full/02_yearly_requirements/run.json', dict(
        status='COMPLETE_REQUIREMENTS_ONLY', inputs=before, samples=len(mapping),
        distinct_admin_units=int(mapping.SPATIAL_UNIT_ID.nunique()),
        distinct_year_admin_pairs=len(units), years=summary,
        upstream_unchanged=True, ac_dates_verified=True, grouped_counts_verified=True,
        geometry_roundtrip_verified=True, source_imagery_dates='UNVERIFIED',
        map_year_rule='SAME_REFERENCE_YEAR_ACQUISITION_TARGET', download_performed=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Total: {len(units):,} year/admin pairs, {mapping.SPATIAL_UNIT_ID.nunique():,} distinct units; no downloads')


if __name__ == '__main__':
    main()
