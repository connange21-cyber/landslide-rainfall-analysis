"""Audit WAMIS probability-map station identities against project KMA STNs."""
from pathlib import Path
import json
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer, Geod

V3 = Path(__file__).resolve().parents[2]
OUT = V3 / 'output/13/full/02_wamis_station_audit'
LOG = V3 / 'log/13/full/02_wamis_station_audit'
ROOT = 'https://map.wamis.go.kr'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    def get(path, name):
        cache = LOG / name
        if not cache.exists():
            r = session.get(ROOT + path, timeout=35)
            r.raise_for_status()
            cache.write_bytes(r.content)
        return cache.read_text(encoding='utf-8')
    payload = json.loads(get('/data/gis/gis_obsvt.json', 'wamis_obsvt.json'))
    w = pd.DataFrame([dict(**f['properties'], X=f['geometry']['coordinates'][0], Y=f['geometry']['coordinates'][1]) for f in payload['features']])
    assert w.OBSCD.is_unique
    # The map uses EPSG:3857; transform its point geometry to WGS84.
    transform = Transformer.from_crs(3857, 4326, always_xy=True)
    w['LON'], w['LAT'] = transform.transform(w.X.to_numpy(), w.Y.to_numpy())
    w['WAMIS_KMA_CODE_STN'] = pd.array([int(c[-3:]) if a == '04' and len(c) == 8 and c[4] in '12' else None for c, a in zip(w.OBSCD, w.AGCCD)], dtype='Int64')
    paths = [V3 / 'base/stations/station_master_final.xlsx', V3 / 'base/stations/station_location_history.xlsx']
    paths += [V3 / 'output/13/full/01_abc_rebuild' / f for f in ['occurrence_selected.parquet', 'b_event_selected.parquet', 'c_event_selected.parquet']]
    signatures = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    master = pd.read_excel(paths[0])
    history = pd.read_excel(paths[1])
    assert master.STN.is_unique
    geod = Geod(ellps='GRS80')
    rows = []
    for r in master.itertuples(index=False):
        distances = geod.inv(np.full(len(w),r.DEFAULT_LON), np.full(len(w),r.DEFAULT_LAT), w.LON.to_numpy(), w.LAT.to_numpy())[2]
        nearest = w.iloc[int(np.argmin(distances))]
        same = w[w.WAMIS_KMA_CODE_STN.eq(r.STN).fillna(False)]
        row = dict(STN=r.STN, DEFAULT_NETWORK=r.DEFAULT_NETWORK, KMA_LAT=r.DEFAULT_LAT, KMA_LON=r.DEFAULT_LON,
                   HAS_LOCATION_CHANGE=bool(r.HAS_LOCATION_CHANGE), NEAREST_WAMIS_CODE=nearest.OBSCD,
                   NEAREST_WAMIS_NAME=nearest.OBSNM, NEAREST_AGCCD=nearest.AGCCD,
                   NEAREST_DISTANCE_M=float(np.min(distances)), STATUS='NO_SAME_STATION_CODE_IN_MAP',
                   WAMIS_CODE=None, WAMIS_NAME=None, DEFAULT_DISTANCE_M=np.nan, HISTORY_MIN_DISTANCE_M=np.nan)
        if len(same) == 1:
            q = same.iloc[0]
            h = history[history.STN.eq(r.STN)]
            hist = geod.inv(np.full(len(h),q.LON),np.full(len(h),q.LAT),h.LON.to_numpy(),h.LAT.to_numpy())[2]
            distance = float(geod.inv(r.DEFAULT_LON,r.DEFAULT_LAT,q.LON,q.LAT)[2])
            best = float(np.min(hist)) if len(hist) else distance
            row.update(WAMIS_CODE=q.OBSCD, WAMIS_NAME=q.OBSNM, WAMIS_LAT=q.LAT, WAMIS_LON=q.LON,
                       DEFAULT_DISTANCE_M=distance, HISTORY_MIN_DISTANCE_M=best,
                       STATUS='SAME_CODE_LOCATION_CONSISTENT' if best <= 100 else 'SAME_CODE_LOCATION_REVIEW',
                       FREQUENCY_APPLICATION='SAME_STATION_REFERENCE_PENDING_FREQUENCY_PERIOD_AND_METADATA_REVIEW')
        elif len(same) > 1:
            row['STATUS'] = 'REVIEW_MULTIPLE_WAMIS_CODES'
        rows.append(row)
    cross = pd.DataFrame(rows)
    cross['NEARBY_IS_IDENTITY'] = False
    cross['LOCATION_TOLERANCE_M'] = 100
    cross.to_parquet(OUT / 'stn_wamis_crosswalk.parquet', index=False)
    w.to_parquet(OUT / 'wamis_probability_map_stations.parquet', index=False)
    usage = []
    for kind, path in zip('ABC', paths[2:]):
        t = pd.read_parquet(path)
        merged = t[['A_REFERENCE_ID','STN']].merge(cross[['STN','STATUS','WAMIS_CODE']],on='STN',validate='many_to_one')
        assert len(merged) == len(t)
        usage.append(dict(sample=kind, rows=len(t), unique_stn=t.STN.nunique(), same_code_rows=int(merged.WAMIS_CODE.notna().sum()),
                          same_code_stn=merged.loc[merged.WAMIS_CODE.notna(),'STN'].nunique(),
                          coordinate_consistent_rows=int(merged.STATUS.eq('SAME_CODE_LOCATION_CONSISTENT').sum())))
        cross['USES_'+kind] = cross.STN.map(t.STN.value_counts()).fillna(0).astype(int)
    # Confirm that actual probability values are returned for one matched station.
    example = get('/popup/rainfall/probability/detail.do?action=select&key=10181108', 'seoul_10181108_probability.html')
    series = re.findall(r"name:\s*'(\d+)년',\s*data:\s*\[([^]]+)\]", example)
    assert len(series) > 0 and '서울' in example
    example_values = {int(year): [float(x) for x in values.split(',')] for year, values in series}
    assert all(len(v)==28 for v in example_values.values())
    cross.to_parquet(OUT / 'stn_wamis_crosswalk.parquet', index=False)
    report = dict(status='COMPLETE_IDENTITY_AUDIT_FREQUENCY_APPLICATION_PENDING', fetched_utc=datetime.now(timezone.utc).isoformat(),
                  wamis_map_stations=len(w), agency_counts=w.AGCCD.value_counts().to_dict(),
                  kma_wamis_map_stations=int(w.AGCCD.eq('04').sum()), project_stn=len(cross),
                  matching_stn=int(cross.WAMIS_CODE.notna().sum()), status_counts=cross.STATUS.value_counts().to_dict(),
                  used_abc_stn=int(cross[['USES_A','USES_B','USES_C']].sum(axis=1).gt(0).sum()), usage=usage,
                  unmatched_wamis_kma=w.loc[w.AGCCD.eq('04') & ~w.WAMIS_KMA_CODE_STN.isin(master.STN),['OBSCD','OBSNM','WAMIS_KMA_CODE_STN']].astype(str).to_dict('records'),
                  example_seoul_stn=108, example_wamis='10181108', example_1h_10year_mm=example_values[10][0],
                  example_duration_hours=list(range(1,25))+[36,48,60,72], example_return_years=list(example_values),
                  frequency_all_stations_downloaded=False, inputs=signatures)
    assert signatures == {str(p): [p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    (LOG / 'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    matched = cross[cross.WAMIS_CODE.notna()]
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>WAMIS-STN 대조</title><style>body{font:15px/1.7 Malgun Gothic;margin:30px}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>WAMIS 확률강우량 관측소 · STN 대조</h1>'
    page += f'<p>지도 관측소 {len(w):,}개 중 기상청 코드(AGCCD 04) {report["kma_wamis_map_stations"]}개. 보유 STN {len(cross)}개 중 동일 표준코드 연결 {len(matched)}개.</p>'
    page += pd.DataFrame(usage).to_html(index=False,border=0)
    page += '<p>동일 관측소 여부와 확률강우량의 직접 적용 가능성은 구분합니다. 기상청 표준코드의 마지막 세 자리만 단독 사용하지 않고 기관·코드 구성·좌표 이력을 함께 검사했습니다. 좌표 100m는 비교용 QC 허용폭이며 관측소 동일성을 증명하는 기준이 아닙니다. 가까운 다른 기관 지점은 동일 STN으로 처리하지 않았습니다.</p>'
    page += '<p>동일 코드도 확률분석 기준연도·관측소 이전·관측기간을 검토해야 합니다. 지점 빈도값을 현재 행정구역 평균 융합강우의 정확한 재현기간으로 직접 해석하지 않습니다. 현재 표는 표준코드 연결 감사표이며 빈도 피처를 기존 ABC에 적용하지 않았습니다.</p>'
    page += f'<p>서울 STN108 ↔ WAMIS10181108에서 실제 확률강우 표 확인: 1시간·10년 {example_values[10][0]}mm, 지속시간 1~24·36·48·60·72시간. 5~15일 빈도로 외삽하지 않습니다. 전체 지점별 확률강우값 다운로드는 미실행.</p>'
    page += '<h2>동일 코드 연결</h2>' + matched[['STN','WAMIS_CODE','WAMIS_NAME','DEFAULT_DISTANCE_M','HISTORY_MIN_DISTANCE_M','STATUS','USES_A','USES_B','USES_C']].round(1).to_html(index=False,border=0)
    example_row = w[w.OBSCD.eq('10184180')].iloc[0]
    example_near = cross.sort_values('NEAREST_DISTANCE_M')
    dist = geod.inv(np.full(len(master),example_row.LON),np.full(len(master),example_row.LAT),master.DEFAULT_LON.to_numpy(),master.DEFAULT_LAT.to_numpy())[2]
    idx=int(np.argmin(dist));near=master.iloc[idx]
    page += f'<p>성남북초교10184180은 기관코드 {example_row.AGCCD}로 기상청 동일코드 변환 대상이 아닙니다. 보유 기본 좌표 기준 가장 가까운 STN은 {int(near.STN)}, 거리 {dist[idx]:,.0f}m이며 동일 관측소라고 확인한 결과가 아닙니다.</p>'
    page += '<p><a href="stn_wamis_crosswalk.parquet">전체 STN 대조표</a> · <a href="wamis_probability_map_stations.parquet">WAMIS 지도 관측소 목록</a> · <a href="https://map.wamis.go.kr/">WAMIS 원출처</a></p></html>'
    (OUT / 'summary.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},ensure_ascii=False),flush=True)
    print('LOCATION_REVIEW',matched[matched.STATUS.eq('SAME_CODE_LOCATION_REVIEW')][['STN','WAMIS_NAME','DEFAULT_DISTANCE_M','HISTORY_MIN_DISTANCE_M']].to_string(index=False),flush=True)
    print('SEONGNAM_NEAREST',int(near.STN),round(float(dist[idx]),1),flush=True)


if __name__ == '__main__':
    main()
