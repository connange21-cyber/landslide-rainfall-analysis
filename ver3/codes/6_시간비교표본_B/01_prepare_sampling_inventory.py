"""06 preparation only: occurrence references and observed-rain availability.

No random sampling, API requests, imputation, or modifications to upstream data.
"""
import argparse
import json
import sys
import time
import traceback
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import V3, fingerprint, write_json


def daily_inventory(frame, date, network):
    expected = ['STN', 'time_start', 'time_end', 'RN_1H']
    if list(frame.columns) != expected:
        raise ValueError('Unexpected rainfall columns')
    if frame.STN.isna().any() or not pd.api.types.is_integer_dtype(frame.STN):
        raise ValueError('Invalid station IDs')
    for col in ('time_start', 'time_end'):
        if frame[col].dt.tz is not None or frame[col].isna().any():
            raise ValueError('Expected non-null naive KST timestamps')
    if (not frame.time_start.dt.normalize().eq(date).all()
            or not frame.time_start.eq(frame.time_start.dt.floor('h')).all()
            or not (frame.time_end-frame.time_start).eq(pd.Timedelta(hours=1)).all()):
        raise ValueError('Invalid hourly interval or partition date')
    if frame.duplicated(['STN', 'time_start', 'time_end']).any():
        raise ValueError('Duplicate raw rainfall keys')
    invalid = frame.RN_1H.notna() & (~np.isfinite(frame.RN_1H) | frame.RN_1H.lt(0))
    f = frame.assign(VALID=frame.RN_1H.notna() & ~invalid, INVALID=invalid,
                     MISSING=frame.RN_1H.isna())
    day = f.groupby('STN').agg(HOURS_PRESENT=('time_start', 'size'),
                              RAW_VALID_HOURS=('VALID', 'sum'),
                              RAW_INVALID_HOURS=('INVALID', 'sum'),
                              RAW_NULL_HOURS=('MISSING', 'sum')).reset_index()
    day['NETWORK'] = network
    day['DATE'] = date
    day['RAW_COMPLETE_DAY'] = day.HOURS_PRESENT.eq(24) & day.RAW_VALID_HOURS.eq(24)
    # Raw availability is not location-history-approved analysis rainfall.
    day['LOCATION_HISTORY_QC'] = 'NOT_APPLIED_RAW_INVENTORY'
    return day


def scan_rain(root, log):
    days, files, catalogs = [], [], {}
    for network in ('AWS', 'ASOS'):
        folder = root / network
        index = pd.read_parquet(folder / 'index.parquet')
        indexed = {(folder / str(v).replace('\\', '/')).resolve()
                   for v in index.loc[index.kind.eq('hourly') & index.table.eq('data'), 'file']}
        actual = {p.resolve() for p in (folder / 'hourly').glob('*/*/*/*.parquet')}
        catalogs[network] = dict(files=len(actual), indexed_missing=sorted(map(str, indexed-actual)),
                                 unindexed=sorted(map(str, actual-indexed)))
        for i, path in enumerate(sorted(actual)):
            stamp = path.stat()
            date = pd.to_datetime(path.stem.removeprefix('data_'), format='%Y%m%d')
            frame = pq.read_table(path, columns=['STN', 'time_start', 'time_end', 'RN_1H']).to_pandas()
            days.append(daily_inventory(frame, date, network))
            after = path.stat()
            if (stamp.st_size, stamp.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source changed during scan: {path}')
            files.append(dict(network=network, path=str(path), rows=len(frame),
                              bytes=stamp.st_size, mtime_ns=stamp.st_mtime_ns))
            if (i+1) % 50 == 0 or i+1 == len(actual):
                print(f'06 raw inventory: {network} {i+1}/{len(actual)} files', flush=True)
        if not actual:
            raise ValueError(f'No rainfall partitions for {network}')
    daily = pd.concat(days, ignore_index=True).sort_values(['NETWORK', 'STN', 'DATE'])
    if daily.duplicated(['NETWORK', 'STN', 'DATE']).any():
        raise ValueError('Duplicate station-day across rainfall files')
    write_json(log / 'rain_source_files.json', dict(catalogs=catalogs, files=files))
    return daily, catalogs


def summarize_coverage(daily):
    rows = []
    for (network, stn), group in daily.groupby(['NETWORK', 'STN'], sort=True):
        first, last = group.DATE.min(), group.DATE.max()
        dates = group.loc[group.RAW_COMPLETE_DAY, 'DATE'].sort_values()
        if len(dates):
            runs = dates.diff().ne(pd.Timedelta(days=1)).cumsum()
            longest = int(dates.groupby(runs).size().max())
        else:
            longest = 0
        rows.append(dict(NETWORK=network, STN=int(stn), FIRST_DATE=first, LAST_DATE=last,
                         CALENDAR_SPAN_DAYS=(last-first).days+1, DAYS_PRESENT=len(group),
                         ABSENT_DAYS_WITHIN_SPAN=(last-first).days+1-len(group),
                         RAW_COMPLETE_DAYS=int(group.RAW_COMPLETE_DAY.sum()),
                         RAW_INCOMPLETE_DAYS=int((~group.RAW_COMPLETE_DAY).sum()),
                         RAW_VALID_HOURS=int(group.RAW_VALID_HOURS.sum()),
                         RAW_NULL_HOURS=int(group.RAW_NULL_HOURS.sum()),
                         RAW_INVALID_HOURS=int(group.RAW_INVALID_HOURS.sum()),
                         LONGEST_RAW_COMPLETE_RUN_DAYS=longest,
                         LOCATION_HISTORY_QC='NOT_APPLIED_RAW_INVENTORY'))
    return pd.DataFrame(rows)


def prepare_occurrences(groups, selection):
    columns = ['SPATIAL_UNIT_ID', 'ADMIN_NAME', 'AREA_COVERAGE_FRACTION', 'ANALYSIS_STATUS', 'REVIEW_REASON']
    a = groups.merge(selection[columns], on='SPATIAL_UNIT_ID', how='left', validate='many_to_one')
    a['A_REFERENCE_ID'] = a.ADMIN_RAIN_GROUP_ID
    a['SAMPLE_TYPE'] = 'A_REPORTED_OCCURRENCE_CANDIDATE'
    a['CUMULATIVE_START_DATE'] = a.END_DATE-pd.Timedelta(days=14)
    a['SPATIAL_ELIGIBLE'] = a.ADMIN_GROUP_RESOLVED & a.ANALYSIS_STATUS.eq('INCLUDED')
    a['RAIN_01_15D_COMPLETE'] = a[[f'RAIN_{n:02d}D_MM' for n in range(1, 16)]].notna().all(axis=1)
    checks = ['LOCATION_REVIEW', 'LOCATION_CHANGED', 'CROSS_NETWORK_REVIEW',
              'CUMULATIVE_LOCATION_REVIEW', 'CUMULATIVE_CROSS_NETWORK_REVIEW']
    a['RAIN_QC_REVIEW'] = a[checks].fillna(True).any(axis=1)
    a['A_REFERENCE_STATUS'] = np.select(
        [~a.SPATIAL_ELIGIBLE, ~a.EVENT_VALID | ~a.END_CONFIRMED | a.RAIN_QC_REVIEW,
         ~a.RAIN_01_15D_COMPLETE],
        ['EXCLUDED_SPATIAL', 'REVIEW_RAIN_EVENT_OR_QC', 'PENDING_CUMULATIVE_RAIN'],
        default='READY_FOR_SAMPLING_DESIGN')
    a['SAME_UNIT_DATE_GROUP_COUNT'] = a.groupby(['SPATIAL_UNIT_ID', 'END_DATE'], dropna=False)[
        'ADMIN_RAIN_GROUP_ID'].transform('size')
    a['CANDIDATE_AMBIGUITY_REVIEW'] = a.MULTIPLE_GROUP_CANDIDATE_EVENT_COUNT.gt(0)
    # READY is data readiness, not resolution of multiple causal-event candidates.
    a['MATCHED_TRIPLET_STATUS'] = 'NOT_SAMPLED'
    if a.A_REFERENCE_ID.duplicated().any():
        raise ValueError('Duplicate A reference IDs')
    return a


def required_rain_review(a, analysis_daily, raw_daily):
    observed = analysis_daily.set_index(['STN', 'date']).DAILY_RAIN_MM
    if observed.index.has_duplicates:
        raise ValueError('Duplicate stage02 analysis daily keys')
    requests = {}
    for row in a.loc[a.SPATIAL_ELIGIBLE].itertuples():
        for date in pd.date_range(row.CUMULATIVE_START_DATE, row.END_DATE, freq='D'):
            key = (int(row.STN), date)
            if pd.isna(observed.get(key, np.nan)):
                requests.setdefault(key, set()).add(row.A_REFERENCE_ID)
    raw = raw_daily.set_index(['NETWORK', 'STN', 'DATE'])
    rows = []
    for (stn, date), groups in sorted(requests.items()):
        record = dict(STN=stn, DATE=date, A_REFERENCE_IDS=sorted(groups), A_REFERENCE_COUNT=len(groups))
        any_present, any_complete = False, False
        for net in ('AWS', 'ASOS'):
            key = (net, stn, date)
            present = key in raw.index
            record[net+'_HOURS_PRESENT'] = int(raw.loc[key, 'HOURS_PRESENT']) if present else 0
            record[net+'_RAW_VALID_HOURS'] = int(raw.loc[key, 'RAW_VALID_HOURS']) if present else 0
            any_present |= present
            any_complete |= bool(raw.loc[key, 'RAW_COMPLETE_DAY']) if present else False
        record['REVIEW_ACTION'] = ('CHECK_LOCATION_HISTORY_AND_ANALYSIS_QC' if any_complete else
                                   'CHECK_RAW_QC_AND_CROSS_NETWORK_COMPLEMENT' if any_present else
                                   'ACQUIRE_MISSING_OBSERVATIONS')
        record['MISSING_SCOPE'] = 'EXISTING_A_WINDOWS_ONLY_B_C_NOT_DEFINED'
        rows.append(record)
    cols = ['STN', 'DATE', 'A_REFERENCE_IDS', 'A_REFERENCE_COUNT', 'AWS_HOURS_PRESENT',
            'AWS_RAW_VALID_HOURS', 'ASOS_HOURS_PRESENT', 'ASOS_RAW_VALID_HOURS', 'REVIEW_ACTION', 'MISSING_SCOPE']
    return pd.DataFrame(rows, columns=cols)


def write_html(out, a, daily, coverage, review):
    ready = a.A_REFERENCE_STATUS.eq('READY_FOR_SAMPLING_DESIGN')
    spatial = a[a.SPATIAL_ELIGIBLE]
    metrics = [('기존 발생 후보 그룹', len(a)), ('공간 사용 가능 그룹', len(spatial)),
               ('사용 가능 발생 구역', spatial.SPATIAL_UNIT_ID.nunique()),
               ('강우 조건 확보 그룹', int(ready.sum())),
               ('공간 보류·미해결 그룹', int((~a.SPATIAL_ELIGIBLE).sum())),
               ('확인 필요한 관측소·날짜', len(review))]
    yearly = daily.groupby(['NETWORK', daily.DATE.dt.year.rename('YEAR')]).agg(
        STATIONS=('STN', 'nunique'), PARTITION_DATES=('DATE', 'nunique'),
        RAW_COMPLETE_STATION_DAYS=('RAW_COMPLETE_DAY', 'sum')).reset_index()
    links = ''.join(f'<li><a href="{name}.parquet">{label}</a></li>' for name, label in [
        ('occurrence_reference', '발생 표본 기준표'),
        ('station_daily_availability', '관측망·관측소·날짜별 원본 보유 현황'),
        ('station_rain_coverage', '관측소별 기간·결측 요약'),
        ('required_rain_review', '발생 표본의 부족 강우·QC 확인 목록')])
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>06 표본 준비</title>'
            '<style>body{font:15px/1.65 "Malgun Gothic",sans-serif;background:#f4f6f8;color:#253046}'
            'main{max-width:960px;margin:32px auto;padding:24px;background:white}'
            'td,th{padding:7px 14px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>'
            '<main><h1>06 · 비교 표본 준비</h1><p>발생 기준표와 AWS·ASOS 보유 범위를 확인했다. 무작위 추출·추가 다운로드는 미실행.</p>'
            '<table>'+''.join(f'<tr><td>{escape(k)}</td><td>{v:,}</td></tr>' for k,v in metrics)+'</table>'
            '<h2>강우 보유 범위</h2>'+yearly.to_html(index=False, border=0)+
            '<p>원본 유효 관측일은 관측소 위치 이력 검증 전 수치다. 시작~끝 사이 모든 날짜가 보유된 것은 아니다.</p>'
            '<p>발생 후보 관계와 같은 구역·날짜의 여러 이벤트를 유지했다. 강우 조건 확보는 인과관계 확정이나 독립 표본 수를 뜻하지 않는다.</p>'
            '<p>B의 탐색 연도·계절, 난수 시드 및 후보 중복 처리 확정 후 1:1:1 추출을 진행한다. 부족 목록은 현재 A의 15일 창만 대상으로 한다.</p>'
            '<ul>'+links+'</ul></main></html>')
    (out/'summary.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out, log = V3/'output/06/full', V3/'log/06/full'
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error('Existing 06 outputs; explicit --overwrite required')
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    config_path = V3/'config/rainfall.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    inputs = [V3/'output/04/full/admin_rain_event_features.parquet',
              V3/'output/05/full/slope_analysis_selection.parquet',
              V3/'output/02/full/rainfall_daily.parquet', V3/'output/03/full/rainfall_events.parquet']
    for path in [V3/'log/04/full/run.json', V3/'log/05/full/02_run.json']:
        if json.loads(path.read_text(encoding='utf-8'))['status'] != 'COMPLETE':
            raise ValueError(f'Upstream not complete: {path}')
    run = dict(status='RUNNING', started=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
               scope='A_REFERENCE_AND_RAW_RAIN_INVENTORY_ONLY',
               inputs=[fingerprint(p) for p in [Path(__file__), config_path, *inputs]],
               sampling_performed=False, rainfall_downloaded=False, imputation=False)
    write_json(log/'run.json', run)
    started = time.monotonic()
    try:
        groups, selection, analysis_daily, events = [pd.read_parquet(p) for p in inputs]
        a = prepare_occurrences(groups, selection)
        print(f'06 A references: {len(a):,}; spatial usable: {int(a.SPATIAL_ELIGIBLE.sum()):,}', flush=True)
        daily, catalogs = scan_rain(Path(config['rain_root']), log)
        coverage = summarize_coverage(daily)
        # Existing stage03 events cover only its selected stations, not all raw stations.
        counts = events.groupby('STN').size()
        confirmed = events[events.EVENT_VALID & events.END_CONFIRMED].groupby('STN').size()
        coverage['EXISTING_STAGE03_EVENTS'] = coverage.STN.map(counts).fillna(0).astype(int)
        coverage['EXISTING_STAGE03_VALID_CONFIRMED_EVENTS'] = coverage.STN.map(confirmed).fillna(0).astype(int)
        coverage['STAGE03_SCOPE'] = 'SELECTED_STATIONS_ONLY_NOT_B_C_ELIGIBILITY'
        review = required_rain_review(a, analysis_daily, daily)
        for name, frame in [('occurrence_reference', a), ('station_daily_availability', daily),
                            ('station_rain_coverage', coverage), ('required_rain_review', review)]:
            path = out/(name+'.parquet')
            temporary = path.with_suffix('.tmp.parquet')
            frame.to_parquet(temporary, index=False)
            assert pq.read_metadata(temporary).num_rows == len(frame)
            temporary.replace(path)
        assert len(a) == len(groups) and set(a.ADMIN_RAIN_GROUP_ID) == set(groups.ADMIN_RAIN_GROUP_ID)
        assert not a.loc[a.ANALYSIS_STATUS.eq('TEMP_EXCLUDED'), 'SPATIAL_ELIGIBLE'].any()
        assert not daily.duplicated(['NETWORK', 'STN', 'DATE']).any()
        assert int(coverage.DAYS_PRESENT.sum()) == len(daily)
        assert not review.duplicated(['STN', 'DATE']).any()
        assert set(sum(review.A_REFERENCE_IDS.tolist(), [])) <= set(a.loc[a.SPATIAL_ELIGIBLE, 'A_REFERENCE_ID'])
        for source in run['inputs']:
            assert fingerprint(Path(source['path']))['sha256'] == source['sha256']
        write_html(out, a, daily, coverage, review)
        run.update(status='COMPLETE', a_groups=len(a), a_status=a.A_REFERENCE_STATUS.value_counts().to_dict(),
                   spatial_units=int(a.loc[a.SPATIAL_ELIGIBLE, 'SPATIAL_UNIT_ID'].nunique()),
                   raw_station_days=len(daily), station_networks=len(coverage),
                   raw_stations=int(daily.STN.nunique()), missing_a_station_dates=len(review),
                   review_actions=review.REVIEW_ACTION.value_counts().to_dict(),
                   catalogs=catalogs, verification='PASS', upstream_inputs_unchanged=True,
                   design_pending=['B_SEARCH_YEARS', 'B_SEASON_POLICY', 'RANDOM_SEED',
                                   'MULTIPLE_CANDIDATE_AND_UNIT_DATE_HANDLING'])
        print(json.dumps({k:run[k] for k in ['status','a_groups','a_status','spatial_units','raw_station_days',
                                            'station_networks','missing_a_station_dates','review_actions']}, ensure_ascii=False), flush=True)
    except BaseException:
        run.update(status='CANCELLED' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'FAILED',
                   error=traceback.format_exc())
        raise
    finally:
        run.update(finished=pd.Timestamp.now(tz='Asia/Seoul').isoformat(), elapsed_seconds=round(time.monotonic()-started,2))
        write_json(log/'run.json', run)


if __name__ == '__main__':
    main()
