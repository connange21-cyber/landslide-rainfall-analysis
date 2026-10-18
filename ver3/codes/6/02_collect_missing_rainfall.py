"""Plan or collect missing observed hours into standardized RAIN_DATAS Parquet.

Default: plan only. --execute: KMA API requests and shared Parquet updates.
Raw responses, backups, progress, errors and value conflicts stay in log/06.
Source definitions: https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=239
and https://apihub.kma.go.kr/apiList.do?apiSeq=2
"""
import argparse
import contextlib
from functools import lru_cache
import getpass
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from html import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import requests
from common import V3, fingerprint, write_json

COLS = ['STN', 'time_start', 'time_end', 'RN_1H']
KEYS = COLS[:3]
URLS = dict(AWS='https://apihub.kma.go.kr/api/typ01/url/awsh.php',
            ASOS='https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php')


def empty_rain():
    return pd.DataFrame({'STN': pd.Series(dtype='int64'),
                         'time_start': pd.Series(dtype='datetime64[ns]'),
                         'time_end': pd.Series(dtype='datetime64[ns]'),
                         'RN_1H': pd.Series(dtype='float64')})


def partition_path(root, network, date):
    return root/network/'hourly'/date.strftime('%Y/%m/%d')/f'data_{date:%Y%m%d}.parquet'


def valid_values(values):
    return values.notna() & np.isfinite(values) & values.ge(0)


@lru_cache(maxsize=1)
def inventory_module():
    spec = importlib.util.spec_from_file_location('stage06_inventory', Path(__file__).with_name('01_prepare_sampling_inventory.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_rain(frame, date):
    inventory_module().daily_inventory(frame, date, 'VALIDATION')
    return frame


def load_day(root, network, date):
    path = partition_path(root, network, date)
    return validate_rain(pd.read_parquet(path), date) if path.exists() else empty_rain()


def history_hours(history, network, stn, times):
    usable = set()
    h = history[history.STN.eq(stn) & history.NETWORK.eq(network)]
    for row in h.itertuples():
        # Do not silently acquire a station/network that cannot be located reliably.
        if bool(row.REVIEW_FLAG):
            continue
        for end in times:
            start = end-pd.Timedelta(hours=1)
            if start >= row.ANALYSIS_FROM and (pd.isna(row.ANALYSIS_TO) or end <= row.ANALYSIS_TO):
                if (pd.notna(row.LAST_OLD_OBSERVED) and pd.notna(row.FIRST_NEW_OBSERVED)
                        and start < row.FIRST_NEW_OBSERVED and end > row.LAST_OLD_OBSERVED):
                    continue
                usable.add(end)
    return sorted(usable)


def build_plan(review, history, root):
    cache = {}
    rows = []
    for row in review.itertuples():
        date, stn = pd.Timestamp(row.DATE).normalize(), int(row.STN)
        expected = list(pd.date_range(date+pd.Timedelta(hours=1), periods=24, freq='h'))
        present = set()
        for network in ('AWS', 'ASOS'):
            key = (network, date)
            if key not in cache:
                cache[key] = load_day(root, network, date)
            f = cache[key]
            present.update(f.loc[f.STN.eq(stn) & valid_values(f.RN_1H), 'time_end'])
        missing = sorted(set(expected)-present)
        eligible = {n: history_hours(history, n, stn, missing) for n in ('AWS', 'ASOS')}
        covered = set(eligible['AWS']) | set(eligible['ASOS'])
        rows.append(dict(STN=stn, DATE=date, REVIEW_ACTION=row.REVIEW_ACTION,
                         RAW_MISSING_HOURS=len(missing), MISSING_END_TIMES=missing,
                         AWS_END_TIMES=eligible['AWS'], ASOS_END_TIMES=eligible['ASOS'],
                         HISTORY_UNRESOLVED_HOURS=len(set(missing)-covered),
                         PLAN_STATUS=('RAW_ALREADY_COVERED_RECHECK_ANALYSIS' if not missing else
                                      'READY_TO_FETCH' if covered else 'REVIEW_STATION_HISTORY'),
                         MAX_AWS_CALLS=len(eligible['AWS']),
                         MAX_ASOS_CALLS=int(bool(eligible['ASOS'])),
                         A_REFERENCE_IDS=list(row.A_REFERENCE_IDS)))
    return pd.DataFrame(rows)


def parse_response(text, network, stn, requested_ends):
    """Fail closed on wrong layout; never convert missing/sentinel rain to zero."""
    lines = text.splitlines()
    if '<html' in text.lower() or '<!doctype' in text.lower():
        raise ValueError('HTML response instead of observation data')
    header = next((l.lstrip('# ').split() for l in lines if l.lstrip().startswith('#')
                   and l.lstrip('# ').startswith('YYMMDDHHMI')), None)
    if header is None:
        raise ValueError('Missing recognized KMA header (possible API error)')
    ncols, rain_col = (14, 6) if network == 'AWS' else (46, 15)
    if len(header) != ncols or header[1] != 'STN' or header[rain_col] != 'RN':
        raise ValueError('Unexpected KMA rainfall layout')
    if network == 'AWS' and not any('HR1' in l for l in lines if l.lstrip().startswith('#')):
        raise ValueError('AWS response does not identify hourly rainfall')
    allowed = set(pd.Timestamp(t) for t in requested_ends)
    records = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values = line.split()
        if len(values) != ncols:
            raise ValueError('Unexpected observation row width')
        end = pd.to_datetime(values[0], format='%Y%m%d%H%M', errors='raise')
        station = int(values[1])
        if end != end.floor('h'):
            raise ValueError('API returned a non-hourly timestamp')
        # Period responses may contain other hours; store requested keys only.
        if station != stn or end not in allowed:
            continue
        amount = float(values[rain_col])
        amount = amount if np.isfinite(amount) and amount >= 0 else np.nan
        records.append((station, end-pd.Timedelta(hours=1), end, amount))
    frame = pd.DataFrame(records, columns=COLS) if records else empty_rain()
    frame = frame.astype({'STN': 'int64', 'RN_1H': 'float64'})
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate observation keys in API response')
    return frame


def merge_observations(old, new):
    if old.duplicated(KEYS).any() or new.duplicated(KEYS).any():
        raise ValueError('Duplicate observation keys before merge')
    a, b = old.set_index(KEYS), new.set_index(KEYS)
    overlap = a.index.intersection(b.index)
    oa, nb = a.loc[overlap, 'RN_1H'], b.loc[overlap, 'RN_1H']
    different = valid_values(oa) & valid_values(nb) & ~np.isclose(oa, nb, rtol=0, atol=1e-9)
    conflicts = pd.DataFrame({'EXISTING_RN_1H': oa[different], 'FETCHED_RN_1H': nb[different]}).reset_index()
    result = a.reindex(a.index.union(b.index))
    incoming = b.RN_1H.reindex(result.index)
    fill = ~valid_values(result.RN_1H) & valid_values(incoming)
    result.loc[fill, 'RN_1H'] = incoming[fill]
    # Actual returned missing rows remain missing; absent API rows are never fabricated.
    result = result.reset_index()[COLS].sort_values(KEYS).reset_index(drop=True)
    protected = a.loc[valid_values(a.RN_1H), 'RN_1H']
    assert np.array_equal(result.set_index(KEYS).loc[protected.index, 'RN_1H'], protected)
    return result, conflicts, int(fill.sum())


def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(temporary), frame, check_dtype=False)
    temporary.replace(path)


@contextlib.contextmanager
def shared_lock(root):
    lock = root/'.ver3_missing_rainfall.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f'pid={os.getpid()}'.encode())
        yield
    finally:
        os.close(fd)
        lock.unlink()


def publish_day(root, network, date, incoming, log):
    """Backup then atomic publication; never replace existing valid observations."""
    with shared_lock(root):
        path = partition_path(root, network, date)
        old = load_day(root, network, date)
        merged, conflicts, filled = merge_observations(old, incoming)
        validate_rain(merged, date)
        changed = not old.equals(merged)
        if merged.empty:
            return conflicts, filled, False
        paths = [path, root/network/'index.parquet']
        # Back up canonical files before touching either file. Cached responses allow resume.
        for item in paths:
            if item.exists():
                digest = fingerprint(item)['sha256'][:16]
                backup = log/'backups'/network/(item.stem+'_'+digest+item.suffix)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(item, backup)
        index_path = root/network/'index.parquet'
        index = (pd.read_parquet(index_path) if index_path.exists() else
                 pd.DataFrame(columns=['kind','table','date','rows','stations','missing','file']))
        relative = str(path.relative_to(root/network))
        match = index.kind.eq('hourly') & index.table.eq('data') & index.date.eq(date.strftime('%Y/%m/%d'))
        if match.sum() > 1:
            raise ValueError('Duplicate index entries for partition')
        entry = dict(kind='hourly', table='data', date=date.strftime('%Y/%m/%d'),
                     rows=len(merged), stations=int(merged.STN.nunique()),
                     missing=int(merged.RN_1H.isna().sum()), file=relative)
        index = pd.concat([index.loc[~match], pd.DataFrame([entry])], ignore_index=True)
        # Reconcile the catalog even after interruption between the two replacements.
        if changed:
            atomic_parquet(merged, path)
        atomic_parquet(index, index_path)
        return conflicts, filled, changed


class Client:
    def __init__(self, key, log, refresh=False, pause=.3):
        self.key, self.refresh, self.pause = key, refresh, pause
        self.session = requests.Session()
        self.db = sqlite3.connect(log/'api_response_cache.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, network TEXT, params TEXT, body TEXT, sha256 TEXT, fetched_at TEXT)')
        self.log = log

    def fetch(self, network, stn, ends):
        ends = sorted(pd.Timestamp(t) for t in ends)
        params = dict(stn=str(stn), help='1')
        if network == 'AWS':
            assert len(ends) == 1
            params.update(var='RN', tm=ends[0].strftime('%Y%m%d%H%M'), disp='0')
        else:
            params.update(tm1=ends[0].strftime('%Y%m%d%H%M'), tm2=ends[-1].strftime('%Y%m%d%H%M'))
        safe = json.dumps(params, sort_keys=True)
        request_id = hashlib.sha256((network+safe).encode()).hexdigest()
        cached = self.db.execute('SELECT body FROM responses WHERE id=?', (request_id,)).fetchone()
        if cached and not self.refresh:
            frame = parse_response(cached[0], network, stn, ends)
            return frame, request_id, 'CACHE'
        error = 'REQUEST_FAILED'
        for attempt in range(1, 4):
            try:
                response = self.session.get(URLS[network], params={**params, 'authKey': self.key}, timeout=(10,60))
                if response.status_code != 200:
                    error = f'HTTP_{response.status_code}'
                    if response.status_code in (401,403):
                        raise PermissionError(error)
                    raise RuntimeError(error)
                try:
                    body = response.content.decode('utf-8')
                except UnicodeDecodeError:
                    body = response.content.decode('cp949')
                body = body.replace(self.key, '[REDACTED]')
                frame = parse_response(body, network, stn, ends)
                self.db.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?)',
                                (request_id, network, safe, body, hashlib.sha256(body.encode()).hexdigest(),
                                 pd.Timestamp.now(tz='Asia/Seoul').isoformat()))
                self.db.commit()
                return frame, request_id, 'API'
            except requests.RequestException:
                error = 'HTTP_OR_CONNECTION_ERROR'  # exception strings may contain credential URLs
            except (ValueError, UnicodeError):
                error = 'INVALID_KMA_RESPONSE_OR_SCHEMA'
            except RuntimeError:
                pass
            finally:
                time.sleep(self.pause)
            if attempt < 3:
                time.sleep(2**attempt)
        if error == 'INVALID_KMA_RESPONSE_OR_SCHEMA':
            raise ValueError(error)  # Stop rather than repeat a bad key/layout across all stations.
        raise RuntimeError(error)

    def close(self):
        self.db.close()
        self.session.close()


def report(out, plan, results, status):
    calls = int(plan.MAX_AWS_CALLS.sum()+plan.MAX_ASOS_CALLS.sum()) if len(plan) else 0
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>06 부족 강우 수집</title>'
            '<style>body{font:16px/1.7 "Malgun Gothic",sans-serif;max-width:900px;margin:40px auto}td,th{padding:6px 14px;text-align:left}</style>'
            f'<h1>06 · 02 부족 강우 수집</h1><p>{escape(status)}</p>'
            f'<p>대상 {len(plan):,} 관측소·날짜 / 원본 부족 {int(plan.RAW_MISSING_HOURS.sum()):,}시간 / 최대 {calls:,}요청.</p>'
            '<p>AWS 우선, 남은 결측에 ASOS 보완. 기존 유효값 유지, 재조회 값 차이는 검토 기록. 원본 결측은 0으로 채우지 않는다.</p>'
            '<p>표준 Parquet과 index.parquet을 갱신한다. 기존 Excel 복제본은 갱신하지 않으므로 분석 입력은 Parquet만 사용한다.</p>'
            '<p>보유한 A 표본의 부족 목록만 대상이며 B/C의 추가 기간은 포함하지 않는다. 수집 이후 위치 이력·이벤트·누적강우를 재검증해야 한다.</p>'
            '<a href="02_rain_collection_plan.parquet">요청 계획</a>')
    if len(results):
        html += results.groupby('STATUS').size().rename('요청 수').to_frame().to_html(border=0)
        html += '<a href="../../../log/06/full/02_collect/request_results.parquet">요청 기록</a>'
    (out/'02_rain_collection_summary.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Actually call API and update RAIN_DATAS; default plans only')
    parser.add_argument('--refresh', action='store_true', help='Requery previously cached API responses')
    parser.add_argument('--max-requests', type=int, default=0, help='Limit logical requests for a small execution; 0=all')
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error('--max-requests must be nonnegative')
    out, log = V3/'output/06/full', V3/'log/06/full/02_collect'
    log.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    review_path = out/'required_rain_review.parquet'
    config = json.loads((V3/'config/rainfall.json').read_text(encoding='utf-8'))
    root = Path(config['rain_root'])
    history_path = V3/'base/stations/station_location_history.xlsx'
    if json.loads((V3/'log/06/full/run.json').read_text(encoding='utf-8'))['status'] != 'COMPLETE':
        raise ValueError('06 preparation incomplete')
    review = pd.read_parquet(review_path)
    history = pd.read_excel(history_path, sheet_name='location_history')
    plan = build_plan(review, history, root)
    atomic_parquet(plan, out/'02_rain_collection_plan.parquet')
    records, conflicts, client = [], [], None
    run = dict(status='PLANNED', started=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
               inputs=[fingerprint(p) for p in [Path(__file__),review_path,history_path]],
               station_dates=len(plan), raw_missing_hours=int(plan.RAW_MISSING_HOURS.sum()),
               max_aws_calls=int(plan.MAX_AWS_CALLS.sum()), max_asos_calls=int(plan.MAX_ASOS_CALLS.sum()),
               history_unresolved_hours=int(plan.HISTORY_UNRESOLVED_HOURS.sum()),
               rainfall_downloaded=False, canonical_parquet_updated=False, source='KMA_APIHUB',
               scope='EXISTING_A_MISSING_WINDOWS_ONLY', excel_mirrors='NOT_UPDATED_USE_PARQUET',
               provenance='api_response_cache.sqlite + requests.jsonl + backups', api_key_logged=False)
    report(out, plan, pd.DataFrame(), '수집 계획 생성 · 다운로드 미실행')
    write_json(log/'run.json', run)
    if not args.execute:
        print(f"계획 완료: {len(plan)} 관측소·날짜, 부족 {run['raw_missing_hours']}시간, "
              f"최대 AWS {run['max_aws_calls']} + ASOS {run['max_asos_calls']}요청. 다운로드 미실행.",flush=True)
        return
    key = os.environ.get('KMA_API_KEY') or getpass.getpass('KMA API허브 인증키 (화면에 표시되지 않음): ')
    if not key:
        raise ValueError('KMA_API_KEY is required')
    run['status']='RUNNING'
    write_json(log/'run.json',run)
    try:
        client = Client(key, log, args.refresh)
        for i,row in enumerate(plan.itertuples()):
            date,stn = pd.Timestamp(row.DATE),int(row.STN)
            remaining = set(pd.Timestamp(x) for x in row.MISSING_END_TIMES)
            for network in ('AWS','ASOS'):
                wanted = sorted(remaining & set(pd.Timestamp(x) for x in getattr(row,network+'_END_TIMES')))
                batches = [[end] for end in wanted] if network=='AWS' else ([wanted] if wanted else [])
                fetched = []
                for ends in batches:
                    if args.max_requests and len(records) >= args.max_requests:
                        break
                    record=dict(NETWORK=network,STN=stn,DATE=date,FIRST_END=ends[0],LAST_END=ends[-1],
                                REQUESTED_HOURS=len(ends),STATUS='FAILED',FILLED_HOURS=0)
                    write_json(log/'progress.json',dict(status='RUNNING',done=i,total=len(plan),
                        requests=len(records),current=f'{network} STN {stn} {ends[0]} ~ {ends[-1]}',
                        updated=pd.Timestamp.now(tz='Asia/Seoul').isoformat()))
                    try:
                        frame,rid,source=client.fetch(network,stn,ends)
                        record.update(REQUEST_ID=rid,SOURCE=source,STATUS='RECEIVED' if len(frame) else 'NO_OBSERVATIONS',
                                      VALID_RETURNED_HOURS=int(valid_values(frame.RN_1H).sum()))
                        fetched.append(frame)
                        remaining -= set(frame.loc[valid_values(frame.RN_1H),'time_end'])
                    except RuntimeError as exc:
                        record['ERROR']=str(exc)  # fixed safe codes from Client only
                    records.append(record)
                    if len(records)%25==0:
                        print(f'06 collection: {i+1}/{len(plan)} station-days, {len(records)} requests',flush=True)
                    with (log/'requests.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
                if fetched:
                    incoming=pd.concat(fetched,ignore_index=True)
                    diff,filled,changed=publish_day(root,network,date,incoming,log)
                    if len(diff):conflicts.append(diff.assign(NETWORK=network))
                    run['canonical_parquet_updated'] |= changed
                    run['rainfall_downloaded'] |= any(r.get('SOURCE')=='API' for r in records)
                    # Filled totals are partition publication metrics, not request response counts.
                    with (log/'publications.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(NETWORK=network,DATE=str(date),STN=stn,
                                                     FILLED_HOURS=filled,VALUE_CONFLICTS=len(diff),CHANGED=changed))+'\n')
                    for record in reversed(records):
                        if record['NETWORK']==network and record['STN']==stn and record['DATE']==date:
                            record['FILLED_HOURS']=filled;break
                if args.max_requests and len(records)>=args.max_requests:break
            write_json(log/'progress.json',dict(status='RUNNING',done=i+1,total=len(plan),requests=len(records),
                                               current=f'{date:%Y-%m-%d} STN {stn}'))
            if (i+1)%10==0 or i+1==len(plan):print(f'06 collection: {i+1}/{len(plan)} days, {len(records)} requests',flush=True)
            if args.max_requests and len(records)>=args.max_requests:
                run['status']='PARTIAL_REQUEST_LIMIT';break
        else:
            run['status']='ATTEMPTS_FINISHED'
    except BaseException as exc:
        run.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                   error_type=type(exc).__name__)  # never serialize URL-bearing exception text
        raise
    finally:
        if client is not None:client.close()
        results=pd.DataFrame(records)
        if len(results):atomic_parquet(results,log/'request_results.parquet')
        if conflicts:atomic_parquet(pd.concat(conflicts,ignore_index=True),log/'value_conflicts.parquet')
        # Verify observed coverage again; successful HTTP is not recovered rainfall.
        post_plan = build_plan(review, history, root)
        atomic_parquet(post_plan, out/'02_rain_remaining_review.parquet')
        run['remaining_raw_missing_hours'] = int(post_plan.RAW_MISSING_HOURS.sum())
        if run['status'] == 'ATTEMPTS_FINISHED':
            run['status'] = ('COMPLETE_WITH_REVIEW' if run['remaining_raw_missing_hours']
                             or len(conflicts) or any(r['STATUS']=='FAILED' for r in records)
                             else 'COMPLETE')
        run.update(finished=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),requests=len(records),
                   filled_hours=sum(r.get('FILLED_HOURS',0) for r in records),
                   value_conflicts=sum(len(x) for x in conflicts))
        write_json(log/'run.json',run)
        write_json(log/'progress.json',dict(status=run['status'],requests=len(records)))
        report(out,plan,results,run['status'])


if __name__=='__main__':
    main()
