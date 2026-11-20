"""Shared file I/O and standalone stage execution for ver3."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT/'ver3'
HOUR = pd.Timedelta(hours=1)

def fingerprint(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def save(frame, stage, name, mode):
    folder = V3/'output'/stage/mode
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(name+'.parquet')
    temporary = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary, index=False)
    if pq.read_metadata(temporary).num_rows != len(frame):
        raise ValueError('Parquet roundtrip row mismatch')
    temporary.replace(path)


def run_stage(number, function):
    parser = argparse.ArgumentParser(description=f'Ver3 stage {number}: file-input execution')
    parser.add_argument('--limit', type=int, default=0, help='Stage 1 smoke sample per location grade')
    parser.add_argument('--smoke', action='store_true', help='Use preceding smoke outputs')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or (args.limit and number != 1):
        parser.error('--limit is only available for stage 1; later stages use --smoke')
    mode = 'smoke' if args.smoke or args.limit else 'full'
    if number == 1 and args.smoke and not args.limit:
        args.limit = 2
    stage = f'{number:02d}'
    folder = V3/'output'/stage/mode
    if folder.exists() and any(folder.iterdir()) and not args.overwrite:
        parser.error(f'Existing output: {folder}; explicit --overwrite required')
    config_path = V3/'config/rainfall.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['rain_priority'] != ['AWS', 'ASOS'] or config['imputation'] != 'NONE_OBSERVED_BASELINE':
        raise ValueError('Unsupported priority/imputation policy')
    import inspect
    code = Path(inspect.getfile(function))
    run = dict(status='RUNNING', stage=number, mode=mode, config=config,
               started=pd.Timestamp.now().isoformat(), inputs=[fingerprint(config_path), fingerprint(code), fingerprint(Path(__file__))])
    for upstream in range(1, number):
        qc = V3/'log'/f'{upstream:02d}'/mode/'qc.json'
        if not qc.is_file():
            raise ValueError(f'Missing upstream QC: {qc}')
        stage_run = V3/'log'/f'{upstream:02d}'/mode/'run.json'
        legacy = V3/'log/01_04'/mode/'run.json'
        previous = json.loads((stage_run if stage_run.exists() else legacy).read_text(encoding='utf-8'))
        if previous.get('status') != 'COMPLETE' or previous['config'] != config:
            raise ValueError('Upstream incomplete or configuration changed; rerun preceding stages')
    def load(n, name):
        path = V3/'output'/f'{n:02d}'/mode/(name+'.parquet')
        run['inputs'].append(fingerprint(path))
        return pd.read_parquet(path)
    log = V3/'log'/stage/mode/'run.json'
    write_json(log, run)
    started = time.monotonic()
    try:
        if number == 1:
            function(config, args.limit, mode)
        elif number == 2:
            links = load(1, 'landslide_station_links')
            history = pd.read_excel(V3/'base/stations/station_location_history.xlsx', sheet_name='location_history')
            function(config, links, history, mode)
        elif number == 3:
            function(config, load(1, 'landslide_station_links'), load(2, 'rainfall_hourly'), mode)
        else:
            function(config, load(3, 'rainfall_events'), load(3, 'landslide_rainfall_candidates'),
                     load(2, 'rainfall_daily'), mode, load(1, 'landslide_station_links'))
        run['status'] = 'COMPLETE'
        write_summary(number, mode)
        print(f'{stage} COMPLETE ({mode})', flush=True)
    except Exception:
        run.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        run.update(elapsed_seconds=round(time.monotonic()-started, 2), finished=pd.Timestamp.now().isoformat())
        write_json(log, run)


def write_summary(number, mode='full'):
    """Compact HTML in each numbered stage output folder."""
    from html import escape
    stage = f'{number:02d}'
    q = json.loads((V3/'log'/stage/mode/'qc.json').read_text(encoding='utf-8'))
    titles = {1: '산사태·관측소 연결', 2: 'AWS·ASOS 강우 구성', 3: '강우 이벤트·산사태 연결', 4: '종료일 기준 누적강우'}
    goals = {1: '산사태 위치와 발생기간에 맞는 관측소를 선정한다.',
             2: 'RAIN_DATAS의 시간강우를 통합하고 완전한 일강우를 계산한다.',
             3: '강우를 이벤트로 나누고 산사태 발생기간과 겹치는 후보를 찾는다.',
             4: '종료일 기준 1~15일 누적강우를 계산하고 같은 행정구역·강우 이벤트를 묶는다.'}
    if number == 1:
        metrics = [('산사태', q['events'], '건'), ('관측소 연결', q['selected'], '건'), ('미연결', q['events']-q['selected'], '건')]
        rows = [(k, v) for k, v in q['selection_status'].items()]
        names = {'SELECTED_PROVISIONAL':'관측소 선정', 'REVIEW_ADMIN_ADDRESS_UNRESOLVED':'주소 미해결',
                 'REVIEW_EVENT_TIME_UNRESOLVED':'발생기간 미해결', 'REVIEW_ROAD_LOCATION_UNRESOLVED':'도로 위치 미해결'}
        rows = [(names.get(k,k), v) for k,v in rows]
        note = '경계 기준연도는 미검증이며 주소·위치 불확실성은 검토 상태로 유지했다.'
    elif number == 2:
        metrics = [('관측소', q['stations'], '개'), ('통합 시간강우', q['rows'], '행'), ('완전한 일강우', q['valid_daily'], '일·관측소')]
        rows = [('분석 결측 시간', q['missing_analysis_hours']), ('관측망 차이 검토', q['cross_network_review'])]
        note = '유효 AWS 우선·ASOS 보완. 결측 복원 없이 관측값 기준으로 계산했다. 하루 24시간이 유효해야 일강우를 확정했다.'
    elif number == 3:
        metrics = [('전체 강우 이벤트', q['events'], '개'), ('연결 산사태', q['linked_events'], '건'), ('겹침 후보', q['candidate_rows'], '행')]
        rows = [('전체 이벤트 중 내부 결측 없음', q['internally_valid']), ('전체 이벤트 중 종료 미확인', q['end_unconfirmed'])]
        note = 'WET ≥ 1mm, 연속 무강우 또는 결측 12시간으로 분리했다. 겹침 후보는 원인 확정이나 독립 산사태 수가 아니다.'
    else:
        metrics = [('연결 산사태', q['unique_landslides'], '건'), ('행정구역×강우 그룹', q.get('admin_rain_groups',q['rows']), '행'), ('여러 산사태를 묶은 그룹', q.get('multi_landslide_groups',0), '개')]
        rows = [('원래 후보 행 보존', q['rows']), ('행정구역 미해결 그룹', q.get('unresolved_admin_groups',0)), ('종료 미확인 그룹',q.get('grouped_end_unconfirmed',q['end_unconfirmed']))]
        rows += [(f'{n}일 누적 계산 가능 그룹', count) for n,count in q.get('group_complete_by_period',q['complete_by_period']).items()]
        spatial_qc = V3/'log/04'/mode/'spatial_qc.json'
        if spatial_qc.exists():
            sq = json.loads(spatial_qc.read_text(encoding='utf-8'))
            rows = [('공간 단위 목록 · 리',sq['ri_units']),('공간 단위 목록 · 리 없는 읍면동',sq['emd_fallback_units']),('상위 경계코드 검토 대상 리',sq['orphan_ri_units'])]+rows
        note = '리 경계가 있으면 리, 없으면 읍·면·동을 사용한다(EPSG:5186). 리를 특정할 수 없는 기록은 사건별 REVIEW로 보존했다. 서로 다른 관측소의 이벤트는 자동 병합하지 않았다. DEM·비교표본·학습은 미수행이다.'
    cards = ''.join(f'<div class="card"><span>{escape(label)}</span><strong>{value:,}<small>{escape(unit)}</small></strong></div>' for label,value,unit in metrics)
    table = ''.join(f'<tr><td>{escape(str(label))}</td><td>{value:,}</td></tr>' for label,value in rows)
    files = sorted((V3/'output'/stage/mode).glob('*.parquet'))
    links = ''.join(f'<a href="{escape(file.name)}">{escape(file.stem)}</a>' for file in files)
    verification = V3/'log/04'/mode/'verification.json'
    badge = '전체 실행' if mode == 'full' else '일부 자료 시험'
    if number == 4 and verification.exists() and json.loads(verification.read_text(encoding='utf-8')).get('status') == 'PASS':
        badge += ' · 누적값 독립 검증 통과'
    summary_date = pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d')
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{stage} {titles[number]}</title><style>
body{{font:15px/1.6 "Segoe UI","Malgun Gothic",sans-serif;background:#f3f5f8;color:#233047;margin:0}}
main{{max-width:900px;margin:44px auto;padding:0 24px}}h1{{font-size:28px;margin:10px 0}}.badge{{color:#35645a;font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:26px 0}}.card{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px}}
.card span{{color:#627086;font-size:13px}}strong{{display:block;font-size:27px;margin-top:8px}}small{{font-size:12px;font-weight:400;margin-left:6px}}
table{{width:100%;background:white;border-collapse:collapse}}td{{padding:10px 16px;border-bottom:1px solid #e9edf3}}td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{padding:16px 0;color:#58667b}}nav a{{display:block;font-size:13px;color:#345ba2;margin:4px 0}}@media(max-width:600px){{.cards{{grid-template-columns:1fr}}}}
</style><main><div class="badge">{escape(badge)} · 요약 {summary_date}</div><h1>{stage} · {titles[number]}</h1><p>{goals[number]}</p>
<section class="cards">{cards}</section><table>{table}</table><p class="note">{note}</p><nav>{links}</nav></main></html>'''
    destination = V3/'output'/stage/mode/'summary.html'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding='utf-8')
    return destination
