"""Collect B/C absent observation rows only; default is a read-only collection plan."""
import argparse
import getpass
import importlib.util
import json
import os
import sys
import time
import math
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def helpers():
    spec=importlib.util.spec_from_file_location('canonical_rain_collector',V3/'scripts/temporal_controls/02_collect_missing_rainfall.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def eligible_ends(targets,history,network,stn):
    """Require known source-network coordinates covering the whole hour."""
    allowed=set()
    for r in history[history.STN.eq(stn)&history.NETWORK.eq(network)].itertuples():
        if bool(r.REVIEW_FLAG):continue
        for t in targets.itertuples():
            start,end=t.time_start,t.time_end
            if not (t.STATION_LAT==r.LAT and t.STATION_LON==r.LON):continue
            if start<r.ANALYSIS_FROM or (pd.notna(r.ANALYSIS_TO) and end>r.ANALYSIS_TO):continue
            if pd.notna(r.LAST_OLD_OBSERVED) and pd.notna(r.FIRST_NEW_OBSERVED):
                if start<r.FIRST_NEW_OBSERVED and end>r.LAST_OLD_OBSERVED:continue
            allowed.add(end)
    return allowed


def build_plan(required,history,root,api):
    cache={};rows=[];started=time.monotonic();last=started
    groups=required.groupby(['STN','DATE'],sort=True);total=groups.ngroups
    print(f'[수집 계획 확인] 0/{total:,} 관측소·날짜 | 기존 파일·위치 이력 확인 중',flush=True)
    for i,((stn,date),group) in enumerate(groups,1):
        stn=int(stn);all_ends=set(group.time_end)
        present=set();valid=set()
        for network in ['AWS','ASOS']:
            key=(network,date)
            if key not in cache:cache[key]=api.load_day(root,network,date)
            f=cache[key];f=f[f.STN.eq(stn)]
            present.update(f.time_end);valid.update(f.loc[api.valid_values(f.RN_1H),'time_end'])
        absent=all_ends-present
        targets=group[group.time_end.isin(absent)]
        eligible={net:eligible_ends(targets,history,net,stn) for net in ['AWS','ASOS']}
        covered=eligible['AWS']|eligible['ASOS']
        rows.append(dict(STN=stn,DATE=date,ORIGINAL_ABSENT_HOURS=len(all_ends),
            ALREADY_VALID_HOURS=len(all_ends&valid),STORED_NULL_OR_INVALID_SKIP_HOURS=len((all_ends&present)-valid),
            CURRENT_ABSENT_HOURS=len(absent),MISSING_END_TIMES=sorted(absent),
            AWS_END_TIMES=sorted(eligible['AWS']),ASOS_END_TIMES=sorted(eligible['ASOS']),
            HISTORY_OR_COORDINATE_REVIEW_HOURS=len(absent-covered),
            MAX_AWS_CALLS=len(eligible['AWS']),MAX_ASOS_CALLS=int(bool(eligible['ASOS'])),
            SAMPLE_TYPES=','.join(sorted(set(group.SAMPLE_TYPE)))))
        now=time.monotonic()
        if i==1 or i==total or now-last>=2:
            elapsed=now-started;eta=elapsed*(total-i)/i
            print(f'[수집 계획 확인] {i:,}/{total:,} ({i/total:.1%}) | 경과 {elapsed:.0f}초 | 남은 약 {eta:.0f}초 | STN {stn} · {pd.Timestamp(date):%Y-%m-%d}',flush=True)
            last=now
    return pd.DataFrame(rows)


def aws_jobs(plan):
    jobs={}
    for r in plan.itertuples():
        for end in r.AWS_END_TIMES:
            jobs.setdefault(pd.Timestamp(end),set()).add(int(r.STN))
    return sorted(jobs.items())


def install_bulk_parser(api):
    # Only this independently loaded helper module is adapted; the older code file is read-only.
    original=api.parse_response
    def parse(text,network,stn,ends):
        if network!='AWS' or stn!=0:return original(text,network,stn,ends)
        lines=text.splitlines()
        comments='\n'.join(l for l in lines if l.lstrip().startswith('#'))
        # Reuse strict header/schema and HR1 checks, including HTML rejection.
        original(text if '<html' in text.lower() or '<!doctype' in text.lower() else comments,network,0,ends)
        allowed=set(pd.Timestamp(t) for t in ends);rows=[];timestamps={}
        for line in lines:
            if not line.strip() or line.lstrip().startswith('#'):continue
            v=line.split()
            if len(v)!=14:raise ValueError('Unexpected AWS bulk row width')
            # A nationwide response repeats one timestamp for every station.
            if v[0] not in timestamps:
                if len(v[0])!=12 or not v[0].isdigit():raise ValueError('Invalid AWS timestamp')
                end=pd.Timestamp(datetime.strptime(v[0],'%Y%m%d%H%M'))
                if end.minute:raise ValueError('Non-hourly AWS bulk timestamp')
                timestamps[v[0]]=(end-pd.Timedelta(hours=1),end)
            start,end=timestamps[v[0]]
            station=int(v[1]);amount=float(v[6])
            if end in allowed:rows.append((station,start,end,amount if math.isfinite(amount) and amount>=0 else float('nan')))
        f=pd.DataFrame(rows,columns=api.COLS) if rows else api.empty_rain()
        f=f.astype({'STN':'int64','RN_1H':'float64'})
        if f.duplicated(api.KEYS).any():raise ValueError('Duplicate AWS bulk keys')
        return f
    api.parse_response=parse


def parallel_aws(jobs,api,client,workers,interval):
    """Bounded prefetch; independent SQLite connections, sessions reused per thread.

    Yield in chronological order. Only the calling thread publishes Parquet/logs.
    Cache keys and cached response bodies retain the previous collector format.
    """
    gate=threading.Lock();next_request=[0.0];local=threading.local();sessions=[]
    def fetch_job(item):
        end,stations=item
        if not hasattr(local,'session'):
            import requests
            local.session=requests.Session();sessions.append(local.session)
            original_get=local.session.get
            def limited_get(*args,**kwargs):
                with gate:
                    delay=max(0.0,next_request[0]-time.monotonic())
                    if delay:time.sleep(delay)
                    next_request[0]=time.monotonic()+interval
                response=original_get(*args,**kwargs)
                if response.status_code==429:
                    raise PermissionError('API_RATE_LIMIT_429_STOPPED')
                return response
            local.session.get=limited_get
        worker=api.Client(client.key,client.log,client.refresh,pause=client.pause)
        worker.session.close();worker.session=local.session
        try:return worker.fetch('AWS',0,[end])
        finally:worker.db.close()
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='aws-rain')
    queue=deque();iterator=iter(jobs)
    try:
        for _ in range(workers):
            item=next(iterator,None)
            if item is not None:queue.append((item,pool.submit(fetch_job,item)))
        while queue:
            item,future=queue.popleft()
            yield item,future
            item=next(iterator,None)
            if item is not None:queue.append((item,pool.submit(fetch_job,item)))
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        for session in sessions:session.close()


def collect_grouped(plan,root,api,client,log,run,records,max_requests,workers=3,interval=.3):
    jobs=aws_jobs(plan);valid=set();pending=[];pending_date=None
    def flush():
        nonlocal pending
        if not pending:return
        frame=pd.concat(pending,ignore_index=True);pending=[]
        old=api.load_day(root,'AWS',pending_date)
        keys=set(zip(old.STN,old.time_end))
        frame=frame[[k not in keys for k in zip(frame.STN,frame.time_end)]]
        if len(frame):
            conflicts,filled,changed=api.publish_day(root,'AWS',pending_date,frame,log)
            if not conflicts.empty:raise ValueError('Unexpected rainfall conflicts')
            run['shared_parquet_updated']|=changed
            print(f'[Parquet 저장] AWS {pending_date:%Y-%m-%d} | 유효값 추가 {filled:,}시간',flush=True)
            with (log/'publications.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(NETWORK='AWS',STN=0,DATE=str(pending_date),FILLED_HOURS=filled,CHANGED=changed))+'\n')
    def request(net,stn,ends,stations,future=None):
        if max_requests and len(records)>=max_requests:return None
        print(f'[요청 #{len(records)+1:,}] {net} | {ends[0]} ~ {ends[-1]} | 대상 {len(stations)}개 관측소',flush=True)
        write_json(log/'progress.json',dict(status='RUNNING',mode='TIME_GROUPED',requests=len(records),network=net,first_end=str(ends[0]),target_stations=len(stations),aws_time_groups=len(jobs)))
        rec=dict(NETWORK=net,STN=stn,FIRST_END=str(ends[0]),LAST_END=str(ends[-1]),REQUESTED_HOURS=len(ends)*len(stations),STATUS='FAILED');started=time.monotonic()
        try:
            f,rid,source=future.result() if future is not None else client.fetch(net,stn,ends)
            f=f[f.STN.isin(stations)].copy()
            rec.update(STATUS='RECEIVED' if len(f) else 'NO_OBSERVATIONS',REQUEST_ID=rid,SOURCE=source,VALID_RETURNED_HOURS=int(api.valid_values(f.RN_1H).sum()))
        except RuntimeError:
            rec['ERROR']='REQUEST_FAILED';f=api.empty_rain()
        records.append(rec)
        with (log/'requests.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(rec)+'\n')
        print(f'[응답 #{len(records):,}] {rec["STATUS"]} | 유효 {rec.get("VALID_RETURNED_HOURS",0)}시간 | {rec.get("SOURCE","실패")} | {time.monotonic()-started:.1f}초',flush=True)
        return f
    selected_jobs=jobs[:max_requests] if max_requests else jobs
    prefetch=parallel_aws(selected_jobs,api,client,workers,interval)
    started=time.monotonic()
    print(f'[AWS 수집] {len(selected_jobs):,}개 시간 묶음 | {workers}병렬 | 전체 요청 간격 최소 {interval:.2f}초 | 날짜 순서로 저장',flush=True)
    try:
        for i,((end,stations),future) in enumerate(prefetch,1):
            date=(end-pd.Timedelta(hours=1)).normalize()
            if pending_date is not None and date!=pending_date:flush()
            pending_date=date
            f=request('AWS',0,[end],stations,future)
            if f is None:return 'PARTIAL_REQUEST_LIMIT'
            pending.append(f)
            good=f[api.valid_values(f.RN_1H)]
            valid.update(zip(good.STN,good.time_end))
            elapsed=time.monotonic()-started;eta=elapsed*(len(selected_jobs)-i)/i
            print(f'[AWS 시간 묶음] {i:,}/{len(jobs):,} ({i/len(jobs):.1%}) | 경과 {elapsed:.0f}초 | 이번 AWS 남은 약 {eta:.0f}초',flush=True)
    finally:
        try:prefetch.close()
        finally:flush()
    if len(selected_jobs)<len(jobs):return 'PARTIAL_REQUEST_LIMIT'
    # ASOS retains its station/day period API and only fills AWS-unrecovered eligible hours.
    for r in plan.itertuples():
        ends=sorted(pd.Timestamp(t) for t in r.ASOS_END_TIMES if (int(r.STN),pd.Timestamp(t)) not in valid)
        if not ends:continue
        f=request('ASOS',int(r.STN),ends,{int(r.STN)})
        if f is None:return 'PARTIAL_REQUEST_LIMIT'
        date=pd.Timestamp(r.DATE);old=api.load_day(root,'ASOS',date)
        keys=set(zip(old.STN,old.time_end));f=f[[k not in keys for k in zip(f.STN,f.time_end)]]
        if len(f):
            conflicts,filled,changed=api.publish_day(root,'ASOS',date,f,log)
            if not conflicts.empty:raise ValueError('Unexpected rainfall conflicts')
            run['shared_parquet_updated']|=changed
            print(f'[Parquet 저장] ASOS STN {r.STN} · {date:%Y-%m-%d} | 유효값 추가 {filled}시간',flush=True)
            with (log/'publications.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(NETWORK='ASOS',STN=int(r.STN),DATE=str(date),FILLED_HOURS=filled,CHANGED=changed))+'\n')
    return 'ATTEMPTS_FINISHED'


def report(out,plan,run):
    fields=[('원래 관측행 없음',int(plan.ORIGINAL_ABSENT_HOURS.sum())),
            ('현재도 관측행 없음',int(plan.CURRENT_ABSENT_HOURS.sum())),
            ('이미 유효값 확보',int(plan.ALREADY_VALID_HOURS.sum())),
            ('현재 저장값 결측: 재요청 제외',int(plan.STORED_NULL_OR_INVALID_SKIP_HOURS.sum())),
            ('좌표/위치 이력 확인 필요',int(plan.HISTORY_OR_COORDINATE_REVIEW_HOURS.sum())),
            ('AWS 시간 묶음 요청',len(aws_jobs(plan))),('기존 개별 AWS 요청',int(plan.MAX_AWS_CALLS.sum())),('최대 ASOS 요청',int(plan.MAX_ASOS_CALLS.sum()))]
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>07 · 05 B/C 부족 강우 수집</title><style>body{font:15px/1.7 Malgun Gothic;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>07 · 05 B/C 부족 강우 수집</h1>'
    page+=f'<p>상태: {run["status"]}. B/C 중복 STN·시간은 한 번만 수집. 기존 null/음수·위치 제외는 자동 재수집하지 않습니다.</p>'
    page+=pd.DataFrame(fields,columns=['항목','시간 또는 요청 수']).to_html(index=False,border=0)
    page+=f'<p>AWS는 같은 시간의 전체 관측소를 한 번에 조회하고 계획에 포함된 부족 관측소만 저장합니다. 최대 {run.get("aws_workers",1)}개 요청을 병렬 처리하고 전체 요청 간격은 최소 {run.get("request_interval",.3):.2f}초입니다. 날짜 순서로 단일 저장하며 중단 시 현재 날짜 버퍼도 저장합니다.</p>'
    page+='<p>기본 실행은 계획만 생성합니다. 실제 수집: --execute. 소규모 시험: --execute --max-requests 25. 기존 요청 응답은 SQLite 캐시로 재사용합니다. 인증키는 KMA_API_KEY 또는 숨김 입력으로 전달하며 로그에 남기지 않습니다.</p>'
    page+='<p>RAIN_DATAS/지상관측/{AWS|ASOS}/hourly/YYYY/MM/DD/data_YYYYMMDD.parquet에 STN,time_start,time_end,RN_1H 형식으로 추가합니다. 기존 유효값은 보존, 백업·검증 후 원자 교체·index 갱신. KST [시작,종료) 1시간, 강우 mm.</p>'
    page+='<p>위치 이력 범위 밖(예: 초기 이력 시작일 이전)이나 선택 좌표와 다른 시간은 보류합니다. 다운로드가 원래 관측 결측까지 해결한다고 가정하지 않습니다. 종료 후 강우 피처/HTML은 별도로 재생성해야 하며 이전 단계 출력은 수정하지 않습니다.</p></html>'
    (out/'summary.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Download and publish; default only plans')
    parser.add_argument('--max-requests',type=int,default=0,help='0=all; includes cache attempts, retries separate')
    parser.add_argument('--refresh',action='store_true',help='Refresh cached responses for still-absent rows only')
    parser.add_argument('--workers',type=int,default=3,help='AWS parallel requests: 1..4 (default 3); Parquet publication stays serial')
    parser.add_argument('--request-interval',type=float,default=.3,help='Global AWS request spacing in seconds, minimum 0.3')
    args=parser.parse_args()
    print('[실행 모드] '+('실제 다운로드 (--execute)' if args.execute else '계획 확인만 수행 · 다운로드하지 않음 · 실제 수집은 --execute 추가'),flush=True)
    if args.max_requests<0:parser.error('--max-requests must be nonnegative')
    if not 1<=args.workers<=4:parser.error('--workers must be 1..4')
    if not math.isfinite(args.request_interval) or args.request_interval<.3:parser.error('--request-interval must be at least 0.3')
    out=V3/'output/07/full/05_collection';log=V3/'log/07/full/05_collection'
    out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
    if args.execute and (log/'.collection_running.lock').exists():
        raise RuntimeError('기존 수집 실행 잠금이 있습니다. 이전 실행을 Ctrl+C로 완전히 종료한 뒤 다시 실행하세요. 강제 종료 후 잠금만 남았다면 PID 종료 확인 후 잠금을 정리해야 합니다.')
    audit=V3/'output/07/full/04_bc_missing/bc_hourly_availability.parquet'
    historypath=V3/'base/stations/station_location_history.xlsx';configpath=V3/'config/rainfall.json'
    print('[입력 읽기] 결측 분석표·관측소 위치 이력',flush=True)
    api=helpers();history=pd.read_excel(historypath,sheet_name='location_history')
    root=Path(json.loads(configpath.read_text(encoding='utf-8'))['rain_root'])
    f=pd.read_parquet(audit)
    required=f[f.REASON.eq('NO_OBSERVATION_ROW')][['SAMPLE_TYPE','STN','STATION_LAT','STATION_LON','time_start']].copy()
    required['time_end']=required.time_start+pd.Timedelta(hours=1)
    required['DATE']=required.time_start.dt.normalize()
    plan=build_plan(required,history,root,api)
    api.atomic_parquet(plan,out/'collection_plan.parquet')
    run=dict(status='PLANNED',started=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),
        inputs=[fingerprint(p) for p in [audit,historypath,configpath,Path(__file__),V3/'scripts/temporal_controls/02_collect_missing_rainfall.py']],
        absent_only=True,original_nulls_skipped=True,original_location_exclusions_skipped=True,
        station_days=len(plan),original_unique_hours=int(plan.ORIGINAL_ABSENT_HOURS.sum()),
        history_review_hours=int(plan.HISTORY_OR_COORDINATE_REVIEW_HOURS.sum()),
        max_aws_calls=len(aws_jobs(plan)),individual_aws_calls=int(plan.MAX_AWS_CALLS.sum()),request_mode='TIME_GROUPED',max_asos_calls=int(plan.MAX_ASOS_CALLS.sum()),
        api_key_logged=False,api_requests=0,shared_parquet_updated=False,
        aws_workers=args.workers,request_interval=args.request_interval)
    write_json(log/'run.json',run);report(out,plan,run)
    print(f'[요청 수] AWS 개별 {run["individual_aws_calls"]:,}회 → 시간 묶음 {run["max_aws_calls"]:,}회 | ASOS 최대 {run["max_asos_calls"]:,}회',flush=True)
    if not args.execute:
        print(f'[계획 완료 · 다운로드 없음] 관측행 없음 {run["original_unique_hours"]:,}시간 · 위치 검토 보류 {run["history_review_hours"]:,}시간\n실제 수집: python "{Path(__file__).resolve()}" --execute',flush=True);return
    key=os.environ.get('KMA_API_KEY') or getpass.getpass('KMA API허브 인증키 (숨김 입력): ')
    if not key:raise ValueError('KMA_API_KEY required')
    lock=log/'.collection_running.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'pid={os.getpid()}'.encode());os.close(fd)
    records=[];client=None;run['status']='RUNNING'
    try:
        client=api.Client(key,log,args.refresh)
        install_bulk_parser(api)
        run['status']=collect_grouped(plan,root,api,client,log,run,records,args.max_requests,args.workers,args.request_interval)
    except BaseException as exc:
        run.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error_type=type(exc).__name__)
        raise
    finally:
        try:
            if client is not None:client.close()
            if records:api.atomic_parquet(pd.DataFrame(records),log/'request_results.parquet')
            post=build_plan(required,history,root,api)
            api.atomic_parquet(post,out/'remaining_review.parquet')
            run.update(finished=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),logical_requests=len(records),
                api_requests=sum(r.get('SOURCE')=='API' for r in records),
                remaining_absent_hours=int(post.CURRENT_ABSENT_HOURS.sum()),
                stored_null_skip_hours=int(post.STORED_NULL_OR_INVALID_SKIP_HOURS.sum()),
                current_valid_hours=int(post.ALREADY_VALID_HOURS.sum()))
            if run['status']=='ATTEMPTS_FINISHED':run['status']='COMPLETE_WITH_REVIEW' if run['remaining_absent_hours'] or run['stored_null_skip_hours'] else 'COMPLETE'
            write_json(log/'run.json',run);report(out,post,run)
            print(json.dumps({k:v for k,v in run.items() if k!='inputs'},default=str))
        finally:
            lock.unlink(missing_ok=True)


if __name__=='__main__':main()
