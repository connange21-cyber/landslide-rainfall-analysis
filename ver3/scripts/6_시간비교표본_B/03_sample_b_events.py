"""Match A to one other rain event at the same STN coordinates and spatial unit.

Uses completed file snapshots, never downloads rain or modifies upstream files.
Final cumulative features and C sampling remain separate steps.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def overlaps(start,end,periods):
    return any(start < b and end > a for a,b in periods)


def maximum_matching(options,rng):
    """Randomized augmenting paths; each A and unit/date has at most one match."""
    shuffled={a:[v[i] for i in rng.permutation(len(v))] for a,v in options.items()}
    owners={}
    def augment(start):
        # Iterative search avoids recursion limits on groups with many A candidates.
        queue=[start];seen_a={start};seen_b=set();parent={}
        for a in queue:
            for b in shuffled[a]:
                if b in seen_b:continue
                seen_b.add(b);parent[b]=a
                if b not in owners:
                    while True:
                        next_a=parent[b]
                        old=next((k for k,v in owners.items() if v==next_a),None)
                        owners[b]=next_a
                        if old is None: return True
                        del owners[old];b=old
                other=owners[b]
                if other not in seen_a:seen_a.add(other);queue.append(other)
        return False
    keys=sorted(shuffled)
    for i in rng.permutation(len(keys)):augment(keys[i])
    return {a:b for b,a in owners.items()}


def event_coordinates(events,hourly,history):
    h=history[['NETWORK','STN','LOCATION_VERSION','LAT','LON']].rename(columns={'NETWORK':'RAIN_SOURCE'})
    if h.duplicated(['RAIN_SOURCE','STN','LOCATION_VERSION']).any():raise ValueError('Duplicate history key')
    good=hourly[hourly.LOCATION_USABLE & hourly.RAIN_1H_ANALYSIS.notna()].merge(
        h,on=['RAIN_SOURCE','STN','LOCATION_VERSION'],how='left',validate='many_to_one')
    groups={int(stn):g.sort_values('time_start') for stn,g in good.groupby('STN')}
    rows=[]
    for stn,ev in events.groupby('STN',sort=True):
        g=groups.get(int(stn))
        for r in ev.itertuples():
            points=[]
            if g is not None:
                times=g.time_start.to_numpy()
                left=times.searchsorted(np.datetime64(r.RAIN_START))
                right=times.searchsorted(np.datetime64(r.RAIN_END_EXCLUSIVE))
                span=g.iloc[left:right]
                points=list(span[['LAT','LON']].dropna().drop_duplicates().itertuples(index=False,name=None))
                # Every hour of the event must be locatable and observed.
                complete=len(span)==int(r.DURATION_HOURS) and span[['LAT','LON']].notna().all().all()
            else:complete=False
            point=points[0] if complete and len(points)==1 else (np.nan,np.nan)
            rows.append(dict(RAIN_EVENT_ID=r.RAIN_EVENT_ID,STATION_LAT=point[0],STATION_LON=point[1],
                             COORDINATES_CONFIRMED=complete and len(points)==1))
    return events.merge(pd.DataFrame(rows),on='RAIN_EVENT_ID',validate='one_to_one')


def related_periods(a,links):
    """Same RI plus parent/uncertain EMD reports, or all children for fallback EMD."""
    level,code,emd=a.ADMIN_LEVEL,str(a.ADMIN_CODE),str(a.EMD_CODE)
    exact=links.ADMIN_LEVEL.eq(level)&links.ADMIN_CODE.astype(str).eq(code)
    same_emd=links.EMD_CODE.astype(str).eq(emd)
    if level=='EMD':mask=exact|same_emd
    else:mask=exact|(same_emd&~links.ADMIN_LEVEL.eq('RI'))
    relevant=links[mask]
    return list(relevant[['EVENT_INTERVAL_START','EVENT_INTERVAL_END_EXCLUSIVE']].itertuples(index=False,name=None))


def candidate_keys(a,events,periods,scope='ALL_AVAILABLE_YEARS',trigger_periods=()):
    selected={};counts=Counter()
    for b in events.itertuples():
        if scope=='SAME_YEAR' and b.END_DATE.year!=a.END_DATE.year:counts['OTHER_YEAR']+=1;continue
        if not (b.EVENT_VALID and b.END_CONFIRMED and b.COORDINATES_CONFIRMED
                and not b.LOCATION_REVIEW and not b.LOCATION_CHANGED and not b.CROSS_NETWORK_REVIEW):
            counts['EVENT_QC_OR_COORDINATES']+=1;continue
        if (b.STATION_LAT,b.STATION_LON)!=(a.STATION_LAT,a.STATION_LON):counts['DIFFERENT_COORDINATES']+=1;continue
        if b.RAIN_EVENT_ID==a.RAIN_EVENT_ID or b.END_DATE==a.END_DATE or overlaps(
                b.RAIN_START,b.RAIN_END_EXCLUSIVE,[(a.RAIN_START,a.RAIN_END_EXCLUSIVE)]):
            counts['SAME_A_EVENT_OR_DATE']+=1;continue
        # Use all linked candidates, including ambiguous/spatially excluded A.
        # Block both the B event and its entire calendar-day cumulative window.
        if overlaps(b.RAIN_START,b.RAIN_END_EXCLUSIVE,trigger_periods) or overlaps(
                b.END_DATE-pd.Timedelta(days=14),b.END_DATE+pd.Timedelta(days=1),trigger_periods):
            counts['TRIGGER_EVENT_OR_15DAY_WINDOW_OVERLAP']+=1;continue
        if overlaps(b.RAIN_START,b.RAIN_END_EXCLUSIVE,periods) or overlaps(
                b.END_DATE,b.END_DATE+pd.Timedelta(days=1),periods):
            counts['KNOWN_LANDSLIDE_OVERLAP']+=1;continue
        selected.setdefault(b.END_DATE,[]).append(b.RAIN_EVENT_ID)
    return selected,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    out=V3/'output/06/full';log=V3/'log/06/full/03_b_sampling'
    target=out/'b_event_samples.parquet'
    if target.exists() and not args.overwrite:parser.error('Existing B outputs; --overwrite required')
    paths=[V3/'config/sampling.json',out/'occurrence_reference.parquet',
           V3/'output/03/full/rainfall_events.parquet',V3/'output/02/full/rainfall_hourly.parquet',
           V3/'base/stations/station_location_history.xlsx',V3/'output/01/full/landslide_station_links.parquet',
           V3/'output/03/full/landslide_rainfall_candidates.parquet']
    fingerprints=[fingerprint(p) for p in paths]
    config=json.loads(paths[0].read_text(encoding='utf-8'))
    if config['b_time_scope'] not in ['ALL_AVAILABLE_YEARS','SAME_YEAR']:raise ValueError('Unknown B scope')
    rng=np.random.default_rng(config['seed'])
    refs=pd.read_parquet(paths[1]);events=pd.read_parquet(paths[2])
    hourly=pd.read_parquet(paths[3],columns=['STN','time_start','RAIN_SOURCE','LOCATION_VERSION','LOCATION_USABLE','RAIN_1H_ANALYSIS'])
    history=pd.read_excel(paths[4],sheet_name='location_history');links=pd.read_parquet(paths[5])
    events=event_coordinates(events,hourly,history)
    trigger_ids=set(pd.read_parquet(paths[6]).RAIN_EVENT_ID)
    triggers=events[events.RAIN_EVENT_ID.isin(trigger_ids)]
    trigger_periods={int(stn):list(g[['RAIN_START','RAIN_END_EXCLUSIVE']].itertuples(index=False,name=None))
                     for stn,g in triggers.groupby('STN')}
    refs=refs.merge(events[['RAIN_EVENT_ID','STATION_LAT','STATION_LON','COORDINATES_CONFIRMED']],on='RAIN_EVENT_ID',validate='many_to_one')
    by_station={int(s):g.sort_values(['END_DATE','RAIN_EVENT_ID']) for s,g in events.groupby('STN')}
    event_lookup=events.set_index('RAIN_EVENT_ID')
    candidates={};eligible={};audit=[];rejections=Counter();period_cache={}
    for i,a in enumerate(refs.sort_values('A_REFERENCE_ID').itertuples()):
        if not a.SPATIAL_ELIGIBLE:status='EXCLUDED_SPATIAL'
        elif not (a.EVENT_VALID and a.END_CONFIRMED and a.COORDINATES_CONFIRMED
                  and not a.LOCATION_REVIEW and not a.LOCATION_CHANGED and not a.CROSS_NETWORK_REVIEW):
            status='REVIEW_A_EVENT_OR_COORDINATES'
        else:status='READY'
        choices={}
        if status=='READY':
            unit=a.SPATIAL_UNIT_ID
            if unit not in period_cache:period_cache[unit]=related_periods(a,links)
            choices,counts=candidate_keys(a,by_station[int(a.STN)],period_cache[unit],config['b_time_scope'],
                                          trigger_periods.get(int(a.STN),[]))
            rejections.update(counts)
            if not choices:status='NO_ELIGIBLE_B_EVENT'
            candidates[a.A_REFERENCE_ID]=choices
            eligible.setdefault(unit,[]).append(a.A_REFERENCE_ID)
        audit.append(dict(A_REFERENCE_ID=a.A_REFERENCE_ID,SPATIAL_UNIT_ID=a.SPATIAL_UNIT_ID,STN=int(a.STN),
            A_END_DATE=a.END_DATE,PREMATCH_STATUS=status,B_CANDIDATE_DATES=len(choices),
            B_CANDIDATE_EVENTS=sum(len(v) for v in choices.values()),STATION_LAT=a.STATION_LAT,STATION_LON=a.STATION_LON))
        if (i+1)%1000==0:print(f'B pool {i+1}/{len(refs)} A candidates',flush=True)
    matches={}
    for unit,ids in sorted(eligible.items()):
        options={a:sorted(candidates[a]) for a in ids}
        matches.update(maximum_matching(options,rng))
    audit=pd.DataFrame(audit)
    audit['B_END_DATE']=audit.A_REFERENCE_ID.map(matches)
    audit['MATCH_STATUS']=np.where(audit.B_END_DATE.notna(),'MATCHED',audit.PREMATCH_STATUS.replace({'READY':'NO_UNIQUE_B_DATE_AVAILABLE'}))
    records=[];ref_lookup=refs.set_index('A_REFERENCE_ID')
    for aid,date in sorted(matches.items()):
        choices=sorted(candidates[aid][date]);eid=choices[int(rng.integers(len(choices)))];b=event_lookup.loc[eid];a=ref_lookup.loc[aid]
        identity=json.dumps([a.SPATIAL_UNIT_ID,eid],ensure_ascii=False)
        record=dict(B_SAMPLE_ID='B_'+hashlib.sha256(identity.encode()).hexdigest()[:20],A_REFERENCE_ID=aid,
            SAMPLE_TYPE='B_SAME_UNIT_STATION_OTHER_RAIN_EVENT',SPATIAL_UNIT_ID=a.SPATIAL_UNIT_ID,
            ADMIN_LEVEL=a.ADMIN_LEVEL,ADMIN_CODE=a.ADMIN_CODE,ADMIN_NAME=a.ADMIN_NAME,EMD_CODE=a.EMD_CODE,
            STN=int(a.STN),STATION_LAT=b.STATION_LAT,STATION_LON=b.STATION_LON,
            A_RAIN_EVENT_ID=a.RAIN_EVENT_ID,A_END_DATE=a.END_DATE,RAIN_EVENT_ID=eid,
            RAIN_START=b.RAIN_START,RAIN_END_EXCLUSIVE=b.RAIN_END_EXCLUSIVE,
            END_DATE=b.END_DATE,END_DATE_RULE='DATE_OF_LAST_WET_INTERVAL_START',
            CUMULATIVE_START_DATE=b.END_DATE-pd.Timedelta(days=14),
            END_CONFIRMATION_TIME=b.END_CONFIRMATION_TIME,EVENT_RAIN_MM=b.EVENT_RAIN_MM,
            EVENT_VALID=b.EVENT_VALID,START_CONFIRMED=b.START_CONFIRMED,END_CONFIRMED=b.END_CONFIRMED,
            LABEL_STATUS=config['label'],LABEL_CERTAINTY='UNRESOLVED_UNREPORTED_IS_NOT_CONFIRMED_ABSENCE',
            CUMULATIVE_STATUS='NOT_COMPUTED_PARTIAL_SUM_AND_MISSING_FLAGS_REQUIRED',
            A_AMBIGUITY_REVIEW=bool(a.CANDIDATE_AMBIGUITY_REVIEW),A_SPATIAL_REVIEW=bool(a.SPATIAL_REVIEW),
            SAMPLING_STATUS=config['status'],SPLIT_GROUP_ID=eid)
        record['TRIGGER_EXCLUSION']='SAME_STN_ALL_LINKED_CANDIDATES_EVENT_AND_15DAY_WINDOW'
        record['TRIGGER_EVENT_OVERLAP']=False
        record['CUMULATIVE_TRIGGER_OVERLAP']=False
        periods=period_cache[a.SPATIAL_UNIT_ID]
        assert not overlaps(b.RAIN_START,b.RAIN_END_EXCLUSIVE,periods)
        assert not overlaps(b.END_DATE,b.END_DATE+pd.Timedelta(days=1),periods)
        triggers_for_station=trigger_periods.get(int(a.STN),[])
        assert not overlaps(b.RAIN_START,b.RAIN_END_EXCLUSIVE,triggers_for_station)
        assert not overlaps(b.END_DATE-pd.Timedelta(days=14),b.END_DATE+pd.Timedelta(days=1),triggers_for_station)
        assert b.END_DATE!=a.END_DATE and (b.STATION_LAT,b.STATION_LON)==(a.STATION_LAT,a.STATION_LON)
        records.append(record)
    samples=pd.DataFrame(records)
    if len(samples):
        assert not samples.A_REFERENCE_ID.duplicated().any()
        assert not samples.duplicated(['SPATIAL_UNIT_ID','END_DATE']).any()
    assert fingerprints==[fingerprint(p) for p in paths],'Upstream inputs changed during run'
    log.mkdir(parents=True,exist_ok=True)
    for frame,name in [(samples,'b_event_samples'),(audit,'b_sampling_status')]:
        p=out/(name+'.parquet');tmp=p.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False)
        assert len(pd.read_parquet(tmp))==len(frame);tmp.replace(p)
    status_counts=audit.MATCH_STATUS.value_counts().to_dict()
    run=dict(status='COMPLETE',config=config,inputs=fingerprints,code=fingerprint(Path(__file__)),a_rows=len(refs),
        b_samples=len(samples),status_counts=status_counts,candidate_rejections=dict(rejections),
        location_equality='EXACT_LAT_LON_FROM_NETWORK_SPECIFIC_HISTORY',
        input_rainfall='READ_ONLY_STAGE02_03_SNAPSHOT_NOT_REBUILT_AFTER_3_HOUR_COLLECTION',
        downloads=False,c_sampling=False,cumulative_computed=False,imputation=False,
        triggering_candidate_events=len(triggers),
        completed=pd.Timestamp.now(tz='Asia/Seoul').isoformat())
    write_json(log/'run.json',run)
    write_json(log/'verification.json',dict(status='PASS',unique_a=True,unique_unit_end_date=True,
        same_stn_coordinates=True,known_occurrence_overlap=False,trigger_event_overlap=False,
        cumulative_trigger_overlap=False,upstream_unchanged=True))
    year_table=samples.groupby(samples.END_DATE.dt.year).size().rename('B 표본 수').to_frame().to_html(border=0) if len(samples) else ''
    html=('<!doctype html><html lang="ko"><meta charset="utf-8"><title>06 B 비교표본</title>'
        '<style>body{font:16px/1.7 "Malgun Gothic",sans-serif;max-width:960px;margin:36px auto}td,th{padding:5px 18px;text-align:left}</style>'
        '<h1>06 · 03 B 비교표본</h1>'
        '<p>같은 행정구역·STN·관측소 좌표에서 보유한 전체 연도의 다른 강우 이벤트를 무작위 추출했다. seed=42.</p>'
        f'<p>A 후보 {len(refs):,}개 / B 매칭 {len(samples):,}개. 구역·종료일이 같은 B는 중복 사용하지 않았다.</p>'
        '<p>이벤트 종료일은 마지막 강우 시간 구간의 시작 날짜다. 기존 1~15일 일단위 누적 기준을 유지한다.</p>'
        '<p>같은 구역 및 상위 읍면동의 알려진 산사태 기간과 B 이벤트·종료일이 겹치면 제외했다. 발생 기록 없음은 실제 미발생 확정이 아니다.</p>'
        '<p>같은 STN의 모든 산사태 연결 후보 이벤트(애매한 후보 포함)와 B 이벤트 또는 이전 15일 누적 구간이 겹치면 제외했다.</p>'
        '<p>현재 보유한 이벤트 목록 기준의 잠정 표본이다. 신규 3시간 반영 재계산·누적 부분합/결측 표시·C 추출·학습/평가 분리는 후속 작업이다.</p>'
        +pd.Series(status_counts,name='A 후보 수').to_frame().to_html(border=0)+year_table+
        '<a href="b_event_samples.parquet">B 표본</a> · <a href="b_sampling_status.parquet">A별 매칭·보류 상태</a></html>')
    (out/'03_b_sampling_summary.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(b_samples=len(samples),status_counts=status_counts),ensure_ascii=False))


if __name__=='__main__':main()
