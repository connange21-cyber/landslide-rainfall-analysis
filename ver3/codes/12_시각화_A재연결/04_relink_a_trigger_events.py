"""One trigger per landslide: exact time, otherwise peak hourly fusion rain in occurrence interval."""
from pathlib import Path
import importlib.util
import gzip
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import csr_matrix

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/12/full/04_a_relink'
LOG=V3/'log/12/full/04_a_relink'

def stamp(p):
    st=p.stat();return [st.st_size,st.st_mtime_ns]

def choose_peak(times,values,candidates):
    """Do not break event ties or choose a lower peak just to obtain a valid event."""
    if not len(values):return dict(STATUS='REVIEW_NO_HOURS')
    if not np.isfinite(values).all():return dict(STATUS='REVIEW_INCOMPLETE_PEAK_WINDOW')
    peak=float(np.max(values))
    if peak<=0:return dict(STATUS='REVIEW_NO_POSITIVE_FUSION_RAIN',PEAK_RAIN_MM=peak)
    winners=np.flatnonzero(np.isclose(values,peak,rtol=0,atol=1e-6))
    found=set();unmatched=0;peak_times=[]
    for i in winners:
        end=times[i]+pd.Timedelta(hours=1)
        # Rain values describe the preceding hour, consistent with existing exact-time association.
        matching=candidates[candidates.RAIN_START.lt(end)&candidates.RAIN_END_EXCLUSIVE.ge(end)]
        if len(matching)!=1:unmatched+=1
        else:found.add(matching.RAIN_EVENT_ID.iloc[0])
        peak_times.append(times[i])
    fields=dict(PEAK_RAIN_MM=peak,PEAK_TIME_START=min(peak_times),PEAK_TIME_END=min(peak_times)+pd.Timedelta(hours=1),
        PEAK_TIED_HOURS=len(winners),PEAK_TIED_EVENT_COUNT=len(found),PEAK_UNMATCHED_HOURS=unmatched)
    if unmatched:return dict(STATUS='REVIEW_PEAK_OUTSIDE_STATION_EVENTS',**fields)
    if len(found)!=1:return dict(STATUS='REVIEW_TIED_MAXIMUM_EVENTS',**fields)
    return dict(STATUS='SELECTED_PEAK_HOURLY_ESTIMATE',SELECTED_RAIN_EVENT_ID=next(iter(found)),**fields)

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    paths=[V3/'output/01/full/landslide_station_links.parquet',V3/'output/04/full/event_spatial_units.parquet',
        V3/'output/03/full/landslide_rainfall_candidates.parquet',V3/'output/03/full/rainfall_events.parquet',
        V3/'output/04/full/landslide_end_date_features.parquet',V3/'output/04/full/admin_rain_event_features.parquet',
        V3/'output/04/full/spatial_units_5186.parquet',V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet',
        Path('/data/RAIN_DATAS/융합기상/INT/grid.parquet'),V3/'codes/10_융합강우_수집/01_collect_fusion_hourly_fixed.py',
        V3/'output/08/full/15_abc_selection/occurrence_selected.parquet',
        V3/'output/08/full/15_abc_selection/b_event_selected.parquet',V3/'output/08/full/15_abc_selection/c_event_selected.parquet',
        V3/'output/11/full/01_rainfall_summary/abc_fusion_rainfall.parquet']
    signatures={str(p):stamp(p) for p in paths}
    original=pd.read_parquet(paths[0]);spatial=pd.read_parquet(paths[1])
    original=original.merge(spatial[['EVENT_ID','SPATIAL_UNIT_ID','SPATIAL_UNIT_STATUS']],on='EVENT_ID',validate='one_to_one')
    assert len(original)==5160 and original.EVENT_ID.is_unique
    candidates=pd.read_parquet(paths[2]);events=pd.read_parquet(paths[3])
    assert not candidates.duplicated(['EVENT_ID','RAIN_EVENT_ID']).any()
    candidates=candidates.merge(events[['RAIN_EVENT_ID','RAIN_START','RAIN_END_EXCLUSIVE']],on='RAIN_EVENT_ID',validate='many_to_one')
    candidate_map={key:g for key,g in candidates.groupby('EVENT_ID')}
    need=original[original.EVENT_ID.isin(candidate_map)&~original.EXACT_TIME.eq(True)&original.SPATIAL_UNIT_ID.ne('')]
    units=sorted(need.SPATIAL_UNIT_ID.unique());ui={u:i for i,u in enumerate(units)}
    registry=gpd.read_parquet(paths[6]).to_crs(5186)
    grid=pd.read_parquet(paths[8]);links=pd.read_parquet(paths[7]);links=links[links.SPATIAL_UNIT_ID.isin(units)]
    missing_units=set(units)-set(links.SPATIAL_UNIT_ID)
    if missing_units:
        print(f'기존 격자 연결 없는 {len(missing_units)}구역 중심점 재확인',flush=True)
        points=gpd.GeoDataFrame(grid[['GRID_ID']],geometry=gpd.points_from_xy(grid.LON,grid.LAT),crs=4326).to_crs(5186)
        extra=gpd.sjoin(points,registry[registry.SPATIAL_UNIT_ID.isin(missing_units)][['SPATIAL_UNIT_ID','geometry']],predicate='within',how='inner')[['GRID_ID','SPATIAL_UNIT_ID']]
        links=pd.concat([links,extra],ignore_index=True).drop_duplicates()
    links.to_parquet(OUT/'admin_grid_links.parquet',index=False)
    gids=np.sort(links.GRID_ID.unique())
    matrix=csr_matrix((np.ones(len(links)),(links.SPATIAL_UNIT_ID.map(ui).to_numpy(),np.searchsorted(gids,links.GRID_ID))),shape=(len(units),len(gids)))
    counts=np.asarray(matrix.sum(axis=1)).ravel()
    days={}
    for r in need.itertuples(index=False):
        if pd.isna(r.EVENT_INTERVAL_START) or pd.isna(r.EVENT_INTERVAL_END_EXCLUSIVE):continue
        for day in pd.date_range(pd.Timestamp(r.EVENT_INTERVAL_START).floor('D'),(pd.Timestamp(r.EVENT_INTERVAL_END_EXCLUSIVE)-pd.Timedelta(nanoseconds=1)).floor('D')):
            days.setdefault(day,set()).add(ui[r.SPATIAL_UNIT_ID])
    spec=importlib.util.spec_from_file_location('trigger_decoder',paths[9]);decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    daily={};raw_sources=[];missing_raw=[]
    for num,(day,required) in enumerate(sorted(days.items()),1):
        vals=np.zeros((len(gids),24),dtype='float32');valid=np.zeros((len(gids),24),dtype=bool)
        for hour in range(24):
            end=day+pd.Timedelta(hours=hour+1);p=decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            if not p.exists():missing_raw.append(str(p));continue
            before=stamp(p);v=decoder.decode(gzip.decompress(p.read_bytes()))[gids];ok=np.isfinite(v)&(v>=0)
            vals[:,hour]=np.where(ok,v,0);valid[:,hour]=ok
            assert before==stamp(p);raw_sources.append(dict(path=str(p),stamp=before))
        total=matrix@vals;number=matrix@valid.astype(float)
        for i in required:
            mean=np.divide(total[i],number[i],out=np.full(24,np.nan),where=number[i]>0)
            mean[(number[i]!=counts[i])|(counts[i]==0)]=np.nan;daily[(i,day)]=mean
        if num%20==0 or num==len(days):
            print(f'A 대표 강우 시각 계산 {num}/{len(days)}일',flush=True)
            (LOG/'progress.json').write_text(json.dumps(dict(status='COMPUTING_PEAKS',done=num,total=len(days))),encoding='utf-8')
    decisions=[];cache={}
    for r in original.itertuples(index=False):
        cand=candidate_map.get(r.EVENT_ID);base=dict(EVENT_ID=r.EVENT_ID,TIME_GRADE=r.TIME_GRADE,
            SPATIAL_UNIT_ID=r.SPATIAL_UNIT_ID,ORIGINAL_CANDIDATE_COUNT=0 if cand is None else len(cand),
            EVENT_INTERVAL_START=r.EVENT_INTERVAL_START,EVENT_INTERVAL_END_EXCLUSIVE=r.EVENT_INTERVAL_END_EXCLUSIVE)
        if cand is None:choice=dict(STATUS='REVIEW_NO_ORIGINAL_CANDIDATE')
        elif r.EXACT_TIME is True or r.EXACT_TIME==True:
            matching=cand[cand.RAIN_START.lt(r.EVENT_INTERVAL_START)&cand.RAIN_END_EXCLUSIVE.ge(r.EVENT_INTERVAL_START)]
            choice=dict(STATUS='SELECTED_EXACT_RECORDED_TIME',SELECTED_RAIN_EVENT_ID=matching.RAIN_EVENT_ID.iloc[0]) if len(matching)==1 else dict(STATUS='REVIEW_EXACT_TIME_NOT_UNIQUE')
        elif not r.SPATIAL_UNIT_ID or r.SPATIAL_UNIT_ID not in ui:choice=dict(STATUS='REVIEW_ADMIN_UNRESOLVED')
        elif counts[ui[r.SPATIAL_UNIT_ID]]==0:choice=dict(STATUS='REVIEW_NO_GRID_CENTERS')
        elif pd.isna(r.EVENT_INTERVAL_START) or pd.isna(r.EVENT_INTERVAL_END_EXCLUSIVE):choice=dict(STATUS='REVIEW_NO_OCCURRENCE_PERIOD')
        else:
            key=(r.SPATIAL_UNIT_ID,r.EVENT_INTERVAL_START,r.EVENT_INTERVAL_END_EXCLUSIVE)
            if key not in cache:
                start=pd.Timestamp(r.EVENT_INTERVAL_START).floor('h');end=pd.Timestamp(r.EVENT_INTERVAL_END_EXCLUSIVE).ceil('h')
                times=pd.date_range(start,end-pd.Timedelta(hours=1),freq='h')
                values=np.array([daily[(ui[r.SPATIAL_UNIT_ID],t.floor('D'))][t.hour] for t in times])
                cache[key]=(times,values)
            times,values=cache[key];choice=choose_peak(times,values,cand)
        decisions.append(dict(**base,**choice))
    selection=pd.DataFrame(decisions)
    if 'SELECTED_RAIN_EVENT_ID' not in selection:selection['SELECTED_RAIN_EVENT_ID']=None
    selection['TRIGGER_CERTAINTY']=np.where(selection.STATUS.eq('SELECTED_EXACT_RECORDED_TIME'),'EXACT_TIME_ASSOCIATION',
        np.where(selection.STATUS.eq('SELECTED_PEAK_HOURLY_ESTIMATE'),'INFERRED_PEAK_HOURLY_RULE','UNRESOLVED'))
    selection.to_parquet(OUT/'event_trigger_selection.parquet',index=False)
    picked=selection[selection.SELECTED_RAIN_EVENT_ID.notna()][['EVENT_ID','SELECTED_RAIN_EVENT_ID','TRIGGER_CERTAINTY']].rename(columns={'SELECTED_RAIN_EVENT_ID':'RAIN_EVENT_ID'})
    assert picked.EVENT_ID.is_unique and len(picked)<=5160
    details=pd.read_parquet(paths[4]);assert not details.duplicated(['EVENT_ID','RAIN_EVENT_ID']).any()
    members=picked.merge(details,on=['EVENT_ID','RAIN_EVENT_ID'],validate='one_to_one')
    assert len(members)==len(picked) and members.EVENT_ID.is_unique
    old_groups=pd.read_parquet(paths[5]);old_a=pd.read_parquet(paths[10]);old_b=pd.read_parquet(paths[11]);old_c=pd.read_parquet(paths[12])
    assert all(f.A_REFERENCE_ID.is_unique for f in [old_a,old_b,old_c])
    assert set(old_a.A_REFERENCE_ID)==set(old_b.A_REFERENCE_ID)==set(old_c.A_REFERENCE_ID)
    damage=original.set_index('EVENT_ID').DAMAGE_HA
    new_rows=[]
    for group,g in members.groupby('ADMIN_RAIN_GROUP_ID',sort=True):
        ha=pd.to_numeric(g.EVENT_ID.map(damage),errors='coerce');valid=np.isfinite(ha)&ha.gt(0)
        new_rows.append(dict(A_REFERENCE_ID=group,LANDSLIDE_EVENT_IDS=g.EVENT_ID.tolist(),LANDSLIDE_COUNT=len(g),
            ORIGINAL_MULTI_CANDIDATE_RECORDS=int(g.EVENT_HAS_MULTIPLE_GROUP_CANDIDATES.sum()),
            DAMAGE_AREA_M2=float(ha[valid].sum()*10000) if valid.any() else np.nan,
            AREA_VALID_RECORDS=int(valid.sum()),AREA_EXCLUDED_RECORDS=int((~valid).sum()),
            EXACT_TIME_RECORDS=int(g.TRIGGER_CERTAINTY.eq('EXACT_TIME_ASSOCIATION').sum()),
            ESTIMATED_TRIGGER_RECORDS=int(g.TRIGGER_CERTAINTY.eq('INFERRED_PEAK_HOURLY_RULE').sum())))
    group_members=pd.DataFrame(new_rows)
    base=old_groups.rename(columns={'ADMIN_RAIN_GROUP_ID':'A_REFERENCE_ID'})
    legacy=['LANDSLIDE_EVENT_IDS','LANDSLIDE_COUNT','MULTIPLE_GROUP_CANDIDATE_EVENT_COUNT','CANDIDATE_ROW_COUNT','LABEL_STATUS','GROUP_STATUS']
    base=base.rename(columns={c:'SOURCE_'+c for c in legacy if c in base})
    all_a=group_members.merge(base,on='A_REFERENCE_ID',validate='one_to_one')
    assert len(all_a)<=len(picked) and all_a.LANDSLIDE_COUNT.sum()==len(picked)
    names=registry.set_index('SPATIAL_UNIT_ID').FULL_NAME
    all_a['ADMIN_NAME']=all_a.SPATIAL_UNIT_ID.map(names)
    all_a['IN_CURRENT_SELECTED_A']=all_a.A_REFERENCE_ID.isin(old_a.A_REFERENCE_ID)
    all_a['LABEL_STATUS']='ONE_TRIGGER_PER_EVENT_WITH_INFERRED_PERIOD_CASES'
    all_a['TRIGGER_RULE']='EXACT_TIME_OR_GLOBAL_PEAK_HOURLY_FUSION_WITHIN_OCCURRENCE_PERIOD'
    selected_a=all_a[all_a.IN_CURRENT_SELECTED_A].copy()
    fusion=pd.read_parquet(paths[13]);fusion=fusion[fusion.SAMPLE_TYPE.eq('A')]
    fields=['A_REFERENCE_ID','GRID_COUNT']+[c for c in fusion if c.startswith('FUSION_')]
    selected_a=selected_a.drop(columns=[c for c in fields if c!='A_REFERENCE_ID' and c in selected_a]).merge(fusion[fields],on='A_REFERENCE_ID',validate='one_to_one')
    assert selected_a.A_REFERENCE_ID.is_unique and set(selected_a.A_REFERENCE_ID)<=set(old_a.A_REFERENCE_ID)
    kept=set(selected_a.A_REFERENCE_ID)
    members['IN_RELINKED_SELECTED_A']=members.ADMIN_RAIN_GROUP_ID.isin(kept)
    assert members.EVENT_ID.is_unique
    # B/C rows remain untouched. This audit records whether their parent A survived.
    audit=old_a[['A_REFERENCE_ID','SPATIAL_UNIT_ID','END_DATE']].copy()
    audit['A_RELINK_DECISION']=np.where(audit.A_REFERENCE_ID.isin(kept),'KEEP_PARENT_A','DROP_PARENT_A')
    audit['OLD_B_AVAILABLE']=audit.A_REFERENCE_ID.isin(old_b.A_REFERENCE_ID)
    audit['OLD_C_AVAILABLE']=audit.A_REFERENCE_ID.isin(old_c.A_REFERENCE_ID)
    all_a.to_parquet(OUT/'a_relinked_all.parquet',index=False)
    selected_a.to_parquet(OUT/'a_relinked_selected.parquet',index=False)
    members.to_parquet(OUT/'a_group_membership.parquet',index=False)
    audit.to_parquet(OUT/'bc_inheritance_audit.parquet',index=False)
    assert signatures=={str(p):stamp(p) for p in paths}
    assert all(s['stamp']==stamp(Path(s['path'])) for s in raw_sources)
    report=dict(status='COMPLETE_WITH_UNRESOLVED',original_events=5160,original_candidate_links=len(candidates),
        selected_one_trigger_records=len(picked),event_status_counts=selection.STATUS.value_counts().to_dict(),
        relinked_all_groups=len(all_a),previous_selected_a=len(old_a),relinked_selected_a=len(selected_a),
        selected_a_unique_event_records=int(members.IN_RELINKED_SELECTED_A.sum()),
        old_a_groups_removed=len(old_a)-len(selected_a),groups_not_in_current_a=int((~all_a.IN_CURRENT_SELECTED_A).sum()),
        inherit_b_rows=len(selected_a),inherit_c_rows=len(selected_a),bc_modified=False,
        repeated_event_assignments=0,peak_source='UNIT_VERIFIED_FUSION_ADMIN_GRID_EQUAL_MEAN',
        event_boundaries_source='UNCHANGED_AWS_ASOS_STAGE03',ties_policy='DIFFERENT_EVENTS_OR_UNMATCHED_MAXIMA_REMAIN_REVIEW',
        exact_time_policy='PRESERVE_RECORDED_EXACT_TIME',missing_raw=missing_raw,
        inputs=signatures,raw_sources=raw_sources,upstream_unchanged=True,elapsed_seconds=time.time()-started)
    render(report,selection,selected_a)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ['inputs','raw_sources']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','raw_sources']},ensure_ascii=False),flush=True)

def render(report,selection,groups):
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>12 A 대표 강우 재연결</title><style>body{font:15px/1.7 Malgun Gothic;margin:30px;color:#21344a}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}details{margin:16px 0}</style><h1>A · 대표 강우 이벤트 재연결</h1>'
    stages=[('원본 산사태 기록',report['original_events']),('기존 후보 연결',report['original_candidate_links']),
        ('대표 이벤트 하나 선택된 기록',report['selected_one_trigger_records']),('재합산 전체 A 묶음',report['relinked_all_groups']),
        ('기존 선별 범위에서 남은 A 묶음',report['relinked_selected_a']),('남은 A에 들어간 고유 사건',report['selected_a_unique_event_records'])]
    page+=pd.DataFrame(stages,columns=['단계','개수']).to_html(index=False,border=0)
    page+='<p><strong>같은 사건의 반복 배정: 0건.</strong> 원본 사건별 대표 이벤트는 최대 하나입니다. 실제 시각이 있는 사건은 시각을 우선하고, 기간만 있는 사건은 그 기간 전체의 최대 시간강우 시각이 들어가는 기존 이벤트를 선택했습니다.</p>'
    page+='<p>최대 시간강우는 단위 보정된 융합기상 행정구역 격자 평균입니다. 이벤트 분리는 기존 AWS·ASOS 기준을 유지했습니다. 동률이 여러 이벤트에 걸치거나 최대 시각이 기존 이벤트 밖이면 미확정으로 남겼습니다. 추정된 대표 이벤트이며 원인 관계의 확정이 아닙니다.</p>'
    page+=f'<p>기존 A {report["previous_selected_a"]:,}개 중 {report["old_a_groups_removed"]:,}개는 빠지고 {report["relinked_selected_a"]:,}개가 남습니다. 남은 A의 B/C는 각각 {report["inherit_b_rows"]:,}개를 그대로 이어받을 수 있습니다. 이번 실행에서 B/C 원본은 수정하지 않았습니다.</p>'
    page+=f'<p>기존 최종 A 범위 밖의 대표 묶음 {report["groups_not_in_current_a"]:,}개는 전체 재연결표에만 남겼습니다. 이전 공간/강우/피복 선별 조건을 통과했다고 간주하거나 B/C를 임의 연결하지 않았습니다.</p>'
    page+='<h2>미확정·선택 상태</h2>'+selection.STATUS.value_counts().rename_axis('상태').reset_index(name='기록 수').to_html(index=False,border=0)
    page+='<details><summary>데이터 구조</summary><p>사건 선택표 → 사건별 단일 배정표 → 행정구역/강우 이벤트별 A 합산표 순서입니다. A_REFERENCE_ID는 기존 키를 유지합니다. 남은 키의 B/C는 그대로 유지하고 빠진 키의 B/C를 제외하면 됩니다. 단, A의 사건 목록과 피해면적 합계는 이번 표로 교체해야 합니다.</p><p>event_trigger_selection.parquet: 원본5160건의 선택/미확정 사유<br>a_group_membership.parquet: 선택된 사건별 하나의 묶음<br>a_relinked_all.parquet: 전체 재합산<br>a_relinked_selected.parquet: 기존 선별 범위에 남은 새 A<br>bc_inheritance_audit.parquet: 기존 B/C 부모 키 유지/제외 감사표</p></details>'
    page+='<p><a href="a_relinked_selected.parquet">새 A 기준표</a> · <a href="event_trigger_selection.parquet">사건 선택·미확정 기록</a> · <a href="bc_inheritance_audit.parquet">B/C 연결 감사표</a></p></html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
