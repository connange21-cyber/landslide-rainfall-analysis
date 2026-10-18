"""Random other-area C controls with fixed A rainfall interval and full-inventory exclusions."""
import sys
import json
from collections import Counter
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'1'))
from common import V3, fingerprint, write_json
from spatial import StationEngine


def overlaps(start,end,periods):
    return any(pd.notna(a) and pd.notna(b) and start<b and end>a for a,b in periods)


def period_index(frame,start,end):
    exact={};parent={};children={}
    for r in frame.itertuples():
        if r.ADMIN_LEVEL not in ('RI','EMD') or not str(r.ADMIN_CODE):continue
        interval=(getattr(r,start),getattr(r,end))
        exact.setdefault((r.ADMIN_LEVEL,str(r.ADMIN_CODE)),[]).append(interval)
        emd=str(r.EMD_CODE)
        children.setdefault(emd,[]).append(interval)
        if r.ADMIN_LEVEL=='EMD':parent.setdefault(emd,[]).append(interval)
    return exact,parent,children


def relevant(unit,index):
    exact,parent,children=index
    if unit.ADMIN_LEVEL=='EMD':return children.get(str(unit.EMD_CODE),[])
    return exact.get(('RI',str(unit.ADMIN_CODE)),[])+parent.get(str(unit.EMD_CODE),[])


class Linker:
    def __init__(self,history,registry,config):
        self.history=history
        self.engine=StationEngine(history,registry.crs,box(*registry.total_bounds).buffer(10000),config)
        self.trees={};self.cache={};self.config=config

    def at(self,unit,when):
        snap=self.engine.snapshot(when)
        if snap is None:return None
        sites,polygons,owners=snap
        key=id(snap)
        resultkey=(unit.SPATIAL_UNIT_ID,key)
        if resultkey in self.cache:return self.cache[resultkey]
        if key not in self.trees:self.trees[key]=STRtree(polygons)
        shares=[];coverage=0
        for j in self.trees[key].query(unit.geometry,predicate='intersects'):
            share=unit.geometry.intersection(polygons[j]).area/unit.geometry.area
            if share<=0:continue
            coverage+=share
            for i in owners[j]:shares.append((share,int(i)))
        if not shares or abs(coverage-1)>1e-5:result=None
        else:
            best=max(s for s,i in shares)
            tied=[i for s,i in shares if abs(s-best)<=self.config['area_tie']]
            result=None if len(tied)!=1 else sites.iloc[tied[0]]
            if result is not None and result.SITE_REVIEW:result=None
        self.cache[resultkey]=result
        return result

    def interval(self,unit,start,end):
        edges=pd.concat([self.history.ANALYSIS_FROM,self.history.ANALYSIS_TO,
                         self.history.LAST_OLD_OBSERVED,self.history.FIRST_NEW_OBSERVED]).dropna()
        times=[start,*sorted(set(edges[(edges>start)&(edges<end)]))]
        sites=[self.at(unit,t) for t in times]
        if any(s is None for s in sites):return None
        keys={(int(s.STN),float(s.LAT),float(s.LON)) for s in sites}
        return sites[0] if len(keys)==1 else None


def main():
    out=V3/'output/07/full';log=V3/'log/07/full/01_sampling'
    if (out/'c_event_samples.parquet').exists():raise FileExistsError('C samples already exist; no automatic overwrite')
    paths=[V3/'output/06/full/05_selection/occurrence_selected.parquet',
           V3/'output/06/full/occurrence_reference.parquet',V3/'output/01/full/landslide_station_links.parquet',
           V3/'output/04/full/spatial_units_5186.parquet',V3/'base/stations/station_location_history.xlsx',
           V3/'output/05/full/slope_analysis_selection.parquet',V3/'config/rainfall.json']
    hashes=[fingerprint(p) for p in paths]
    a,all_a,links=[pd.read_parquet(p) for p in paths[:3]]
    registry=gpd.read_parquet(paths[3]);history=pd.read_excel(paths[4],sheet_name='location_history')
    slope=pd.read_parquet(paths[5]);config=json.loads(paths[6].read_text(encoding='utf-8'))
    for col in ['ANALYSIS_FROM','ANALYSIS_TO','LAST_OLD_OBSERVED','FIRST_NEW_OBSERVED']:
        history[col]=pd.to_datetime(history[col])
    excluded=set(slope.loc[slope.ANALYSIS_STATUS.eq('TEMP_EXCLUDED'),'SPATIAL_UNIT_ID'])
    pool=registry[~registry.SPATIAL_UNIT_ID.isin(excluded)&registry.geometry.notna()&~registry.geometry.is_empty].sort_values('SPATIAL_UNIT_ID')
    assert pool.geometry.is_valid.all() and pool.SPATIAL_UNIT_ID.is_unique
    units=list(pool.itertuples());occurrence=period_index(links,'EVENT_INTERVAL_START','EVENT_INTERVAL_END_EXCLUSIVE')
    triggers=period_index(all_a,'RAIN_START','RAIN_END_EXCLUSIVE')
    occ={u.SPATIAL_UNIT_ID:relevant(u,occurrence) for u in units}
    rain={u.SPATIAL_UNIT_ID:relevant(u,triggers) for u in units}
    linker=Linker(history,registry,config);rng=np.random.default_rng(42)
    used=set();samples=[];audit=[];reject=Counter();slope_lookup=slope.set_index('SPATIAL_UNIT_ID')
    order=a.sort_values('A_REFERENCE_ID')
    for number,r in enumerate(order.itertuples()):
        chosen=None;tries=0
        # Fresh random ordering means uniform proposals without replacement per A.
        for i in rng.permutation(len(units)):
            u=units[i];tries+=1
            if u.SPATIAL_UNIT_ID==r.SPATIAL_UNIT_ID:
                reject['SAME_A_UNIT']+=1;continue
            key=(u.SPATIAL_UNIT_ID,r.END_DATE)
            if key in used:reject['DUPLICATE_C_UNIT_DATE']+=1;continue
            if overlaps(r.RAIN_START,r.RAIN_END_EXCLUSIVE,occ[u.SPATIAL_UNIT_ID]):
                reject['KNOWN_OCCURRENCE_OVERLAP']+=1;continue
            if overlaps(r.RAIN_START,r.RAIN_END_EXCLUSIVE,rain[u.SPATIAL_UNIT_ID]):
                reject['KNOWN_TRIGGER_EVENT_OVERLAP']+=1;continue
            site=linker.interval(u,r.RAIN_START,r.RAIN_END_EXCLUSIVE)
            if site is None:reject['STATION_HISTORY_OR_TIE_REVIEW']+=1;continue
            chosen=(u,site);used.add(key);break
        audit.append(dict(A_REFERENCE_ID=r.A_REFERENCE_ID,STATUS='MATCHED' if chosen else 'NO_ELIGIBLE_AREA',PROPOSALS=tries))
        if chosen:
            u,site=chosen
            state=slope_lookup.loc[u.SPATIAL_UNIT_ID,'ANALYSIS_STATUS'] if u.SPATIAL_UNIT_ID in slope_lookup.index else 'PENDING_SLOPE_ANALYSIS'
            samples.append(dict(C_SAMPLE_ID='C_'+r.A_REFERENCE_ID,A_REFERENCE_ID=r.A_REFERENCE_ID,SAMPLE_TYPE='C',
                A_SPATIAL_UNIT_ID=r.SPATIAL_UNIT_ID,A_STN=int(r.STN),A_RAIN_EVENT_ID=r.RAIN_EVENT_ID,
                SPATIAL_UNIT_ID=u.SPATIAL_UNIT_ID,ADMIN_LEVEL=u.ADMIN_LEVEL,ADMIN_CODE=u.ADMIN_CODE,
                ADMIN_NAME=u.ADMIN_NAME,EMD_CODE=u.EMD_CODE,STN=int(site.STN),
                STATION_LAT=float(site.LAT),STATION_LON=float(site.LON),NETWORK=site.NETWORK,
                RAIN_START=r.RAIN_START,RAIN_END_EXCLUSIVE=r.RAIN_END_EXCLUSIVE,END_DATE=r.END_DATE,
                CUMULATIVE_START_DATE=r.END_DATE-pd.Timedelta(days=14),
                TIME_RULE='FIXED_A_RAIN_EVENT_INTERVAL_AND_END_DATE',STATION_METHOD='ADMIN_THIESSEN_MAX_OVERLAP_HISTORY',
                LABEL_STATUS='NO_REPORTED_LANDSLIDE',LABEL_CERTAINTY='NOT_CONFIRMED_ABSENCE',
                RAIN_STATUS='PENDING_SHARED_PARQUET_COVERAGE_CHECK',SLOPE_STATUS=state,
                KNOWN_OCCURRENCE_OVERLAP=False,KNOWN_TRIGGER_EVENT_OVERLAP=False,
                BOUNDARY_TIME_VALIDITY=u.BOUNDARY_TIME_VALIDITY,SPLIT_GROUP_ID=r.RAIN_EVENT_ID))
        if number%100==0 or number+1==len(order):
            print(f'07 C sampling: {number+1:,}/{len(order):,}, matched={len(samples):,}',flush=True)
            write_json(log/'progress.json',dict(done=number+1,total=len(order),matched=len(samples)))
    c=pd.DataFrame(samples);status=pd.DataFrame(audit)
    assert c.A_REFERENCE_ID.is_unique and not c.duplicated(['SPATIAL_UNIT_ID','END_DATE']).any()
    check=c.merge(a[['A_REFERENCE_ID','RAIN_START','RAIN_END_EXCLUSIVE','END_DATE','SPATIAL_UNIT_ID']],on='A_REFERENCE_ID',validate='one_to_one',suffixes=('','_A'))
    for col in ['RAIN_START','RAIN_END_EXCLUSIVE','END_DATE']:assert check[col].eq(check[col+'_A']).all()
    assert check.SPATIAL_UNIT_ID.ne(check.SPATIAL_UNIT_ID_A).all()
    lookup={u.SPATIAL_UNIT_ID:u for u in units}
    for r in c.itertuples():
        u=lookup[r.SPATIAL_UNIT_ID]
        assert not overlaps(r.RAIN_START,r.RAIN_END_EXCLUSIVE,relevant(u,occurrence))
        assert not overlaps(r.RAIN_START,r.RAIN_END_EXCLUSIVE,relevant(u,triggers))
    assert hashes==[fingerprint(p) for p in paths]
    out.mkdir(parents=True,exist_ok=True)
    c.to_parquet(out/'c_event_samples.parquet',index=False)
    status.to_parquet(out/'c_sampling_status.parquet',index=False)
    registry[registry.SPATIAL_UNIT_ID.isin(c.SPATIAL_UNIT_ID)].to_parquet(out/'c_spatial_units_5186.parquet',index=False)
    unlocated=int((~links.ADMIN_LEVEL.isin(['RI','EMD'])).sum())
    result=dict(status='COMPLETE_SELECTION_ONLY',seed=42,a_samples=len(a),c_samples=len(c),
        c_spatial_units=int(c.SPATIAL_UNIT_ID.nunique()),c_stations=int(c.STN.nunique()),
        same_stn_as_a=int(c.STN.eq(c.A_STN).sum()),pending_slope=int(c.SLOPE_STATUS.eq('PENDING_SLOPE_ANALYSIS').sum()),
        rejected_proposals=dict(reject),unresolved_inventory_location_records=unlocated,
        inputs=hashes,upstream_unchanged=True,rain_downloaded=False,dem_computed=False,
        excluded_trigger_inventory_included=True,whole_a_interval_preserved=True,
        occurrence_and_linked_trigger_overlap_count=0)
    write_json(log/'run.json',result)
    summary=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>07 공간 비교표본 C</title><style>body{{font:15px/1.7 Malgun Gothic;margin:35px;color:#24364a}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style>
<h1>07 공간 비교표본 C</h1><p>A의 시간 구간 고정 · 다른 행정구역 무작위 추출 · seed 42</p>
<p>A {len(a):,}개 → C {len(c):,}개 · 행정구역 {result['c_spatial_units']:,}개 · STN {result['c_stations']:,}개</p>
<p>후보 구역에 A 기간과 겹치는 발생 기록 또는 산사태 연결 강우 이벤트가 있으면 폐기 후 재추출. 전체 보유 기록(분석 제외 A 포함)을 확인했습니다. 다른 구역의 발생이나 동일 STN이라는 이유로는 제외하지 않습니다.</p>
<p><a href="c_event_samples.parquet">C 표본</a> · <a href="c_sampling_status.parquet">매칭 상태</a> · <a href="c_spatial_units_5186.parquet">C 경계 EPSG:5186</a></p>
<p>강우는 아직 다운로드·누적 계산하지 않았습니다. 신규 C 구역의 경사 확보 여부도 후속 점검이 필요합니다.</p>
<p>위치가 행정구역으로 확정되지 않은 발생 기록 {unlocated:,}건은 구역별 제외에 적용하지 못했습니다. 미포착은 실제 미발생 확정이 아닙니다.</p>
<h2>표본 예시 20개</h2>{c[['A_REFERENCE_ID','ADMIN_NAME','STN','RAIN_START','RAIN_END_EXCLUSIVE','SLOPE_STATUS']].head(20).to_html(index=False,border=0,escape=True)}</html>'''
    (out/'summary.html').write_text(summary,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','rejected_proposals']}))


if __name__=='__main__':main()
