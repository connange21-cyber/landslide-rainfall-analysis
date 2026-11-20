"""Replace only no-grid C controls; preserve A times and reuse full-area rain cache."""
from pathlib import Path
import importlib.util
import json
import gzip
import time
from collections import Counter
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import csr_matrix

V3 = Path(__file__).resolve().parents[2]
OUT = V3 / 'output/13/full/03_c_grid_replacement'
LOG = V3 / 'log/13/full/03_c_grid_replacement'
SEED = 20261016


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    sampler = load_module('original_c_sampler', V3 / 'scripts/7_공간비교표본_C/01_sample_spatial_controls.py')
    decoder = load_module('fusion_decoder', V3 / 'scripts/10_융합강우_수집/01_collect_fusion_hourly_fixed.py')
    root = V3 / 'output/13/full/01_abc_rebuild'
    paths = [root/'occurrence_selected.parquet', root/'c_event_selected.parquet',
             V3/'output/04/full/spatial_units_5186.parquet',
             V3/'output/01/full/landslide_station_links.parquet',
             V3/'output/06/full/occurrence_reference.parquet',
             V3/'base/stations/station_location_history.xlsx', V3/'config/rainfall.json',
             V3/'output/10/full/01_fusion_hourly/admin_grid_links.parquet',
             V3/'output/08/full/14_admin_coverage/admin_landcover_coverage.parquet',
             V3/'output/05/full/slope_analysis_selection.parquet',
             V3/'output/09/full/02_c_slope/c_admin_slope_features.parquet']
    signatures = {str(p): stamp(p) for p in paths}
    a, old_c = [pd.read_parquet(p) for p in paths[:2]]
    registry = gpd.read_parquet(paths[2])
    inventory, all_triggers = [pd.read_parquet(p) for p in paths[3:5]]
    history = pd.read_excel(paths[5])
    for col in ['ANALYSIS_FROM','ANALYSIS_TO','LAST_OLD_OBSERVED','FIRST_NEW_OBSERVED']:
        history[col] = pd.to_datetime(history[col])
    config = json.loads(paths[6].read_text(encoding='utf-8'))
    grid_links, cover, slope_old, slope_new = [pd.read_parquet(p) for p in paths[7:]]
    grid_links = grid_links.drop_duplicates(['SPATIAL_UNIT_ID','GRID_ID'])
    grid_count = grid_links.groupby('SPATIAL_UNIT_ID').size()
    slope_ok = set(slope_old.loc[slope_old.ANALYSIS_STATUS.eq('INCLUDED') & slope_old.VALID_CELLS.gt(0) & slope_old.AREA_COVERAGE_FRACTION.ge(.9),'SPATIAL_UNIT_ID'])
    slope_ok |= set(slope_new.loc[slope_new.STATUS.eq('COMPUTED') & slope_new.VALID_CELLS.gt(0) & slope_new.AREA_COVERAGE_FRACTION.ge(.9),'SPATIAL_UNIT_ID'])
    excluded = set(slope_old.loc[slope_old.ANALYSIS_STATUS.eq('TEMP_EXCLUDED'),'SPATIAL_UNIT_ID'])
    pool = registry[registry.SPATIAL_UNIT_ID.isin(grid_count.index) & registry.SPATIAL_UNIT_ID.isin(slope_ok) & ~registry.SPATIAL_UNIT_ID.isin(excluded)].sort_values('SPATIAL_UNIT_ID')
    assert pool.SPATIAL_UNIT_ID.is_unique and pool.geometry.is_valid.all()
    units = list(pool.itertuples())
    assert cover[['REFERENCE_YEAR','SPATIAL_UNIT_ID']].duplicated().sum() == 0
    coverage = cover.set_index(['REFERENCE_YEAR','SPATIAL_UNIT_ID']).COVERAGE_PERCENT.to_dict()
    targets = old_c[old_c.NO_GRID | old_c.GRID_COUNT.eq(0)].sort_values('A_REFERENCE_ID')
    assert len(targets) == 67 and old_c.A_REFERENCE_ID.is_unique
    kept = old_c[~old_c.A_REFERENCE_ID.isin(targets.A_REFERENCE_ID)].copy()
    occurrence = sampler.period_index(inventory,'EVENT_INTERVAL_START','EVENT_INTERVAL_END_EXCLUSIVE')
    triggers = sampler.period_index(all_triggers,'RAIN_START','RAIN_END_EXCLUSIVE')
    occ = {u.SPATIAL_UNIT_ID:sampler.relevant(u,occurrence) for u in units}
    rain = {u.SPATIAL_UNIT_ID:sampler.relevant(u,triggers) for u in units}
    linker = sampler.Linker(history,registry,config)
    rng = np.random.default_rng(SEED)
    used = set(zip(kept.SPATIAL_UNIT_ID,kept.END_DATE))
    parents = a.set_index('A_REFERENCE_ID')
    proposals = Counter()
    replacement = []
    audits = []
    print(f'C 격자 연결 구역 교체: {len(targets)}개 · 경사/격자 확보 후보 {len(pool):,}구역',flush=True)
    for number, old in enumerate(targets.to_dict('records'),1):
        parent = parents.loc[old['A_REFERENCE_ID']]
        chosen = None
        for index in rng.permutation(len(units)):
            u = units[index]
            if u.SPATIAL_UNIT_ID == parent.SPATIAL_UNIT_ID:
                proposals['SAME_A_UNIT'] += 1; continue
            if (u.SPATIAL_UNIT_ID,parent.END_DATE) in used:
                proposals['DUPLICATE_C_UNIT_DATE'] += 1; continue
            pct = coverage.get((parent.END_DATE.year,u.SPATIAL_UNIT_ID),np.nan)
            if not np.isfinite(pct) or pct < 99.9:
                proposals['LANDCOVER_YEAR_COVERAGE'] += 1; continue
            if sampler.overlaps(parent.RAIN_START,parent.RAIN_END_EXCLUSIVE,occ[u.SPATIAL_UNIT_ID]):
                proposals['KNOWN_OCCURRENCE_OVERLAP'] += 1; continue
            if sampler.overlaps(parent.RAIN_START,parent.RAIN_END_EXCLUSIVE,rain[u.SPATIAL_UNIT_ID]):
                proposals['KNOWN_TRIGGER_EVENT_OVERLAP'] += 1; continue
            site = linker.interval(u,parent.RAIN_START,parent.RAIN_END_EXCLUSIVE)
            if site is None:
                proposals['STATION_HISTORY_REVIEW'] += 1; continue
            chosen = (u,site,pct)
            break
        if chosen is None:
            raise RuntimeError(f'No eligible replacement for {old["A_REFERENCE_ID"]}; no forced match')
        u,site,pct = chosen
        row = dict(old)
        # Clear all old rain features before any values from the new location are attached.
        for key in row:
            if key.startswith('FUSION_') or key in ['ABSENT_GRID_HOURS','INVALID_GRID_HOURS']:
                row[key] = None
        row.update(SPATIAL_UNIT_ID=u.SPATIAL_UNIT_ID,ADMIN_LEVEL=u.ADMIN_LEVEL,ADMIN_CODE=u.ADMIN_CODE,
                   ADMIN_NAME=u.FULL_NAME if pd.notna(u.FULL_NAME) else u.ADMIN_NAME,EMD_CODE=u.EMD_CODE,
                   STN=int(site.STN),STATION_LAT=float(site.LAT),STATION_LON=float(site.LON),NETWORK=site.NETWORK,
                   GRID_COUNT=int(grid_count[u.SPATIAL_UNIT_ID]),NO_GRID=False,
                   C_ESTIMATED_COVERAGE_PERCENT=float(pct),C_COVERAGE_PASS=True,
                   LANDCOVER_COVERAGE_BASIS='ACTUAL_FOOTPRINT_COVERAGE_08_14',
                   RAIN_STATUS='PENDING_REPLACEMENT_FUSION_CACHE_EXTRACTION',SLOPE_STATUS='AVAILABLE_COVERAGE_GE_90_PERCENT',
                   BOUNDARY_TIME_VALIDITY=u.BOUNDARY_TIME_VALIDITY,REBUILD_METHOD='REPLACE_NO_GRID_C_RANDOM_ELIGIBLE_ADMIN',
                   FUSION_RAIN_STATUS='PENDING_CACHE_EXTRACTION')
        row['PREVIOUS_C_SPATIAL_UNIT_ID'] = old['SPATIAL_UNIT_ID']
        row['LOCATION_REPLACED'] = True
        used.add((u.SPATIAL_UNIT_ID,parent.END_DATE))
        replacement.append(row)
        audits.append(dict(A_REFERENCE_ID=old['A_REFERENCE_ID'],OLD_SPATIAL_UNIT_ID=old['SPATIAL_UNIT_ID'],
                           OLD_ADMIN_NAME=old['ADMIN_NAME'],NEW_SPATIAL_UNIT_ID=u.SPATIAL_UNIT_ID,NEW_ADMIN_NAME=row['ADMIN_NAME'],
                           END_DATE=parent.END_DATE,STN=int(site.STN),GRID_COUNT=row['GRID_COUNT'],
                           LANDCOVER_COVERAGE_PERCENT=float(pct),STATUS='REPLACED_KNOWN_OVERLAPS_EXCLUDED'))
        if number%10==0 or number==len(targets):
            print(f'C 재추출 {number}/{len(targets)}',flush=True)
            (LOG/'progress.json').write_text(json.dumps(dict(status='SAMPLING',done=number,total=len(targets))),encoding='utf-8')
    new = pd.DataFrame(replacement)
    wanted = sorted(new.SPATIAL_UNIT_ID.unique())
    links = grid_links[grid_links.SPATIAL_UNIT_ID.isin(wanted)].copy()
    links.to_parquet(OUT/'replacement_admin_grid_links.parquet',index=False)
    # Full-area raw files already hold every grid; check them before proposing downloads.
    required = {}
    for r in new.itertuples():
        for d in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE):
            required.setdefault(d,set()).add(r.SPATIAL_UNIT_ID)
    missing = []
    for day in sorted(required):
        for hour in range(24):
            end = day+pd.Timedelta(hours=hour+1)
            path = decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
            if not path.exists():missing.append(dict(TIME_END=end,RAW_PATH=str(path)))
    (LOG/'missing_raw_hours.json').write_text(json.dumps(missing,ensure_ascii=False,default=str,indent=2),encoding='utf-8')
    source_stamps = {}
    if not missing:
        grids = np.sort(links.GRID_ID.unique())
        ui = {u:i for i,u in enumerate(wanted)}
        matrix = csr_matrix((np.ones(len(links)),(links.SPATIAL_UNIT_ID.map(ui).to_numpy(),np.searchsorted(grids,links.GRID_ID))),shape=(len(wanted),len(grids)))
        counts = np.asarray(matrix.sum(axis=1)).ravel()
        daily = {}
        for number,(day,needed) in enumerate(sorted(required.items()),1):
            vals = []
            for hour in range(24):
                end = day+pd.Timedelta(hours=hour+1)
                path = decoder.RAW/end.strftime('%Y/%m')/f'rn_60m_{end:%Y%m%d%H%M}.nc.gz'
                source_stamps[str(path)] = stamp(path)
                vals.append(decoder.decode(gzip.decompress(path.read_bytes()))[grids])
            values = np.stack(vals,axis=1)
            valid = np.isfinite(values)&(values>=0)
            sums = matrix@np.where(valid,values,0)
            nv = matrix@valid.astype(float)
            for unit in needed:
                i = ui[unit]
                partial = np.divide(sums[i],nv[i],out=np.full(24,np.nan),where=nv[i]>0)
                strict = np.where(nv[i]==counts[i],partial,np.nan)
                daily[(unit,day)] = (strict,partial,int((counts[i]-nv[i]).sum()))
            if number%20==0 or number==len(required):
                print(f'C 교체 구역 강우 캐시 계산 {number}/{len(required)}일',flush=True)
                (LOG/'progress.json').write_text(json.dumps(dict(status='EXTRACTING_CACHED_RAIN',done=number,total=len(required))),encoding='utf-8')
        for idx,r in new.iterrows():
            days=[daily[(r.SPATIAL_UNIT_ID,d)] for d in pd.date_range(r.END_DATE-pd.Timedelta(days=14),r.END_DATE)]
            strict=np.concatenate([d[0] for d in days]);partial=np.concatenate([d[1] for d in days])
            for n in range(1,16):
                x=strict[-n*24:];y=partial[-n*24:];nv=int(np.isfinite(x).sum())
                fields={f'FUSION_{n:02d}D_MM':float(x.sum()) if nv==n*24 else np.nan,
                        f'FUSION_{n:02d}D_OBSERVED_SUM_MM':float(np.nansum(x)) if nv else np.nan,
                        f'FUSION_{n:02d}D_VALID_HOURS':nv,f'FUSION_{n:02d}D_MISSING_HOURS':n*24-nv,
                        f'FUSION_{n:02d}D_PARTIAL_GRID_OBSERVED_SUM_MM':float(np.nansum(y)) if np.isfinite(y).any() else np.nan}
                for key,val in fields.items():new.at[idx,key]=val
            complete=np.isfinite(strict).all()
            new.at[idx,'ABSENT_GRID_HOURS']=0
            new.at[idx,'INVALID_GRID_HOURS']=sum(d[2] for d in days)
            new.at[idx,'FUSION_RAIN_STATUS']='COMPLETE_360_HOURS' if complete else 'REVIEW_INCOMPLETE_360_HOURS'
            new.at[idx,'RAIN_STATUS']='FUSION_CACHE_EXTRACTED_SEPARATE_FROM_STATION_RAIN'
            new.at[idx,'VALUE_PROVENANCE']='RAW_NETCDF_DATA_SCALE_APPLIED;CANONICAL_PARQUET_UNIT_REPAIR_PENDING'
    kept['LOCATION_REPLACED']=False
    kept['PREVIOUS_C_SPATIAL_UNIT_ID']=None
    combined=pd.concat([kept,new],ignore_index=True).sort_values('A_REFERENCE_ID').reset_index(drop=True)
    assert len(combined)==len(old_c) and combined.A_REFERENCE_ID.is_unique
    assert set(combined.A_REFERENCE_ID)==set(a.A_REFERENCE_ID)
    assert not combined.duplicated(['SPATIAL_UNIT_ID','END_DATE']).any()
    expected=parents.loc[combined.A_REFERENCE_ID]
    for field in ['RAIN_START','RAIN_END_EXCLUSIVE','END_DATE']:
        assert combined[field].tolist()==expected[field].tolist()
    assert combined.SPATIAL_UNIT_ID.ne(expected.SPATIAL_UNIT_ID.to_numpy()).all()
    for row in new.itertuples():
        u=next(u for u in units if u.SPATIAL_UNIT_ID==row.SPATIAL_UNIT_ID)
        assert not sampler.overlaps(row.RAIN_START,row.RAIN_END_EXCLUSIVE,sampler.relevant(u,occurrence))
        assert not sampler.overlaps(row.RAIN_START,row.RAIN_END_EXCLUSIVE,sampler.relevant(u,triggers))
    pd.testing.assert_frame_equal(old_c.loc[old_c.A_REFERENCE_ID.isin(kept.A_REFERENCE_ID)].sort_values('A_REFERENCE_ID').reset_index(drop=True),combined.loc[~combined.LOCATION_REPLACED,old_c.columns].reset_index(drop=True),check_dtype=False)
    assert signatures=={str(p):stamp(p) for p in paths}
    assert all(stamp(Path(p))==sig for p,sig in source_stamps.items())
    combined.to_parquet(OUT/'c_event_selected.parquet',index=False)
    pd.DataFrame(audits).to_parquet(OUT/'replacement_audit.parquet',index=False)
    report=dict(status='COMPLETE' if not missing else 'SELECTION_COMPLETE_RAIN_PENDING',seed=SEED,
                c_samples=len(combined),replaced=len(new),unchanged=len(kept),candidate_units=len(pool),
                complete_15d=int(combined.FUSION_15D_MISSING_HOURS.eq(0).sum()),
                incomplete_15d=int(combined.FUSION_15D_MISSING_HOURS.ne(0).sum()),
                cached_hour_files=len(source_stamps),missing_raw_hours=len(missing),api_requests=0,
                known_occurrence_and_trigger_overlap=0,unresolved_inventory_location_records=int((~inventory.ADMIN_LEVEL.isin(['RI','EMD'])).sum()),
                rejected_proposals=dict(proposals),inputs=signatures,raw_sources=source_stamps,upstream_unchanged=True,elapsed_seconds=time.time()-started)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ['inputs','raw_sources']},ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>C 격자 없는 구역 교체</title><style>body{font:15px/1.7 Malgun Gothic;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>C · 강우 격자 없는 67개 교체</h1>'
    page+=f'<p>C {len(combined):,}개 유지 · 67개 구역 교체 · 기존 1,545개 유지 · 15일 강우 완전 {report["complete_15d"]:,}개 / 검토 {report["incomplete_15d"]:,}개.</p>'
    page+='<p>A 이벤트 구간·종료일 고정. 강우 격자 중심점 1개 이상, 유효 경사 면적 90% 이상, 해당 연도 실제 피복 확보율 99.9% 이상인 구역에서 무작위 재추출했습니다. 같은 A 구역, 중복 C 구역·종료일, 전체 위치확인 발생 기록 및 기존 연결 트리거와 시간 중첩되는 구역을 제외했습니다. 위치 미확정 발생 기록은 구역별 검사에 적용할 수 없으므로 미포착은 미발생 확정이 아닙니다.</p>'
    page+=f'<p>새 격자의 시간강우는 기존 전국 NetCDF 원응답 {len(source_stamps):,}시간에서 단위를 적용해 추출했습니다. API 요청 0회, 원응답 부족 {len(missing)}시간. 원본과 기존 Parquet은 변경하지 않았습니다.</p>'
    page+='<p>후속 C 입력은 이 폴더의 c_event_selected.parquet을 사용합니다. A/B는 13/01 파일을 유지합니다. 기존 C의 강우·경사·피복 연결을 새 위치에 그대로 복사하지 않았으며, 경사·피복 수치 피처의 후속 통합은 새 SPATIAL_UNIT_ID와 연도로 수행해야 합니다. WAMIS 매칭은 아직 보류입니다.</p><p><a href="c_event_selected.parquet">새 C 표본</a> · <a href="replacement_audit.parquet">67개 변경표</a> · <a href="replacement_admin_grid_links.parquet">새 구역 격자 연결</a></p>'
    page+=pd.DataFrame(audits)[['OLD_ADMIN_NAME','NEW_ADMIN_NAME','END_DATE','STN','GRID_COUNT']].to_html(index=False,border=0)+'</html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','raw_sources','rejected_proposals']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
