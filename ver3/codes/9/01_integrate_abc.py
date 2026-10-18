"""Integrate selected ABC samples with current rain and available spatial features."""
import importlib.util
import json
from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd

V3=Path(__file__).resolve().parents[2]
OUT=V3/'output/09/full/01_integration'
LOG=V3/'log/09/full/01_integration'

def render_summary(table):
    original_paths={'A':V3/'output/06/full/occurrence_reference.parquet',
                    'B':V3/'output/06/full/b_event_samples.parquet',
                    'C':V3/'output/07/full/c_event_samples.parquet'}
    original_counts={k:len(pd.read_parquet(p,columns=['A_REFERENCE_ID'])) for k,p in original_paths.items()}
    rain_ok=table.RAIN_15D_MISSING_HOURS.eq(0)
    slope_ok=table.SLOPE_STATUS.eq('COMPUTED') & table.SLOPE_AREA_COVERAGE_FRACTION.ge(.9)
    checked_ok=rain_ok & slope_ok
    records=[]
    for kind in 'ABC':
        mask=table.SAMPLE_TYPE.eq(kind);total=int(mask.sum())
        good=int((mask & checked_ok).sum())
        records.append(dict(표본=kind,최초전체=original_counts[kind],현재선택=total,
            선별제외=original_counts[kind]-total,현재항목_문제없음=good,
            보완필요=total-good,강우_문제없음=int((mask & rain_ok).sum()),
            경사_문제없음=int((mask & slope_ok).sum())))
    triples=table[['TRIPLET_ID']].assign(PASS=checked_ok).groupby('TRIPLET_ID').PASS.all()
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>09 ABC 표본 점검</title><style>body{font:15px/1.7 Malgun Gothic;margin:32px;color:#24364a;background:#f6f8fb}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:white;padding:22px;border:1px solid #dce3eb;border-radius:12px;min-width:220px}.big{font-size:29px;font-weight:bold;color:#176447}small{color:#657588}table{background:white;border-collapse:collapse;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #dce3eb}.note{background:#fff5de;padding:15px;border-radius:10px}</style><h1>09 · ABC 표본 점검</h1>'
    page+='<p>분모는 각 표본의 최초 생성 파일 전체 수입니다. 이후 선별하여 현재 선택한 수는 ABC 각각 5,673개입니다.</p><div class="cards">'
    for row in records:
        good=row['현재항목_문제없음'];total=row['최초전체']
        page+=f'<section class="card"><h2>{row["표본"]} 표본</h2><div class="big">{good:,} / {total:,}개</div><p>현재 선택 중 강우·경사 통과 / 최초 전체 · {good/total:.1%}</p><small>현재 선택 {row["현재선택"]:,}개<br>선별 제외 {row["선별제외"]:,}개 · 선택 중 보완 필요 {row["보완필요"]:,}개</small></section>'
    page+='</div><p><strong>ABC 모두 강우·경사 통과: '+f'{int(triples.sum()):,} / {len(triples):,}묶음</strong></p>'
    page+='<p>현재 항목의 “문제없음” = 종료일 포함 이전 15일 강우 360시간에 결측이 없고, 경사가 계산되어 유효 면적 확보율이 기존 기준 90% 이상인 표본입니다. 강우와 경사를 각각 통과한 수는 서로 겹칠 수 있습니다.</p>'
    page+='<p>통과 수는 현재 선택된 표본에서 계산했습니다. 이미 선별 제외된 표본은 이번 QC에 포함하지 않았으며, “선별 제외”와 “선택 중 보완 필요”는 별개입니다.</p>'
    display=pd.DataFrame(records).rename(columns={'최초전체':'최초 전체','현재선택':'현재 선택','선별제외':'선별 제외','현재항목_문제없음':'선택 중 강우·경사 통과','보완필요':'선택 중 보완 필요','강우_문제없음':'선택 중 강우 통과','경사_문제없음':'선택 중 경사 통과'})
    page+=display.to_html(index=False,border=0)
    page+='<p class="note"><strong>피복까지 모든 항목 검증 완료: 현재 0개.</strong> 피복 종류별 면적 비율이 아직 없고, 실제 폴리곤 확보율·촬영시점도 확인 전입니다. 위 통과 수는 학습자료 최종 사용 가능 수가 아닙니다.</p>'
    page+='<p><a href="abc_integrated.parquet">ABC 통합 기준표</a> · <a href="pending_slope_units.parquet">경사 보완 구역</a></p><p>표본을 추가 제외하거나 결측을 0으로 채우지 않았습니다. B·C는 발생기록 미포착 표본입니다.</p></html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    return dict(samples=records,abc_checked_complete_triplets=int(triples.sum()),total_triplets=len(triples),
                criterion='15D_NO_MISSING_AND_COMPUTED_SLOPE_COVERAGE_GE_90_PERCENT',
                all_features_verified=0,landcover='PENDING')

def stats(values,prefix):
    values=np.asarray(values,dtype=float)
    n=int(np.isfinite(values).sum())
    return {prefix+'_OBSERVED_SUM_MM':float(np.nansum(values)) if n else np.nan,
            prefix+'_COMPLETE_SUM_MM':float(values.sum()) if n==len(values) else np.nan,
            prefix+'_VALID_HOURS':n,prefix+'_MISSING_HOURS':len(values)-n,
            prefix+'_COVERAGE':n/len(values)}

def main():
    if '--report-only' in sys.argv:
        source=OUT/'abc_integrated.parquet';before=(source.stat().st_size,source.stat().st_mtime_ns)
        report=render_summary(pd.read_parquet(source))
        assert before==(source.stat().st_size,source.stat().st_mtime_ns)
        (LOG/'summary_qc.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False));return
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Existing integration; no automatic overwrite')
    LOG.mkdir(parents=True,exist_ok=True);started=time.time()
    selection=V3/'output/08/full/15_abc_selection'
    paths=[selection/n for n in ['occurrence_selected.parquet','b_event_selected.parquet','c_event_selected.parquet']]
    paths += [V3/'config/rainfall.json',V3/'base/stations/station_location_history.xlsx',
              V3/'output/05/full/admin_slope_features.parquet']
    stamps=[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    a,b,c=[pd.read_parquet(p) for p in paths[:3]]
    ids=set(a.A_REFERENCE_ID)
    assert len(ids)==5673 and all(len(f)==5673 and f.A_REFERENCE_ID.is_unique and set(f.A_REFERENCE_ID)==ids for f in [a,b,c])
    coords=b[['A_REFERENCE_ID','STATION_LAT','STATION_LON']]
    a=a.merge(coords,on='A_REFERENCE_ID',validate='one_to_one')
    frames=[]
    for typ,f in zip('ABC',[a,b,c]):
        f=f.copy();f['SOURCE_SAMPLE_TYPE']=f.SAMPLE_TYPE;f['SAMPLE_TYPE']=typ
        f['SAMPLE_ID']=f.A_REFERENCE_ID if typ=='A' else f[f'{typ}_SAMPLE_ID']
        legacy_qc=['RAIN_STATUS','SLOPE_STATUS','EVENT_RAIN_MM','RAIN_01_15D_COMPLETE',
                   'RAIN_QC_REVIEW','CUMULATIVE_STATUS','A_REFERENCE_STATUS','MATCHED_TRIPLET_STATUS']
        legacy_qc += [x for x in f if x.startswith('VALID_DAYS_')]
        for col in legacy_qc:
            if col in f:f=f.rename(columns={col:'SOURCE_'+col})
        f['LABEL']=1 if typ=='A' else 0
        f['LABEL_CERTAINTY']='REPORTED_OCCURRENCE_CANDIDATE' if typ=='A' else 'UNRESOLVED_UNREPORTED_IS_NOT_CONFIRMED_ABSENCE'
        f['REFERENCE_YEAR']=pd.to_datetime(f.END_DATE).dt.year
        f['TRIPLET_ID']=f.A_REFERENCE_ID
        frames.append(f)
    table=pd.concat(frames,ignore_index=True)
    assert table.SAMPLE_ID.is_unique
    assert (a.set_index('A_REFERENCE_ID').END_DATE.sort_index()==c.set_index('A_REFERENCE_ID').END_DATE.sort_index()).all()
    assert (a.set_index('A_REFERENCE_ID').SPATIAL_UNIT_ID.sort_index()==b.set_index('A_REFERENCE_ID').SPATIAL_UNIT_ID.sort_index()).all()
    config=json.loads(paths[3].read_text(encoding='utf-8'))
    history=pd.read_excel(paths[4],sheet_name='location_history')
    spec=importlib.util.spec_from_file_location('integration_rain_loader',V3/'codes/6/04_build_hourly_comparison.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    dates=set()
    for day in pd.to_datetime(table.END_DATE).unique():
        dates.update(pd.date_range(pd.Timestamp(day).normalize()-pd.Timedelta(days=14),pd.Timestamp(day).normalize()))
    print(f'RAIN_DATAS 읽기: {len(dates):,}일, {table.STN.nunique():,} STN',flush=True)
    (LOG/'progress.json').write_text(json.dumps(dict(status='READING_RAIN_DATAS',days=len(dates))),encoding='utf-8')
    rain,sources=loader.current_rain(config,sorted(table.STN.unique().astype(int)),dates,history)
    empty=pd.DataFrame({'VALUE':pd.Series(dtype=float),'USABLE':pd.Series(dtype=bool),
        'LAT':pd.Series(dtype=float),'LON':pd.Series(dtype=float),'SOURCE':pd.Series(dtype=str)},index=pd.DatetimeIndex([]))
    cache={};features=[]
    for i,r in enumerate(table.itertuples(),1):
        end=pd.Timestamp(r.END_DATE).normalize()+pd.Timedelta(days=1)
        start=end-pd.Timedelta(days=15)
        key=(int(r.STN),float(r.STATION_LAT),float(r.STATION_LON),end)
        if key not in cache:
            values,q=loader.sequence(rain.get(int(r.STN),empty),start,end,r.STATION_LAT,r.STATION_LON)
            assert len(values)==360
            f={}
            for n in range(1,16):
                f.update(stats([np.nan if v is None else v for v in values[-24*n:]],f'RAIN_{n:02d}D'))
                f[f'RAIN_{n:02d}D_MM']=f[f'RAIN_{n:02d}D_COMPLETE_SUM_MM']
            f.update(RAIN_NO_VALID_OBSERVATION_HOURS=sum(x==2 for x in q),
                     RAIN_LOCATION_HISTORY_UNUSABLE_HOURS=sum(x==3 for x in q),
                     RAIN_COORDINATE_MISMATCH_HOURS=sum(x==4 for x in q),
                     RAIN_STATUS='COMPLETE_15D' if f['RAIN_15D_MISSING_HOURS']==0 else 'REVIEW_MISSING',
                     RAIN_TIME_RULE='END_DATE_INCLUDED_PREVIOUS_1_TO_15_CALENDAR_DAYS_KST')
            cache[key]=f
        features.append(cache[key])
        if i%1000==0 or i==len(table):
            print(f'누적강우 {i:,}/{len(table):,}',flush=True)
            (LOG/'progress.json').write_text(json.dumps(dict(status='CALCULATING_RAIN',done=i,total=len(table))),encoding='utf-8')
    rain_features=pd.DataFrame(features)
    table=table.drop(columns=[x for x in rain_features if x in table])
    table=pd.concat([table.reset_index(drop=True),rain_features],axis=1)
    slope=pd.read_parquet(paths[5]);assert slope.SPATIAL_UNIT_ID.is_unique
    fields=['SPATIAL_UNIT_ID','VALID_AREA_M2','AREA_COVERAGE_FRACTION','MEAN_SLOPE_DEG','STATUS']
    fields += [f'GE_{n}DEG_{s}' for n in [20,30,40] for s in ['AREA_M2','FRACTION','PERCENT']]
    slope=slope[fields].rename(columns={x:'SLOPE_'+x for x in fields if x!='SPATIAL_UNIT_ID'})
    table=table.merge(slope,on='SPATIAL_UNIT_ID',how='left',validate='many_to_one')
    table['SLOPE_STATUS']=table.SLOPE_STATUS.fillna('PENDING_NO_ADMIN_FEATURES')
    table['LANDCOVER_FEATURE_STATUS']='PENDING_CLASS_AREA_RATIOS'
    table['LANDCOVER_TEMPORAL_STATUS']='REVIEW_SOURCE_IMAGE_DATE'
    coverage=V3/'output/08/full/14_admin_coverage/admin_landcover_coverage.parquet'
    if coverage.exists():
        cover=pd.read_parquet(coverage)[['REFERENCE_YEAR','SPATIAL_UNIT_ID','COVERAGE_PERCENT','STATUS']]
        cover=cover.rename(columns={'COVERAGE_PERCENT':'LANDCOVER_ACTUAL_COVERAGE_PERCENT','STATUS':'LANDCOVER_ACTUAL_COVERAGE_STATUS'})
        table=table.merge(cover,on=['REFERENCE_YEAR','SPATIAL_UNIT_ID'],how='left',validate='many_to_one')
    else:
        table['LANDCOVER_ACTUAL_COVERAGE_PERCENT']=np.nan
        table['LANDCOVER_ACTUAL_COVERAGE_STATUS']='PENDING_POLYGON_COVERAGE'
    # Leakage groups are retained, not used as predictors or randomly split here.
    amap=a.set_index('A_REFERENCE_ID').RAIN_EVENT_ID
    table['A_TRIGGER_GROUP_ID']=table.A_REFERENCE_ID.map(amap)
    table['TRAINING_READY']=False
    assert len(table)==17019 and table.groupby('TRIPLET_ID').SAMPLE_TYPE.nunique().eq(3).all()
    for n in range(1,16):
        assert (table[f'RAIN_{n:02d}D_VALID_HOURS']+table[f'RAIN_{n:02d}D_MISSING_HOURS']).eq(n*24).all()
        assert table.loc[table[f'RAIN_{n:02d}D_MISSING_HOURS'].gt(0),f'RAIN_{n:02d}D_MM'].isna().all()
    assert stamps==[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    for s in sources:
        st=Path(s['path']).stat();assert (st.st_size,st.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
    OUT.mkdir(parents=True)
    table.to_parquet(OUT/'abc_integrated.parquet',index=False)
    pending=table.loc[table.SLOPE_STATUS.eq('PENDING_NO_ADMIN_FEATURES'),['SPATIAL_UNIT_ID','ADMIN_NAME','ADMIN_LEVEL']].drop_duplicates()
    pending.to_parquet(OUT/'pending_slope_units.parquet',index=False)
    summary=table.groupby('SAMPLE_TYPE').agg(SAMPLES=('SAMPLE_ID','size'),RAIN_15D_COMPLETE=('RAIN_15D_MISSING_HOURS',lambda s:int(s.eq(0).sum())),SLOPE_LINKED=('SLOPE_MEAN_SLOPE_DEG',lambda s:int(s.notna().sum())))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>09 데이터 통합</title><style>body{font:15px/1.7 Malgun Gothic;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>09 · 데이터 통합</h1><p>ABC 5,673묶음, 17,019행. 최신 RAIN_DATAS 강우를 로드하여 종료일 포함 이전 1~15일을 재계산하고 행정구역 경사를 연결했습니다.</p>'
    page+=summary.to_html()+'<p>강우 부분합과 완전합을 구분했습니다. 결측은 0으로 채우지 않았고 위치 이력·좌표 불일치 시간도 표시했습니다.</p>'
    page+=f'<p>경사 미계산 구역 {len(pending):,}개. 토지피복 분류별 면적 비율은 계산 전이며, 실제 폴리곤 확보율·촬영시점 적합성도 별도 확인 대상입니다. 따라서 학습 준비 완료로 표시하지 않았습니다.</p>'
    page+='<p><a href="abc_integrated.parquet">ABC 통합 기준표</a> · <a href="pending_slope_units.parquet">경사 추가 계산 필요 구역</a></p><p>B·C의 0 라벨은 발생기록 미포착을 뜻하며 확정 미발생이 아닙니다. 묶음·강우 이벤트 키를 보존했고 학습/시험 분할은 수행하지 않았습니다.</p></html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    render_summary(table)
    report=dict(status='COMPLETE_INTEGRATION_WITH_PENDING_FEATURES',triplets=5673,rows=len(table),
        summary=summary.to_dict('index'),pending_slope_units=len(pending),
        rainfall_source='CURRENT_RAIN_DATAS_PARQUET',source_files=sources,
        inputs=[str(p) for p in paths],upstream_unchanged=True,zero_imputation=False,
        landcover_ratios='PENDING',train_test_split='NOT_PERFORMED',elapsed_seconds=time.time()-started)
    (LOG/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOG/'progress.json').write_text(json.dumps(dict(status=report['status'],rows=len(table))),encoding='utf-8')
    print(summary.to_string(),flush=True)

if __name__=='__main__':main()
