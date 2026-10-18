"""Select matched ABC triples using the approved index-coverage criterion."""
import argparse
import json
from pathlib import Path
import pandas as pd

V3=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--criterion',choices=['all','c'])
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    paths=[V3/'output/06/full/05_selection/occurrence_selected.parquet',
           V3/'output/06/full/05_selection/b_event_selected.parquet',
           V3/'output/07/full/c_event_samples.parquet',
           V3/'output/08/full/14_admin_coverage/admin_tile_coverage_estimate.parquet']
    signatures=[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    frames=[pd.read_parquet(p) for p in paths[:3]]
    coverage=pd.read_parquet(paths[3])
    decision=None
    for kind,frame in zip('ABC',frames):
        assert frame.A_REFERENCE_ID.is_unique
        data=frame[['A_REFERENCE_ID','SPATIAL_UNIT_ID','END_DATE']].copy()
        data['REFERENCE_YEAR']=pd.to_datetime(data.END_DATE).dt.year
        data=data.merge(coverage[['REFERENCE_YEAR','SPATIAL_UNIT_ID','ESTIMATED_COVERAGE_PERCENT']],
                        on=['REFERENCE_YEAR','SPATIAL_UNIT_ID'],how='left',validate='many_to_one')
        assert data.ESTIMATED_COVERAGE_PERCENT.notna().all()
        data=data[['A_REFERENCE_ID','ESTIMATED_COVERAGE_PERCENT']].rename(
            columns={'ESTIMATED_COVERAGE_PERCENT':f'{kind}_ESTIMATED_COVERAGE_PERCENT'})
        decision=data if decision is None else decision.merge(data,on='A_REFERENCE_ID',validate='one_to_one')
    assert len(decision)==len(frames[0])==len(frames[1])==len(frames[2])
    for k in 'ABC':decision[f'{k}_COVERAGE_PASS']=decision[f'{k}_ESTIMATED_COVERAGE_PERCENT'].ge(99.9)
    decision['ALL_COVERAGE_PASS']=decision[[f'{k}_COVERAGE_PASS' for k in 'ABC']].all(axis=1)
    print(json.dumps(dict(total=len(decision),c_only=int(decision.C_COVERAGE_PASS.sum()),
                         all_abc=int(decision.ALL_COVERAGE_PASS.sum())),ensure_ascii=False))
    if not args.execute:return
    if args.criterion is None:parser.error('--execute requires --criterion all or c')
    out=V3/'output/08/full/15_abc_selection'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Selection exists; no overwrite')
    decision['USE_FOR_ANALYSIS']=decision.ALL_COVERAGE_PASS if args.criterion=='all' else decision.C_COVERAGE_PASS
    decision['DECISION_REASON']=decision.USE_FOR_ANALYSIS.map({True:'INCLUDED',False:'EXCLUDED_INDEX_COVERAGE'})
    ids=set(decision.loc[decision.USE_FOR_ANALYSIS,'A_REFERENCE_ID'])
    selected=[f[f.A_REFERENCE_ID.isin(ids)].copy() for f in frames]
    assert all(len(f)==len(ids) and set(f.A_REFERENCE_ID)==ids for f in selected)
    assert signatures==[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    out.mkdir(parents=True)
    for k,f,name in zip('ABC',selected,['occurrence_selected.parquet','b_event_selected.parquet','c_event_selected.parquet']):
        qc=decision[['A_REFERENCE_ID',f'{k}_ESTIMATED_COVERAGE_PERCENT',f'{k}_COVERAGE_PASS']]
        f=f.merge(qc,on='A_REFERENCE_ID',validate='one_to_one')
        f['LANDCOVER_COVERAGE_BASIS']='INDEX_BASED_ESTIMATE'
        f.to_parquet(out/name,index=False)
    decision.to_parquet(out/'abc_selection.parquet',index=False)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>ABC 피복 선택</title><style>body{font:15px/1.8 Malgun Gothic;margin:35px}</style><h1>ABC 피복 분석 표본 선택</h1>'
    page+=f'<p>A·B·C 각각 <strong>{len(ids):,}개</strong>, 합계 {3*len(ids):,}개. 이전 5,866묶음 중 {len(decision)-len(ids):,}묶음 제외. A_REFERENCE_ID로 1:1:1 연결을 유지합니다.</p>'
    page+=f'<p>선택 기준: {"ABC 모두" if args.criterion=="all" else "C"} 도엽 인덱스 면적 확보율 ≥99.9%. 실제 SHP 피복 면적 및 강우 결측 검증 완료를 의미하지 않습니다. 이전 원본 표본은 유지했습니다.</p>'
    page+='<ul><li><a href="occurrence_selected.parquet">A 선택 표본</a></li><li><a href="b_event_selected.parquet">B 선택 표본</a></li><li><a href="c_event_selected.parquet">C 선택 표본</a></li><li><a href="abc_selection.parquet">전체 묶음별 선택·제외 및 확보율</a></li></ul></html>'
    (out/'summary.html').write_text(page,encoding='utf-8')
    log=V3/'log/08/full/15_abc_selection';log.mkdir(parents=True,exist_ok=True)
    report=dict(status='COMPLETE',criterion=args.criterion,threshold_percent=99.9,
                selected_triples=len(ids),excluded_triples=len(decision)-len(ids),upstream_unchanged=True,
                coverage_basis='INDEX_BASED_ESTIMATE',inputs=[str(p) for p in paths])
    (log/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
