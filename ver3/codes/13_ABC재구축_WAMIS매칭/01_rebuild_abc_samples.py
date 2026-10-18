"""Rebuild matched ABC sets by inheriting controls for the relinked A keys."""
from pathlib import Path
from html import escape
import json
import pandas as pd

V3 = Path(__file__).resolve().parents[2]
OUT = V3 / 'output/13/full/01_abc_rebuild'
LOG = V3 / 'log/13/full/01_abc_rebuild'


def stamp(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def main():
    paths = [V3 / 'output/12/full/04_a_relink/a_relinked_selected.parquet',
             V3 / 'output/12/full/04_a_relink/a_group_membership.parquet',
             V3 / 'output/08/full/15_abc_selection/b_event_selected.parquet',
             V3 / 'output/08/full/15_abc_selection/c_event_selected.parquet',
             V3 / 'output/11/full/01_rainfall_summary/abc_fusion_rainfall.parquet']
    signatures = {str(p): stamp(p) for p in paths}
    a, members, old_b, old_c, fusion = [pd.read_parquet(p) for p in paths]
    assert a.A_REFERENCE_ID.is_unique and members.EVENT_ID.is_unique
    assert len(members[members.IN_RELINKED_SELECTED_A]) == int(a.LANDSLIDE_COUNT.sum())
    keys = set(a.A_REFERENCE_ID)
    b = old_b[old_b.A_REFERENCE_ID.isin(keys)].copy()
    c = old_c[old_c.A_REFERENCE_ID.isin(keys)].copy()
    assert all(f.A_REFERENCE_ID.is_unique and set(f.A_REFERENCE_ID) == keys for f in [a, b, c])
    parent = a.set_index('A_REFERENCE_ID')
    for controls in [b, c]:
        expected = parent.loc[controls.A_REFERENCE_ID]
        assert controls.A_RAIN_EVENT_ID.tolist() == expected.RAIN_EVENT_ID.tolist()
    expected_b = parent.loc[b.A_REFERENCE_ID]
    assert b.SPATIAL_UNIT_ID.tolist() == expected_b.SPATIAL_UNIT_ID.tolist()
    assert b.STN.tolist() == expected_b.STN.tolist()
    assert b.RAIN_EVENT_ID.ne(b.A_RAIN_EVENT_ID).all()
    assert b.A_END_DATE.tolist() == expected_b.END_DATE.tolist()
    assert not b.TRIGGER_EVENT_OVERLAP.any() and not b.CUMULATIVE_TRIGGER_OVERLAP.any()
    expected_c = parent.loc[c.A_REFERENCE_ID]
    assert c.SPATIAL_UNIT_ID.ne(c.A_SPATIAL_UNIT_ID).all()
    assert c.A_SPATIAL_UNIT_ID.tolist() == expected_c.SPATIAL_UNIT_ID.tolist()
    assert c.A_STN.tolist() == expected_c.STN.tolist()
    for field in ['RAIN_START', 'RAIN_END_EXCLUSIVE', 'END_DATE']:
        assert c[field].tolist() == expected_c[field].tolist()
    assert not c.KNOWN_OCCURRENCE_OVERLAP.any() and not c.KNOWN_TRIGGER_EVENT_OVERLAP.any()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    result = []
    files = {'A': 'occurrence_selected.parquet', 'B': 'b_event_selected.parquet', 'C': 'c_event_selected.parquet'}
    for kind, table in [('A', a), ('B', b), ('C', c)]:
        rain = fusion[fusion.SAMPLE_TYPE.eq(kind) & fusion.A_REFERENCE_ID.isin(keys)].copy()
        assert rain.A_REFERENCE_ID.is_unique and set(rain.A_REFERENCE_ID) == keys
        aligned = rain.set_index('A_REFERENCE_ID').loc[table.A_REFERENCE_ID]
        for field in ['SPATIAL_UNIT_ID', 'END_DATE', 'STN']:
            assert table[field].tolist() == aligned[field].tolist()
        assert rain.VALUE_PROVENANCE.str.startswith('RAW_NETCDF').all()
        fields = ['A_REFERENCE_ID', 'GRID_COUNT', 'ABSENT_GRID_HOURS', 'INVALID_GRID_HOURS', 'NO_GRID', 'VALUE_PROVENANCE']
        fields += [f for f in rain if f.startswith('FUSION_')]
        table = table.drop(columns=[f for f in fields if f != 'A_REFERENCE_ID' and f in table])
        table = table.merge(rain[fields], on='A_REFERENCE_ID', validate='one_to_one').sort_values('A_REFERENCE_ID').reset_index(drop=True)
        table['SAMPLE_CLASS'] = kind
        table['TARGET_REPORTED_OCCURRENCE'] = int(kind == 'A')
        table['REBUILD_METHOD'] = 'RELINKED_A_KEY_SUBSET_NO_RESAMPLING'
        table['PARENT_A_TRIGGER_RULE'] = parent.loc[table.A_REFERENCE_ID, 'TRIGGER_RULE'].to_numpy()
        table['PARENT_A_CANDIDATE_SELECTION_STATUS'] = 'ONE_TRIGGER_PER_EVENT_SELECTED'
        table['FUSION_RAIN_STATUS'] = table.FUSION_15D_MISSING_HOURS.eq(0).map({True: 'COMPLETE_360_HOURS', False: 'REVIEW_INCOMPLETE_360_HOURS'})
        table.to_parquet(OUT / files[kind], index=False)
        result.append(dict(sample=kind, previous_rows=len(a) if kind == 'A' else len(old_b if kind == 'B' else old_c),
                           rows=len(table), unique_admin_units=table.SPATIAL_UNIT_ID.nunique(),
                           complete_15d=int(table.FUSION_15D_MISSING_HOURS.eq(0).sum()),
                           incomplete_15d=int(table.FUSION_15D_MISSING_HOURS.ne(0).sum()),
                           repeated_admin_date_rows=int(table.duplicated(['SPATIAL_UNIT_ID', 'END_DATE']).sum())))
    assert signatures == {str(p): stamp(p) for p in paths}
    report = dict(status='COMPLETE', matched_triplets=len(a), total_rows=3 * len(a),
                  unique_landslide_records=int(a.LANDSLIDE_COUNT.sum()), results=result,
                  removed_old_control_rows=len(old_b)-len(b), resampled=False,
                  inherited_exclusion_flags_pass=True, upstream_unchanged=True, inputs=signatures)
    (LOG / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    table_html = pd.DataFrame(result).rename(columns={'sample':'표본','previous_rows':'입력 행 수','rows':'저장 행 수','unique_admin_units':'고유 행정구역','complete_15d':'15일 강우 완전','incomplete_15d':'15일 강우 검토','repeated_admin_date_rows':'구역·종료일 반복 행'}).to_html(index=False, border=0)
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>13 ABC 재구축</title><style>body{font:15px/1.7 Malgun Gothic;margin:32px;color:#21344a}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>새 A 기준 ABC 재구축</h1>'
    page += f'<p><strong>A·B·C 각각 {len(a):,}개 · {len(a):,}개 매칭 묶음 · 총 {3*len(a):,}행</strong></p>' + table_html
    page += '<p>A는 사건별 대표 강우 이벤트를 하나 선택한 결과입니다. B는 같은 구역·STN의 다른 이벤트, C는 A와 같은 이벤트 시간대의 다른 행정구역입니다. A_REFERENCE_ID로 1:1:1 연결합니다.</p>'
    page += '<p>기존 B/C에서 남은 A의 키만 추렸습니다. 무작위 재추출은 하지 않았고, B/C 각각 기존 5,673개 중 4,061개를 제외했습니다. 기존 발생·트리거 중첩 제외 결과를 유지하고 공간·시간 연결과 제외 플래그를 검증했습니다. 전체 발생 기록을 대상으로 중첩 검사를 새로 수행한 것은 아닙니다.</p>'
    page += '<p>원본 NetCDF 단위를 보정해 계산한 기존 융합강우 피처를 연결했습니다. 이벤트 종료일 포함 이전 1~15일이며 결측을 0으로 대체하지 않았습니다. 강우 검토 표본도 표본 수를 유지하며 상태로 구분합니다. RAIN_DATAS hourly Parquet 단위 복구는 별도 보류 상태입니다.</p>'
    page += '<p>B/C의 0 라벨은 발생 기록 미포착을 뜻하며 확실한 미발생이 아닙니다. 이전 LABEL_STATUS·RAIN_STATUS·SLOPE_STATUS 등은 원래 단계의 기록으로 보존했습니다. 새 강우 상태는 FUSION_RAIN_STATUS를 사용합니다. 경사·피복 통합 및 모델 학습은 아직 수행하지 않았습니다.</p>'
    page += '<p>서로 다른 A가 같은 B/C 구역·날짜를 공유할 수 있습니다. 표본 행 수를 독립 관측 수로 해석하지 않고, 후속 학습 분할 시 공유 공간·시간을 확인해야 합니다.</p><ul>'
    page += ''.join(f'<li><a href="{escape(name)}">{kind} · {escape(name)}</a></li>' for kind, name in files.items()) + '</ul></html>'
    (OUT / 'summary.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'inputs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
