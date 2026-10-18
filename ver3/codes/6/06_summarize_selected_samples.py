"""Summarize the approved A/B selection without duplicating rainfall or rasters."""
import html
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    folder = V3/'output/06/full/05_selection'
    paths = [folder/'occurrence_selected.parquet', folder/'b_event_selected.parquet', folder/'event_selection.parquet']
    before = [fingerprint(p) for p in paths]
    a, b, decision = [pd.read_parquet(p) for p in paths]
    assert len(a) == len(b) and a.A_REFERENCE_ID.is_unique and b.A_REFERENCE_ID.is_unique
    pair = a[['A_REFERENCE_ID', 'SPATIAL_UNIT_ID', 'STN']].merge(
        b[['A_REFERENCE_ID', 'SPATIAL_UNIT_ID', 'STN']], on='A_REFERENCE_ID', validate='one_to_one', suffixes=('_A','_B'))
    assert pair.SPATIAL_UNIT_ID_A.eq(pair.SPATIAL_UNIT_ID_B).all() and pair.STN_A.eq(pair.STN_B).all()
    assert not b.TRIGGER_EVENT_OVERLAP.any() and not b.CUMULATIVE_TRIGGER_OVERLAP.any()
    selected = decision.loc[decision.USE_FOR_ANALYSIS, ['STN', 'RAIN_EVENT_ID']]
    assert set(map(tuple, selected.values)) == set(map(tuple, a[['STN','RAIN_EVENT_ID']].drop_duplicates().values))
    counts = dict(a_samples=len(a), b_samples=len(b), pairs=len(pair), total_samples=len(a)+len(b),
        a_unique_stn_events=len(a[['STN','RAIN_EVENT_ID']].drop_duplicates()),
        b_unique_stn_events=len(b[['STN','RAIN_EVENT_ID']].drop_duplicates()),
        spatial_units=int(a.SPATIAL_UNIT_ID.nunique()), stations=int(a.STN.nunique()))
    table = a.groupby(['SPATIAL_UNIT_ID','ADMIN_NAME'], dropna=False).agg(
        A=('A_REFERENCE_ID','size'), STN_COUNT=('STN','nunique'),
        A_EVENTS=('RAIN_EVENT_ID','nunique')).reset_index()
    table = table.merge(b.groupby('SPATIAL_UNIT_ID').size().rename('B'),on='SPATIAL_UNIT_ID',validate='one_to_one')
    assert table.A.eq(table.B).all()
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
        [r.ADMIN_NAME,r.SPATIAL_UNIT_ID,r.A,r.B,r.STN_COUNT,r.A_EVENTS])+'</tr>' for r in table.itertuples())
    exclusions = decision.loc[~decision.USE_FOR_ANALYSIS,'DECISION_REASON'].value_counts()
    reasons = {'EXCLUDED_SPATIAL_EVENT':'공간 문제 연결 이벤트',
        'EXCLUDED_A_END_UNCONFIRMED':'A 종료 미확인', 'EXCLUDED_A_MISSING':'A 이벤트 내부 결측',
        'EXCLUDED_A_MISSING_AND_END_UNCONFIRMED':'A 결측 + 종료 미확인'}
    exclude_rows = ''.join(f'<tr><td>{reasons[k]}</td><td>{v:,}</td></tr>' for k,v in exclusions.items())
    cards = ''.join(f'<div class="card"><span>{label}</span><b>{value:,}</b></div>' for label,value in
        [('A 발생 기록',len(a)),('B 미포착',len(b)),('A–B 쌍',len(pair)),('전체 표본',len(a)+len(b))])
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A·B 분석 표본 정리</title>
<style>body{margin:0;background:#f3f6fa;color:#24364a;font:15px/1.6 "Malgun Gothic",sans-serif}main{max-width:1100px;margin:30px auto;padding:0 22px}h1{font-size:26px}h2{font-size:19px;margin-top:28px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,section{background:white;border:1px solid #dce3ec;border-radius:9px;padding:16px}.card span{color:#65758a}.card b{display:block;font-size:28px}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #e4e9ef}th{background:#eef3f8;position:sticky;top:0}.muted{color:#65758a;font-size:13px}.scroll{max-height:550px;overflow:auto}input{padding:10px;width:min(500px,100%);border:1px solid #bfcbd8;border-radius:5px;margin-bottom:12px}a{color:#236b9a}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}.scroll{overflow:auto}td,th{white-space:nowrap}}</style><main>
<h1>A·B 분석 표본 정리</h1><p>행정구역–STN–강우 이벤트 단위로 A와 B를 1:1 연결 · 2026-10-08 확정 제외 기준</p>
'''+f'<div class="cards">{cards}</div><p class="muted">A 고유 STN·강우 이벤트 {counts["a_unique_stn_events"]:,}개 · B 고유 STN·강우 이벤트 {counts["b_unique_stn_events"]:,}개 · 행정구역 {counts["spatial_units"]:,}개 · STN {counts["stations"]:,}개</p>'+'''
<h2>사용할 파일</h2><section><p><a href="occurrence_selected.parquet">occurrence_selected.parquet</a> — A 산사태 발생 기록 표본</p>
<p><a href="b_event_selected.parquet">b_event_selected.parquet</a> — B 같은 행정구역·STN 좌표의 다른 강우 이벤트</p>
<p><a href="event_selection.parquet">event_selection.parquet</a> — 전체 1,556개 A 이벤트의 포함·제외 결정</p>
<p class="muted">연결 키: A_REFERENCE_ID. 행정구역: 리 우선, 없으면 읍면동. 원시 강우는 RAIN_DATAS Parquet에서 로드하고 지형은 output/05/full/admin_slope의 SPATIAL_UNIT_ID별 TIFF를 참조합니다.</p></section>
<h2>제외 기준</h2><section><table><tr><th>사유</th><th>고유 A 이벤트 수</th></tr>'''+exclude_rows+'''
</table><p class="muted">공간 문제 43개 구역에 연결된 이벤트 68개는 정상 구역과 공유하더라도 전체 제외. 추가 A 강우 검토 73개 제외 → 1,415개 유지. 행정구역 미확정 6행은 기존처럼 사용하지 않으며 원본에 남아 있습니다.</p></section>
<h2>행정구역별 표본 수</h2><input id="search" placeholder="행정구역 이름 또는 코드 검색" aria-label="행정구역 검색"><section class="scroll"><table id="units"><thead><tr><th>행정구역</th><th>공간 ID</th><th>A</th><th>B</th><th>STN 수</th><th>A 이벤트 수</th></tr></thead><tbody>'''+rows+'''
</tbody></table></section><p class="muted">B는 산사태 미포착이며 실제 미발생 확정이 아닙니다. B 이벤트 및 이전 15일 창은 알려진 같은 STN의 산사태 연결 이벤트와 겹치지 않습니다. 제외한 A도 이 겹침 금지 목록에는 유지합니다.</p>
<p class="muted">이 폴더는 선택 목록입니다. 강우 재수집·누적 피처 재계산·학습/평가 분리는 수행하지 않았습니다. 기존 시간당 강우 비교 HTML은 제외 전 스냅샷입니다.</p></main>
<script>document.getElementById('search').addEventListener('input',function(){const q=this.value.trim().toLowerCase();for(const row of document.querySelectorAll('#units tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(q);});</script></html>'''
    assert before == [fingerprint(p) for p in paths]
    target = folder/'summary.html'
    target.write_text(page, encoding='utf-8')
    write_json(V3/'log/06/full/05_selection/summary_verification.json',dict(status='PASS',**counts,
        same_unit_stn_pairing=True, overlapping_b_events=0, upstream_unchanged=True,
        admin_table_rows=len(table),inputs=before,visual_render_verified=False))
    print(counts)
    print(target)


if __name__ == '__main__':
    main()
