"""Plot damage/rainfall distributions for the one-trigger-per-event A selection."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[2]
OUT = V3 / 'output/12/full/05_relinked_a_distribution'
LOG = V3 / 'log/12/full/05_relinked_a_distribution'


def main():
    paths = [V3 / 'output/12/full/04_a_relink/a_relinked_selected.parquet',
             V3 / 'output/12/full/04_a_relink/a_group_membership.parquet',
             V3 / 'output/12/full/03_damage_rainfall/damage_rainfall.parquet']
    stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    a, membership, previous = [pd.read_parquet(p) for p in paths]
    assert a.A_REFERENCE_ID.is_unique and membership.EVENT_ID.is_unique
    retained = membership[membership.IN_RELINKED_SELECTED_A]
    assert len(retained) == int(a.LANDSLIDE_COUNT.sum())
    assert previous.A_REFERENCE_ID.is_unique
    # These retained keys keep the same admin, end date and member records.
    # Reuse the already checked raw-NetCDF rainfall features without new downloads.
    indexed = previous.set_index('A_REFERENCE_ID')
    for row in a.itertuples(index=False):
        old = indexed.loc[row.A_REFERENCE_ID]
        assert old.SPATIAL_UNIT_ID == row.SPATIAL_UNIT_ID
        assert pd.Timestamp(old.END_DATE) == pd.Timestamp(row.END_DATE)
        assert set(old.AREA_EVENT_IDS) | set(old.EXCLUDED_AREA_EVENT_IDS) == set(row.LANDSLIDE_EVENT_IDS)
        assert np.isclose(old.DAMAGE_AREA_M2, row.DAMAGE_AREA_M2, equal_nan=True)
    table = previous[previous.A_REFERENCE_ID.isin(a.A_REFERENCE_ID)].copy()
    assert len(table) == len(a)
    assert table.VALUE_PROVENANCE.str.startswith('RAW_NETCDF').all()
    for n in [1, 3, 5, 7, 10, 15]:
        matched = a.set_index('A_REFERENCE_ID').loc[table.A_REFERENCE_ID, f'FUSION_{n:02d}D_MM']
        assert np.allclose(table[f'FUSION_{n:02d}D_MM'], matched, equal_nan=True)
    table['TRIGGER_RULE'] = a.set_index('A_REFERENCE_ID').loc[table.A_REFERENCE_ID, 'TRIGGER_RULE'].to_numpy()
    used = table[table.DAMAGE_AREA_M2.notna()].copy()
    report = dict(status='COMPLETE', a_samples=len(table), area_valid_samples=len(used),
                  area_excluded_samples=int(table.DAMAGE_AREA_M2.isna().sum()),
                  partial_area_samples=int(used.DAMAGE_AREA_STATUS.eq('PARTIAL_RECORDED_AREA').sum()),
                  unique_event_records=len(retained), repeated_event_assignments=0,
                  periods=[1, 3, 5, 7, 10, 15], rainfall_reused_after_key_member_date_area_validation=True,
                  original_data_unchanged=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / 'damage_rainfall.parquet', index=False)
    spec = importlib.util.spec_from_file_location('damage_plot', V3 / 'scripts/12_시각화_A재연결/03_plot_damage_rainfall.py')
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    plot.OUT = OUT
    plot.render(used, report)
    html = OUT / 'summary.html'
    page = html.read_text(encoding='utf-8')
    page = page.replace('피해면적과 강우 분포 · A 발생 표본', '피해면적과 강우 분포 · 재연결 A')
    page = page.replace('EVENT_ID는 묶음 안에서 중복 제거합니다.', '원본 사건마다 대표 강우 이벤트를 하나 선택한 뒤 행정구역–이벤트별로 합산했습니다. 사건의 여러 묶음 반복 배정은 0건입니다.')
    page = page.replace('서로 다른 묶음에 같은 사건이 연결될 수 있으므로 점들의 면적 합계를 전국 피해면적으로 해석하지 않습니다.', '표시된 합계는 이번 선별 표본의 기록된 피해면적이며, 전국 전체 피해면적을 뜻하지 않습니다.')
    page = page.replace('<p id="counts"></p>', '<p id="counts"></p><p class="note">새 A 1,612묶음 · 고유 산사태 기록 3,994건. 최대 시간강우로 선택한 이벤트는 추정 연결이며, 실제 발생 시각이 있는 기록은 해당 시각을 유지했습니다.</p>')
    page = page.replace('x:items.map(r=>r.area)', 'x:items.map(r=>Math.min(r.area,50000))')
    page = page.replace('[r.admin,r.date,r.id,r.events,r.excluded]', '[r.admin,r.date,r.id,r.events,r.excluded,r.area]')
    page = page.replace('피해면적 %{x:,.1f} m²', '실제 피해면적 %{customdata[5]:,.1f} m²')
    page = page.replace("type:log?'log':'linear',rangemode:log?undefined:'tozero'",
                        "type:log?'log':'linear',range:log?[Math.log10(Math.min(...chosen.map(r=>r.area)))-.1,Math.log10(55000)]:[0,55000],tickmode:'array',tickvals:log?[10,100,1000,10000,50000]:[0,10000,20000,30000,40000,50000],ticktext:log?['10','100','1k','10k','50k 이상']:['0','10k','20k','30k','40k','50k 이상']")
    page = page.replace('<p class="note">점에 마우스를', '<p class="note">50,000m² 이상은 X=50k에 모아 표시하며 실제 면적은 마우스를 올려 확인합니다.</p><p class="note">점에 마우스를')
    html.write_text(page, encoding='utf-8')
    assert stamps == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    (LOG / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
