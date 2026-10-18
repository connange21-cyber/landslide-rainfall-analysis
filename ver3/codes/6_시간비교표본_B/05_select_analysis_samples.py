"""Apply user-approved whole-event exclusions; preserve previous A/B outputs."""
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    root = V3/'output/06/full'
    out = root/'05_selection'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Selection already exists; no automatic overwrite')
    paths = [root/'occurrence_reference.parquet', root/'b_event_samples.parquet',
             V3/'output/05/full/slope_analysis_selection.parquet']
    before = [fingerprint(p) for p in paths]
    a, b, slope = [pd.read_parquet(p) for p in paths]
    keys = ['STN', 'RAIN_EVENT_ID']
    events = a[keys+['EVENT_VALID', 'END_CONFIRMED']].drop_duplicates()
    assert not events.duplicated(keys).any()
    excluded_units = set(slope.loc[slope.ANALYSIS_STATUS.eq('TEMP_EXCLUDED'), 'SPATIAL_UNIT_ID'])
    spatial = a.loc[a.SPATIAL_UNIT_ID.isin(excluded_units), keys].drop_duplicates()
    spatial['EXCLUDE_SPATIAL_EVENT'] = True
    decision = events.merge(spatial, on=keys, how='left', validate='one_to_one')
    decision['EXCLUDE_SPATIAL_EVENT'] = decision.EXCLUDE_SPATIAL_EVENT.fillna(False).astype(bool)
    decision['EXCLUDE_A_RAIN_QC'] = ~decision.EVENT_VALID | ~decision.END_CONFIRMED
    decision['USE_FOR_ANALYSIS'] = ~decision.EXCLUDE_SPATIAL_EVENT & ~decision.EXCLUDE_A_RAIN_QC
    decision['DECISION_REASON'] = decision.apply(lambda r: 'EXCLUDED_SPATIAL_EVENT' if r.EXCLUDE_SPATIAL_EVENT
        else 'EXCLUDED_A_MISSING_AND_END_UNCONFIRMED' if not r.EVENT_VALID and not r.END_CONFIRMED
        else 'EXCLUDED_A_MISSING' if not r.EVENT_VALID
        else 'EXCLUDED_A_END_UNCONFIRMED' if not r.END_CONFIRMED else 'INCLUDED', axis=1)
    joined = a.merge(decision[keys+['USE_FOR_ANALYSIS']], on=keys, validate='many_to_one')
    selected_a = joined.loc[joined.USE_FOR_ANALYSIS & joined.SPATIAL_ELIGIBLE].drop(columns='USE_FOR_ANALYSIS')
    selected_b = b[b.A_REFERENCE_ID.isin(selected_a.A_REFERENCE_ID)].copy()
    assert set(selected_a.A_REFERENCE_ID) == set(selected_b.A_REFERENCE_ID)
    assert selected_a.SPATIAL_ELIGIBLE.all()
    assert selected_b.A_REFERENCE_ID.is_unique
    assert before == [fingerprint(p) for p in paths]
    out.mkdir(parents=True)
    decision.to_parquet(out/'event_selection.parquet', index=False)
    selected_a.to_parquet(out/'occurrence_selected.parquet', index=False)
    selected_b.to_parquet(out/'b_event_selected.parquet', index=False)
    result = dict(status='COMPLETE', total_events=len(events),
        spatial_excluded_events=int(decision.EXCLUDE_SPATIAL_EVENT.sum()),
        additional_qc_excluded_events=int((~decision.EXCLUDE_SPATIAL_EVENT & decision.EXCLUDE_A_RAIN_QC).sum()),
        selected_events=int(decision.USE_FOR_ANALYSIS.sum()), selected_pairs=len(selected_b),
        unresolved_admin_rows_retained_in_original=int((joined.USE_FOR_ANALYSIS & ~joined.SPATIAL_ELIGIBLE).sum()),
        inputs=before, upstream_unchanged=True, user_decision_date='2026-10-08',
        existing_hourly_html='PRE_SELECTION_SNAPSHOT_NOT_UPDATED',
        trigger_exclusions='KEEP_ALL_KNOWN_TRIGGER_EVENTS_FOR_B_SAFETY')
    write_json(V3/'log/06/full/05_selection/run.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))


if __name__ == '__main__':
    main()
