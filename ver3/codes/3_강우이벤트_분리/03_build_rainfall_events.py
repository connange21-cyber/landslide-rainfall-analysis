"""Ver3 stage 3: standalone execution from file inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import ROOT, V3, HOUR, save, write_json, run_stage


def segment_station(frame, config):
    """Partition using consecutive DRY/UNKNOWN runs; endpoint = last WET end."""
    frame = frame.set_index('time_start').sort_index()
    index = pd.date_range(frame.index.min(), frame.index.max(), freq='h')
    values = frame.RAIN_1H_ANALYSIS.reindex(index).to_numpy(dtype=float)
    states = np.where(~np.isfinite(values), 'UNKNOWN', np.where(values >= config['wet_mm'], 'WET', 'DRY'))
    wet = np.flatnonzero(states == 'WET')
    if not len(wet):
        return []
    breaks = np.zeros(len(index), dtype=bool)
    dry = unknown = 0
    for i, state in enumerate(states):
        dry = dry+1 if state == 'DRY' else 0
        unknown = unknown+1 if state == 'UNKNOWN' else 0
        breaks[i] = dry >= config['dry_split_hours'] or unknown >= config['unknown_split_hours']
    # Prefix counts make gap split checks independent of gap length.
    prefix = np.r_[0, np.cumsum(breaks)]
    splits = [0] + [j for j in range(1, len(wet)) if prefix[wet[j]]-prefix[wet[j-1]+1] > 0] + [len(wet)]
    def dry_before(a):
        return a >= config['dry_split_hours'] and np.all(states[a-config['dry_split_hours']:a] == 'DRY')
    def dry_after(b):
        return b+config['dry_split_hours'] < len(states) and np.all(states[b+1:b+1+config['dry_split_hours']] == 'DRY')
    rows = []
    stn = int(frame.STN.iloc[0])
    for left, right in zip(splits[:-1], splits[1:]):
        a, b = wet[left], wet[right-1]
        span = values[a:b+1]
        location = frame.reindex(index[a:b+1])
        missing = int(np.isnan(span).sum())
        start, end = index[a], index[b]+HOUR
        rows.append(dict(STN=stn, RAIN_EVENT_ID=f'{stn}_{start:%Y%m%d%H}_{end:%Y%m%d%H}',
            RAIN_START=start, RAIN_END_EXCLUSIVE=end, LAST_WET_START=index[b],
            END_DATE=index[b].normalize(), END_DATE_RULE='DATE_OF_LAST_WET_INTERVAL_START',
            END_CONFIRMED=dry_after(b), START_CONFIRMED=dry_before(a),
            END_CONFIRMATION_TIME=end+pd.Timedelta(hours=config['dry_split_hours']) if dry_after(b) else pd.NaT,
            UNKNOWN_HOURS_WITHIN_EVENT=missing, EVENT_VALID=missing == 0,
            EVENT_RAIN_MM=float(span.sum()) if missing == 0 else np.nan,
            OBSERVED_PARTIAL_SUM_MM=float(np.nansum(span)), DURATION_HOURS=b-a+1,
            LOCATION_REVIEW=bool(location.LOCATION_REVIEW.fillna(False).any()),
            LOCATION_CHANGED=location.LOCATION_VERSION.nunique() > 1,
            CROSS_NETWORK_REVIEW=bool(location.CROSS_NETWORK_REVIEW.fillna(False).any())))
    return rows

def stage03(config, links, rain, mode):
    records = []
    for stn, frame in rain.groupby('STN', sort=True):
        records.extend(segment_station(frame, config))
    events = pd.DataFrame(records)
    if events.empty:
        raise ValueError('No WET events in input')
    assoc, summaries = [], []
    lookup = {int(stn): frame for stn, frame in events.groupby('STN')}
    for row in links.itertuples():
        count = 0
        if pd.notna(row.SELECTED_STN) and int(row.SELECTED_STN) in lookup:
            frame = lookup[int(row.SELECTED_STN)]
            # Preserve ver2 exact-time association: start < t <= last wet end.
            # Storage remains [start,end); this is the event association rule.
            overlap = frame.RAIN_START.lt(row.EVENT_INTERVAL_START) & frame.RAIN_END_EXCLUSIVE.ge(row.EVENT_INTERVAL_START) if row.EXACT_TIME else (
                frame.RAIN_START.lt(row.EVENT_INTERVAL_END_EXCLUSIVE) & frame.RAIN_END_EXCLUSIVE.gt(row.EVENT_INTERVAL_START))
            for candidate in frame[overlap].itertuples():
                assoc.append(dict(EVENT_ID=row.EVENT_ID, RAIN_EVENT_ID=candidate.RAIN_EVENT_ID, STN=int(row.SELECTED_STN),
                    TIME_GRADE=row.TIME_GRADE, LOCATION_GRADE=row.LOCATION_GRADE, ADMIN_LEVEL=row.ADMIN_LEVEL,
                    ADMIN_CODE=row.ADMIN_CODE, EMD_CODE=row.EMD_CODE, LOCATION_STATUS=row.LOCATION_STATUS,
                    SELECTION_STATUS=row.SELECTION_STATUS, EVENT_VALID=candidate.EVENT_VALID,
                    END_CONFIRMED=candidate.END_CONFIRMED, START_CONFIRMED=candidate.START_CONFIRMED,
                    END_DATE=candidate.END_DATE, RELATION='OVERLAP_CANDIDATE_NOT_CAUSAL_CONFIRMATION'))
                count += 1
        summaries.append(dict(EVENT_ID=row.EVENT_ID, CANDIDATE_COUNT=count,
                              STATUS='LINKED_CANDIDATES' if count else 'NO_CANDIDATE_UNRESOLVED'))
    assoc = pd.DataFrame(assoc)
    save(events, '03', 'rainfall_events', mode)
    save(assoc, '03', 'landslide_rainfall_candidates', mode)
    save(pd.DataFrame(summaries), '03', 'event_link_status', mode)
    write_json(V3/'log/03'/mode/'qc.json', dict(events=len(events), candidate_rows=len(assoc),
        linked_events=sum(r['CANDIDATE_COUNT'] > 0 for r in summaries), internally_valid=int(events.EVENT_VALID.sum()),
        end_unconfirmed=int((~events.END_CONFIRMED).sum()), full_available_collection_used=True,
        exact_point_convention='RAIN_START < t <= RAIN_END_EXCLUSIVE (ver2 association rule)', zero_negative_labels_created=True))
    return events, assoc

if __name__ == "__main__":
    run_stage(3, stage03)
