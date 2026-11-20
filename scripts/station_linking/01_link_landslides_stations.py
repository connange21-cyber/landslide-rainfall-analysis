"""Preprocessing stage 1: standalone execution from file inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import ROOT, V3, HOUR, save, write_json, run_stage
from spatial import boundaries, locate, StationEngine


def event_interval(row):
    exact = pd.to_datetime(row.EVENT_DATETIME, errors='coerce')
    if pd.notna(exact):
        return exact, exact+pd.Timedelta(nanoseconds=1), True
    start = pd.to_datetime(row.EVENT_DATE_START, errors='coerce')
    end = pd.to_datetime(row.EVENT_DATE_END, errors='coerce')
    if pd.isna(start) or pd.isna(end) or start > end:
        return None
    return start.normalize(), end.normalize()+pd.Timedelta(days=1), False

def stage01(config, limit, mode):
    base = V3/'base'
    events = pd.read_csv(base/'landslide/landslide_event_master.csv')
    if events.EVENT_ID.isna().any() or events.EVENT_ID.duplicated().any():
        raise ValueError('Invalid event IDs')
    if limit:
        # Include different precision levels in smoke tests, deterministically.
        events = events.sort_values(['LOCATION_GRADE', 'EVENT_ID']).groupby('LOCATION_GRADE', dropna=False).head(limit)
    history = pd.read_excel(base/'stations/station_location_history.xlsx', sheet_name='location_history')
    master = pd.read_excel(base/'stations/station_master_final.xlsx', sheet_name='station_master')
    history = history[history.STN.isin(master.STN)].copy()
    for col in ('ANALYSIS_FROM', 'ANALYSIS_TO', 'LAST_OLD_OBSERVED', 'FIRST_NEW_OBSERVED'):
        history[col] = pd.to_datetime(history[col], errors='raise')
    if history.ANALYSIS_FROM.isna().any() or (history.ANALYSIS_TO.notna() & history.ANALYSIS_TO.le(history.ANALYSIS_FROM)).any():
        raise ValueError('Invalid station history intervals')
    if not history.NETWORK.isin(['AWS', 'ASOS']).all():
        raise ValueError('Invalid history network')
    if not history.LAT.between(-90, 90).all() or not history.LON.between(-180, 180).all():
        raise ValueError('Invalid station coordinates')
    history['REVIEW_FLAG'] = history.REVIEW_FLAG.fillna(False).astype(bool)
    ri, emd, crs, extent = boundaries(base)
    engine = StationEngine(history, crs, extent, config)
    links, candidates = [], []
    for i, (_, row) in enumerate(events.iterrows()):
        result = row.to_dict()
        kind, geom, level, code, name, location_status = locate(row, ri, emd, engine.transform)
        result.update(GEOMETRY_KIND=kind, ADMIN_LEVEL=level, ADMIN_CODE=code, ADMIN_NAME=name,
                      EMD_CODE=code[:8]+'00' if level == 'RI' else code if level == 'EMD' else '',
                      LOCATION_STATUS=location_status, ADMIN_BOUNDARY_TIME_VALIDITY=config['boundary_time_validity'],
                      SELECTED_STN=None, SELECTION_STATUS=location_status, ELEVATION_CHECK='NOT_ASSESSED_NO_DEM')
        interval = event_interval(row)
        if interval is None:
            result['SELECTION_STATUS'] = 'REVIEW_EVENT_TIME_UNRESOLVED'
        else:
            start, end, exact = interval
            result.update(EVENT_INTERVAL_START=start, EVENT_INTERVAL_END_EXCLUSIVE=end, EXACT_TIME=exact)
            if start < pd.Timestamp(config['research_start']):
                result['SELECTION_STATUS'] = 'OUTSIDE_RESEARCH_PERIOD'
            elif kind != 'NONE':
                edges = pd.concat([history.ANALYSIS_FROM, history.ANALYSIS_TO]).dropna()
                edges = [start, *sorted(set(edges[(edges > start) & (edges < end)])), end]
                outcomes = []
                for a, b in zip(edges[:-1], edges[1:]):
                    stn, status, rows = engine.select(kind, geom, a)
                    outcomes.append((stn, status))
                    for candidate in rows:
                        candidate.update(EVENT_ID=row.EVENT_ID, PERIOD_START=a, PERIOD_END_EXCLUSIVE=b,
                                         SELECTED_IN_PERIOD=stn is not None and candidate['STN'] == stn,
                                         METHOD='POINT_DISTANCE' if kind == 'POINT' else 'ADMIN_THIESSEN_MAX_OVERLAP')
                    candidates.extend(rows)
                chosen = {x[0] for x in outcomes}
                if len(chosen) == 1 and None not in chosen:
                    result['SELECTED_STN'] = next(iter(chosen))
                    result['SELECTION_STATUS'] = 'SELECTED_HISTORY_REVIEW' if any('REVIEW' in x[1] for x in outcomes) else 'SELECTED_PROVISIONAL'
                else:
                    result['SELECTION_STATUS'] = 'REVIEW_TEMPORAL_SELECTION' if len(outcomes) > 1 else outcomes[0][1]
        links.append(result)
        if (i+1) % 500 == 0:
            print(f'01: {i+1}/{len(events)} events', flush=True)
    links = pd.DataFrame(links)
    links['SELECTED_STN'] = pd.array(links.SELECTED_STN, dtype='Int64')
    save(links, '01', 'landslide_station_links', mode)
    save(pd.DataFrame(candidates), '01', 'station_candidates', mode)
    write_json(V3/'log/01'/mode/'qc.json', dict(events=len(links), selected=int(links.SELECTED_STN.notna().sum()),
        selection_status=links.SELECTION_STATUS.value_counts().to_dict(), location_status=links.LOCATION_STATUS.value_counts().to_dict(),
        boundary_crs=crs.to_string(), boundary_time_validity='NOT_VERIFIED', dem_read=False))
    return links, history

if __name__ == "__main__":
    run_stage(1, stage01)
