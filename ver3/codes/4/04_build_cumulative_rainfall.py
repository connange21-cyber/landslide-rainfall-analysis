"""Ver3 stage 4: standalone execution from file inputs."""
import sys
import hashlib
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import ROOT, V3, HOUR, save, write_json, run_stage


def spatial_units(links, mode):
    """RI where child boundaries exist; EMD only where no RI boundary exists."""
    sys.path.insert(0, str(V3/'codes/1'))
    from spatial import boundaries
    import geopandas as gpd
    ri, emd, _, _ = boundaries(V3/'base')
    ri, emd = ri.to_crs(5186), emd.to_crs(5186)
    ri['EMD_CODE'] = ri.ADMIN_CODE.str[:8]+'00'
    parents = set(ri.EMD_CODE)
    fallback = emd[~emd.ADMIN_CODE.isin(parents)].copy()
    fallback['EMD_CODE'] = fallback.ADMIN_CODE
    units = gpd.GeoDataFrame(pd.concat([ri, fallback], ignore_index=True), geometry='geometry', crs='EPSG:5186')
    units['SPATIAL_UNIT_ID'] = units.ADMIN_LEVEL+'_'+units.ADMIN_CODE
    units['BOUNDARY_TIME_VALIDITY'] = 'NOT_VERIFIED'
    units['PARENT_BOUNDARY_PRESENT'] = units.EMD_CODE.isin(set(emd.ADMIN_CODE))
    if units.SPATIAL_UNIT_ID.duplicated().any() or set(fallback.ADMIN_CODE) & parents:
        raise ValueError('Overlapping administrative hierarchy in unit registry')
    out = V3/'output/04'/mode
    out.mkdir(parents=True, exist_ok=True)
    temporary = out/'spatial_units_5186.tmp.parquet'
    units.to_parquet(temporary, index=False)
    temporary.replace(out/'spatial_units_5186.parquet')
    ri_codes, emd_codes = set(ri.ADMIN_CODE), set(emd.ADMIN_CODE)
    mapping = links[['EVENT_ID','ADMIN_LEVEL','ADMIN_CODE']].copy()
    mapping = mapping.rename(columns={'ADMIN_LEVEL':'ORIGINAL_ADMIN_LEVEL','ADMIN_CODE':'ORIGINAL_ADMIN_CODE'})
    levels, codes, statuses = [], [], []
    for row in mapping.itertuples():
        level, code = row.ORIGINAL_ADMIN_LEVEL, str(row.ORIGINAL_ADMIN_CODE) if pd.notna(row.ORIGINAL_ADMIN_CODE) else ''
        if level == 'RI' and code in ri_codes:
            a, b, status = 'RI', code, 'RI_RESOLVED'
        elif level == 'EMD' and code in emd_codes and code not in parents:
            a, b, status = 'EMD', code, 'EMD_NO_RI_BOUNDARY'
        elif level == 'EMD' and code in parents:
            a, b, status = '', '', 'REVIEW_RI_REQUIRED_BUT_NOT_IDENTIFIED'
        else:
            a, b, status = '', '', 'REVIEW_ADMIN_UNRESOLVED'
        levels.append(a); codes.append(b); statuses.append(status)
    mapping['ADMIN_LEVEL'], mapping['ADMIN_CODE'], mapping['SPATIAL_UNIT_STATUS'] = levels, codes, statuses
    mapping['SPATIAL_UNIT_ID'] = [a+'_'+b if a else '' for a,b in zip(levels,codes)]
    save(mapping, '04', 'event_spatial_units', mode)
    write_json(V3/'log/04'/mode/'spatial_qc.json', dict(crs='EPSG:5186', ri_units=len(ri), emd_fallback_units=len(fallback),
        total_units=len(units), event_status=mapping.SPATIAL_UNIT_STATUS.value_counts().to_dict(),
        orphan_ri_units=int((~units.PARENT_BOUNDARY_PRESENT).sum()),
        boundary_time_validity='NOT_VERIFIED', absence_of_ri_is_based_on_current_boundary_inventory=True,
        station_selection_unchanged=True, dem_read=False))
    return mapping


def cumulative_one(stn, end_date, daily, days):
    """Calendar-day backward sums: missing day/hour => NaN, never sum past gaps."""
    index = pd.date_range(end_date-pd.Timedelta(days=max(days)-1), end_date, freq='D')
    frame = daily[daily.STN.eq(stn)].set_index('date').reindex(index)
    result = {}
    for n in days:
        window = frame.iloc[-n:]
        count = int(window.DAILY_RAIN_MM.notna().sum())
        result[f'RAIN_{n:02d}D_MM'] = float(window.DAILY_RAIN_MM.sum()) if count == n else np.nan
        result[f'VALID_DAYS_{n:02d}D'] = count
    result['CUMULATIVE_LOCATION_REVIEW'] = bool(frame.LOCATION_REVIEW.fillna(False).any())
    result['CUMULATIVE_CROSS_NETWORK_REVIEW'] = bool(frame.CROSS_NETWORK_REVIEW.fillna(False).any())
    return result

def group_admin_events(result, days):
    """Same boundary level/code + station-defined rain event; retain lineage."""
    result = result.copy()
    result['ADMIN_LEVEL'] = result.ADMIN_LEVEL.fillna('')
    result['ADMIN_CODE'] = result.ADMIN_CODE.fillna('').astype(str)
    known = result.ADMIN_LEVEL.isin(['RI', 'EMD']) & result.ADMIN_CODE.str.fullmatch(r'\d{10}')
    identities = [json.dumps([r.ADMIN_LEVEL, r.ADMIN_CODE, r.RAIN_EVENT_ID,
                               '' if ok else r.EVENT_ID], ensure_ascii=False)
                  for r, ok in zip(result.itertuples(), known)]
    result['ADMIN_RAIN_GROUP_ID'] = ['ARG_'+hashlib.sha256(x.encode()).hexdigest()[:20] for x in identities]
    result['ADMIN_GROUP_RESOLVED'] = known
    choices = result.groupby('EVENT_ID').ADMIN_RAIN_GROUP_ID.nunique()
    result['EVENT_HAS_MULTIPLE_GROUP_CANDIDATES'] = result.EVENT_ID.map(choices).gt(1)
    shared = ['STN', 'RAIN_EVENT_ID', 'END_DATE', 'RAIN_START', 'RAIN_END_EXCLUSIVE',
              'END_CONFIRMATION_TIME', 'EVENT_RAIN_MM', 'EVENT_VALID', 'END_CONFIRMED', 'START_CONFIRMED',
              'UNKNOWN_HOURS_WITHIN_EVENT', 'LOCATION_CHANGED', 'LOCATION_REVIEW', 'CROSS_NETWORK_REVIEW',
              'CUMULATIVE_LOCATION_REVIEW', 'CUMULATIVE_CROSS_NETWORK_REVIEW', 'REFERENCE_STATUS',
              'CALENDAR_DAY_INCLUDES_POST_EVENT_HOURS']
    shared += [f'RAIN_{n:02d}D_MM' for n in days]+[f'VALID_DAYS_{n:02d}D' for n in days]
    rows = []
    for group_id, members in result.groupby('ADMIN_RAIN_GROUP_ID', sort=True):
        if any(members[c].nunique(dropna=False) != 1 for c in shared):
            raise ValueError(f'Conflicting rain features within group {group_id}')
        first = members.iloc[0]
        row = {c: first[c] for c in shared}
        spatial_review = bool(members.LOCATION_STATUS.fillna('').str.contains('REVIEW|REPAIRED|AMBIGUOUS').any()
                              or members.SELECTION_STATUS.fillna('').str.contains('REVIEW|AMBIGUOUS').any())
        resolved = bool(members.ADMIN_GROUP_RESOLVED.all())
        row.update(ADMIN_RAIN_GROUP_ID=group_id, ADMIN_LEVEL=first.ADMIN_LEVEL, ADMIN_CODE=first.ADMIN_CODE,
                   EMD_CODE=first.EMD_CODE, LANDSLIDE_EVENT_IDS=sorted(members.EVENT_ID.unique().tolist()),
                   LANDSLIDE_COUNT=members.EVENT_ID.nunique(), CANDIDATE_ROW_COUNT=len(members),
                   TIME_GRADES=sorted(members.TIME_GRADE.dropna().unique().tolist()),
                   LOCATION_GRADES=sorted(members.LOCATION_GRADE.dropna().unique().tolist()),
                   ADMIN_GROUP_RESOLVED=resolved, SPATIAL_REVIEW=spatial_review,
                   MULTIPLE_GROUP_CANDIDATE_EVENT_COUNT=members.loc[members.EVENT_HAS_MULTIPLE_GROUP_CANDIDATES, 'EVENT_ID'].nunique(),
                   GROUP_STATUS='REVIEW_ADMIN_UNRESOLVED' if not resolved else 'REVIEW_SPATIAL' if spatial_review else 'GROUPED',
                   BOUNDARY_TIME_VALIDITY='NOT_VERIFIED',
                   LABEL_STATUS='REPORTED_LANDSLIDE_GROUP_WITH_OVERLAP_CANDIDATES')
        if 'SPATIAL_UNIT_STATUS' in members:
            row['SPATIAL_UNIT_STATUSES'] = sorted(members.SPATIAL_UNIT_STATUS.unique().tolist())
            row['SPATIAL_UNIT_ID'] = first.SPATIAL_UNIT_ID
        rows.append(row)
    grouped = pd.DataFrame(rows)
    if grouped.ADMIN_RAIN_GROUP_ID.duplicated().any():
        raise ValueError('Group ID collision')
    return result, grouped


def stage04(config, events, assoc, daily, mode, links):
    write_json(V3/'log/04'/mode/'verification.json', {'status':'PENDING', 'reason':'Stage 04 regenerated; independent verification required'})
    mapping = spatial_units(links, mode)
    unique = assoc[['STN', 'END_DATE']].drop_duplicates()
    records = [dict(STN=int(r.STN), END_DATE=r.END_DATE, **cumulative_one(int(r.STN), r.END_DATE, daily, config['cumulative_days']))
               for r in unique.itertuples()]
    totals = pd.DataFrame(records)
    result = assoc.merge(totals, on=['STN', 'END_DATE'], how='left', validate='many_to_one')
    details = events[['RAIN_EVENT_ID', 'RAIN_START', 'RAIN_END_EXCLUSIVE', 'END_CONFIRMATION_TIME',
                      'EVENT_RAIN_MM', 'UNKNOWN_HOURS_WITHIN_EVENT', 'LOCATION_CHANGED', 'LOCATION_REVIEW', 'CROSS_NETWORK_REVIEW']]
    result = result.merge(details, on='RAIN_EVENT_ID', how='left', validate='many_to_one')
    result['REFERENCE_STATUS'] = np.where(result.END_CONFIRMED, 'END_CONFIRMED', 'REVIEW_END_UNCONFIRMED')
    result['CALENDAR_DAY_INCLUDES_POST_EVENT_HOURS'] = result.RAIN_END_EXCLUSIVE.lt(result.END_DATE+pd.Timedelta(days=1))
    result['LABEL_STATUS'] = 'REPORTED_LANDSLIDE_WITH_OVERLAP_CANDIDATE'
    result = result.drop(columns=['ADMIN_LEVEL','ADMIN_CODE']).merge(mapping, on='EVENT_ID', how='left', validate='many_to_one')
    if result.SPATIAL_UNIT_STATUS.isna().any():
        raise ValueError('Missing spatial assignment')
    result, grouped = group_admin_events(result, config['cumulative_days'])
    save(totals, '04', 'station_end_date_cumulative', mode)
    save(result, '04', 'landslide_end_date_features', mode)
    save(grouped, '04', 'admin_rain_event_features', mode)
    write_json(V3/'log/04'/mode/'qc.json', dict(rows=len(result), unique_station_dates=len(totals),
        unique_landslides=result.EVENT_ID.nunique(), periods=config['cumulative_days'],
        complete_by_period={str(n): int(result[f'RAIN_{n:02d}D_MM'].notna().sum()) for n in config['cumulative_days']},
        end_unconfirmed=int((~result.END_CONFIRMED).sum()), missing_not_filled=True,
        admin_rain_groups=len(grouped), resolved_groups=int(grouped.ADMIN_GROUP_RESOLVED.sum()),
        unresolved_admin_groups=int((~grouped.ADMIN_GROUP_RESOLVED).sum()),
        multi_landslide_groups=int(grouped.LANDSLIDE_COUNT.gt(1).sum()),
        spatial_review_groups=int(grouped.SPATIAL_REVIEW.sum()),
        grouped_end_unconfirmed=int((~grouped.END_CONFIRMED).sum()),
        group_complete_by_period={str(n): int(grouped[f'RAIN_{n:02d}D_MM'].notna().sum()) for n in config['cumulative_days']},
        grouping_key=['ADMIN_LEVEL', 'ADMIN_CODE', 'RAIN_EVENT_ID'],
        mixed_boundary_levels_not_merged=True, physical_storm_across_stations_not_inferred=True,
        dem_processed=False, comparison_samples_created=False, model_ready=False))
    return result

if __name__ == "__main__":
    run_stage(4, stage04)
