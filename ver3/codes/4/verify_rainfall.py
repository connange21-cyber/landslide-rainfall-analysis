"""Independent synthetic checks for time boundaries, missingness and splitting."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'common.py'
import types
modules = {}
for number, filename in [(2, '02_build_rainfall.py'), (3, '03_build_rainfall_events.py'), (4, '04_build_cumulative_rainfall.py')]:
    spec = importlib.util.spec_from_file_location(f'stage{number}', path.parent / str(number) / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules[number] = module
p = types.SimpleNamespace(segment_station=modules[3].segment_station,
                          cumulative_one=modules[4].cumulative_one,
                          validate_partition=modules[2].validate_partition)
config = {'wet_mm': 1.0, 'dry_split_hours': 12, 'unknown_split_hours': 12}


def frame(values, start='2023-07-01 00:00'):
    times = pd.date_range(start, periods=len(values), freq='h')
    return pd.DataFrame(dict(STN=1, time_start=times, RAIN_1H_ANALYSIS=values,
                             LOCATION_REVIEW=False, LOCATION_VERSION=1, CROSS_NETWORK_REVIEW=False))


# 11 dry hours stay internal; 12 dry hours split. Sub-threshold rainfall is summed.
x = p.segment_station(frame([0]*12+[2]+[0]*10+[0.4]+[3]+[0]*12+[4]+[0]*12), config)
assert len(x) == 2 and x[0]['EVENT_RAIN_MM'] == 5.4
assert x[0]['END_CONFIRMED'] and x[0]['START_CONFIRMED']
# Short missing span invalidates; 12 UNKNOWN splits, with unconfirmed boundary.
x = p.segment_station(frame([0]*12+[2, np.nan, 3]+[0]*12), config)
assert len(x) == 1 and not x[0]['EVENT_VALID'] and np.isnan(x[0]['EVENT_RAIN_MM'])
x = p.segment_station(frame([2]+[np.nan]*12+[3]), config)
assert len(x) == 2 and not x[0]['END_CONFIRMED'] and not x[1]['START_CONFIRMED']
# Midnight end belongs to the date of the last wet interval, not the next date.
x = p.segment_station(frame([2]+[0]*12, '2023-07-01 23:00'), config)
assert x[0]['RAIN_END_EXCLUSIVE'] == pd.Timestamp('2023-07-02')
assert x[0]['END_DATE'] == pd.Timestamp('2023-07-01')
# Backward calendar sums include anchor day; missing day is not zero-filled.
d = pd.DataFrame(dict(STN=[1]*3, date=pd.date_range('2023-07-01', periods=3),
    DAILY_RAIN_MM=[10., np.nan, 30.], LOCATION_REVIEW=False, CROSS_NETWORK_REVIEW=False))
r = p.cumulative_one(1, pd.Timestamp('2023-07-03'), d, [1,2,3,15])
assert r['RAIN_01D_MM'] == 30 and np.isnan(r['RAIN_02D_MM']) and np.isnan(r['RAIN_15D_MM'])
assert r['VALID_DAYS_03D'] == 2
# Partition rules use start date, and reject duplicate intervals.
f = pd.DataFrame(dict(STN=pd.Series([1], dtype='int64'), time_start=[pd.Timestamp('2023-07-01 23:00')],
                     time_end=[pd.Timestamp('2023-07-02')], RN_1H=[2.]))
p.validate_partition(f, pd.Timestamp('2023-07-01'))
try:
    p.validate_partition(pd.concat([f, f]), pd.Timestamp('2023-07-01'))
except ValueError:
    pass
else:
    raise AssertionError('Duplicate interval accepted')
print('PASS: dry/unknown splits, missing values, midnight endpoint, calendar sums, duplicate keys')

if '--full' in sys.argv:
    v = path.resolve().parents[1]
    links = pd.read_parquet(v/'output/01/full/landslide_station_links.parquet')
    daily = pd.read_parquet(v/'output/02/full/rainfall_daily.parquet')
    events = pd.read_parquet(v/'output/03/full/rainfall_events.parquet')
    assoc = pd.read_parquet(v/'output/03/full/landslide_rainfall_candidates.parquet')
    features = pd.read_parquet(v/'output/04/full/landslide_end_date_features.parquet')
    assert len(links) == 5160 and not links.EVENT_ID.duplicated().any()
    assert not events.RAIN_EVENT_ID.duplicated().any()
    assert not assoc.duplicated(['EVENT_ID', 'RAIN_EVENT_ID']).any()
    assert len(features) == len(assoc)
    grouped = pd.read_parquet(v/'output/04/full/admin_rain_event_features.parquet')
    import pyarrow.parquet as pq
    unit_path = v/'output/04/full/spatial_units_5186.parquet'
    registry = pd.read_parquet(unit_path, columns=['ADMIN_LEVEL','ADMIN_CODE','EMD_CODE','SPATIAL_UNIT_ID'])
    geo = json.loads(pq.read_schema(unit_path).metadata[b'geo'])
    assert str(geo['columns']['geometry']['crs']['id']['code']) == '5186'
    parents = set(registry.loc[registry.ADMIN_LEVEL.eq('RI'),'EMD_CODE'])
    assert not set(registry.loc[registry.ADMIN_LEVEL.eq('EMD'),'ADMIN_CODE']) & parents
    assert not registry.SPATIAL_UNIT_ID.duplicated().any()
    mapping = pd.read_parquet(v/'output/04/full/event_spatial_units.parquet')
    valid_units = mapping.ADMIN_LEVEL.ne('')
    assert mapping.loc[valid_units,'SPATIAL_UNIT_ID'].isin(registry.SPATIAL_UNIT_ID).all()
    require_ri = mapping.ORIGINAL_ADMIN_LEVEL.eq('EMD') & mapping.ORIGINAL_ADMIN_CODE.isin(parents)
    assert mapping.loc[require_ri,'SPATIAL_UNIT_STATUS'].eq('REVIEW_RI_REQUIRED_BUT_NOT_IDENTIFIED').all()
    assert mapping.loc[require_ri,'ADMIN_CODE'].eq('').all()
    assert grouped.loc[grouped.ADMIN_GROUP_RESOLVED,'SPATIAL_UNIT_ID'].isin(registry.SPATIAL_UNIT_ID).all()
    membership = grouped[['ADMIN_RAIN_GROUP_ID', 'LANDSLIDE_EVENT_IDS']].explode('LANDSLIDE_EVENT_IDS').rename(columns={'LANDSLIDE_EVENT_IDS':'EVENT_ID'})
    actual = features[['ADMIN_RAIN_GROUP_ID','EVENT_ID']].drop_duplicates().sort_values(['ADMIN_RAIN_GROUP_ID','EVENT_ID']).reset_index(drop=True)
    expected_membership = membership.sort_values(['ADMIN_RAIN_GROUP_ID','EVENT_ID']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected_membership, check_dtype=False)
    assert grouped.LANDSLIDE_COUNT.sum() == len(actual)
    assert grouped.CANDIDATE_ROW_COUNT.sum() == len(features)
    assert not grouped.ADMIN_RAIN_GROUP_ID.duplicated().any()
    known = grouped[grouped.ADMIN_GROUP_RESOLVED]
    assert not known.duplicated(['ADMIN_LEVEL','ADMIN_CODE','RAIN_EVENT_ID']).any()
    assert grouped.loc[~grouped.ADMIN_GROUP_RESOLVED,'LANDSLIDE_COUNT'].le(1).all()
    joined = features.merge(grouped, on='ADMIN_RAIN_GROUP_ID', suffixes=('_member','_group'),validate='many_to_one')
    for col in ['STN','RAIN_EVENT_ID','ADMIN_LEVEL','ADMIN_CODE']:
        assert joined[col+'_member'].eq(joined[col+'_group']).all()
    for n in range(1,16):
        assert np.allclose(joined[f'RAIN_{n:02d}D_MM_member'],joined[f'RAIN_{n:02d}D_MM_group'],equal_nan=True)
    # Same district and rain merges; different rain/district and unknown district do not.
    sample = pd.concat([features.head(1)]*6,ignore_index=True)
    sample['EVENT_ID'] = ['A','B','C','D','E','F']
    sample['ADMIN_LEVEL'] = 'RI';sample['ADMIN_CODE'] = '1234567890'
    sample.loc[2,'RAIN_EVENT_ID'] = 'DIFFERENT_RAIN'
    sample.loc[3,'ADMIN_CODE'] = '1234567891'
    sample.loc[4:5,'ADMIN_CODE'] = ''
    members, groups = modules[4].group_admin_events(sample,list(range(1,16)))
    assert len(groups)==5 and sorted(groups.LANDSLIDE_COUNT)==[1,1,1,1,2]
    assert features.EVENT_ID.isin(links.EVENT_ID).all()
    assert events.EVENT_VALID.eq(events.UNKNOWN_HOURS_WITHIN_EVENT.eq(0)).all()
    assert events.END_DATE.eq((events.RAIN_END_EXCLUSIVE-pd.Timedelta(nanoseconds=1)).dt.normalize()).all()
    # Check one complete daily sum per year against RAIN_DATAS directly.
    config = json.loads((v/'config/rainfall.json').read_text(encoding='utf-8'))
    complete = daily[daily.DAILY_RAIN_MM.notna()].copy()
    complete['year'] = complete.date.dt.year
    raw_checks = 0
    for row in complete.groupby('year').head(1).itertuples():
        raw = []
        for network in ['AWS', 'ASOS']:
            source = Path(config['rain_root'])/network/'hourly'/row.date.strftime('%Y/%m/%d')/f'data_{row.date:%Y%m%d}.parquet'
            if source.exists():
                data = pd.read_parquet(source, filters=[('STN', '=', int(row.STN))])
                raw.append(data.set_index('time_start').RN_1H.rename(network))
        paired = pd.concat(raw, axis=1)
        aws = paired['AWS'] if 'AWS' in paired else pd.Series(np.nan, index=paired.index)
        asos = paired['ASOS'] if 'ASOS' in paired else pd.Series(np.nan, index=paired.index)
        values = aws.combine_first(asos)
        assert len(values) == 24 and values.notna().all()
        assert np.isclose(values.sum(), row.DAILY_RAIN_MM, atol=1e-7)
        raw_checks += 1
    lookup = links.set_index('EVENT_ID')
    assert features.STN.eq(features.EVENT_ID.map(lookup.SELECTED_STN)).all()
    exact = features.EVENT_ID.map(lookup.EXACT_TIME).fillna(False).astype(bool)
    times = features.EVENT_ID.map(lookup.EVENT_INTERVAL_START)
    assert ((features.loc[exact, 'RAIN_START'] < times[exact]) & (features.loc[exact, 'RAIN_END_EXCLUSIVE'] >= times[exact])).all()
    # Recompute all calendar sums independently with indexed rolling daily series.
    anchors = features[['STN', 'END_DATE']].drop_duplicates()
    expected = []
    for stn, group in anchors.groupby('STN'):
        source = daily[daily.STN.eq(stn)].set_index('date').DAILY_RAIN_MM
        dates = pd.date_range(min(source.index.min(), group.END_DATE.min()-pd.Timedelta(days=14)),
                              max(source.index.max(), group.END_DATE.max()), freq='D')
        source = source.reindex(dates)
        out = group.copy()
        for n in range(1, 16):
            out[f'EXPECTED_{n}'] = source.rolling(n, min_periods=n).sum().reindex(group.END_DATE).to_numpy()
        expected.append(out)
    expected = pd.concat(expected, ignore_index=True)
    checked = features.merge(expected, on=['STN', 'END_DATE'], validate='many_to_one')
    for n in range(1, 16):
        actual = checked[f'RAIN_{n:02d}D_MM'].to_numpy()
        target = checked[f'EXPECTED_{n}'].to_numpy()
        assert np.allclose(actual, target, equal_nan=True, rtol=1e-10, atol=1e-7)
        assert checked[f'RAIN_{n:02d}D_MM'].notna().eq(checked[f'VALID_DAYS_{n:02d}D'].eq(n)).all()
    result = dict(status='PASS', original_events=len(links), candidate_rows=len(features),
                  cumulative_columns_verified=15, synthetic_checks='PASS', missing_preserved=True,
                  exact_overlap_verified=True, raw_daily_checks=raw_checks, dem_read=False,
                  admin_rain_groups=len(grouped), group_membership_verified=True,
                  group_key_and_features_verified=True, grouping_synthetic_checks='PASS',
                  spatial_crs='EPSG:5186', mixed_hierarchy_policy_verified=True,
                  spatial_unit_registry_rows=len(registry), ri_required_unresolved_events=int(require_ri.sum()))
    (v/'log/04/full/verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
