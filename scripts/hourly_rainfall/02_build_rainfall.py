"""Preprocessing stage 2: standalone execution from file inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import ROOT, V3, HOUR, save, write_json, run_stage


def validate_partition(frame, date):
    if list(frame.columns) != ['STN', 'time_start', 'time_end', 'RN_1H']:
        raise ValueError('Unexpected rain schema')
    if frame.STN.isna().any() or not pd.api.types.is_integer_dtype(frame.STN):
        raise ValueError('Invalid STN type')
    if frame.time_start.dt.tz is not None or frame.time_end.dt.tz is not None:
        raise ValueError('Expected naive KST timestamps')
    if frame[['time_start', 'time_end']].isna().any().any() or not (frame.time_end-frame.time_start).eq(HOUR).all():
        raise ValueError('Invalid hourly interval')
    if not frame.time_start.eq(frame.time_start.dt.floor('h')).all():
        raise ValueError('Non-hourly interval alignment')
    if not frame.time_start.dt.normalize().eq(date).all():
        raise ValueError('Partition start-date mismatch')
    if frame.duplicated(['STN', 'time_start', 'time_end']).any():
        raise ValueError('Duplicate rain keys')
    return frame

def stage02(config, links, history, mode):
    stns = sorted(links.SELECTED_STN.dropna().astype(int).unique().tolist())
    root = Path(config['rain_root'])
    valid = links[links.SELECTED_STN.notna()]
    lower = valid.EVENT_INTERVAL_START.min().normalize()-pd.Timedelta(days=max(config['cumulative_days']))
    # Read through the last available partition: an event may extend after occurrence.
    catalogs, files, parts, qc = {}, [], {}, {}
    for network in config['rain_priority']:
        folder = root/network
        index = pd.read_parquet(folder/'index.parquet')
        actual = {p.resolve() for p in (folder/'hourly').glob('*/*/*/*.parquet')}
        indexed = { (folder/str(x).replace('\\', '/')).resolve() for x in index.loc[index.kind.eq('hourly') & index.table.eq('data'), 'file'] }
        catalogs[network] = dict(actual_files=len(actual), indexed_missing=len(indexed-actual), unindexed=len(actual-indexed))
        frames = []
        for path in sorted(actual):
            date = pd.Timestamp(path.stem.removeprefix('data_'))
            if date < lower:
                continue
            table = pq.read_table(path, columns=['STN', 'time_start', 'time_end', 'RN_1H'], filters=[('STN', 'in', stns)])
            frame = validate_partition(table.to_pandas(), date)
            invalid = frame.RN_1H.notna() & (~np.isfinite(frame.RN_1H) | frame.RN_1H.lt(0))
            qc[network+'_invalid_values'] = qc.get(network+'_invalid_values', 0)+int(invalid.sum())
            frame.loc[invalid, 'RN_1H'] = np.nan
            frames.append(frame)
            files.append(dict(network=network, path=str(path), rows=len(frame), bytes=path.stat().st_size,
                              mtime_ns=path.stat().st_mtime_ns))
        if not frames:
            raise ValueError(f'No rainfall files for {network}')
        frame = pd.concat(frames, ignore_index=True)
        if frame.duplicated(['STN', 'time_start']).any():
            raise ValueError('Cross-file duplicate keys')
        parts[network] = frame.rename(columns={'RN_1H': network+'_RN_1H'})
        print(f'02: {network} {len(frame):,} rows loaded', flush=True)
    aws = parts['AWS'].assign(AWS_ROW_PRESENT=True)
    asos = parts['ASOS'].assign(ASOS_ROW_PRESENT=True)
    rain = aws.merge(asos, on=['STN', 'time_start', 'time_end'], how='outer', validate='one_to_one')
    rain['RN_1H'] = rain.AWS_RN_1H.combine_first(rain.ASOS_RN_1H)
    rain['RAIN_SOURCE'] = np.select([rain.AWS_RN_1H.notna(), rain.ASOS_RN_1H.notna()], ['AWS', 'ASOS'], default='MISSING')
    rain['CROSS_NETWORK_DIFF_MM'] = (rain.AWS_RN_1H-rain.ASOS_RN_1H).abs()
    rain['CROSS_NETWORK_REVIEW'] = rain.CROSS_NETWORK_DIFF_MM.ge(5)
    for col in ('AWS_ROW_PRESENT', 'ASOS_ROW_PRESENT'):
        rain[col] = rain[col].fillna(False).astype(bool)
    rain['IS_IMPUTED'] = False
    rain = rain.sort_values(['STN', 'time_start']).reset_index(drop=True)
    rain['LOCATION_USABLE'] = False
    rain['LOCATION_REVIEW'] = False
    rain['LOCATION_VERSION'] = pd.array([None]*len(rain), dtype='Int64')
    # Match within station groups; source-network history is never silently borrowed.
    histories = {int(stn): h for stn, h in history.groupby('STN')}
    for stn, group in rain.groupby('STN', sort=False):
        h = histories.get(int(stn))
        if h is None:
            continue
        hits = np.zeros(len(group), dtype=np.int16)
        review = np.zeros(len(group), dtype=bool)
        versions = np.full(len(group), np.nan)
        for r in h.itertuples():
            same = group.RAIN_SOURCE.eq(r.NETWORK).to_numpy()
            mask = same & group.time_start.ge(r.ANALYSIS_FROM).to_numpy()
            if pd.notna(r.ANALYSIS_TO):
                mask &= group.time_end.le(r.ANALYSIS_TO).to_numpy()
            hits[mask] += 1
            versions[mask] = int(r.LOCATION_VERSION)
            review[mask] |= bool(r.REVIEW_FLAG)
            for edge in (r.ANALYSIS_FROM, r.ANALYSIS_TO):
                if pd.notna(edge):
                    review |= same & group.time_start.lt(edge).to_numpy() & group.time_end.gt(edge).to_numpy()
            if pd.notna(r.LAST_OLD_OBSERVED) and pd.notna(r.FIRST_NEW_OBSERVED):
                review |= same & group.time_start.lt(r.FIRST_NEW_OBSERVED).to_numpy() & group.time_end.gt(r.LAST_OLD_OBSERVED).to_numpy()
        if (hits > 1).any():
            raise ValueError('Overlapping same-network station history')
        rain.loc[group.index, 'LOCATION_USABLE'] = (hits == 1) & ~review
        rain.loc[group.index, 'LOCATION_REVIEW'] = review
        rain.loc[group.index, 'LOCATION_VERSION'] = pd.array(versions, dtype='Int64')
    rain['RAIN_1H_ANALYSIS'] = rain.RN_1H.where(rain.LOCATION_USABLE)
    daily = rain.assign(date=rain.time_start.dt.normalize()).groupby(['STN', 'date']).agg(
        HOURS_PRESENT=('time_start', 'size'), VALID_HOURS=('RAIN_1H_ANALYSIS', 'count'),
        OBSERVED_PARTIAL_SUM_MM=('RAIN_1H_ANALYSIS', 'sum'), LOCATION_REVIEW=('LOCATION_REVIEW', 'max'),
        CROSS_NETWORK_REVIEW=('CROSS_NETWORK_REVIEW', 'max'), LOCATION_VERSIONS=('LOCATION_VERSION', 'nunique')).reset_index()
    daily['DAILY_RAIN_MM'] = daily.OBSERVED_PARTIAL_SUM_MM.where(daily.VALID_HOURS.eq(24) & daily.HOURS_PRESENT.eq(24))
    daily['STATUS'] = np.where(daily.DAILY_RAIN_MM.notna(), 'COMPLETE_OBSERVED', 'INCOMPLETE')
    save(rain, '02', 'rainfall_hourly', mode)
    save(daily, '02', 'rainfall_daily', mode)
    qc.update(rows=len(rain), stations=len(stns), missing_analysis_hours=int(rain.RAIN_1H_ANALYSIS.isna().sum()),
              valid_daily=int(daily.DAILY_RAIN_MM.notna().sum()), cross_network_review=int(rain.CROSS_NETWORK_REVIEW.sum()),
              location_unusable=int((~rain.LOCATION_USABLE).sum()), imputation=config['imputation'], catalogs=catalogs)
    write_json(V3/'log/02'/mode/'qc.json', qc)
    write_json(V3/'log/02'/mode/'source_files.json', files)
    return rain, daily

if __name__ == "__main__":
    run_stage(2, stage02)
