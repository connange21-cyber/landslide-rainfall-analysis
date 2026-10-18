"""Clip existing slope to each processed RI/EMD unit without recalculation."""
import argparse
import html
import json
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import V3, write_json
from dem import Progress


def blocks(width, height, size=512):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def verify(path, shape, transform, crs, expected, signature, unit_id):
    count = 0
    with rasterio.open(path) as ds:
        assert (ds.height, ds.width) == shape and ds.transform == transform
        assert ds.crs == crs and ds.nodata == -9999 and ds.dtypes == ('float32',)
        assert ds.tags().get('source_signature') == signature
        assert ds.tags().get('SPATIAL_UNIT_ID') == unit_id
        for _, win in ds.block_windows(1):
            values = ds.read(1, window=win, masked=True).compressed()
            assert np.isfinite(values).all() and (values >= 0).all() and (values <= 90).all()
            count += values.size
    assert count == expected, f'{unit_id}: valid cells {count} != {expected}'
    return int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Compatibility option; resuming is now automatic')
    parser.add_argument('--limit', type=int, default=0, help='Test first N units in separate smoke folder')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    mode = 'smoke' if args.limit else 'full'
    source = V3/'output/05/full/slope_degrees.tif'
    folder = V3/'output/05'/mode/'admin_slope'
    log = V3/'log/05'/mode/'03_export_admin_slope'
    progress = Progress(log, '03 행정구역별 경사 TIFF 저장')
    run = dict(status='RUNNING', source=str(source), recalculated=False,
               started=pd.Timestamp.now(tz='Asia/Seoul').isoformat())
    records = []
    try:
        selection = pd.read_parquet(V3/'output/05/full/slope_analysis_selection.parquet')
        ids = pd.read_parquet(V3/'output/05/full/processed_units.parquet').SPATIAL_UNIT_ID
        registry = gpd.read_parquet(V3/'output/04/full/spatial_units_5186.parquet')
        units = registry[registry.SPATIAL_UNIT_ID.isin(ids)].sort_values('SPATIAL_UNIT_ID')
        assert len(units) == len(ids) and units.SPATIAL_UNIT_ID.is_unique
        units = units.merge(selection[['SPATIAL_UNIT_ID', 'VALID_CELLS', 'ANALYSIS_STATUS', 'REVIEW_REASON']],
                            on='SPATIAL_UNIT_ID', validate='one_to_one')
        assert units.VALID_CELLS.notna().all()
        if args.limit:
            units = units.head(args.limit)
        stamp = source.stat()
        signature = f'{stamp.st_size}:{stamp.st_mtime_ns}'
        extract = json.loads((V3/'log/05/full/01_run.json').read_text(encoding='utf-8'))
        assert extract['status'] == 'COMPLETE'
        assert stamp.st_size == extract['slope_file']['bytes']
        assert stamp.st_mtime_ns == extract['slope_file']['mtime_ns']
        folder.mkdir(parents=True, exist_ok=True)
        with rasterio.open(source) as src:
            units = units.to_crs(src.crs)
            for i, unit in enumerate(units.itertuples()):
                uid = unit.SPATIAL_UNIT_ID
                target = folder/f'{uid}_slope_deg.tif'
                win = geometry_window(src, [unit.geometry])
                width, height = int(win.width), int(win.height)
                transform = src.window_transform(win)
                progress.update('행정구역 저장', i, len(units), uid, force=True)
                if target.exists():
                    verify(target, (height, width), transform, src.crs,
                           int(unit.VALID_CELLS), signature, uid)
                    status = 'VERIFIED_EXISTING'
                else:
                    profile = dict(driver='GTiff', width=width, height=height, count=1,
                                   dtype='float32', crs=src.crs, transform=transform, nodata=-9999.,
                                   tiled=True, blockxsize=512, blockysize=512,
                                   compress='deflate', predictor=3, BIGTIFF='IF_SAFER')
                    temporary = target.with_suffix('.tmp.tif')
                    total = ((width+511)//512)*((height+511)//512)
                    with rasterio.open(temporary, 'w', **profile) as dst:
                        dst.update_tags(**src.tags())
                        dst.update_tags(SPATIAL_UNIT_ID=uid, ADMIN_NAME=str(unit.ADMIN_NAME),
                                        source_signature=signature, source_path=str(source),
                                        clipping='PIXEL_CENTER_NO_RESAMPLING',
                                        ANALYSIS_STATUS=str(unit.ANALYSIS_STATUS))
                        for j, local in enumerate(blocks(width, height)):
                            original = Window(win.col_off+local.col_off, win.row_off+local.row_off,
                                              local.width, local.height)
                            values = src.read(1, window=original)
                            inside = geometry_mask([unit.geometry], values.shape,
                                                   dst.window_transform(local), invert=True,
                                                   all_touched=False)
                            values[~inside] = -9999.
                            dst.write(values, 1, window=local)
                            progress.update('구역 내부 블록', j+1, total, f'{i+1}/{len(units)} · {uid}')
                    verify(temporary, (height, width), transform, src.crs,
                           int(unit.VALID_CELLS), signature, uid)
                    temporary.replace(target)
                    status = 'CREATED'
                records.append(dict(SPATIAL_UNIT_ID=uid, ADMIN_NAME=unit.ADMIN_NAME,
                                    FILE=target.name, VALID_CELLS=int(unit.VALID_CELLS),
                                    ANALYSIS_STATUS=unit.ANALYSIS_STATUS, REVIEW_REASON=unit.REVIEW_REASON,
                                    BYTES=target.stat().st_size, EXPORT_STATUS=status))
                progress.update('행정구역 완료', i+1, len(units), uid, force=True)
        after = source.stat()
        assert (stamp.st_size, stamp.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        table = pd.DataFrame(records)
        temporary = folder/'index.tmp.parquet'
        table.to_parquet(temporary, index=False)
        temporary.replace(folder/'index.parquet')
        rows = ''.join(f'<tr><td>{html.escape(str(r.ADMIN_NAME))}</td><td><a href="{r.FILE}">{r.SPATIAL_UNIT_ID}</a></td><td>{r.VALID_CELLS:,}</td><td>{r.ANALYSIS_STATUS}</td></tr>' for r in table.itertuples())
        (folder/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>행정구역별 경사</title><style>body{font:15px Malgun Gothic;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
            f'<h1>행정구역별 경사 TIFF</h1><p>기존 경사각을 경계로 잘라 저장. 재계산·재투영·보간 없음. {len(table):,}개 파일.</p>'
            '<p>단위 °, 원본 5m 격자 유지. 경계 밖·결측은 NoData(-9999). TEMP_EXCLUDED도 확인용으로 저장하며 분석 제외 상태를 유지합니다.</p>'
            '<table><tr><th>행정구역</th><th>파일</th><th>유효 격자</th><th>분석 상태</th></tr>'+rows+'</table>', encoding='utf-8')
        run.update(status='COMPLETE', files=len(table), bytes=int(table.BYTES.sum()),
                   no_valid_files=int(table.VALID_CELLS.eq(0).sum()), output=str(folder),
                   verification='ALL_FILES_REOPENED_VALID_CELL_COUNTS_MATCH', source_unchanged=True)
        progress.finish('COMPLETE', f'{len(table):,}개 TIFF 저장 완료')
    except BaseException:
        run.update(status='FAILED_OR_INTERRUPTED', error=traceback.format_exc())
        progress.finish('FAILED_OR_INTERRUPTED', '중단됨. 로그 확인 후 --resume으로 재개')
        raise
    finally:
        run['finished'] = pd.Timestamp.now(tz='Asia/Seoul').isoformat()
        write_json(log/'run.json', run)


if __name__ == '__main__':
    main()
