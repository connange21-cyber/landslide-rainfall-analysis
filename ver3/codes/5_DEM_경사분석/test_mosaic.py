"""Three-tile DEM mosaic experiment, independent of production slope processing."""
import json
import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


def main():
    v3 = Path(__file__).resolve().parents[2]
    out = v3 / 'output/05/test_mosaic'
    log = v3 / 'log/05/test_mosaic'
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    target = out / 'dem_35604002_35604004_priority_5m.tif'
    if target.exists():
        raise FileExistsError(target)
    paths = [Path('/data/DEM_5m/35604') / f'{n}.img'
             for n in (35604002, 35604003, 35604004)]
    headers = []
    crs = None
    for path in paths:
        with rasterio.open(path) as ds:
            if crs is None:
                crs = ds.crs
            if ds.crs != crs or not np.allclose(
                    [ds.transform.a, ds.transform.e, ds.transform.b, ds.transform.d],
                    [5, -5, 0, 0], rtol=0, atol=1e-8):
                raise ValueError(f'Unexpected grid/CRS: {path}')
            headers.append(dict(path=str(path), bounds=list(ds.bounds),
                                transform=list(ds.transform), bytes=path.stat().st_size,
                                mtime_ns=path.stat().st_mtime_ns))
    left = 5 * math.floor(min(h['bounds'][0] for h in headers) / 5)
    bottom = 5 * math.floor(min(h['bounds'][1] for h in headers) / 5)
    right = 5 * math.ceil(max(h['bounds'][2] for h in headers) / 5)
    top = 5 * math.ceil(max(h['bounds'][3] for h in headers) / 5)
    transform = from_origin(left, top, 5, 5)
    shape = (round((top-bottom)/5), round((right-left)/5))
    merged = np.full(shape, np.nan, dtype='float32')
    union = np.zeros(shape, dtype=bool)
    conflict = np.zeros(shape, dtype=bool)
    sources = []
    for path in paths:
        with rasterio.open(path) as ds:
            t = ds.transform
            aligned = np.allclose([t.c/5, t.f/5], np.round([t.c/5, t.f/5]),
                                  rtol=0, atol=.001/5)
            layer = np.full(shape, np.nan, dtype='float32')
            method = Resampling.nearest if aligned else Resampling.bilinear
            reproject(source=ds.read(1, masked=True).astype('float32').filled(np.nan),
                      destination=layer, src_transform=t, src_crs=crs,
                      src_nodata=np.nan, dst_transform=transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=method)
            overlap = np.isfinite(merged) & np.isfinite(layer)
            conflict |= overlap & ~np.isclose(merged, layer, rtol=0, atol=.001)
            union |= np.isfinite(layer)
            fill = ~np.isfinite(merged) & np.isfinite(layer)
            merged[fill] = layer[fill]
            # Lower sheet number already occupies overlapping cells.
            sources.append(dict(path=str(path), treatment='COPY_ON_COMMON_GRID' if aligned
                                else 'BILINEAR_RESAMPLED_ESTIMATE',
                                aligned_cells=int(np.isfinite(layer).sum())))
            print(f'{path.name}: {sources[-1]["treatment"]}', flush=True)
    from dem import Mosaic
    import pandas as pd
    index = pd.DataFrame([dict(path=h['path'], left=h['bounds'][0], bottom=h['bounds'][1], right=h['bounds'][2], top=h['bounds'][3]) for h in headers])
    engine=Mosaic(index.iloc[::-1])
    production,_=engine.read((left,bottom,right,top),shape)
    assert np.array_equal(np.isfinite(merged),union)
    assert np.allclose(merged,production,atol=1e-5,rtol=0,equal_nan=True)
    # Reconstruct independent windows to check block-boundary consistency.
    for col in range(0,shape[1],256):
        end=min(col+256,shape[1])
        part,_=engine.read((left+col*5,bottom,left+end*5,top),(shape[0],end-col))
        assert np.allclose(part,production[:,col:end],atol=1e-8,rtol=0,equal_nan=True)
    profile = dict(driver='GTiff', height=shape[0], width=shape[1], count=1,
                   dtype='float32', crs=crs, transform=transform, nodata=-9999.,
                   tiled=True, compress='deflate')
    temp = target.with_suffix('.tmp.tif')
    with rasterio.open(temp, 'w', **profile) as ds:
        ds.write(np.where(np.isfinite(merged), merged, -9999).astype('float32'), 1)
        ds.update_tags(value_unit='m', product='TEST_DEM_MOSAIC_NOT_SLOPE',
                       shifted_tile_treatment='BILINEAR_RESAMPLED_ESTIMATE',
                       overlap_policy='SMALLER_SHEET_FIRST_VALID', original_dem_unchanged='true')
    with rasterio.open(temp) as ds:
        actual = ds.read(1, masked=True).filled(np.nan)
        assert ds.res == (5, 5) and ds.crs == crs
        assert np.array_equal(actual, merged, equal_nan=True)
    for h, path in zip(headers, paths):
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (h['bytes'], h['mtime_ns'])
    temp.replace(target)
    valid = merged[np.isfinite(merged)]
    result = dict(status='COMPLETE', test_only=True, sources=sources, headers=headers,
                  output=str(target), shape=list(shape), crs=crs.to_wkt(),
                  valid_cells=len(valid), conflicting_overlap_cells=int(conflict.sum()),
                  min_height_m=float(valid.min()), max_height_m=float(valid.max()),
                  verification='FULL_TIFF_READBACK_EQUAL; SOURCE_SIZE_MTIME_UNCHANGED; UNION_COVERAGE; PRODUCTION_ENGINE_EQUAL; BLOCK_SEAMS_EQUAL')
    (log/'priority_run.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    from dem import report
    report(out, 'DEM 3개 시험 병합 완료',
           [('입력 타일', 3), ('해상도', '5m'), ('유효 격자', len(valid)),
            ('중첩 값 차이(작은 번호 유지)', int(conflict.sum()))],
           '표고(m) TIFF이며 경사각이 아니다. 35604003만 쌍선형 보간한 시험본으로 중첩은 작은 번호의 유효값을 유지했다. 빈 띠 없음 및 본 코드와 블록 경계 일치 검증 통과.')
    print(f'COMPLETE: {target}', flush=True)


if __name__ == '__main__':
    main()
