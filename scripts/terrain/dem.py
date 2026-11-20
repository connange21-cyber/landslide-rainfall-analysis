"""Native-grid mosaics and Horn slope; never changes source rasters."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.strtree import STRtree
from pyproj import CRS


class Progress:
    """Throttled console + atomic JSON/auto-refresh HTML, no DEM reads."""
    def __init__(self, folder, step):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.step=step;self.started=time.monotonic();self.phase_started=self.started
        self.phase=None;self.last_write=-float('inf')
        self.warned_locked_files=set()

    def update(self,phase,done=0,total=None,current='',status='RUNNING',force=False):
        now=time.monotonic()
        changed=phase!=self.phase
        if changed:
            self.phase=phase;self.phase_started=now
        if not force and not changed and now-self.last_write<2 and done!=total:
            return
        self.last_write=now
        phase_elapsed=now-self.phase_started
        eta=(total-done)*phase_elapsed/done if total and done>=5 and phase_elapsed>=3 and done<total else None
        percent=100*done/total if total else None
        state=dict(step=self.step,status=status,phase=phase,done=int(done),total=total,
                   percent=round(percent,1) if percent is not None else None,current=str(current),
                   elapsed_seconds=round(now-self.started,1),phase_elapsed_seconds=round(phase_elapsed,1),
                   phase_eta_seconds=round(eta,1) if eta is not None else None,
                   updated_at=pd.Timestamp.now(tz='Asia/Seoul').isoformat())
        def duration(seconds):
            if seconds is None:return '산정 중'
            seconds=int(seconds);return f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
        count=f'{done:,}/{total:,}' if total is not None else f'{done:,}'
        pct=f'{percent:.1f}%' if percent is not None else '준비 중'
        print(f'[{self.step} | {phase}] {count} ({pct}) | 경과 {duration(now-self.started)} | 단계 남은 시간 {duration(eta)} | {current}',flush=True)
        from html import escape
        refresh='<meta http-equiv="refresh" content="5">' if status=='RUNNING' else ''
        bar=min(100,max(0,percent or 0))
        html=f'''<!doctype html><html lang="ko"><meta charset="utf-8">{refresh}<title>05 진행 상황</title>
<style>body{{font:16px/1.7 "Malgun Gothic",sans-serif;background:#f3f5f8;color:#233047}}main{{max-width:820px;margin:50px auto;padding:24px}}.track{{background:#dce3ec;height:16px;border-radius:8px}}.bar{{background:#39756b;height:16px;border-radius:8px;width:{bar}%}}td{{padding:8px 20px 8px 0}}small{{color:#68778a}}</style>
<main><h1>05 · {escape(self.step)}</h1><p>{escape(status)} · {escape(phase)}</p><div class="track"><div class="bar"></div></div>
<h2>{count} · {pct}</h2><table><tr><td>전체 경과</td><td>{duration(now-self.started)}</td></tr><tr><td>현재 단계 예상 남은 시간</td><td>{duration(eta)}</td></tr><tr><td>현재 처리 항목</td><td>{escape(str(current))}</td></tr></table>
<small>진행률·예상 시간은 현재 단계 기준입니다. 실행 중 5초마다 새로고침합니다.<br>마지막 갱신: {escape(state['updated_at'])}</small></main></html>'''
        for name,text in [('progress.json',json.dumps(state,ensure_ascii=False,indent=2)),('progress.html',html)]:
            import os
            path=self.folder/name;temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
            try:
                temporary.write_text(text,encoding='utf-8');temporary.replace(path)
                self.warned_locked_files.discard(name)
            except PermissionError:
                if name not in self.warned_locked_files:
                    print(f'[진행도 갱신 보류] {path}: 파일 접근 거부. 처리 작업은 계속합니다.',flush=True)
                    self.warned_locked_files.add(name)
                # Progress is supplementary: a viewer/file lock must not abort analysis.
                try:
                    temporary.unlink(missing_ok=True)
                except PermissionError:
                    pass


    def finish(self,status,current=''):
        self.update('완료' if status=='COMPLETE' else '중단' if status=='CANCELLED' else '오류',
                    current=current,status=status,force=True)


def horn(z, dx, dy, scale):
    if dx <= 0 or dy <= 0 or scale <= 0:
        raise ValueError('Positive horizontal spacing and confirmed vertical scale required')
    z = np.asarray(z, dtype='float64') * scale
    a,b,c = z[:-2,:-2],z[:-2,1:-1],z[:-2,2:]
    d,e,f = z[1:-1,:-2],z[1:-1,1:-1],z[1:-1,2:]
    g,h,i = z[2:,:-2],z[2:,1:-1],z[2:,2:]
    valid = np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)&np.isfinite(e)&np.isfinite(f)&np.isfinite(g)&np.isfinite(h)&np.isfinite(i)
    gx = ((c+2*f+i)-(a+2*d+g))/(8*dx)
    gy = ((g+2*h+i)-(a+2*b+c))/(8*dy)
    return np.where(valid, np.degrees(np.arctan(np.hypot(gx,gy))), np.nan).astype('float32')


def require_vertical(config):
    scale = config.get('vertical_scale_to_m')
    if scale is None or not np.isfinite(scale) or scale <= 0 or not config.get('vertical_unit_confirmation'):
        raise ValueError('DEM height unit unconfirmed: set vertical_scale_to_m and vertical_unit_confirmation after checking the source documentation')
    return float(scale)


def source_signature(path):
    """Invalidate cached metadata when the raster or georeferencing sidecars change."""
    files=[path,Path(str(path)+'.aux.xml'),Path(str(path)+'.xml'),path.with_suffix('.aux'),path.with_suffix('.prj')]
    values=[]
    for candidate in files:
        try:
            stat=candidate.stat();values.append([str(candidate),stat.st_size,stat.st_mtime_ns])
        except FileNotFoundError:
            values.append([str(candidate),None,None])
    return json.dumps(values,separators=(',',':'))


def read_header(path,signature):
    # Each worker owns and closes its GDAL dataset; no elevation arrays are read.
    with rasterio.open(path) as ds:
        if ds.crs is None:raise ValueError(f'Missing DEM CRS: {path}')
        t=ds.transform
        if not np.allclose([t.a,t.e,t.b,t.d],[5,-5,0,0],rtol=0,atol=1e-8):
            raise ValueError(f'Unexpected raster grid: {path}')
        # Shifted origins are aligned during window reading, not rejected.
        row=dict(path=str(path),left=ds.bounds.left,bottom=ds.bounds.bottom,right=ds.bounds.right,top=ds.bounds.top,
                 width=ds.width,height=ds.height,nodata=ds.nodata,unit=ds.units[0],crs_wkt=ds.crs.to_wkt(),
                 bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    if source_signature(path)!=signature:raise ValueError(f'DEM changed during header read: {path}')
    return row


def sheet_number(path):
    stem=Path(path).stem
    if stem.endswith('_projected'):stem=stem[:-10]
    if not stem.isdigit():raise ValueError(f'Unknown DEM filename: {path}')
    return int(stem)


def select_dem_paths(root):
    """Use one file per sheet; explicitly prefer user-created projected IMG."""
    chosen={};replaced=[]
    for path in sorted(Path(root).rglob('*.img')):
        number=sheet_number(path)
        previous=chosen.get(number)
        if previous is None:chosen[number]=path;continue
        if path.stem.endswith('_projected') == previous.stem.endswith('_projected'):
            raise ValueError(f'Ambiguous duplicate DEM sheet: {previous}, {path}')
        preferred=path if path.stem.endswith('_projected') else previous
        replaced.append(dict(sheet=number,selected=str(preferred),excluded=str(previous if preferred==path else path)))
        chosen[number]=preferred
    return [chosen[k] for k in sorted(chosen)],replaced


def build_index(root, folder, log, progress=None, workers=4, executor='process',target_crs='EPSG:5186'):
    """Parallel headers, persistent incremental cache, atomic final index."""
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
    if not isinstance(workers,int) or not 1<=workers<=32:
        raise ValueError('header_workers must be between 1 and 32')
    if executor not in ('process','thread'):raise ValueError('Unknown header executor')
    root=Path(root).resolve();folder=Path(folder);log=Path(log)
    log.mkdir(parents=True,exist_ok=True);folder.mkdir(parents=True,exist_ok=True)
    if progress:progress.update('DEM 파일 목록 확인',current=str(root),force=True)
    paths,replacements=select_dem_paths(root)
    if not paths:raise ValueError('No DEM .img files')
    cache_path=log.parent/'dem_header_cache.sqlite'
    db=sqlite3.connect(cache_path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS headers (path TEXT PRIMARY KEY, signature TEXT NOT NULL, payload TEXT NOT NULL, version INTEGER NOT NULL)')
    cached={path:(signature,payload,version) for path,signature,payload,version in db.execute('SELECT path,signature,payload,version FROM headers')}
    rows={};missing=[];pool=None;futures={};reused=0;started=time.monotonic()
    try:
        for i,path in enumerate(paths):
            signature=source_signature(path);saved=cached.get(str(path))
            if saved and saved[0]==signature and saved[2]==1:
                rows[str(path)]=json.loads(saved[1]);reused+=1
            else:missing.append((path,signature))
            if progress:progress.update('헤더 캐시 확인',i+1,len(paths),f'재사용 {reused:,}개')
        print(f'DEM headers: cached={reused:,}, to_read={len(missing):,}, workers={workers}, executor={executor}',flush=True)
        if missing:
            pool=ProcessPoolExecutor(max_workers=workers) if executor=='process' else ThreadPoolExecutor(max_workers=workers,thread_name_prefix='dem_header')
            remaining=iter(missing)
            def submit():
                item=next(remaining,None)
                if item is not None:
                    path,signature=item;futures[pool.submit(read_header,path,signature)]=(path,signature)
            for _ in range(min(workers*2,len(missing))):submit()
            done=0;last_commit=time.monotonic()
            while futures:
                complete,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in complete:
                    path,signature=futures.pop(future);row=future.result()
                    rows[str(path)]=row
                    db.execute('INSERT OR REPLACE INTO headers VALUES (?,?,?,?)',(str(path),signature,json.dumps(row),1))
                    done+=1
                    if done%32==0 or time.monotonic()-last_commit>=2:
                        db.commit();last_commit=time.monotonic()
                    if progress:progress.update('DEM 헤더 병렬 확인',done,len(missing),f'{path.name} · 캐시 재사용 {reused:,}개 · {workers}병렬')
                    submit()
            db.commit()
        # Keep source definitions, transform footprints into the common analysis CRS.
        from rasterio.warp import transform_bounds
        crs=CRS.from_user_input(target_crs);seen=set()
        for row in rows.values():
            wkt=row['crs_wkt']
            if wkt in seen:continue
            current=CRS.from_wkt(wkt)
            if not current.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in current.axis_info[:2]):
                raise ValueError(f'Metric projected DEM CRS required: {row["path"]}')
            seen.add(wkt)
        indexed=[]
        for path in paths:
            row=dict(rows[str(path)])
            original=[row[k] for k in ('left','bottom','right','top')]
            transformed=transform_bounds(row['crs_wkt'],crs,*original,densify_pts=21)
            row.update(zip(('left','bottom','right','top'),transformed))
            row.update(sheet_number=sheet_number(path),source_bounds=original,
                       user_projected=path.stem.endswith('_projected'))
            indexed.append(row)
        frame=pd.DataFrame(indexed)
        temporary=folder/'dem_tile_index.tmp.parquet';frame.to_parquet(temporary,index=False)
        temporary.replace(folder/'dem_tile_index.parquet')
        metadata=dict(tiles=len(frame),crs_wkt=crs.to_wkt(),header_only=True,workers=workers,executor=executor,
                      cache_reused=reused,headers_read=len(missing),cache=str(cache_path),elapsed_seconds=round(time.monotonic()-started,2),
                      source_crs_definitions=sorted(seen),projected_replacements=replacements,
                      crs_policy='TRANSFORM_FROM_RECORDED_SOURCE_DEFINITION',source_datum_validity='NOT_INDEPENDENTLY_VERIFIED')
        (log/'dem_index.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return frame,crs
    finally:
        # Keep every finished/collected header even on Ctrl+C or an error.
        for future in futures:future.cancel()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        db.commit();db.close()


class Mosaic:
    def __init__(self, index, target_crs=None):
        self.index = index.reset_index(drop=True)
        self.tree = STRtree([box(r.left,r.bottom,r.right,r.top) for r in self.index.itertuples()])
        self.resampled_paths=set()
        self.target_crs=target_crs

    def read(self, bounds, shape):
        left,bottom,right,top = bounds
        values = np.full(shape,np.nan,dtype='float64')
        conflict = np.zeros(shape,dtype=bool)
        tiles = self.tree.query(box(*bounds),predicate='intersects')
        # STRtree query order is unspecified; numeric sheet order is explicit.
        def priority(idx):
            p=Path(self.index.iloc[int(idx)].path)
            return sheet_number(p),str(p)
        for idx in sorted(tiles,key=priority):
            path = self.index.iloc[int(idx)].path
            with rasterio.open(path) as ds:
                from rasterio.warp import transform_bounds
                destination_crs=self.target_crs or ds.crs
                try:
                    source_bounds=transform_bounds(destination_crs,ds.crs,*bounds,densify_pts=21)
                    raw=from_bounds(*source_bounds,ds.transform)
                    import math
                    # Extra source cells keep bilinear kernels identical across blocks.
                    c0=math.floor(raw.col_off)-2;r0=math.floor(raw.row_off)-2
                    c1=math.ceil(raw.col_off+raw.width)+2;r1=math.ceil(raw.row_off+raw.height)+2
                    win=Window(c0,r0,c1-c0,r1-r0).intersection(Window(0,0,ds.width,ds.height))
                except rasterio.errors.WindowError:
                    continue
                if win.width <= 0 or win.height <= 0:
                    continue
                z = ds.read(1,window=win,masked=True).astype('float64').filled(np.nan)
                from rasterio.warp import reproject,Resampling
                from rasterio.transform import from_origin
                t=ds.transform
                aligned=(CRS.from_user_input(ds.crs).equals(CRS.from_user_input(destination_crs),ignore_axis_order=True)
                         and np.allclose([t.c/5,t.f/5],np.round([t.c/5,t.f/5]),rtol=0,atol=.001/5))
                layer=np.full(shape,np.nan,dtype='float64')
                reproject(z,layer,src_transform=ds.window_transform(win),src_crs=ds.crs,
                          src_nodata=np.nan,dst_transform=from_origin(left,top,5,5),dst_crs=destination_crs,
                          dst_nodata=np.nan,resampling=Resampling.nearest if aligned else Resampling.bilinear)
                if not aligned:self.resampled_paths.add(path)
                both=np.isfinite(values)&np.isfinite(layer)
                conflict |= both & ~np.isclose(values,layer,rtol=0,atol=.001)
                fill=~np.isfinite(values)&np.isfinite(layer)
                values[fill]=layer[fill]
        return values,int(conflict.sum())


def summarize(values, cell_area, thresholds):
    valid = values[np.isfinite(values)]
    result = dict(VALID_CELLS=int(len(valid)),VALID_AREA_M2=float(len(valid)*cell_area),
                  MEAN_SLOPE_DEG=float(valid.mean()) if len(valid) else np.nan)
    for threshold in thresholds:
        count = int((valid>=threshold).sum())
        result[f'GE_{threshold}DEG_CELLS'] = count
        result[f'GE_{threshold}DEG_AREA_M2'] = float(count*cell_area)
        result[f'GE_{threshold}DEG_FRACTION'] = count/len(valid) if len(valid) else np.nan
    return result


def report(folder, status, rows, note):
    from html import escape
    folder.mkdir(parents=True,exist_ok=True)
    table = ''.join(f'<tr><td>{escape(str(a))}</td><td>{escape(str(b))}</td></tr>' for a,b in rows)
    links = ''.join(f'<p><a href="{escape(p.name)}">{escape(p.name)}</a></p>' for p in sorted(folder.glob('*.parquet')))
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>05 경사 분석</title>
<style>body{{font:16px/1.7 "Malgun Gothic",sans-serif;background:#f3f5f8;color:#233047}}main{{max-width:900px;margin:40px auto;padding:24px}}table{{background:white;width:100%;border-collapse:collapse}}td{{padding:12px;border-bottom:1px solid #e1e6ed}}td:last-child{{text-align:right}}a{{color:#345ba2}}.status{{color:#8a5a14}}</style>
<main><h1>05 · 경사각과 급경사 면적 비율</h1><p>Horn 3×3 경사각을 계산하고 행정구역별 20°·30°·40° 이상 면적을 집계한다.</p>
<p class="status">{escape(status)}</p><table>{table}</table><p>{escape(note)}</p>{links}</main></html>'''
    (folder/'summary.html').write_text(html,encoding='utf-8')
