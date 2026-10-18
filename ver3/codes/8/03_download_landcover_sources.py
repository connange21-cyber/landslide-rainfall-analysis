"""Download original land-cover SHP ZIPs using an official tile index and download URLs.

Index (GeoParquet/GeoJSON/SHP): MAP_YEAR, TILE_ID, DOWNLOAD_URL, geometry, CRS.
DOWNLOAD_URL must be an authorized official original-file URL, not WMS GetMap.
Default: plan only. No undocumented download endpoint or inferred tile number.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import geopandas as gpd
import pandas as pd
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json

OFFICIAL_HOSTS = {'aid.mcee.go.kr', 'api.mcee.go.kr', 'egis.me.go.kr', 'api.me.go.kr'}


def validate_url(url):
    u = urlparse(str(url))
    if u.scheme != 'https' or u.hostname not in OFFICIAL_HOSTS or u.username or u.password:
        raise ValueError('Expected an HTTPS official original-file URL')
    q = {k.lower():v for k,v in parse_qs(u.query).items()}
    if 'wms' in u.path.lower() or any(x.lower() == 'getmap' for x in q.get('request', [])):
        raise ValueError('WMS map image is not a classified original SHP source')


def validate_zip(path):
    """Reject HTML/PNG, corrupt archives and incomplete shapefile packages."""
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP checksum failure')
        names = {n.lower().replace('\\','/') for n in z.namelist()}
        shps = [n for n in names if n.endswith('.shp')]
        if not shps:
            raise ValueError('No original SHP found in ZIP')
        for shp in shps:
            stem = shp[:-4]
            if not all(stem+ext in names for ext in ['.shx','.dbf','.prj']):
                raise ValueError('Incomplete SHP package: SHX/DBF/PRJ required')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_plan(units, tiles):
    required = {'MAP_YEAR','TILE_ID','DOWNLOAD_URL','geometry'}
    if not required.issubset(tiles.columns) or tiles.crs is None:
        raise ValueError('Tile index requires MAP_YEAR,TILE_ID,DOWNLOAD_URL,geometry and CRS')
    tiles = tiles.reset_index(drop=True).copy()
    tiles['MAP_YEAR'] = pd.to_numeric(tiles.MAP_YEAR, errors='raise').astype(int)
    tiles['TILE_ID'] = tiles.TILE_ID.astype(str)
    if tiles.duplicated(['MAP_YEAR','TILE_ID']).any():
        raise ValueError('Duplicate year/tile keys in index')
    if not tiles.TILE_ID.map(lambda s:bool(re.fullmatch(r'[A-Za-z0-9_-]+',s))).all():
        raise ValueError('Unsafe tile filename')
    if tiles.geometry.isna().any() or tiles.geometry.is_empty.any() or not tiles.geometry.is_valid.all():
        raise ValueError('Invalid official tile geometry')
    tiles = tiles.to_crs(units.crs)
    plans = []
    coverage = []
    for year, group in units.groupby('REQUEST_MAP_YEAR'):
        t = tiles[tiles.MAP_YEAR.eq(year)]
        links = gpd.sjoin(group[['SPATIAL_UNIT_ID','geometry']], t[['TILE_ID','geometry']], how='inner', predicate='intersects')
        # A touching edge alone does not require that tile.
        links = links[[r.geometry.intersection(t.loc[r.index_right].geometry).area > 0 for r in links.itertuples()]] if len(links) else links
        for tid, matches in links.groupby('TILE_ID'):
            row = t[t.TILE_ID.eq(tid)].iloc[0]
            validate_url(row.DOWNLOAD_URL)
            plans.append(dict(MAP_YEAR=int(year), TILE_ID=tid, DOWNLOAD_URL=row.DOWNLOAD_URL,
                              ADMIN_UNIT_COUNT=matches.SPATIAL_UNIT_ID.nunique(),
                              SPATIAL_UNIT_IDS=sorted(matches.SPATIAL_UNIT_ID.unique())))
        covered = set(links.SPATIAL_UNIT_ID)
        for r in group.itertuples():
            selected = t[t.TILE_ID.isin(links.loc[links.SPATIAL_UNIT_ID.eq(r.SPATIAL_UNIT_ID),'TILE_ID'])]
            ratio = r.geometry.intersection(selected.geometry.union_all()).area/r.geometry.area if len(selected) and r.geometry.area else 0.
            coverage.append(dict(MAP_YEAR=int(year), SPATIAL_UNIT_ID=r.SPATIAL_UNIT_ID,
                                 TILE_COVERAGE_FRACTION=ratio,
                                 STATUS='INDEX_COVERED' if ratio >= .999999 else 'REVIEW_INDEX_COVERAGE'))
    return pd.DataFrame(plans, columns=['MAP_YEAR','TILE_ID','DOWNLOAD_URL','ADMIN_UNIT_COUNT','SPATIAL_UNIT_IDS']), pd.DataFrame(coverage)


def download(session, url, destination):
    validate_url(url)
    if destination.exists():
        return 'EXISTING_VERIFIED', validate_zip(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    if temporary.exists():
        raise FileExistsError('Previous partial file exists; inspect it before retrying')
    try:
        # Do not forward authentication cookies through automatic redirects.
        with session.get(url, stream=True, timeout=(15,120), allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError(f'HTTP_{response.status_code}: original download URL may require renewal/login')
            content_type = response.headers.get('Content-Type','').lower()
            if 'html' in content_type or 'image/' in content_type:
                raise ValueError('Received login page or rendered image instead of original ZIP')
            with temporary.open('xb') as stream:
                for block in response.iter_content(1024*1024):
                    if block:stream.write(block)
        digest = validate_zip(temporary)
        if destination.exists():
            raise FileExistsError('Destination appeared during download; refusing overwrite')
        temporary.rename(destination)
        return 'DOWNLOADED_VERIFIED', digest
    finally:
        # Remove only the incomplete file owned by this invocation.
        if temporary.exists():temporary.unlink()


def report(out, run):
    rows = pd.DataFrame(run.get('year_summary', []))
    (out/'summary.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>08 · 03 피복 원자료 수집</title>
<style>body{font:15px/1.7 Malgun Gothic;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}</style>
<h1>08 · 03 연도별 토지피복 원자료 수집</h1>'''+f'<p>상태: {run["status"]}</p>'+
        '<p>세분류 1m급 SHP ZIP 원자료 수집용입니다. WMS 표시 이미지를 분류코드 자료로 대체하지 않습니다.</p>'+
        '<p>입력: 공식 도엽 경계 인덱스(GeoParquet/GeoJSON/SHP), MAP_YEAR·TILE_ID·DOWNLOAD_URL·geometry 및 CRS. 다운로드 주소는 로그인·신청 후 제공되는 정식 원자료 주소가 필요합니다. 사이트의 신청 절차를 자동으로 우회하지 않습니다.</p>'+
        '<p>기본 실행은 계획만 생성. --execute 실제 수집, --limit 2 소량 확인. base/landcover/YYYY/TILE_ID.zip 저장. ZIP 무결성 및 SHP 구성 파일 확인, 기존 파일 보존·검증, 진행 및 오류는 log/08/full/03_download에 기록.</p>'+
        (rows.to_html(index=False,border=0) if len(rows) else '')+'</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tile-index',type=Path,default=V3/'base/landcover/tile_index.parquet')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if args.limit < 0:parser.error('--limit must be nonnegative')
    out = V3/'output/08/full/03_download';log = V3/'log/08/full/03_download'
    out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
    run = dict(status='PREPARING',started=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),downloaded=0,
               inputs=[fingerprint(Path(__file__))],mode='EXECUTE' if args.execute else 'PLAN')
    if not args.tile_index.is_file():
        run['status']='WAITING_OFFICIAL_TILE_INDEX_AND_DOWNLOAD_URLS'
        write_json(log/'run.json',run);report(out,run)
        print('다운로드 준비 보류: 공식 도엽 인덱스와 원자료 DOWNLOAD_URL이 필요합니다. '+str(args.tile_index),flush=True)
        return
    files = sorted((V3/'output/08/full/02_yearly_requirements').glob('*/required_admin_units.parquet'))
    if not files:raise FileNotFoundError('Run 08/02 first')
    run['inputs'] += [fingerprint(p) for p in [args.tile_index,*files]]
    units = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(p) for p in files],ignore_index=True),crs=gpd.read_parquet(files[0]).crs)
    tiles = gpd.read_parquet(args.tile_index) if args.tile_index.suffix=='.parquet' else gpd.read_file(args.tile_index)
    plan, coverage = make_plan(units,tiles)
    plan.drop(columns='DOWNLOAD_URL').to_parquet(out/'required_tiles.parquet',index=False)
    coverage.to_parquet(out/'index_coverage.parquet',index=False)
    run['year_summary'] = plan.groupby('MAP_YEAR').size().rename('TILES').reset_index().to_dict('records')
    run['status']='PLANNED';write_json(log/'run.json',run);report(out,run)
    if not args.execute:
        print(f'계획 완료: {len(plan):,}개 도엽. 실제 수집은 --execute 추가',flush=True);return
    if args.limit:plan=plan.head(args.limit)
    lock=log/'.download.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    session=requests.Session()
    # Optional caller-provided authorized session; never log it or URL query strings.
    if os.environ.get('LANDCOVER_COOKIE'):session.headers['Cookie']=os.environ['LANDCOVER_COOKIE']
    run['status']='RUNNING'
    try:
        for i,r in enumerate(plan.itertuples(),1):
            path=V3/'base/landcover'/str(r.MAP_YEAR)/(r.TILE_ID+'.zip')
            print(f'[원자료 수집] {i:,}/{len(plan):,} | {r.MAP_YEAR} · {r.TILE_ID}',flush=True)
            status,digest=download(session,r.DOWNLOAD_URL,path)
            run['downloaded']+=int(status=='DOWNLOADED_VERIFIED')
            write_json(log/'progress.json',dict(done=i,total=len(plan),year=r.MAP_YEAR,tile=r.TILE_ID,status=status))
            with (log/'files.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(year=r.MAP_YEAR,tile=r.TILE_ID,status=status,bytes=path.stat().st_size,sha256=digest))+'\n')
            print(f'[{status}] {path.name}',flush=True);time.sleep(.3)
        run['status']='COMPLETE_LIMITED' if args.limit else 'COMPLETE_DOWNLOAD_ATTEMPTS'
    except BaseException as exc:
        run.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error_type=type(exc).__name__)
        raise RuntimeError('원자료 수집 중단: 세션/공식 다운로드 주소와 파일 구성을 확인하세요. 로그에는 인증정보를 저장하지 않습니다.') from None
    finally:
        session.close();lock.unlink(missing_ok=True)
        run['finished']=pd.Timestamp.now(tz='Asia/Seoul').isoformat()
        write_json(log/'run.json',run);report(out,run)


if __name__=='__main__':main()
