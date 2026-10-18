"""Read-only audit of downloaded ZIP inventory and extracted member sizes."""
import json
import re
import time
import zipfile
import zlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

V3 = Path(__file__).resolve().parents[2]
SOURCE = V3 / 'output/08/full/11_yearly_application'
EXTRACTED = V3 / 'output/08/full/12_extracted'
OUT = V3 / 'output/08/full/13_file_audit'
LOG = V3 / 'log/08/full/13_file_audit'
COMPLETED = {}


def audit(path):
    relative = path.relative_to(SOURCE)
    dest = EXTRACTED / relative.parent / path.stem
    partial = dest.with_name(dest.name + '.partial')
    row = dict(YEAR=int(relative.parts[0]), ZIP=path.name, FINAL_FOLDER_EXISTS=dest.is_dir(),
               PARTIAL_FOLDER_EXISTS=partial.is_dir(), STATUS='OK', ISSUES='')
    match = re.fullmatch(r'SG05_(\d{8})_\d{8}', path.stem)
    row['TILE_ID'] = match[1] if match else ''
    problems = []
    try:
        previous = COMPLETED.get(relative.as_posix())
        if previous and dest.is_dir():
            # The extraction ledger already records a successful per-member CRC
            # check. Use cached directory-entry metadata for a fast current check.
            source_stat = path.stat()
            if (source_stat.st_size,source_stat.st_mtime_ns) != previous[:2]:
                problems.append('SOURCE_ZIP_CHANGED_SINCE_EXTRACTION')
            actual=[]
            def scan(folder):
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):scan(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):actual.append((Path(entry.path),entry.stat().st_size))
                        else:problems.append('UNEXPECTED_SYMLINK_OR_ENTRY:' + entry.name)
            scan(dest)
            row['MEMBER_COUNT']=len(actual)
            if len(actual)!=previous[2]:problems.append('EXTRACTED_FILE_COUNT_MISMATCH')
            if sum(size for _,size in actual)!=previous[3]:problems.append('EXTRACTED_TOTAL_SIZE_MISMATCH')
            names={p.name.casefold() for p,size in actual}
            shapes=[p for p,size in actual if p.suffix.lower()=='.shp']
            if not shapes:problems.append('EXTRACTED_NO_SHP')
            for shape in shapes:
                for ext in ['.dbf','.shx','.prj']:
                    if shape.with_suffix(ext).name.casefold() not in names:
                        problems.append('EXTRACTED_MISSING_COMPONENT:'+shape.stem+ext)
            if problems:row.update(STATUS='REVIEW',ISSUES='; '.join(problems))
            return row
        with zipfile.ZipFile(path) as archive:
            files = [i for i in archive.infolist() if not i.is_dir()]
            row['MEMBER_COUNT'] = len(files)
            if not files:problems.append('ZIP_EMPTY')
            names = {i.filename.casefold() for i in files}
            shapes = [i for i in files if i.filename.lower().endswith('.shp')]
            if not shapes:problems.append('ZIP_NO_SHP')
            for shape in shapes:
                for ext in ['.dbf', '.shx', '.prj']:
                    if (shape.filename[:-4]+ext).casefold() not in names:
                        problems.append('ZIP_MISSING_COMPONENT:' + shape.filename[:-4]+ext)
            folder = dest if dest.is_dir() else partial
            if not dest.is_dir():problems.append('FINAL_FOLDER_MISSING')
            for info in files:
                rel = Path(info.filename.replace('\\', '/'))
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe ZIP member path')
                target = folder / rel
                if not target.is_file():problems.append('FILE_MISSING:' + info.filename);continue
                if target.stat().st_size != info.file_size:problems.append('SIZE_MISMATCH:' + info.filename)
                # Failed publication folders get a full extracted CRC check.
                if not dest.is_dir() and partial.is_dir():
                    crc = 0
                    with target.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024*1024), b''):crc=zlib.crc32(block,crc)
                    if crc & 0xffffffff != info.CRC:problems.append('CRC_MISMATCH:' + info.filename)
            if not dest.is_dir() and partial.is_dir():
                row['PARTIAL_CRC_MATCH'] = not any(p.startswith(('FILE_MISSING:', 'SIZE_MISMATCH:', 'CRC_MISMATCH:')) for p in problems)
    except Exception as error:problems.append(type(error).__name__ + ':' + str(error))
    if problems:row.update(STATUS='REVIEW', ISSUES='; '.join(problems))
    return row


def main():
    global COMPLETED
    OUT.mkdir(parents=True, exist_ok=True);LOG.mkdir(parents=True, exist_ok=True)
    print('[도엽 점검] ZIP 목록 읽기', flush=True)
    paths = sorted(SOURCE.glob('20[0-9][0-9]/*.zip'))
    database = V3/'log/08/full/12_extracted/extraction_state.sqlite'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        COMPLETED={r[0]:tuple(r[1:]) for r in db.execute('SELECT source,size,mtime_ns,files,bytes FROM completed')}
    results=[];started=time.monotonic();last=started
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(audit,paths),1):
            results.append(row);now=time.monotonic()
            if i==1 or i==len(paths) or now-last>=5:
                print(f'[도엽 점검] {i:,}/{len(paths):,} | 경과 {now-started:.0f}초',flush=True);last=now
    frame=pd.DataFrame(results)
    required=pd.read_parquet(V3/'output/08/full/05_tile_index/required_tiles_by_year.parquet')
    inventory={(r.YEAR,r.TILE_ID) for r in frame.itertuples()}
    missing=required[[ (int(r.REFERENCE_YEAR),str(r.TILE_ID)) not in inventory for r in required.itertuples() ]].copy()
    missing['STATUS']='REVIEW_NO_DOWNLOADED_ZIP'
    issues=frame[frame.STATUS.ne('OK')]
    frame.to_parquet(OUT/'archive_inventory.parquet',index=False)
    issues.to_parquet(OUT/'extraction_issues.parquet',index=False)
    missing.to_parquet(OUT/'missing_required_tiles.parquet',index=False)
    years=[]
    for year, group in required.groupby('REFERENCE_YEAR'):
        f=frame[frame.YEAR.eq(year)]
        years.append(dict(YEAR=int(year),REQUIRED=len(group),ZIPS=len(f),EXTRACTED_OK=int(f.STATUS.eq('OK').sum()),EXTRACTION_REVIEW=int(f.STATUS.ne('OK').sum()),MISSING_ZIPS=int(missing.REFERENCE_YEAR.eq(year).sum())))
    info=dict(status='AUDIT_COMPLETE',total_zips=len(frame),ok=int(frame.STATUS.eq('OK').sum()),
              extraction_review=len(issues),missing_required_zips=len(missing),years=years,
              verification='Completed extraction ledger + current ZIP signature + extracted file count/total size/SHP components; ZIP member and extracted CRC checks for failed partial folders only',
              source_changed=False,elapsed_seconds=time.monotonic()-started)
    (LOG/'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>13 토지피복 파일 점검</title><style>body{font:15px/1.7 Malgun Gothic;margin:36px}td,th{padding:6px}</style><h1>13 · 토지피복 파일 점검</h1>'
    page+='<p>목적: ZIP·압축 해제 결과·필요 도엽을 대조. 기존 파일 변경 없음.</p>'
    page+=pd.DataFrame(years).to_html(index=False,border=0)
    page+='<h2>압축 해제 검토</h2>'+issues.to_html(index=False,border=0)
    page+='<h2>필요 목록 중 ZIP 없음</h2>'+missing[['REFERENCE_YEAR','TILE_ID','TILE_NAME','STATUS']].to_html(index=False,border=0)
    page+='<p>전수 CRC 및 지리정보 내용 검증은 미수행. ZIP 미보유는 제공 목록 제외인지 다운로드 누락인지 별도 확인 필요.</p></html>'
    (OUT/'summary.html').write_text(page,encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False))
    print(issues.to_json(orient='records',force_ascii=False))


if __name__=='__main__':main()
