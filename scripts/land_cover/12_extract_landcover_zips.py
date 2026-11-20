"""Extract yearly landcover ZIPs; preserve originals and resume completed archives."""
import argparse
import html
import json
import os
import shutil
import sqlite3
import stat
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path, PurePosixPath

V3 = Path(__file__).resolve().parents[2]
SOURCE = V3 / 'output/08/full/11_yearly_application'
OUTPUT = V3 / 'output/08/full/12_extracted'
LOG = V3 / 'log/08/full/12_extracted'


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def members(archive):
    result = []; seen = set()
    for info in archive.infolist():
        name = info.filename.replace('\\', '/')
        parts = PurePosixPath(name).parts
        if not parts or name.startswith('/') or any(p in ('..', '.') or ':' in p for p in parts):
            raise ValueError(f'Unsafe ZIP path: {name}')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'Symlink ZIP entry: {name}')
        key = '/'.join(parts).casefold()
        if key in seen:raise ValueError(f'Duplicate ZIP path: {name}')
        seen.add(key)
        if info.flag_bits & 1:raise ValueError('Encrypted ZIP is unsupported')
        if not info.is_dir():result.append((info, Path(*parts)))
    if not result:raise ValueError('ZIP contains no files')
    return result


def safe_target(folder, relative):
    target = folder / relative
    if not target.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f'Output path escapes archive folder: {relative}')
    return target


def matches(path, info, check_crc=False):
    if not path.is_file() or path.stat().st_size != info.file_size:return False
    if check_crc:
        crc = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):crc = zlib.crc32(block, crc)
        if crc & 0xffffffff != info.CRC:return False
    return True


def extract_one(source, destination, completed=False, verify_existing=False):
    """Atomic archive publication; interrupted staging files can be resumed."""
    if destination.is_symlink():raise ValueError('Output archive folder is a symlink')
    with zipfile.ZipFile(source) as archive:
        entries = members(archive)
        if destination.exists():
            # Unrecorded final folders get a CRC check before adoption.
            if all(matches(safe_target(destination, rel), info,
                           verify_existing or not completed) for info, rel in entries):
                return 'SKIPPED', len(entries), sum(i.file_size for i, _ in entries)
            raise FileExistsError(f'Existing output differs or is incomplete: {destination}')
        staging = destination.with_name(destination.name + '.partial')
        if staging.is_symlink():raise ValueError('Staging folder is a symlink')
        staging.mkdir(parents=True, exist_ok=True)
        for info, relative in entries:
            target = safe_target(staging, relative)
            if target.exists():
                if matches(target, info, True):continue
                raise FileExistsError(f'Existing staging file differs: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            # ZIP reader validates CRC as each member reaches EOF.
            with archive.open(info) as incoming, temporary.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size:raise ValueError('Extracted size mismatch')
            temporary.replace(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return 'EXTRACTED', len(entries), sum(i.file_size for i, _ in entries)


def summary(out, counts, status):
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>' for k,v in counts.items())
    (out / 'summary.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>12 토지피복 ZIP 압축 해제</title>
<style>body{font:16px/1.7 "Malgun Gothic";margin:36px}td{padding:8px 24px;border-bottom:1px solid #ddd}</style>
<h1>12 · 토지피복 ZIP 압축 해제</h1><p>목적: 연도별 다운로드 ZIP을 원본 보존 상태로 풀어 후속 분석 입력을 준비합니다.</p>
<p>상태: ''' + html.escape(status) + '</p><table>' + rows + '''</table>
<p>저장: 12_extracted/연도/ZIP파일명(확장자 제외)/원래 내부 파일 경로.</p>
<p>완료 ZIP은 재실행 시 건너뜁니다. 신규 압축 해제는 ZIP CRC를 검증합니다. 기본 재개 검증은 완료된 파일의 존재·크기 확인이며 --verify-existing 사용 시 CRC도 재검증합니다.</p>
<p>압축 해제 성공은 토지피복 분류·좌표·내용의 분석 검증 완료를 의미하지 않습니다. 상세 기록은 log/08/full/12_extracted/에 있습니다.</p></html>''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=range(2021,2026), help='Only this year; default all years')
    parser.add_argument('--limit', type=int, default=0, help='First N ZIPs for a small trial; 0=all')
    parser.add_argument('--verify-existing', action='store_true', help='Also CRC-check completed extracted files')
    args = parser.parse_args()
    if args.limit < 0:parser.error('--limit must be nonnegative')
    if not SOURCE.is_dir():raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True);LOG.mkdir(parents=True, exist_ok=True)
    lock = LOG / '.extract_running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f'pid={os.getpid()}'.encode());os.close(fd)
    db = None;counts = dict(total=0, extracted=0, skipped=0, failed=0);status='RUNNING'
    started=time.monotonic()
    try:
        print('[12 압축 해제] ZIP 목록 확인 중…', flush=True)
        files = sorted(p for p in (SOURCE / str(args.year) if args.year else SOURCE).rglob('*')
                       if p.is_file() and p.suffix.lower()=='.zip')
        if args.limit:files=files[:args.limit]
        if not files:raise FileNotFoundError('No ZIP files found')
        counts['total']=len(files)
        db=sqlite3.connect(LOG / 'extraction_state.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS completed (source TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, files INTEGER, bytes INTEGER)')
        print(f'[12 압축 해제] 대상 {len(files):,}개 | 저장 {OUTPUT}',flush=True)
        last=0
        with (LOG/'extraction.jsonl').open('a',encoding='utf-8') as records:
            for i, source in enumerate(files,1):
                relative=source.relative_to(SOURCE);destination=OUTPUT / relative.parent / source.stem
                st=source.stat();previous=db.execute('SELECT size,mtime_ns FROM completed WHERE source=?',(relative.as_posix(),)).fetchone()
                completed=previous==(st.st_size,st.st_mtime_ns)
                record=dict(source=str(relative),destination=str(destination.relative_to(OUTPUT)))
                try:
                    if previous and not completed:raise ValueError('Previously extracted source ZIP changed; review before replacing output')
                    state,nfiles,nbytes=extract_one(source,destination,completed,args.verify_existing)
                    db.execute('INSERT OR REPLACE INTO completed VALUES (?,?,?,?,?)',(relative.as_posix(),st.st_size,st.st_mtime_ns,nfiles,nbytes));db.commit()
                    counts['extracted' if state=='EXTRACTED' else 'skipped']+=1
                    record.update(status=state,files=nfiles,bytes=nbytes)
                except Exception as error:
                    counts['failed']+=1;record.update(status='FAILED',error_type=type(error).__name__,error=str(error))
                    print(f'[실패] {relative}: {error}',flush=True)
                records.write(json.dumps(record,ensure_ascii=False)+'\n');records.flush()
                now=time.monotonic()
                if i==1 or i==len(files) or now-last>=2:
                    elapsed=now-started;eta=elapsed*(len(files)-i)/i
                    print(f'[12 압축 해제] {i:,}/{len(files):,} ({i/len(files):.1%}) | 경과 {elapsed/60:.1f}분 | 남은 약 {eta/60:.1f}분 | 신규 {counts["extracted"]:,} · 건너뜀 {counts["skipped"]:,} · 실패 {counts["failed"]:,} | {source.name}',flush=True)
                    write_json(LOG/'progress.json',dict(status='RUNNING',done=i,**counts));last=now
        status='COMPLETE_WITH_REVIEW' if counts['failed'] else 'COMPLETE'
    except KeyboardInterrupt:
        status='CANCELLED';print('\n[중단] 완료 ZIP은 보존했습니다. 같은 명령으로 재개하세요.',flush=True)
    except BaseException:
        status='FAILED';raise
    finally:
        if db is not None:db.close()
        write_json(LOG/'run.json',dict(status=status,finished=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-started,**counts))
        summary(OUTPUT,counts,status);lock.unlink(missing_ok=True)
    if counts['failed']:raise SystemExit(1)


if __name__=='__main__':main()
