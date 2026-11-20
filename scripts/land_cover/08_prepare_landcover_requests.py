"""Reconstruct public metadata lookup and prepare application batches; never submit."""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

V3 = Path(__file__).resolve().parents[2]
LOG = V3 / 'log/08/full/08_request_structure'
OUT = V3 / 'output/08/full/08_request_structure'
LOOKUP_URL = 'https://aid.mcee.go.kr/map/getMetaList.do'
LOOKUP_PARAMS = dict(type03='', type04='', type05='32', type09='',
                     emd_code='', mapsheet_name='', mapsheet_num='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Refresh public 2021 metadata (no cookies).')
    parser.add_argument('--batch-size', type=int, default=300, help='Planning size, not a confirmed site limit.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    LOG.mkdir(parents=True, exist_ok=True)
    cache = LOG / '2021_metadata.json'
    if args.fetch:
        response = requests.post(LOOKUP_URL, data=LOOKUP_PARAMS, timeout=60)
        response.raise_for_status()
        payload = response.json()['ajaxData']
        assert len(payload['list']) == int(payload['total']), 'Incomplete national list'
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    payload = json.loads(cache.read_text(encoding='utf-8'))
    catalog = pd.DataFrame(payload['list'])
    assert len(catalog) == int(payload['total'])
    assert catalog.mapsheetNumber.is_unique and catalog.mapDtlSeq.is_unique
    requirements = pd.read_parquet(V3 / 'output/08/full/05_tile_index/required_tiles_by_year.parquet')
    requirements = requirements[requirements.REFERENCE_YEAR.eq(2021)].copy()
    merged = requirements.merge(catalog[['mapsheetNumber', 'mapsheetName', 'mapDtlSeq', 'dataSize', 'metaStdVer']],
                                left_on='TILE_ID', right_on='mapsheetNumber', how='left', validate='one_to_one')
    matched = merged[merged.mapDtlSeq.notna()].copy()
    missing = merged[merged.mapDtlSeq.isna()].copy()
    matched['mapDtlSeq'] = matched.mapDtlSeq.astype('int64')
    matched['BATCH'] = range(len(matched))
    matched['BATCH'] = matched.BATCH // args.batch_size + 1
    missing['STATUS'] = 'REVIEW_NOT_IN_2021_WEBSITE_CATALOG'
    OUT.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(OUT / '2021_matched_tiles.parquet', index=False)
    missing.to_parquet(OUT / '2021_unmatched_tiles.parquet', index=False)
    drafts = []
    for batch, frame in matched.groupby('BATCH', sort=True):
        drafts.append(dict(batch=int(batch), count=len(frame), bytes=int(frame.dataSize.sum()),
                           method='POST', url='https://aid.mcee.go.kr/req/insertInfRqtPost.do',
                           fields=dict(reqType='SearchType1', regNm=f'산사태 ABC 토지피복 2021_{batch:02d}',
                                       mapDtlSeqs=','.join(frame.mapDtlSeq.astype(str))),
                           status='DRAFT_ONLY_NEEDS_AUTHENTICATED_FORM_AND_USER_CAPTCHA'))
    (OUT / 'application_drafts.json').write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding='utf-8')
    status = dict(status='METADATA_MATCHED_APPLICATION_NOT_SUBMITTED', year=2021,
                  national_count=len(catalog), required=len(requirements), matched=len(matched),
                  unmatched=len(missing), bytes=int(matched.dataSize.sum()), batches=len(drafts),
                  batch_size_reason='Operational choice; >300 warning in client, not confirmed server limit',
                  cookies_used=False, captcha_bypassed=False, submitted=False, downloaded=False,
                  download_request_status='ATTACHMENT_LIST_KNOWN_RAON_TRANSFER_NEEDS_ACTUAL_REQUEST')
    (LOG / 'run.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>2021 요청 구조</title><style>body{font:16px/1.7 Malgun Gothic;margin:32px}td,th{padding:8px}</style><h1>2021 토지피복 요청 준비</h1>'
    page += f'<p>전국 {len(catalog):,}개 조회 완료 → 필요 {len(requirements):,}개 중 {len(matched):,}개 연결, 미연결 {len(missing)}개 검토. 합계 {status["bytes"]/1024**3:.2f}GiB.</p>'
    page += '<p>조회는 로그인 쿠키 없이 POST /map/getMetaList.do, type05=32. 화면 체크는 클라이언트 내부 선택이며 서버 신청은 mapDtlSeqs 파일 ID 목록을 쉼표로 전달합니다.</p>'
    page += f'<p>{args.batch_size}개씩 신청 초안만 생성했습니다. 이는 운영 단위이며 서버 제한으로 확정한 값이 아닙니다. 신청/다운로드는 미실행입니다. 다운로드 첨부 목록은 GET /req/download.do?infRqtSeq=신청번호, 실제 전송은 RAONK 에이전트 및 /raonkupload/handler/raonkhandler.jsp를 사용합니다. 실제 전송 요청은 아직 미검증입니다.</p>'
    page += missing[['TILE_ID', 'TILE_NAME', 'STATUS']].to_html(index=False, border=0)
    page += '<p><a href="application_drafts.json">신청 초안</a> · <a href="2021_matched_tiles.parquet">연결된 도엽</a> · <a href="2021_unmatched_tiles.parquet">미연결 도엽</a></p></html>'
    (OUT / 'summary.html').write_text(page, encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
