"""Embed existing year/tile requirements into a browser HTTP application runner."""
import json
from pathlib import Path

import pandas as pd

V3 = Path(__file__).resolve().parents[2]


def main():
    source = V3 / 'output/08/full/05_tile_index/required_tiles_by_year.parquet'
    frame = pd.read_parquet(source)
    assert not frame.duplicated(['REFERENCE_YEAR', 'TILE_ID']).any()
    manifest = {}
    for year, group in frame.groupby('REFERENCE_YEAR', sort=True):
        numbers = sorted(group.TILE_ID.astype(str).tolist())
        assert all(len(n) == 8 and n.isdigit() for n in numbers)
        manifest[str(int(year))] = ','.join(numbers)
    assert set(manifest) == {'2021', '2022', '2023', '2024', '2025'}
    template = Path(__file__).with_name('11_apply_yearly_landcover_http.js').read_text(encoding='utf-8')
    marker = '/*MANIFEST*/ {}'
    assert template.count(marker) == 1
    script = template.replace(marker, json.dumps(manifest, ensure_ascii=False))
    out = V3 / 'output/08/full/11_yearly_application'
    log = V3 / 'log/08/full/11_yearly_application'
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    (out / 'apply_yearly_http.js').write_text(script, encoding='utf-8')
    counts = {year: len(value.split(',')) for year, value in manifest.items()}
    (log / 'build.json').write_text(json.dumps(dict(status='CODE_GENERATED_NOT_SUBMITTED',
        required_counts=counts, source=str(source), cookies_stored=False,
        captcha_stored=False, network_calls=0), ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ''.join(f'<tr><td>{year}</td><td>{count:,}</td></tr>' for year, count in counts.items())
    (out / 'summary.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>연도별 토지피복 HTTP 신청</title>
<style>body{font:16px/1.7 "Malgun Gothic";max-width:960px;margin:36px auto}td,th{padding:8px 24px;border-bottom:1px solid #ddd}</style>
<h1>연도별 토지피복 HTTP 신청 코드</h1><p>상태: 코드 생성 완료. 전체 신청은 아직 실행하지 않았습니다.</p>
<table><tr><th>연도</th><th>필요 도엽</th></tr>''' + rows + '''</table>
<p>로그인된 https://aid.mcee.go.kr/req/write.do 에서 도엽종류를 세분류로 선택하고 F12 → Console에 <a href="apply_yearly_http.js">apply_yearly_http.js</a> 전체 내용을 붙여넣습니다.</p>
<ol><li>패널에서 연도 선택 → 목록 조회. 사이트 전국 목록에서 도엽번호를 내부 ID로 연결합니다.</li>
<li>미조회 도엽과 신청 수를 확인하고 현재 사이트 보안문자 입력 → 다음 신청 POST.</li>
<li>다음 연도 신청 전 사이트 보안문자를 새로고침합니다. 같은 절차를 반복합니다.</li>
<li>신청 내역에서 등록을 확인하고 기록 JSON 저장 버튼으로 기록을 내려받아 ver3/log/08/full/11_yearly_application/에 보관합니다.</li></ol>
<p>한 신청당 도엽 수 0은 연도 전체 1회 요청입니다. 300개 초과의 사이트 속도 경고와 별개로 서버 허용 한도는 미확인입니다. 필요하면 조회 전에 분할 수를 지정할 수 있습니다. 오류 시 자동으로 쪼개거나 재요청하지 않습니다.</p>
<p>미조회 도엽은 REVIEW로 기록합니다. 같은 도구의 성공 기록은 다음 요청에서 제외하며 결과 미확정 요청은 차단합니다. 기록은 이 브라우저의 localStorage에 보관됩니다. 다른 브라우저나 기록 삭제 후에는 중복 방지 기록이 이어지지 않습니다. 이전 시험 신청은 자동 제외하지 않습니다.</p>
<p>쿠키·보안문자는 기록하지 않으며 다운로드는 수행하지 않습니다. 코드의 문법 및 목록 일치 검증과 실제 서버 신청 성공은 별개입니다.</p></html>''', encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
