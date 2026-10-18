// Template: run 11_build_yearly_application.py, then paste the generated JS
// in DevTools Console on the logged-in /req/write.do page.
(() => {
  'use strict';
  const REQUIRED_BY_YEAR = /*MANIFEST*/ {};
  if (location.origin !== 'https://aid.mcee.go.kr' || location.pathname !== '/req/write.do') {
    throw new Error('로그인된 https://aid.mcee.go.kr/req/write.do 에서 실행하세요.');
  }
  if (document.getElementById('landcover-http-panel')) {
    throw new Error('신청 패널이 이미 열려 있습니다. 기존 패널을 사용하세요.');
  }
  if (!Object.keys(REQUIRED_BY_YEAR).length) throw new Error('생성된 실행 파일을 사용하세요.');
  const KEY = 'landslide-landcover-http-annual-v1';
  let ledger;
  try { ledger = JSON.parse(localStorage.getItem(KEY) || '{"attempts":[]}'); }
  catch { throw new Error('저장된 신청 기록을 읽을 수 없습니다. 기록을 먼저 확인하세요.'); }
  if (!Array.isArray(ledger.attempts)) throw new Error('신청 기록 형식 오류');
  let prepared = null, busy = false;
  const box = document.createElement('section');
  box.id = 'landcover-http-panel';
  box.style.cssText = 'position:fixed;right:18px;top:20px;z-index:999999;background:white;border:2px solid #2463a6;padding:18px;width:440px;max-height:85vh;overflow:auto;box-shadow:0 4px 24px #777;font:14px/1.6 sans-serif;color:#222';
  box.innerHTML = `<b>ABC 토지피복 연도별 HTTP 신청</b><p>연도 선택 → 목록 조회 → 보안문자 입력 → 신청.<br>파일 다운로드는 수행하지 않습니다.</p>
    <label>연도 <select id="lc-year">${Object.keys(REQUIRED_BY_YEAR).map(y => `<option>${y}</option>`).join('')}</select></label>
    <label>한 신청당 도엽 수 <input id="lc-size" type="number" min="0" step="1" value="0" style="width:80px"></label>
    <p>0 = 해당 연도 전체를 한 번에 신청. 300개 초과 시 사이트의 속도 경고가 있으며 서버 허용 한도는 미확인입니다. 필요하면 조회 전에 분할 수를 입력하세요.</p>
    <button id="lc-prepare">1. 연도 목록 조회</button>
    <pre id="lc-status" style="white-space:pre-wrap">준비 전</pre>
    <label>현재 사이트 보안문자 <input id="lc-captcha" autocomplete="off" style="width:130px"></label>
    <p>사이트 아래쪽의 보안문자를 입력하세요. 다음 신청 전 사이트의 ‘새로고침’으로 새 문자를 받으세요.</p>
    <button id="lc-submit" disabled>2. 다음 신청 POST</button>
    <button id="lc-export">기록 JSON 저장</button>
    <p>신청 결과가 불확실하면 재요청을 차단합니다. 신청 내역을 확인하고 기록을 보존하세요.</p>`;
  document.body.append(box);
  const $ = id => box.querySelector(id);
  const status = text => { $('#lc-status').textContent = text; console.info('[토지피복 HTTP]', text); };
  const save = () => localStorage.setItem(KEY, JSON.stringify(ledger));
  const post = async (path, body) => {
    const response = await fetch(path, {
      method: 'POST', credentials: 'same-origin', redirect: 'error',
      headers: {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Requested-With': 'XMLHttpRequest'},
      body: new URLSearchParams(body).toString(),
    });
    const raw = await response.text();
    let payload;
    try { payload = JSON.parse(raw); }
    catch { throw new Error(`HTTP ${response.status}: JSON이 아닌 응답. 로그인 및 Network 확인 필요`); }
    if (!response.ok || !payload.ajaxData) throw new Error(`HTTP ${response.status}: 응답 구조 확인 필요`);
    return payload.ajaxData;
  };
  const refresh = () => {
    if (!prepared) return;
    const confirmed = new Set(ledger.attempts.filter(a => a.year === prepared.year && a.state === 'SERVER_SUCCESS').flatMap(a => a.tiles));
    const uncertain = ledger.attempts.some(a => a.year === prepared.year && ['SENDING', 'UNRESOLVED'].includes(a.state));
    const remaining = prepared.matched.filter(t => !confirmed.has(t.number));
    $('#lc-submit').disabled = busy || uncertain || !remaining.length;
    status(`${prepared.year}년: 필요 ${prepared.required.length}개 / 제공 ${prepared.matched.length}개 / 미조회 ${prepared.missing.length}개\n이 실행 도구에서 서버 성공 확인 ${prepared.matched.length - remaining.length}개 / 남은 ${remaining.length}개\n${uncertain ? '결과 미확정 요청이 있어 차단됨. 신청 내역 확인 필요.' : '다음 신청: ' + Math.min(prepared.size || remaining.length, remaining.length) + '개'}\n미조회: ${prepared.missing.join(', ') || '없음'}`);
    return remaining;
  };
  $('#lc-prepare').onclick = async () => {
    if (busy) return;
    busy = true; $('#lc-submit').disabled = true;
    try {
      const year = Number($('#lc-year').value);
      const size = Number($('#lc-size').value);
      if (!Number.isInteger(size) || size < 0) throw new Error('분할 수는 0 이상의 정수여야 합니다.');
      if (!document.querySelector('select#idx_type')?.selectedOptions[0]?.textContent.includes('세분류')) {
        throw new Error('사이트의 도엽종류를 세분류로 선택한 뒤 조회하세요.');
      }
      const options = [...(document.querySelector('select#mapMstSeq')?.options || [])];
      const choices = options.filter(o => o.textContent.includes(`[${year}]`) && o.textContent.includes('전국'));
      if (choices.length !== 1) throw new Error(`${year}년 전국 사업 ID를 확정할 수 없습니다.`);
      status(`${year}년 전국 제공 목록 조회 중…`);
      const data = await post('/map/getMetaList.do', {
        type03: '', type04: '', type05: choices[0].value, type09: '',
        emd_code: '', mapsheet_name: '', mapsheet_num: '',
      });
      if (!Array.isArray(data.list) || data.list.length !== Number(data.total)) throw new Error('전국 목록 응답이 불완전합니다.');
      const catalog = new Map();
      const ids = new Set();
      for (const row of data.list) {
        const number = String(row.mapsheetNumber), id = String(row.mapDtlSeq);
        if (!/^\d{8}$/.test(number) || !/^\d+$/.test(id) || catalog.has(number) || ids.has(id)) throw new Error('목록 ID 형식 또는 중복 오류');
        catalog.set(number, {number, id, name: row.mapsheetName}); ids.add(id);
      }
      const required = REQUIRED_BY_YEAR[year].split(',');
      const matched = required.filter(n => catalog.has(n)).map(n => catalog.get(n));
      const missing = required.filter(n => !catalog.has(n));
      if (!matched.length) throw new Error('신청할 수 있는 도엽이 없습니다.');
      prepared = {year, size, required, matched, missing};
      ledger.catalogs ||= {};
      ledger.catalogs[year] = {checkedAt: new Date().toISOString(), required: required.length, matched: matched.length, missing};
      save();
      console.table(matched); if (missing.length) console.warn('REVIEW_NOT_IN_CATALOG', missing);
    } catch (error) { prepared = null; status(error.message); }
    finally { busy = false; if (prepared) refresh(); }
  };
  $('#lc-submit').onclick = async () => {
    if (busy || !prepared) return;
    const remaining = refresh();
    if ($('#lc-submit').disabled || !remaining?.length) return;
    const captcha = $('#lc-captcha').value.trim();
    if (!captcha) return status('현재 사이트 보안문자를 먼저 입력하세요.');
    const batch = remaining.slice(0, prepared.size || remaining.length);
    const year = prepared.year;
    const number = ledger.attempts.filter(a => a.year === year).length + 1;
    const name = `산사태 ABC 토지피복 ${year}_HTTP_${String(number).padStart(2, '0')}_${batch.length}개`;
    if (!confirm(`${name}\n필요 도엽 ${batch.length}개를 POST 1회로 신청합니다.\n미조회 ${prepared.missing.length}개는 포함되지 않습니다.\n등록할까요?`)) return;
    const entry = {year, name, tiles: batch.map(t => t.number), ids: batch.map(t => t.id), startedAt: new Date().toISOString(), state: 'SENDING'};
    busy = true; $('#lc-submit').disabled = true;
    try {
      ledger.attempts.push(entry); save(); // Persist before mutation; failure stops the request.
      status(`${year}년 ${batch.length}개 신청 POST 중… 중복 실행하지 마세요.`);
      const date = new Intl.DateTimeFormat('sv-SE', {timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit'}).format(new Date()).replaceAll('-', '');
      const data = await post('/req/insertInfRqtPost.do', {
        reqType: 'SearchType1', infRqtSeq: '0', infRqtSecudtaSeq: '', regNm: name,
        mapDtlSeqs: batch.map(t => t.id).join(','), telno: '', instInfo: '', vistDay: date,
        reqstMattr: '', fileIds: '', userCaptcha: captcha,
      });
      entry.state = data.success === true ? 'SERVER_SUCCESS' : 'SERVER_REJECTED';
      entry.applicationId = data.infrqtseq ?? data.infRqtSeq ?? null;
      entry.message = data.msg;
      entry.finishedAt = new Date().toISOString(); save();
      console.log('신청 서버 응답', {year, name, count: batch.length, state: entry.state, applicationId: entry.applicationId, message: entry.message});
      if (entry.state === 'SERVER_REJECTED') alert(`서버가 신청을 거절했습니다: ${entry.message}\n신청 내역을 확인한 뒤 원인을 해결하세요. 자동 재시도하지 않습니다.`);
      else alert(`신청 성공 응답: ${year}년 ${batch.length}개\n신청번호: ${entry.applicationId ?? '응답에 없음'}\n신청 내역에서도 확인하세요. 다음 요청 전 보안문자를 새로 받으세요.`);
    } catch (error) {
      entry.state = 'UNRESOLVED'; entry.error = error.message;
      try { save(); } catch { /* Keep in-memory state; never resubmit automatically. */ }
      alert('결과 미확정: ' + error.message + '\n신청 내역 확인 전 이 연도 재요청은 차단됩니다. 기록 JSON을 저장하세요.');
    } finally { $('#lc-captcha').value = ''; busy = false; refresh(); }
  };
  $('#lc-export').onclick = () => {
    const blob = new Blob([JSON.stringify(ledger, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = '11_yearly_application_log.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  console.info('연도별 신청 패널 준비. 보안문자·쿠키는 기록하지 않습니다. 기록 JSON은 프로젝트 ver3/log/08/full/11_yearly_application/에 보관하세요.');
})();
