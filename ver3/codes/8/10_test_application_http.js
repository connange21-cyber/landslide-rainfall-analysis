// Run once in the DevTools Console of the authenticated request page.
// Creates one NEW application for ten required 2021 tiles. Does not download files.
// Uses the current browser session; never exports cookies or stores CAPTCHA.
(async () => {
  'use strict';
  const expectedOrigin = 'https://aid.mcee.go.kr';
  if (location.origin !== expectedOrigin || location.pathname !== '/req/write.do') {
    throw new Error('로그인한 https://aid.mcee.go.kr/req/write.do 에서 실행하세요.');
  }
  if (window.__landcoverHttpTest10Attempted) {
    throw new Error('이미 요청을 시도했습니다. 신청 내역을 확인하기 전 재실행하지 마세요.');
  }
  const captcha = document.querySelector('#userCaptcha')?.value.trim()
    || prompt('현재 신청 화면에 표시된 보안문자를 입력하세요.');
  if (!captcha) throw new Error('보안문자 입력 취소: 요청하지 않았습니다.');
  // Required-list sample, seed 202110; excludes previously tested tiles.
  const tiles = [
    { number: '35607047', name: '부안047', id: 86509 },
    { number: '35715003', name: '하동003', id: 88720 },
    { number: '35715056', name: '하동056', id: 88773 },
    { number: '35814004', name: '함안004', id: 90221 },
    { number: '36616031', name: '한산031', id: 92191 },
    { number: '36802085', name: '단양085', id: 94045 },
    { number: '36813082', name: '김천082', id: 95142 },
    { number: '37716067', name: '엄정067', id: 98581 },
    { number: '37805030', name: '청일030', id: 98994 },
    { number: '37914021', name: '죽변021', id: 100417 },
  ];
  console.table(tiles);
  const regNm = '산사태 ABC HTTP 재현시험_2021_신규10개';
  const fields = new URLSearchParams({
    reqType: 'SearchType1', infRqtSeq: '0', infRqtSecudtaSeq: '',
    regNm, mapDtlSeqs: tiles.map(tile => tile.id).join(','),
    telno: '', instInfo: '',
    vistDay: document.querySelector('#vistDay')?.value.replaceAll('-', '') || '20261008',
    reqstMattr: '', fileIds: '', userCaptcha: captcha,
  });
  if (!confirm(`${regNm}\n${tiles.map(tile => tile.name).join('·')}\n2021년 필요 도엽 ${tiles.length}개를 새로 신청합니다.\n체크박스 선택과 관계없이 POST 요청 1회를 보낼까요?`)) {
    return console.info('취소: 요청하지 않았습니다.');
  }
  window.__landcoverHttpTest10Attempted = true;
  console.info(`HTTP 신청 POST 시작: 도엽 ${tiles.length}개. 응답 전 중복 실행하지 마세요.`);
  try {
    const response = await fetch('/req/insertInfRqtPost.do', {
      method: 'POST', credentials: 'same-origin', redirect: 'error',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
      },
      body: fields.toString(),
    });
    const raw = await response.text();
    let payload;
    try { payload = JSON.parse(raw); }
    catch { throw new Error(`HTTP ${response.status}: JSON 응답이 아닙니다. Network의 Response와 로그인 상태를 확인하세요.`); }
    const data = payload.ajaxData;
    console.log('HTTP 재현 결과', {
      httpStatus: response.status, applicationName: regNm, tileCount: tiles.length,
      success: data?.success, message: data?.msg,
      applicationId: data?.infrqtseq ?? data?.infRqtSeq ?? null,
    });
    if (!response.ok || data?.success !== true) {
      console.warn('신청 성공 미확인. Network 응답을 확인하세요. 자동 재시도하지 않습니다.');
      return;
    }
    console.info('서버가 신청 성공을 반환했습니다. /req/list.do 에서 위 신청명을 확인하세요. 다운로드는 하지 않았습니다.');
  } catch (error) {
    console.error('요청 결과 미확정:', error.message,
      '신청 내역을 먼저 확인하세요. 중복 방지를 위해 자동 재시도하지 않습니다.');
  }
})();
