import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';

// The junction resolves only Codex bundled dependencies, not project packages.
const runtimeRequire=createRequire(path.join(os.tmpdir(),'landslide-rain-excel-20261008','loader.mjs'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const v3=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const log=path.join(v3,'log/06/full/02_collect');
const data=JSON.parse(await fs.readFile(path.join(log,'excel_review_data.json'),'utf8'));
const output=path.join(v3,'output/06/full/rainfall_missing_review.xlsx');
const wb=Workbook.create();
const summary=wb.worksheets.add('요약');
const stations=wb.worksheets.add('관측소별');
const segments=wb.worksheets.add('연속부족구간');
const detail=wb.worksheets.add('시간별상세');
const font='Arial';
function column(n){let out='';while(n){n--;out=String.fromCharCode(65+n%26)+out;n=Math.floor(n/26)}return out;}
function excelDate(value){return value==null?null:(Date.parse(value.endsWith('Z')?value:value+'Z')-Date.UTC(1899,11,30))/86400000;}
function table(sheet,headers,rows,name,widths,dateCols=[]){
  const last=column(headers.length),end=rows.length+1;
  sheet.showGridLines=false;
  sheet.getRange(`A1:${last}${end}`).values=[headers,...rows];
  sheet.getRange(`A1:${last}${end}`).format.font={name:font,size:10,color:'#243447'};
  sheet.getRange(`A1:${last}${end}`).format.rowHeight=20;
  sheet.getRange(`A1:${last}${end}`).format.verticalAlignment='center';
  sheet.getRange(`A1:${last}1`).format={fill:'#244A66',font:{name:font,size:10,bold:true,color:'#FFFFFF'},rowHeight:32,wrapText:true,horizontalAlignment:'center'};
  sheet.tables.add(`A1:${last}${end}`,true,name);
  widths.forEach((width,i)=>sheet.getRange(`${column(i+1)}1:${column(i+1)}${end}`).format.columnWidth=width);
  dateCols.forEach(i=>sheet.getRange(`${column(i)}2:${column(i)}${end}`).setNumberFormat('yyyy-mm-dd hh:mm'));
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(sheet===detail?3:1);
  return end;
}
const detailCols=[
  ['STN','관측소 번호'],['START','구간 시작 KST'],['END','구간 종료 KST'],['CATEGORY','부족 분류'],
  ['AWS_SAVED_STATE','AWS 저장 상태'],['AWS_SAVED_VALUE','AWS 저장값 mm'],['AWS_REQUEST_STATUS','AWS 요청 결과'],
  ['AWS_API_RN','AWS 원응답 RN_HR1'],['AWS_RE_SUM','AWS RE_SUM 분'],['AWS_RE_QCM','AWS RE_QCM 개'],
  ['ASOS_SAVED_STATE','ASOS 저장 상태'],['ASOS_SAVED_VALUE','ASOS 저장값 mm'],['ASOS_REQUEST_STATUS','ASOS 요청 결과'],
  ['ASOS_API_RN','ASOS 원응답 RN'],['ASOS_IR','ASOS 원응답 IR'],['REVIEW','해석 상태'],
  ['A_COUNT','연결 A 후보 수'],['A_IDS','연결 A 후보 ID'],['AWS_URL','AWS 확인 URL 인증키 미포함'],['ASOS_URL','ASOS 단일시각 확인 URL']
];
const detailRows=data.details.map(r=>detailCols.map(([key])=>['START','END'].includes(key)?excelDate(r[key]):r[key]??null));
const detailEnd=table(detail,detailCols.map(x=>x[1]),detailRows,'MissingHours',
  [13,22,22,29,18,19,24,23,21,22,18,20,24,23,22,19,21,48,65,65],[2,3]);
for(const col of ['F','H','L','N'])detail.getRange(`${col}2:${col}${detailEnd}`).setNumberFormat('0.0');
detail.getRange(`G2:G${detailEnd}`).conditionalFormats.add('containsText',{text:'요청 실패',format:{fill:'#FCE4D6',font:{color:'#9C2B1B'}}});
detail.getRange(`D2:D${detailEnd}`).conditionalFormats.add('containsText',{text:'미요청',format:{fill:'#FFF2CC'}});

const stationCols=[['STN','관측소 번호'],['FIRST','첫 부족 구간 시작'],['LAST','마지막 부족 구간 종료'],['DAYS','해당 날짜 수'],
 ['HOURS','부족 시간 수'],['MINUS99','AWS -99 반환'],['OTHER_NEGATIVE','AWS 기타 음수'],['NO_ROW','AWS 관측 행 없음'],
 ['FAILED','AWS 요청 실패'],['NOT_REQUESTED','미요청']];
table(stations,stationCols.map(x=>x[1]),data.stations.map(r=>stationCols.map(([k])=>['FIRST','LAST'].includes(k)?excelDate(r[k]):r[k])),
 'StationMissing',[13,24,24,18,19,22,23,24,23,19],[2,3]);
const segmentCols=[['STN','관측소 번호'],['START','연속 구간 시작 KST'],['END','연속 구간 종료 KST'],['HOURS','연속 부족 시간 수'],['CATEGORIES','포함된 부족 분류']];
table(segments,segmentCols.map(x=>x[1]),data.segments.map(r=>segmentCols.map(([k])=>['START','END'].includes(k)?excelDate(r[k]):r[k])),
 'ContinuousMissing',[13,25,25,23,75],[2,3]);

summary.showGridLines=false;summary.tabColor='#244A66';
summary.getRange('A1:E33').format.font={name:font,size:11,color:'#243447'};
summary.getRange('A1:E33').format.rowHeight=23;
summary.getRange('A2').values=[['강우 부족 시간 검토']];
summary.getRange('A2').format.font={name:font,size:15,bold:true};
summary.getRange('A4').values=[['확인 시각 KST']];
summary.getRange('B4').values=[[excelDate(data.as_of.slice(0,19))]];summary.getRange('B4').setNumberFormat('yyyy-mm-dd hh:mm');
summary.getRange('A5').values=[['대상']];summary.getRange('B5').values=[['A 발생 후보의 종료일 포함 이전 15일 창 중 현재 부족한 시간']];
summary.getRange('A8:D8').values=[['부족 분류','시간 수','비율','의미']];
const labels=['AWS -99 반환','AWS 기타 음수 반환','AWS 관측 행 없음','AWS 요청 실패','위치 이력 등으로 미요청'];
const meanings=['강우 항목 -99 반환. 실제 결측 여부는 미확정','다른 음수 코드 반환. 상세에서 원래 값 확인',
 '응답에 해당 관측소·시각 행이 없음','503/504 등 요청 실패. 실제 관측 결측과 구분','위치 이력 등 조건으로 자동 요청 보류'];
for(let i=0;i<labels.length;i++){
 const row=i+9;summary.getRange(`A${row}`).values=[[labels[i]]];
 summary.getRange(`B${row}`).formulas=[[`=COUNTIFS('시간별상세'!$D$2:$D$${detailEnd},A${row})`]];
 summary.getRange(`C${row}`).formulas=[[`=B${row}/$B$14`]];
 summary.getRange(`D${row}`).values=[[meanings[i]]];
}
summary.getRange('A14').values=[['합계']];summary.getRange('B14').formulas=[['=SUM(B9:B13)']];summary.getRange('C14').formulas=[['=SUM(C9:C13)']];
summary.getRange('B9:B14').setNumberFormat('#,##0');summary.getRange('C9:C14').setNumberFormat('0.00%');
summary.getRange('A8:D8').format={fill:'#244A66',font:{name:font,bold:true,color:'#FFFFFF'},rowHeight:26};
summary.getRange('A14:D14').format.font={bold:true};summary.getRange('A14:D14').format.borders={bottom:{style:'thin',color:'#9FB2C1'}};
summary.getRange('A17:B21').values=[['최초 부족 시간',data.initial_hours],['현재 부족 시간',null],['추가 확보 시간',null],['관측소 수',data.stations.length],['연속 부족 구간 수',data.segments.length]];
summary.getRange('B18').formulas=[['=B14']];summary.getRange('B19').formulas=[['=B17-B18']];summary.getRange('B17:B21').setNumberFormat('#,##0');
summary.getRange('A24').values=[['해석']];summary.getRange('B24').values=[['부족 분류는 조회 결과이며 실제 관측 결측 확정이 아님. 해석 상태는 UNRESOLVED.']];
summary.getRange('B25').values=[['빈 숫자 칸은 값 없음. 관측된 0mm와 구분. 시간 구간은 시작 포함·종료 제외.']];
summary.getRange('B26').values=[['관측소별·연속 구간은 부족 시간 내림차순. 시간별 상세는 관측소·시간순. 필터 사용 가능.']];
summary.getRange('B27').values=[['전체 RAIN_DATAS 결측률이 아닌 현재 A 표본 15일 창의 부족 목록임. B/C는 미포함.']];
summary.getRange('B29').values=[['출처: 02_rain_collection_plan.parquet, request_results.parquet, api_response_cache.sqlite 및 현재 RAIN_DATAS.']];
summary.getRange('B30').values=[['AWS URL은 요청 시각 그대로. ASOS URL은 해당 한 시간을 확인하기 위한 주소이며 원래는 기간 요청.']];
summary.getRange('B31').values=[['ASOS RN의 무강우/결측 구분과 IR 코드 해석은 검토 중. 음수값을 0으로 바꾸지 않음.']];
for(const [col,width]of[['A',31],['B',25],['C',15],['D',68],['E',3]])summary.getRange(`${col}1:${col}33`).format.columnWidth=width;
wb.recalculate();
const expected=data.categories;
for(let i=0;i<labels.length;i++)if(summary.getRange(`B${i+9}`).values[0][0]!==expected[labels[i]])throw new Error('Category total mismatch');
if(summary.getRange('B14').values[0][0]!==data.remaining_hours)throw new Error('Hour reconciliation failed');
const inspected=await wb.inspect({kind:'table',range:'요약!A8:D14',include:'values,formulas',tableMaxRows:7,tableMaxCols:4,maxChars:2400});
await fs.writeFile(path.join(log,'excel_review_inspect.json'),inspected.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1800});
await fs.writeFile(path.join(log,'excel_review_formula_scan.json'),errors.ndjson);
for(const [name,range]of[['요약','A2:D21'],['관측소별','A1:J9'],['연속부족구간','A1:E9'],['시간별상세','A1:J9']]){
 const image=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
 await fs.writeFile(path.join(log,`excel_preview_${name}.png`),new Uint8Array(await image.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(output);
console.log(JSON.stringify({output,hours:data.remaining_hours,stations:data.stations.length,segments:data.segments.length}));
