"""Read-only rainfall audit extraction for the companion Excel builder. No API calls."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import V3, fingerprint, write_json


def main():
    out=V3/'output/06/full'
    log=V3/'log/06/full/02_collect'
    plan_path=out/'02_rain_collection_plan.parquet'
    requests_path=log/'request_results.parquet'
    cache_path=log/'api_response_cache.sqlite'
    paths=[plan_path,requests_path,cache_path,V3/'config/rainfall.json']
    before=[fingerprint(p) for p in paths]
    plan=pd.read_parquet(plan_path)
    requests=pd.read_parquet(requests_path)
    root=Path(json.loads(paths[-1].read_text(encoding='utf-8'))['rain_root'])
    raw,read_files={},[]
    for date in sorted(plan.DATE.unique()):
        date=pd.Timestamp(date)
        for network in ['AWS','ASOS']:
            p=root/network/'hourly'/date.strftime('%Y/%m/%d')/f'data_{date:%Y%m%d}.parquet'
            if not p.exists():continue
            stamp=p.stat()
            frame=pd.read_parquet(p)
            for stn,end,value in frame[['STN','time_end','RN_1H']].itertuples(index=False,name=None):
                raw[(network,int(stn),pd.Timestamp(end))]=None if pd.isna(value) else float(value)
            after=p.stat()
            assert (stamp.st_size,stamp.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            read_files.append(dict(path=str(p),bytes=stamp.st_size,mtime_ns=stamp.st_mtime_ns))
    api={}
    db=sqlite3.connect(cache_path.resolve().as_uri()+'?mode=ro',uri=True)
    for network,params,body in db.execute('SELECT network,params,body FROM responses'):
        params=json.loads(params)
        for line in body.splitlines():
            if not line.strip() or line.lstrip().startswith('#'):continue
            values=line.split()
            assert len(values)==(14 if network=='AWS' else 46)
            stn,end=int(values[1]),pd.to_datetime(values[0],format='%Y%m%d%H%M')
            if stn!=int(params['stn']):continue
            api[(network,stn,end)]=dict(rain=float(values[6 if network=='AWS' else 15]),
                ir=int(values[44]) if network=='ASOS' else None,
                re_sum=int(values[2]) if network=='AWS' else None,
                re_qcm=int(values[3]) if network=='AWS' else None)
    db.close()
    rows=[]
    for row in plan.itertuples():
        stn,date=int(row.STN),pd.Timestamp(row.DATE)
        req=requests[requests.STN.eq(stn)&requests.DATE.eq(date)]
        for end in row.MISSING_END_TIMES:
            end=pd.Timestamp(end);start=end-pd.Timedelta(hours=1)
            valid=lambda x:x is not None and np.isfinite(x) and x>=0
            if any(valid(raw.get((n,stn,end))) for n in ['AWS','ASOS']):continue
            record={'STN':stn,'DATE':date,'START':start,'END':end}
            for net in ['AWS','ASOS']:
                key=(net,stn,end);value=raw.get(key)
                record[net+'_SAVED_STATE']=('행 없음' if key not in raw else '값 공란' if value is None
                    else '음수' if value<0 else '비유한값' if not np.isfinite(value) else '유효값')
                record[net+'_SAVED_VALUE']=value
                observation=api.get(key,{})
                candidate=req[req.NETWORK.eq(net)&req.FIRST_END.le(end)&req.LAST_END.ge(end)]
                eligible=end in set(pd.Timestamp(t) for t in getattr(row,net+'_END_TIMES'))
                requested=len(candidate)>0 and eligible
                status='미요청'
                if requested:
                    latest=candidate.iloc[-1]
                    status=('요청 실패 '+str(latest.get('ERROR','')) if latest.STATUS=='FAILED' else
                        '관측 행 없음' if not observation else '음수 반환' if observation['rain']<0 else '유효값 반환')
                record[net+'_REQUEST_STATUS']=status
                record[net+'_API_RN']=observation.get('rain') if requested else None
                if net=='AWS':
                    record['AWS_RE_SUM']=observation.get('re_sum') if requested else None
                    record['AWS_RE_QCM']=observation.get('re_qcm') if requested else None
                else:record['ASOS_IR']=observation.get('ir') if requested else None
            value=record['AWS_API_RN'];status=record['AWS_REQUEST_STATUS']
            record['CATEGORY']=('AWS -99 반환' if value==-99 else 'AWS 기타 음수 반환' if value is not None and value<0
                else 'AWS 관측 행 없음' if status=='관측 행 없음' else 'AWS 요청 실패' if status.startswith('요청 실패')
                else '위치 이력 등으로 미요청')
            record['REVIEW']='UNRESOLVED'
            record['A_COUNT']=len(row.A_REFERENCE_IDS)
            record['A_IDS']=', '.join(row.A_REFERENCE_IDS)
            record['AWS_URL']=f'https://apihub.kma.go.kr/api/typ01/url/awsh.php?var=RN&tm={end:%Y%m%d%H%M}&stn={stn}&disp=0&help=1&authKey={{인증키}}'
            record['ASOS_URL']=(f'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={end:%Y%m%d%H%M}&tm2={end:%Y%m%d%H%M}&stn={stn}&help=1&authKey={{인증키}}'
                if len(getattr(row,'ASOS_END_TIMES')) else None)
            rows.append(record)
    f=pd.DataFrame(rows).sort_values(['STN','START']).reset_index(drop=True)
    assert not f.duplicated(['STN','START','END']).any()
    summary=[]
    for stn,g in f.groupby('STN'):
        summary.append(dict(STN=int(stn),FIRST=g.START.min(),LAST=g.END.max(),DAYS=g.DATE.nunique(),
            HOURS=len(g),MINUS99=g.CATEGORY.eq('AWS -99 반환').sum(),
            OTHER_NEGATIVE=g.CATEGORY.eq('AWS 기타 음수 반환').sum(),NO_ROW=g.CATEGORY.eq('AWS 관측 행 없음').sum(),
            FAILED=g.CATEGORY.eq('AWS 요청 실패').sum(),NOT_REQUESTED=g.CATEGORY.eq('위치 이력 등으로 미요청').sum()))
    stations=pd.DataFrame(summary).sort_values(['HOURS','STN'],ascending=[False,True])
    segments=[]
    for stn,g in f.groupby('STN'):
        runs=g.START.diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _,run in g.groupby(runs):
            segments.append(dict(STN=int(stn),START=run.START.min(),END=run.END.max(),HOURS=len(run),
                CATEGORIES=', '.join(sorted(run.CATEGORY.unique()))))
    segments=pd.DataFrame(segments).sort_values(['HOURS','STN','START'],ascending=[False,True,True])
    assert int(stations.HOURS.sum())==len(f)==int(segments.HOURS.sum())
    after=[fingerprint(p) for p in paths]
    assert before==after,'Inputs changed during export'
    categories=f.CATEGORY.value_counts().to_dict()
    data=dict(as_of=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),initial_hours=int(plan.RAW_MISSING_HOURS.sum()),
        remaining_hours=len(f),recovered_hours=int(plan.RAW_MISSING_HOURS.sum())-len(f),categories=categories,
        details=f.to_dict('records'),stations=stations.to_dict('records'),segments=segments.to_dict('records'))
    def encode(value):
        if isinstance(value,pd.Timestamp):return value.isoformat()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value))
    # Replace NaN with JSON null; numeric zero remains zero.
    data=json.loads(json.dumps(data,default=encode,ensure_ascii=False))
    def clean(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,list):return [clean(x) for x in value]
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        return value
    write_json(log/'excel_review_data.json',clean(data))
    write_json(log/'excel_review_verification.json',dict(status='EXTRACTION_PASS',remaining_hours=len(f),
        stations=len(stations),station_dates=len(f[['STN','DATE']].drop_duplicates()),segments=len(segments),
        categories=categories,api_requests=0,shared_rainfall_modified=False,inputs=before,read_files=read_files))
    print(json.dumps(dict(hours=len(f),stations=len(stations),segments=len(segments),categories=categories),ensure_ascii=False,default=encode))


if __name__=='__main__':main()
