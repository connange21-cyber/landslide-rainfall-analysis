"""Offline tests only: temporary partitions, recorded fixtures, no real API calls."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('02_collect_missing_rainfall.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
D = pd.Timestamp('2023-07-14')


def rain(values, hours=None):
    hours = hours or list(range(len(values)))
    starts = [D+pd.Timedelta(hours=h) for h in hours]
    return pd.DataFrame({'STN':pd.Series([90]*len(values),dtype='int64'),
                         'time_start':starts,'time_end':[s+pd.Timedelta(hours=1) for s in starts],
                         'RN_1H':pd.Series(values,dtype='float64')})


class CollectionTests(unittest.TestCase):
    def test_merge_protect_fill_conflict(self):
        old = rain([0,2,np.nan,-99])
        new = rain([9,2,0,1,5])
        merged, conflicts, filled = c.merge_observations(old,new)
        self.assertEqual(merged.RN_1H.tolist(),[0,2,0,1,5])
        self.assertEqual((len(conflicts),filled),(1,3))
        merged2,_,filled2 = c.merge_observations(merged,new)
        pd.testing.assert_frame_equal(merged,merged2)
        self.assertEqual(filled2,0)

    def test_empty_missing_duplicate(self):
        merged,_,filled = c.merge_observations(c.empty_rain(),rain([np.nan,0]))
        self.assertTrue(pd.isna(merged.RN_1H.iloc[0]))
        self.assertEqual(filled,1)
        with self.assertRaises(ValueError):
            c.merge_observations(merged,pd.concat([merged,merged]))
        c.validate_rain(merged,D)

    def test_recorded_api_layout_and_midnight(self):
        root = Path(__file__).resolve().parents[3]/'ver2/5_rain_download/rainfall_datas/sample_test'
        aws = (root/'AWS/raw/AWS_RN_202307140100.txt').read_text(encoding='utf-8')
        asos = (root/'ASOS/raw/ASOS_202307140100_202307150000.txt').read_text(encoding='cp949')
        ends = pd.date_range(D+pd.Timedelta(hours=1),periods=24,freq='h')
        a = c.parse_response(aws,'AWS',90,ends)
        self.assertAlmostEqual(a.RN_1H.iloc[0],0.6)  # HR1; 1.4 is the moving-60m maximum
        b = c.parse_response(asos,'ASOS',90,ends)
        self.assertEqual(len(b),24)
        self.assertEqual(b.time_end.iloc[-1],D+pd.Timedelta(days=1))
        c.validate_rain(b,D)
        sentinel = aws.replace('0.6','-99.9')
        self.assertTrue(pd.isna(c.parse_response(sentinel,'AWS',90,ends).RN_1H.iloc[0]))
        with self.assertRaises(ValueError): c.parse_response('<html>error</html>','AWS',90,ends)

    def test_publication_and_interrupted_index_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'rain';root.mkdir()
            log=Path(tmp)/'log';log.mkdir()
            _,filled,changed=c.publish_day(root,'AWS',D,rain([0,3]),log)
            self.assertEqual((filled,changed),(2,True))
            index=root/'AWS/index.parquet'
            pd.read_parquet(index).iloc[:0].to_parquet(index,index=False)
            _,filled,changed=c.publish_day(root,'AWS',D,rain([9,3]),log)
            self.assertEqual((filled,changed),(0,False))
            self.assertEqual(len(pd.read_parquet(index)),1)
            self.assertEqual(pd.read_parquet(c.partition_path(root,'AWS',D)).RN_1H.tolist(),[0,3])
            self.assertTrue(list((log/'backups/AWS').glob('*.parquet')))
            self.assertFalse((root/'.ver3_missing_rainfall.lock').exists())

    def test_planning_union_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for net,f in [('AWS',rain([0], [0])),('ASOS',rain([2], [1]))]:
                c.atomic_parquet(f,c.partition_path(root,net,D))
            history=pd.DataFrame([dict(STN=90,NETWORK='AWS',REVIEW_FLAG=False,
                ANALYSIS_FROM=D,ANALYSIS_TO=pd.NaT,LAST_OLD_OBSERVED=pd.NaT,FIRST_NEW_OBSERVED=pd.NaT)])
            review=pd.DataFrame([dict(STN=90,DATE=D,REVIEW_ACTION='CHECK',A_REFERENCE_IDS=['A1'])])
            plan=c.build_plan(review,history,root)
            self.assertEqual(plan.RAW_MISSING_HOURS.iloc[0],22)
            self.assertEqual(plan.MAX_AWS_CALLS.iloc[0],22)
            history['REVIEW_FLAG']=True
            plan=c.build_plan(review,history,root)
            self.assertEqual(plan.HISTORY_UNRESOLVED_HOURS.iloc[0],22)

    def test_mock_http_cache_and_credential_redaction(self):
        root=Path(__file__).resolve().parents[3]/'ver2/5_rain_download/rainfall_datas/sample_test'
        body=(root/'AWS/raw/AWS_RN_202307140100.txt').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            client=c.Client('FAKE_TEST_KEY',Path(tmp),pause=0)
            response=type('Response',(),{'status_code':200,'content':body})()
            with patch.object(client.session,'get',return_value=response) as get:
                first,_,source=client.fetch('AWS',90,[D+pd.Timedelta(hours=1)])
                self.assertEqual(source,'API')
                second,_,source=client.fetch('AWS',90,[D+pd.Timedelta(hours=1)])
                self.assertEqual(source,'CACHE')
                self.assertEqual(get.call_count,1)
                pd.testing.assert_frame_equal(first,second)
            stored=client.db.execute('SELECT params,body FROM responses').fetchone()
            self.assertNotIn('FAKE_TEST_KEY',str(stored))
            client.close()


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(CollectionTests))
    c.write_json(c.V3/'log/06/full/02_collect/verification.json',
                 dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                      real_api_calls=0,shared_rainfall_modified=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
