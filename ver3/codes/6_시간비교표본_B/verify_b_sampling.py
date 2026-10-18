"""Small boundary tests for trigger exclusion, coordinates, and unique matching."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('b_sampling',Path(__file__).with_name('03_sample_b_events.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
T=pd.Timestamp


class SamplingTests(unittest.TestCase):
    def test_interval_boundaries(self):
        periods=[(T('2023-07-07 08:00'),T('2023-07-07 10:00'))]
        self.assertTrue(b.overlaps(T('2023-07-07 09:00'),T('2023-07-07 11:00'),periods))
        self.assertFalse(b.overlaps(T('2023-07-07 10:00'),T('2023-07-07 11:00'),periods))

    def test_exclude_event_and_cumulative_trigger(self):
        a=SimpleNamespace(END_DATE=T('2023-08-20'),RAIN_START=T('2023-08-20 01:00'),
            RAIN_END_EXCLUSIVE=T('2023-08-20 03:00'),RAIN_EVENT_ID='A',STATION_LAT=37.,STATION_LON=127.)
        rows=[]
        for eid,start,end,lat in [('safe','2023-07-01 01:00','2023-07-01 03:00',37.),
                ('cum','2023-07-20 01:00','2023-07-20 03:00',37.),
                ('direct','2023-07-07 09:00','2023-07-07 11:00',37.),
                ('moved','2023-06-01 01:00','2023-06-01 03:00',38.)]:
            rows.append(dict(RAIN_EVENT_ID=eid,RAIN_START=T(start),RAIN_END_EXCLUSIVE=T(end),END_DATE=T(start).normalize(),
                EVENT_VALID=True,END_CONFIRMED=True,COORDINATES_CONFIRMED=True,LOCATION_REVIEW=False,
                LOCATION_CHANGED=False,CROSS_NETWORK_REVIEW=False,STATION_LAT=lat,STATION_LON=127.))
        pool,counts=b.candidate_keys(a,pd.DataFrame(rows),[],trigger_periods=[(T('2023-07-07 08:00'),T('2023-07-07 10:00'))])
        self.assertEqual(pool,{T('2023-07-01'):['safe']})
        self.assertEqual(counts['TRIGGER_EVENT_OR_15DAY_WINDOW_OVERLAP'],2)
        self.assertEqual(counts['DIFFERENT_COORDINATES'],1)

    def test_maximum_matching_is_unique_and_reproducible(self):
        options={'A1':[1,2],'A2':[1],'A3':[3]}
        for seed in range(10):
            matches=b.maximum_matching(options,np.random.default_rng(seed))
            self.assertEqual(len(matches),3)
            self.assertEqual(len(set(matches.values())),3)
        self.assertEqual(b.maximum_matching(options,np.random.default_rng(42)),b.maximum_matching(options,np.random.default_rng(42)))

    def test_same_version_different_network_coordinates(self):
        events=pd.DataFrame([dict(STN=1,RAIN_EVENT_ID='E1',RAIN_START=T('2023-07-01'),
            RAIN_END_EXCLUSIVE=T('2023-07-01 02:00'),DURATION_HOURS=2)])
        h=pd.DataFrame([dict(NETWORK='AWS',STN=1,LOCATION_VERSION=1,LAT=37.,LON=127.),
                        dict(NETWORK='ASOS',STN=1,LOCATION_VERSION=1,LAT=38.,LON=127.)])
        hours=pd.DataFrame(dict(STN=[1,1],time_start=[T('2023-07-01'),T('2023-07-01 01:00')],
            RAIN_SOURCE=['AWS','ASOS'],LOCATION_VERSION=[1,1],LOCATION_USABLE=[True,True],RAIN_1H_ANALYSIS=[1.,2.]))
        result=b.event_coordinates(events,hours,h)
        self.assertFalse(result.COORDINATES_CONFIRMED.iloc[0])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    b.write_json(b.V3/'log/06/full/03_b_sampling/code_verification.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun))
    raise SystemExit(0 if result.wasSuccessful() else 1)
