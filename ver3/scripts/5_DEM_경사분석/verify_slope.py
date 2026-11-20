"""Synthetic metre-based tests; never assumes real DEM height units."""
import json
import sys
import tempfile
import importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from common import V3,write_json
from dem import horn,Mosaic,summarize,require_vertical,select_dem_paths


def main():
    yy,xx=np.mgrid[:12,:12]
    flat=horn(np.zeros((12,12)),5,5,1)
    assert np.all(flat==0)
    for angle in [20,30,40]:
        z=xx*5*np.tan(np.radians(angle))
        assert np.allclose(horn(z,5,5,1),angle,atol=1e-5)
        assert np.allclose(horn(z/100,5,5,100),angle,atol=1e-5)
    diagonal=(xx+yy)*5
    assert np.allclose(horn(diagonal,5,5,1),np.degrees(np.arctan(np.sqrt(2))),atol=1e-5)
    z=xx.astype(float);z[5,5]=np.nan
    assert int(np.isnan(horn(z,5,5,1)).sum())==9
    result=summarize(np.array([10,20,30,40,np.nan]),25,[20,30,40])
    assert result['VALID_AREA_M2']==100
    assert [result[f'GE_{a}DEG_FRACTION'] for a in [20,30,40]]==[.75,.5,.25]
    assert np.isnan(summarize(np.array([np.nan]),25,[20])['GE_20DEG_FRACTION'])
    try:require_vertical({'vertical_scale_to_m':None})
    except ValueError:pass
    else:raise AssertionError('Unknown vertical unit accepted')
    with tempfile.TemporaryDirectory(prefix='ver3_slope_test_') as folder:
        rows=[]
        for tile,left in enumerate([0,50]):
            p=Path(folder)/f'{tile}.img'
            values=np.tile((left+np.arange(10)*5+2.5)*np.tan(np.radians(30)),(10,1))
            with rasterio.open(p,'w',driver='GTiff',width=10,height=10,count=1,dtype='float64',
                               crs='EPSG:5186',transform=from_origin(left,50,5,5),nodata=-9999) as dst:
                dst.write(values,1)
            rows.append(dict(path=str(p),left=left,bottom=0,right=left+50,top=50))
        values,conflicts=Mosaic(pd.DataFrame(rows)).read((-5,-5,105,55),(12,22))
        slope=horn(values,5,5,1)
        assert conflicts==0 and np.allclose(slope[1:-1,8:12],30,atol=1e-5)
        assert np.all(np.isnan(slope[:,[0,-1]]))
        # Smaller numeric sheet retains valid values; larger fills only NoData.
        p=Path(folder)/'2.img'
        with rasterio.open(rows[0]['path']) as src:
            profile=src.profile;values=src.read(1)+100
        with rasterio.open(p,'w',**profile) as dst:dst.write(values,1)
        conflict_row=dict(rows[0],path=str(p))
        bad,n=Mosaic(pd.DataFrame(rows+[conflict_row])).read((0,0,50,50),(10,10))
        assert n==100 and np.allclose(bad,values-100,rtol=0,atol=1e-12)
        # Smaller-sheet NoData is filled from a larger sheet, preserving valid zero.
        with rasterio.open(rows[0]['path'],'r+') as ds:
            smaller=ds.read(1);smaller[0,0]=-9999;smaller[0,1]=0;ds.write(smaller,1)
        filled,_=Mosaic(pd.DataFrame([conflict_row,rows[0]])).read((0,0,50,50),(10,10))
        assert filled[0,0]==values[0,0] and filled[0,1]==0
        projected=Path(folder)/'0_projected.img'
        with rasterio.open(projected,'w',**profile) as dst:dst.write(smaller,1)
        selected,replaced=select_dem_paths(folder)
        assert projected in selected and Path(rows[0]['path']) not in selected and len(replaced)==1
        # Mixed source projections are indexed in target coordinates and actually warped.
        from rasterio.warp import transform_bounds,reproject,Resampling
        p=Path(folder)/'3.img'
        with rasterio.open(p,'w',driver='GTiff',width=10,height=10,count=1,dtype='float64',
                           crs='EPSG:3857',transform=from_origin(14100000,4500000,5,5),nodata=-9999) as dst:
            dst.write(np.full((10,10),42.),1)
        with rasterio.open(p) as ds:
            b=transform_bounds(ds.crs,'EPSG:5186',*ds.bounds)
            import math
            left=math.floor(b[0]/5)*5;top=math.ceil(b[3]/5)*5
            width=math.ceil((b[2]-left)/5);height=math.ceil((top-b[1])/5)
            transformed,_=Mosaic(pd.DataFrame([dict(path=str(p),left=b[0],bottom=b[1],right=b[2],top=b[3])]),'EPSG:5186').read(
                (left,top-height*5,left+width*5,top),(height,width))
            expected=np.full((height,width),np.nan)
            reproject(ds.read(1),expected,src_transform=ds.transform,src_crs=ds.crs,src_nodata=-9999,
                      dst_transform=from_origin(left,top,5,5),dst_crs='EPSG:5186',dst_nodata=np.nan,resampling=Resampling.bilinear)
            assert np.isfinite(transformed).any() and np.allclose(transformed,expected,equal_nan=True)
    # End-to-end extraction and administrative aggregation in an isolated synthetic project.
    with tempfile.TemporaryDirectory(prefix='ver3_slope_pipeline_test_') as folder:
        import geopandas as gpd
        from shapely.geometry import box
        v=Path(folder)
        (v/'config').mkdir();(v/'dem').mkdir();(v/'output/04/full').mkdir(parents=True);(v/'log/04/full').mkdir(parents=True)
        configuration=dict(dem_root=str(v/'dem'),vertical_scale_to_m=1,vertical_unit_confirmation='SYNTHETIC_METRES_ONLY',block_size=512,thresholds_deg=[20,30,40])
        (v/'config/slope.json').write_text(json.dumps(configuration),encoding='utf-8')
        (v/'log/04/full/run.json').write_text('{"status":"COMPLETE"}',encoding='utf-8')
        pd.DataFrame(dict(SPATIAL_UNIT_ID=['RI_test'],ADMIN_GROUP_RESOLVED=[True])).to_parquet(v/'output/04/full/admin_rain_event_features.parquet')
        gpd.GeoDataFrame(dict(SPATIAL_UNIT_ID=['RI_test'],ADMIN_LEVEL=['RI'],ADMIN_CODE=['1234567890'],ADMIN_NAME=['Synthetic'],PARENT_BOUNDARY_PRESENT=[True]),geometry=[box(0,0,100,50)],crs=5186).to_parquet(v/'output/04/full/spatial_units_5186.parquet')
        for tile,left in enumerate([0,50]):
            values=np.tile((left+np.arange(10)*5+2.5)*np.tan(np.radians(30)),(10,1))
            with rasterio.open(v/'dem'/f'{tile}.img','w',driver='GTiff',width=10,height=10,count=1,dtype='float64',crs=5186,transform=from_origin(left,50,5,5),nodata=-9999) as dst:dst.write(values,1)
        previous_argv=sys.argv
        try:
            for filename,arguments in [('01_extract_slope.py',['--limit','1']),('02_summarize_slope.py',['--smoke'])]:
                path=Path(__file__).with_name(filename)
                spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.V3=v;sys.argv=[str(path),*arguments];module.main()
            table=pd.read_parquet(v/'output/05/smoke/admin_slope_features.parquet')
            assert table.VALID_CELLS.iloc[0]==144
            assert table.GE_20DEG_FRACTION.iloc[0]==1 and table.GE_30DEG_FRACTION.iloc[0]==1 and table.GE_40DEG_FRACTION.iloc[0]==0
            assert np.isclose(table.AREA_COVERAGE_FRACTION.iloc[0],.72)
        finally:
            sys.argv=previous_argv
    result=dict(status='CODE_TESTS_PASS',flat_and_known_planes=True,vertical_scale_conversion=True,
                all_9_required=True,tile_seam_halo=True,smaller_sheet_overlap_priority=True,
                smaller_nodata_fallback_preserves_zero=True,projected_file_selection=True,mixed_crs_actual_reprojection=True,
                inclusive_thresholds_and_valid_area_denominator=True,unknown_unit_refused=True,
                synthetic_end_to_end_extraction_and_aggregation=True,real_dem_slope_computed=False)
    write_json(V3/'log/05/full/verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
