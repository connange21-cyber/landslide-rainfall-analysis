"""Run inside ArcGIS Pro's Python window; add GDB layers and enable hourly time."""
from pathlib import Path
import json
import arcpy

V3=Path('/data/project/ver3')
OUT=V3/'output/12/full/01_time_slider'
LAYER_OUT=V3/'output/12/full/02_time_layers'
LOG=V3/'log/12/full/02_time_layers'
GDB=OUT/'abc_rainfall_examples.gdb'
if not GDB.exists():raise FileNotFoundError(GDB)
project=arcpy.mp.ArcGISProject('CURRENT')
maps=project.listMaps('12 ABC 시간강우')
map_view=maps[0] if maps else project.createMap('12 ABC 시간강우')
map_view.spatialReference=arcpy.SpatialReference(5186)
LAYER_OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
for name in ['admin_boundaries','A_hourly','B_hourly','C_hourly']:
    source=str(GDB/name)
    existing=[l for l in map_view.listLayers() if l.supports('DATASOURCE') and Path(l.dataSource)==Path(source)]
    layer=existing[0] if existing else map_view.addDataFromPath(source)
    if name.endswith('_hourly'):
        layer.enableTime('START_TIME')
        if not layer.isTimeEnabled:raise RuntimeError(f'Time enable failed: {name}')
        layer.time.timeZone='Korea Standard Time'
        layer.time.daylightSavings=False
        layer.time.timeStepInterval=1
        layer.time.timeStepIntervalUnits='HOURS'
        arcpy.management.SaveToLayerFile(layer,str(LAYER_OUT/f'{name}_time.lyrx'),'ABSOLUTE')
        print(f'{name}: START_TIME, 한국시간, 1시간 간격 설정 완료')
print('시간 슬라이더를 켜고 RN_1H로 심볼을 설정하세요. CASE_NO = 1~5로 묶음을 선택할 수 있습니다.')
(LOG/'run.json').write_text(json.dumps(dict(status='COMPLETE',map='12 ABC 시간강우',gdb=str(GDB),
    output=str(LAYER_OUT),time_field='START_TIME',time_zone='Korea Standard Time',step_hours=1),ensure_ascii=False,indent=2),encoding='utf-8')
