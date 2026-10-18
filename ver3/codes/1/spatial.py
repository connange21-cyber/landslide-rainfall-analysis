"""New ver3 spatial linking; preserves ver2 selection rules, no DEM reads."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, box
from pyproj import CRS, Transformer


def norm(value):
    if pd.isna(value):
        return ''
    text = str(value).replace('강원특별자치도', '강원도').replace('전북특별자치도', '전라북도')
    return re.sub(r'\s+', ' ', text).strip()


def polygon_only(geom):
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    return shapely.union_all([polygon_only(x) for x in geom.geoms])


def read_layer(path, level, code, name, encoding=None):
    layer = gpd.read_file(path, encoding=encoding)
    prj = path.with_suffix('.prj')
    if prj.exists():
        layer = layer.set_crs(CRS.from_wkt(prj.read_text(encoding='utf-8-sig')), allow_override=True)
    xml = Path(str(path) + '.xml')
    if xml.exists():
        for el in ET.parse(xml).getroot().iter():
            if el.tag.split('}')[-1] == 'peXml' and el.text:
                for node in ET.fromstring(el.text).iter():
                    if node.tag.split('}')[-1] == 'WKT' and node.text:
                        meta = CRS.from_wkt(node.text)
                        if layer.crs is None:
                            layer = layer.set_crs(meta)
                        elif not CRS(layer.crs).equals(meta, ignore_axis_order=True):
                            raise ValueError(f'Conflicting boundary CRS: {path}')
    if layer.crs is None:
        raise ValueError(f'Missing CRS: {path}')
    layer['ADMIN_CODE'] = layer[code].astype(str).str.strip()
    if not layer.ADMIN_CODE.str.fullmatch(r'\d{10}').all():
        raise ValueError('Invalid boundary code')
    layer['ADMIN_NAME'] = layer[name].map(norm)
    layer['ADMIN_LEVEL'] = level
    layer['GEOMETRY_REPAIRED'] = ~layer.geometry.is_valid
    layer.geometry = layer.geometry.map(lambda g: polygon_only(shapely.make_valid(g)))
    if layer.geometry.is_empty.any() or not layer.geometry.is_valid.all():
        raise ValueError('Unusable boundary geometry')
    layer = layer[['ADMIN_CODE', 'ADMIN_NAME', 'ADMIN_LEVEL', 'GEOMETRY_REPAIRED', 'geometry']]
    if layer.groupby('ADMIN_CODE').ADMIN_NAME.nunique().gt(1).any():
        raise ValueError('Conflicting boundary names')
    return layer.dissolve(by='ADMIN_CODE', as_index=False, aggfunc={
        'ADMIN_NAME': 'first', 'ADMIN_LEVEL': 'first', 'GEOMETRY_REPAIRED': 'max'})


def boundaries(base):
    ri = read_layer(base/'boundaries/LX법정구역경계_리_전국/RI.shp', 'RI', 'li_cd', 'adzn_nm')
    emd = read_layer(base/'boundaries/N3A_G0110000/N3A_G0110000.shp', 'EMD', 'BJCD', 'NAME', 'cp949')
    crs = CRS(emd.crs)
    if not crs.is_projected or any(abs(x.unit_conversion_factor-1) > 1e-9 for x in crs.axis_info[:2]):
        raise ValueError('Boundary CRS must use projected metres')
    ri = ri.to_crs(crs)
    parents, districts = {}, {}
    for r in ri.itertuples():
        parents.setdefault(r.ADMIN_CODE[:8]+'00', set()).add(r.ADMIN_NAME.rsplit(' ', 1)[0])
        districts.setdefault(r.ADMIN_CODE[:5], set()).add(r.ADMIN_NAME.rsplit(' ', 2)[0])
    def fullname(row):
        names = parents.get(row.ADMIN_CODE, set())
        if len(names) == 1:
            return next(iter(names))
        names = districts.get(row.ADMIN_CODE[:5], set())
        return next(iter(names))+' '+row.ADMIN_NAME if len(names) == 1 else ''
    emd['FULL_NAME'] = emd.apply(fullname, axis=1)
    ri['FULL_NAME'] = ri.ADMIN_NAME
    for layer in (ri, emd):
        layer['_MATCH_LEAF'] = layer.ADMIN_NAME.map(lambda name: name.split()[-1] if name else '')
    bounds = np.vstack([ri.total_bounds, emd.total_bounds])
    extent = box(bounds[:, 0].min()-100000, bounds[:, 1].min()-100000,
                 bounds[:, 2].max()+100000, bounds[:, 3].max()+100000)
    return ri, emd, crs, extent


def locate(row, ri, emd, transform):
    lat, lon = row.get('LAT'), row.get('LON')
    if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
        geom = Point(*transform.transform(lon, lat))
        for layer in (ri, emd):
            hits = layer.iloc[layer.sindex.query(geom, predicate='intersects')]
            if len(hits) == 1:
                r = hits.iloc[0]
                return 'POINT', geom, r.ADMIN_LEVEL, r.ADMIN_CODE, r.FULL_NAME, 'COORDINATE'
            if len(hits) > 1:
                return 'POINT', geom, '', '', '', 'REVIEW_BOUNDARY_POINT'
        return 'POINT', geom, '', '', '', 'REVIEW_POINT_OUTSIDE_BOUNDARY'
    if str(row.LOCATION_GRADE) == 'S2':
        return 'NONE', None, '', '', '', 'REVIEW_ROAD_LOCATION_UNRESOLVED'
    tokens = norm(row.LOCATION_NAME).split()
    leaves = [i for i, t in enumerate(tokens) if re.fullmatch(r'[가-힣0-9·]+(?:리|읍|면|동|가)', t)]
    if not leaves or leaves[-1] == 0:
        return 'NONE', None, '', '', '', 'REVIEW_ADMIN_ADDRESS_UNRESOLVED'
    supplied = tokens[:leaves[-1]+1]
    layer = ri if supplied[-1].endswith('리') else emd
    def agrees(full):
        parts = norm(full).split()
        if not parts or parts[-1] != supplied[-1]:
            return False
        ancestors = iter(parts[:-1])
        return all(any(part == token for part in ancestors) for token in supplied[:-1])
    candidates = layer[layer._MATCH_LEAF.eq(supplied[-1])]
    hits = candidates[candidates.FULL_NAME.map(agrees)]
    if len(hits) != 1:
        return 'NONE', None, '', '', '', 'AMBIGUOUS_ADMIN_ADDRESS' if len(hits) > 1 else 'REVIEW_ADMIN_ADDRESS_UNRESOLVED'
    r = hits.iloc[0]
    status = 'FULL_ADDRESS_MATCH' if r.FULL_NAME == ' '.join(supplied) else 'REVIEW_HIERARCHY_OMISSION_MATCH'
    if r.GEOMETRY_REPAIRED:
        status += '_GEOMETRY_REPAIRED'
    return 'ADMIN', r.geometry, r.ADMIN_LEVEL, r.ADMIN_CODE, r.FULL_NAME, status


class StationEngine:
    def __init__(self, history, crs, extent, config):
        self.history, self.extent, self.config = history, extent, config
        self.transform = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        self.cache = {}

    def snapshot(self, when):
        h = self.history
        active = h[(h.ANALYSIS_FROM <= when) & (h.ANALYSIS_TO.isna() | (when < h.ANALYSIS_TO))]
        uncertain = h[h.LAST_OLD_OBSERVED.notna() & h.FIRST_NEW_OBSERVED.notna() &
                      (h.LAST_OLD_OBSERVED <= when) & (when < h.FIRST_NEW_OBSERVED)]
        key = (tuple(active.index), tuple(sorted(uncertain.STN)))
        if key in self.cache:
            return self.cache[key]
        rows = []
        for stn, group in active.groupby('STN', sort=True):
            group = group.assign(priority=group.NETWORK.map({'AWS': 0, 'ASOS': 1})).sort_values(['priority', 'LOCATION_VERSION'])
            r = group.iloc[0]
            conflict = len(group[['LAT', 'LON']].round(7).drop_duplicates()) > 1 or group.NETWORK.duplicated().any()
            rows.append(dict(STN=int(stn), NETWORK=r.NETWORK, LOCATION_VERSION=int(r.LOCATION_VERSION),
                             LAT=r.LAT, LON=r.LON, SITE_REVIEW=bool(group.REVIEW_FLAG.any() or conflict or stn in set(uncertain.STN))))
        if not rows:
            return None
        sites = pd.DataFrame(rows)
        xy = [self.transform.transform(r.LON, r.LAT) for r in sites.itertuples()]
        sites['X'], sites['Y'] = zip(*xy)
        owners = {}
        for i, pt in enumerate(xy):
            owners.setdefault(pt, []).append(i)
        points = list(owners)
        polygons = [self.extent] if len(points) == 1 else list(shapely.voronoi_polygons(
            MultiPoint(points), extend_to=self.extent, ordered=True).geoms)
        if len(polygons) != len(points) or any(not g.covers(Point(p)) for p, g in zip(points, polygons)):
            raise ValueError('Voronoi station correspondence failed')
        result = sites, polygons, list(owners.values())
        self.cache[key] = result
        return result

    def select(self, kind, geom, when):
        snapshot = self.snapshot(when)
        if snapshot is None:
            return None, 'NO_HISTORY_STATIONS', []
        sites, polygons, owners = snapshot
        rows = []
        if kind == 'POINT':
            distances = np.hypot(sites.X-geom.x, sites.Y-geom.y)
            for i in np.flatnonzero(distances <= self.config['point_max_distance_m']):
                r = sites.iloc[i].to_dict()
                r.update(DISTANCE_M=float(distances.iloc[i]), AREA_SHARE=np.nan)
                rows.append(r)
            if not rows:
                return None, 'NO_ELIGIBLE_POINT_STATION', rows
            best = min(r['DISTANCE_M'] for r in rows)
            tied = [r for r in rows if abs(r['DISTANCE_M']-best) <= self.config['distance_tie_m']]
        else:
            coverage = 0
            for polygon, ids in zip(polygons, owners):
                share = geom.intersection(polygon).area / geom.area
                if share <= 0:
                    continue
                coverage += share
                for i in ids:
                    r = sites.iloc[i].to_dict()
                    r.update(AREA_SHARE=share, DISTANCE_M=geom.distance(Point(r['X'], r['Y'])))
                    rows.append(r)
            if abs(coverage-1) > 1e-5:
                raise ValueError('Incomplete Thiessen coverage')
            best = max(r['AREA_SHARE'] for r in rows)
            tied = [r for r in rows if abs(r['AREA_SHARE']-best) <= self.config['area_tie']]
        if len(tied) != 1:
            return None, 'AMBIGUOUS_STATION_TIE', rows
        winner = tied[0]
        return winner['STN'], 'SELECTED_HISTORY_REVIEW' if winner['SITE_REVIEW'] else 'SELECTED_PROVISIONAL', rows
