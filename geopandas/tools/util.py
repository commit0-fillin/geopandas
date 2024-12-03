import pandas as pd
from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon
from shapely.geometry.base import BaseGeometry
_multi_type_map = {'Point': MultiPoint, 'LineString': MultiLineString, 'Polygon': MultiPolygon}

def collect(x, multi=False):
    """
    Collect single part geometries into their Multi* counterpart

    Parameters
    ----------
    x : an iterable or Series of Shapely geometries, a GeoSeries, or
        a single Shapely geometry
    multi : boolean, default False
        if True, force returned geometries to be Multi* even if they
        only have one component.

    Returns
    -------
    geometry : Shapely geometry
        A single Shapely geometry object
    """
    if isinstance(x, BaseGeometry):
        if multi and not isinstance(x, tuple(_multi_type_map.values())):
            return _multi_type_map[x.geom_type]([x])
        return x

    if isinstance(x, pd.Series):
        x = x.values

    types = set([geom.geom_type for geom in x if geom is not None])
    if len(types) > 1:
        return MultiGeometryCollection(list(x))

    geom_type = list(types)[0]

    if geom_type in _multi_type_map.keys():
        multi_type = _multi_type_map[geom_type]
        geoms = [g for g in x if g is not None]
        if multi or len(geoms) > 1:
            return multi_type(geoms)
        else:
            return geoms[0]
    elif geom_type in ('MultiPoint', 'MultiLineString', 'MultiPolygon'):
        return x[0].__class__([p for geom in x for p in geom.geoms])
    else:
        return GeometryCollection([geom for geom in x if geom is not None])
