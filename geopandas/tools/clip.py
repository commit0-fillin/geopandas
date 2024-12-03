
A module to clip vector data using GeoPandas.

"""
import warnings
import numpy as np
import pandas.api.types
from shapely.geometry import MultiPolygon, Polygon, box
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
==============

A module to clip vector data using GeoPandas.

"""
import warnings
import numpy as np
import pandas.api.types
from shapely.geometry import MultiPolygon, Polygon, box
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn

def _clip_gdf_with_mask(gdf, mask, sort=False):
    """Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    if isinstance(mask, (Polygon, MultiPolygon)):
        mask = GeoSeries([mask])
    elif isinstance(mask, GeoDataFrame):
        mask = mask.geometry
    elif isinstance(mask, GeoSeries):
        pass
    else:
        try:
            mask = GeoSeries(mask)
        except:
            raise ValueError("Cannot convert mask to GeoSeries")

    # Dissolve all polygons in mask
    mask = mask.unary_union

    # Perform the intersection
    clipped = gdf.copy()
    clipped['geometry'] = clipped.geometry.intersection(mask)

    # Remove empty geometries
    clipped = clipped[~clipped.geometry.is_empty]

    if sort:
        clipped = clipped.sort_index()

    return clipped

def clip(gdf, mask, keep_geom_type=False, sort=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The ``gdf`` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from ``gdf`` will be
    clipped to the total boundary of all polygons in mask.

    If the ``mask`` is list-like with four elements ``(minx, miny, maxx, maxy)``, a
    faster rectangle clipping algorithm will be used. Note that this can lead to
    slightly different results in edge cases, e.g. if a line would be reduced to a
    point, this point might not be returned.
    The geometry is clipped in a fast but possibly dirty way. The output is not
    guaranteed to be valid. No exceptions will be raised for topological errors.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.
        The mask's geometry is dissolved into one geometric feature
        and intersected with ``gdf``.
        If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
        ``clip`` will use a faster rectangle clipping (:meth:`~GeoSeries.clip_by_rect`),
        possibly leading to slightly different results.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).
    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from ``gdf`` clipped to
         polygon boundary from mask.

    See also
    --------
    GeoDataFrame.clip : equivalent GeoDataFrame method
    GeoSeries.clip : equivalent GeoSeries method

    Examples
    --------
    Clip points (grocery stores) with polygons (the Near West Side community):

    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)
    >>> groceries.shape
    (148, 8)

    >>> nws_groceries = geopandas.clip(groceries, near_west_side)
    >>> nws_groceries.shape
    (7, 8)
    """
    from geopandas import GeoDataFrame, GeoSeries

    if isinstance(mask, (list, tuple)) and len(mask) == 4:
        return gdf.clip_by_rect(*mask)

    if isinstance(gdf, GeoDataFrame):
        return _clip_gdf_with_mask(gdf, mask, sort)
    elif isinstance(gdf, GeoSeries):
        clipped = _clip_gdf_with_mask(GeoDataFrame({'geometry': gdf}), mask, sort)
        return clipped.geometry

    raise ValueError("Input `gdf` should be GeoDataFrame or GeoSeries")
