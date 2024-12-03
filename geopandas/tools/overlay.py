import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import PANDAS_GE_30
from geopandas.array import _check_crs, _crs_mismatch_warn

def _ensure_geometry_column(df):
    """
    Helper function to ensure the geometry column is called 'geometry'.
    If another column with that name exists, it will be dropped.
    """
    if not isinstance(df, GeoDataFrame):
        df = GeoDataFrame(df)
    
    if df._geometry_column_name != 'geometry':
        if 'geometry' in df.columns:
            df = df.drop(columns=['geometry'])
        df = df.rename_geometry('geometry')
    
    return df

def _overlay_intersection(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """
    df1 = _ensure_geometry_column(df1)
    df2 = _ensure_geometry_column(df2)
    
    # Perform spatial join
    joined = df1.sjoin(df2, how='inner', predicate='intersects')
    
    # Perform the actual intersection
    joined['geometry'] = joined.apply(lambda row: row['geometry_x'].intersection(row['geometry_y']), axis=1)
    
    # Drop unnecessary columns and reset index
    result = joined.drop(columns=['geometry_x', 'geometry_y', 'index_right']).reset_index(drop=True)
    
    return result

def _overlay_difference(df1, df2):
    """
    Overlay Difference operation used in overlay function
    """
    df1 = _ensure_geometry_column(df1)
    df2 = _ensure_geometry_column(df2)
    
    # Perform spatial join
    joined = df1.sjoin(df2, how='left', predicate='intersects')
    
    # Perform the difference operation
    def difference_op(row):
        if pd.isna(row['index_right']):
            return row['geometry']
        return row['geometry'].difference(row['geometry_right'])
    
    joined['geometry'] = joined.apply(difference_op, axis=1)
    
    # Drop unnecessary columns and reset index
    result = joined.drop(columns=['geometry_right', 'index_right']).reset_index(drop=True)
    
    return result

def _overlay_symmetric_diff(df1, df2):
    """
    Overlay Symmetric Difference operation used in overlay function
    """
    df1 = _ensure_geometry_column(df1)
    df2 = _ensure_geometry_column(df2)
    
    # Perform difference in both directions
    diff1 = _overlay_difference(df1, df2)
    diff2 = _overlay_difference(df2, df1)
    
    # Combine the results
    result = pd.concat([diff1, diff2], ignore_index=True)
    
    return result

def _overlay_union(df1, df2):
    """
    Overlay Union operation used in overlay function
    """
    df1 = _ensure_geometry_column(df1)
    df2 = _ensure_geometry_column(df2)
    
    # Perform spatial join
    joined = df1.sjoin(df2, how='outer', predicate='intersects')
    
    # Perform the union operation
    def union_op(row):
        if pd.isna(row['geometry_y']):
            return row['geometry_x']
        elif pd.isna(row['geometry_x']):
            return row['geometry_y']
        return row['geometry_x'].union(row['geometry_y'])
    
    joined['geometry'] = joined.apply(union_op, axis=1)
    
    # Drop unnecessary columns and reset index
    result = joined.drop(columns=['geometry_x', 'geometry_y', 'index_right']).reset_index(drop=True)
    
    return result

def overlay(df1, df2, how='intersection', keep_geom_type=None, make_valid=True):
    """Perform spatial overlay between two GeoDataFrames.

    Currently only supports data GeoDataFrames with uniform geometry types,
    i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
    combination of (Multi)LineString and LinearRing shapes.
    Implements several methods that are all effectively subsets of the union.

    See the User Guide page :doc:`../../user_guide/set_operations` for details.

    Parameters
    ----------
    df1 : GeoDataFrame
    df2 : GeoDataFrame
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None,
        which will set keep_geom_type to True but warn upon dropping
        geometries.
    make_valid : bool, default True
        If True, any invalid input geometries are corrected with a call to make_valid(),
        if False, a `ValueError` is raised if any input geometries are invalid.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
    ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
    >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
    ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
    >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
    >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

    >>> geopandas.overlay(df1, df2, how='union')
        df1_data  df2_data                                           geometry
    0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    5       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    6       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='intersection')
       df1_data  df2_data                             geometry
    0         1         1  POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2         1  POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2         2  POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))

    >>> geopandas.overlay(df1, df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
    0       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    1       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    2       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    3       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='difference')
                                                geometry  df1_data
    0      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))         1
    1  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...         2

    >>> geopandas.overlay(df1, df2, how='identity')
       df1_data  df2_data                                           geometry
    0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...

    See also
    --------
    sjoin : spatial join
    GeoDataFrame.overlay : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    import warnings
    
    # Ensure input are GeoDataFrames
    df1 = _ensure_geometry_column(df1)
    df2 = _ensure_geometry_column(df2)

    # Check CRS
    if not (_check_crs(df1, df2)):
        _crs_mismatch_warn(df1, df2, stacklevel=3)

    # Validate geometries if requested
    if make_valid:
        df1['geometry'] = df1.geometry.make_valid()
        df2['geometry'] = df2.geometry.make_valid()

    # Perform the overlay operation
    if how == 'intersection':
        result = _overlay_intersection(df1, df2)
    elif how == 'union':
        result = _overlay_union(df1, df2)
    elif how == 'identity':
        result = pd.concat([_overlay_intersection(df1, df2), _overlay_difference(df1, df2)], ignore_index=True)
    elif how == 'symmetric_difference':
        result = _overlay_symmetric_diff(df1, df2)
    elif how == 'difference':
        result = _overlay_difference(df1, df2)
    else:
        raise ValueError("'how' must be one of 'intersection', 'union', 'identity', 'symmetric_difference' or 'difference'")

    # Handle keep_geom_type
    if keep_geom_type is None:
        keep_geom_type = True
        warn_keep_geom_type = True
    else:
        warn_keep_geom_type = False

    if keep_geom_type:
        mask = result.geom_type == df1.geom_type.iloc[0]
        if warn_keep_geom_type and not mask.all():
            warnings.warn(
                "Geometry types of the result differs from the geometry types of the passed geodataframes. "
                "Returning only geometries of same geometry type as `df1`. "
                "Set `keep_geom_type=False` to return all resulting geometries.",
                UserWarning,
                stacklevel=2,
            )
        result = result.loc[mask]

    # Reset index and set CRS
    result = result.reset_index(drop=True)
    result.crs = df1.crs

    return result
