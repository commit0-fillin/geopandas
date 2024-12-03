"""
Testing functionality for geopandas objects.
"""
import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype

def _isna(this):
    """isna version that works for both scalars and (Geo)Series"""
    if isinstance(this, (pd.Series, pd.Index)):
        return this.isna()
    return pd.isna(this)

def _geom_equals_mask(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    Series
        boolean Series, True if geometries in left equal geometries in right
    """
    this_na = _isna(this)
    that_na = _isna(that)
    empty_eq = (this.is_empty & that.is_empty) | (this_na & that_na)
    eq = this.equals(that)
    return empty_eq | eq

def geom_equals(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    bool
        True if all geometries in left equal geometries in right
    """
    return _geom_equals_mask(this, that).all()

def _geom_almost_equals_mask(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects

    Returns
    -------
    Series
        boolean Series, True if geometries in left almost equal geometries in right
    """
    this_na = _isna(this)
    that_na = _isna(that)
    empty_eq = (this.is_empty & that.is_empty) | (this_na & that_na)
    eq = this.geom_almost_equals(that)
    return empty_eq | eq

def geom_almost_equals(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)

    Returns
    -------
    bool
        True if all geometries in left almost equal geometries in right
    """
    return _geom_almost_equals_mask(this, that).all()

def assert_geoseries_equal(left, right, check_dtype=True, check_index_type=False, check_series_type=True, check_less_precise=False, check_geom_type=False, check_crs=True, normalize=False):
    """
    Test util for checking that two GeoSeries are equal.

    Parameters
    ----------
    left, right : two GeoSeries
    check_dtype : bool, default False
        If True, check geo dtype [only included so it's a drop-in replacement
        for assert_series_equal].
    check_index_type : bool, default False
        Check that index types are equal.
    check_series_type : bool, default True
        Check that both are same type (*and* are GeoSeries). If False,
        will attempt to convert both into GeoSeries.
    check_less_precise : bool, default False
        If True, use geom_equals_exact with relative error of 0.5e-6.
        If False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_series_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    from geopandas import GeoSeries

    if check_series_type:
        assert isinstance(left, GeoSeries)
        assert isinstance(right, GeoSeries)
        assert_series_equal(left, right, check_dtype=False, check_index_type=check_index_type, check_series_type=check_series_type, obj="GeoSeries")
    else:
        left = GeoSeries(left)
        right = GeoSeries(right)
        assert_series_equal(left, right, check_dtype=False, check_index_type=check_index_type, check_series_type=False, obj="GeoSeries")

    if check_dtype:
        assert left.dtype == right.dtype, "DTYPEs are not equal"

    if check_geom_type:
        assert (left.geom_type == right.geom_type).all(), "Geometry types are not all equal"

    if check_crs:
        assert left.crs == right.crs, "CRS are not equal"

    if normalize:
        left = left.normalize()
        right = right.normalize()

    if check_less_precise:
        assert geom_almost_equals(left, right)
    else:
        assert geom_equals(left, right)

def _truncated_string(geom):
    """Truncated WKT repr of geom"""
    if geom is None:
        return 'None'
    if hasattr(geom, '__geo_interface__'):
        return str(geom)[:80]
    return 'Geometry'

def assert_geodataframe_equal(left, right, check_dtype=True, check_index_type='equiv', check_column_type='equiv', check_frame_type=True, check_like=False, check_less_precise=False, check_geom_type=False, check_crs=True, normalize=False):
    """
    Check that two GeoDataFrames are equal/

    Parameters
    ----------
    left, right : two GeoDataFrames
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type, check_column_type : bool, default 'equiv'
        Check that index types are equal.
    check_frame_type : bool, default True
        Check that both are same type (*and* are GeoDataFrames). If False,
        will attempt to convert both into GeoDataFrame.
    check_like : bool, default False
        If true, ignore the order of rows & columns
    check_less_precise : bool, default False
        If True, use geom_equals_exact. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_frame_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    from geopandas import GeoDataFrame

    # Check that the type of the frames is identical
    if check_frame_type:
        assert isinstance(left, GeoDataFrame), "'left' is not a GeoDataFrame"
        assert isinstance(right, GeoDataFrame), "'right' is not a GeoDataFrame"

    # Check that the geometry column is identical
    assert left._geometry_column_name == right._geometry_column_name, "Geometry column names are not equal"

    # Check CRS
    if check_crs:
        assert left.crs == right.crs, "CRS are not equal"

    # Normalize geometries if requested
    if normalize:
        left = left.copy()
        right = right.copy()
        left[left._geometry_column_name] = left.geometry.normalize()
        right[right._geometry_column_name] = right.geometry.normalize()

    # Check geometries
    assert_geoseries_equal(left.geometry, right.geometry, check_dtype=check_dtype, check_less_precise=check_less_precise, check_geom_type=check_geom_type, normalize=False)

    # Check the rest of the DataFrame
    assert_frame_equal(left.drop(columns=[left._geometry_column_name]),
                       right.drop(columns=[right._geometry_column_name]),
                       check_dtype=check_dtype,
                       check_index_type=check_index_type,
                       check_column_type=check_column_type,
                       check_frame_type=False,
                       check_like=check_like,
                       obj="GeoDataFrame")
