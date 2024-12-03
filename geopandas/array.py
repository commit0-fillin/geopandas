import inspect
import numbers
import operator
import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkt
from shapely.geometry.base import BaseGeometry
from ._compat import HAS_PYPROJ, requires_pyproj
from .sindex import SpatialIndex
if HAS_PYPROJ:
    from pyproj import Transformer
    TransformerFromCRS = lru_cache(Transformer.from_crs)
_names = {'MISSING': None, 'NAG': None, 'POINT': 'Point', 'LINESTRING': 'LineString', 'LINEARRING': 'LinearRing', 'POLYGON': 'Polygon', 'MULTIPOINT': 'MultiPoint', 'MULTILINESTRING': 'MultiLineString', 'MULTIPOLYGON': 'MultiPolygon', 'GEOMETRYCOLLECTION': 'GeometryCollection'}
type_mapping = {p.value: _names[p.name] for p in shapely.GeometryType}
geometry_type_ids = list(type_mapping.keys())
geometry_type_values = np.array(list(type_mapping.values()), dtype=object)

class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = 'geometry'
    na_value = np.nan
register_extension_dtype(GeometryDtype)

def _check_crs(left, right, allow_none=False):
    """
    Check if the projection of both arrays is the same.

    If allow_none is True, empty CRS is treated as the same.
    """
    if allow_none and (left is None or right is None):
        return True
    elif left is None or right is None:
        return False
    else:
        return left == right

def _crs_mismatch_warn(left, right, stacklevel=3):
    """
    Raise a CRS mismatch warning with the information on the assigned CRS.
    """
    warnings.warn(
        f"CRS mismatch between the CRS of left geometries and the CRS of right geometries.\n"
        f"Left CRS: {left}\n"
        f"Right CRS: {right}\n"
        f"Use `to_crs()` to reproject geometries to the same CRS before comparison.",
        UserWarning,
        stacklevel=stacklevel,
    )

def isna(value):
    """
    Check if scalar value is NA-like (None, np.nan or pd.NA).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)

def from_shapely(data, crs=None):
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.

    Parameters
    ----------
    data : array-like
        list or array of shapely objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    """
    data = np.asarray(data, dtype=object)
    return GeometryArray(data, crs=crs)

def to_shapely(geoms):
    """
    Convert GeometryArray to numpy object array of shapely objects.
    """
    if isinstance(geoms, GeometryArray):
        return geoms._data
    return np.asarray(geoms, dtype=object)

def from_wkb(data, crs=None, on_invalid='raise'):
    """
    Convert a list or array of WKB objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKB objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    on_invalid: {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKB input geometry is invalid.
        - warn: a warning will be raised and invalid WKB geometries will be returned as
          None.
        - ignore: invalid WKB geometries will be returned as None without a warning.

    """
    pass

def to_wkb(geoms, hex=False, **kwargs):
    """
    Convert GeometryArray to a numpy object array of WKB objects.
    """
    pass

def from_wkt(data, crs=None, on_invalid='raise'):
    """
    Convert a list or array of WKT objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKT objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    on_invalid : {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKT input geometry is invalid.
        - warn: a warning will be raised and invalid WKT geometries will be
          returned as ``None``.
        - ignore: invalid WKT geometries will be returned as ``None`` without a warning.

    """
    pass

def to_wkt(geoms, **kwargs):
    """
    Convert GeometryArray to a numpy object array of WKT objects.
    """
    pass

def points_from_xy(x, y, z=None, crs=None):
    """
    Generate GeometryArray of shapely Point geometries from x, y(, z) coordinates.

    In case of geographic coordinates, it is assumed that longitude is captured by
    ``x`` coordinates and latitude by ``y``.

    Parameters
    ----------
    x, y, z : iterable
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2], 'z': [0, 1, 2]})
    >>> df
       x  y  z
    0  0  0  0
    1  1  1  1
    2  2  2  2
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(
    ...     df, geometry=geopandas.points_from_xy(df['x'], df['y']))

    Having geographic coordinates:

    >>> df = pd.DataFrame({'longitude': [-140, 0, 123], 'latitude': [-65, 1, 48]})
    >>> df
       longitude  latitude
    0       -140       -65
    1          0         1
    2        123        48
    >>> geometry = geopandas.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326")

    Returns
    -------
    output : GeometryArray
    """
    pass

class GeometryArray(ExtensionArray):
    """
    Class wrapping a numpy array of Shapely objects and
    holding the array-based implementations.
    """
    _dtype = GeometryDtype()

    def __init__(self, data, crs=None):
        if isinstance(data, self.__class__):
            if not crs:
                crs = data.crs
            data = data._data
        elif not isinstance(data, np.ndarray):
            raise TypeError("'data' should be array of geometry objects. Use from_shapely, from_wkb, from_wkt functions to construct a GeometryArray.")
        elif not data.ndim == 1:
            raise ValueError("'data' should be a 1-dimensional array of geometry objects.")
        self._data = data
        self._crs = None
        self.crs = crs
        self._sindex = None

    @property
    def has_sindex(self):
        """Check the existence of the spatial index without generating it.

        Use the `.sindex` attribute on a GeoDataFrame or GeoSeries
        to generate a spatial index if it does not yet exist,
        which may take considerable time based on the underlying index
        implementation.

        Note that the underlying spatial index may not be fully
        initialized until the first use.

        See Also
        ---------
        GeoDataFrame.has_sindex

        Returns
        -------
        bool
            `True` if the spatial index has been generated or
            `False` if not.
        """
        pass

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns None if the CRS is not set, and to set the value it
        :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        pass

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        pass

    def check_geographic_crs(self, stacklevel):
        """Check CRS and warn if the planar operation is done in a geographic CRS"""
        pass

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self._data[idx]
        idx = pd.api.indexers.check_array_indexer(self, idx)
        return GeometryArray(self._data[idx], crs=self.crs)

    def __setitem__(self, key, value):
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, (list, np.ndarray)):
            value = from_shapely(value)
        if isinstance(value, GeometryArray):
            if isinstance(key, numbers.Integral):
                raise ValueError('cannot set a single element with an array')
            self._data[key] = value._data
        elif isinstance(value, BaseGeometry) or isna(value):
            if isna(value):
                value = None
            elif isinstance(value, BaseGeometry):
                value = from_shapely([value])._data[0]
            else:
                raise TypeError('should be valid geometry')
            if isinstance(key, (slice, list, np.ndarray)):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self._data[key] = value_array
            else:
                self._data[key] = value
        else:
            raise TypeError('Value should be either a BaseGeometry or None, got %s' % str(value))
        self._sindex = None

    def __getstate__(self):
        return (shapely.to_wkb(self._data), self._crs)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            geoms = shapely.from_wkb(state[0])
            self._crs = state[1]
            self._sindex = None
            self._data = geoms
            self.base = None
        else:
            if 'data' in state:
                state['_data'] = state.pop('data')
            if '_crs' not in state:
                state['_crs'] = None
            self.__dict__.update(state)

    @requires_pyproj
    def to_crs(self, crs=None, epsg=None):
        """Returns a ``GeometryArray`` with all geometries transformed to a new
        coordinate reference system.

        Transform all geometries in a GeometryArray to a different coordinate
        reference system.  The ``crs`` attribute on the current GeometryArray must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects.  It has no notion
        of projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.

        Returns
        -------
        GeometryArray

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> from geopandas.array import from_shapely, to_wkt
        >>> a = from_shapely([Point(1, 1), Point(2, 2), Point(3, 3)], crs=4326)
        >>> to_wkt(a)
        array(['POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        >>> a = a.to_crs(3857)
        >>> to_wkt(a)
        array(['POINT (111319.490793 111325.142866)',
               'POINT (222638.981587 222684.208506)',
               'POINT (333958.47238 334111.171402)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        """
        pass

    @requires_pyproj
    def estimate_utm_crs(self, datum_name='WGS 84'):
        """Returns the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        pyproj.CRS

        Examples
        --------
        >>> import geodatasets
        >>> df = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_commpop")
        ... )
        >>> df.geometry.values.estimate_utm_crs()  # doctest: +SKIP
        <Derived Projected CRS: EPSG:32616>
        Name: WGS 84 / UTM zone 16N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: Between 90°W and 84°W, northern hemisphere between equator and 84°N,...
        - bounds: (-90.0, 0.0, -84.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 16N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        pass

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries"""
        pass

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries"""
        pass

    @property
    def z(self):
        """Return the z location of point geometries in a GeoSeries"""
        pass

    def fillna(self, value=None, method=None, limit=None, copy=True):
        """
        Fill NA values with geometry (or geometries) or using the specified method.

        Parameters
        ----------
        value : shapely geometry object or GeometryArray
            If a geometry value is passed it is used to fill all missing values.
            Alternatively, an GeometryArray 'value' can be given. It's expected
            that the GeometryArray has the same length as 'self'.

        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap

        limit : int, default None
            The maximum number of entries where NA values will be filled.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.

        Returns
        -------
        GeometryArray
        """
        pass

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        pass

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        pass

    def value_counts(self, dropna: bool=True):
        """
        Compute a histogram of the counts of non-null values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN

        Returns
        -------
        pd.Series
        """
        pass

    def unique(self):
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        pass

    def shift(self, periods=1, fill_value=None):
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional (default None)
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        GeometryArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.
        """
        pass

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        pass

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        pass

    def _values_for_factorize(self):
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        pass

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        pass

    def _values_for_argsort(self):
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        pass

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        pass

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        pass

    def __array__(self, dtype=None, copy=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        if copy and (dtype is None or dtype == np.dtype('object')):
            return self._data.copy()
        return self._data

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def __ne__(self, other):
        return self._binop(other, operator.ne)

    def __contains__(self, item):
        """
        Return for `item in self`.
        """
        if isna(item):
            if item is self.dtype.na_value or isinstance(item, self.dtype.type) or item is None:
                return self.isna().any()
            else:
                return False
        return (self == item).any()
