from __future__ import annotations
import os
import urllib.request
import warnings
from io import IOBase
from packaging.version import Version
from pathlib import Path
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_netloc, uses_params, uses_relative
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import shapely
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import HAS_PYPROJ, PANDAS_GE_20
from geopandas.io.util import vsi_path
_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')
_VALID_URLS.discard('file')
fiona = None
fiona_env = None
fiona_import_error = None
FIONA_GE_19 = False
pyogrio = None
pyogrio_import_error = None
_EXTENSION_TO_DRIVER = {'.bna': 'BNA', '.dxf': 'DXF', '.csv': 'CSV', '.shp': 'ESRI Shapefile', '.dbf': 'ESRI Shapefile', '.json': 'GeoJSON', '.geojson': 'GeoJSON', '.geojsonl': 'GeoJSONSeq', '.geojsons': 'GeoJSONSeq', '.gpkg': 'GPKG', '.gml': 'GML', '.xml': 'GML', '.gpx': 'GPX', '.gtm': 'GPSTrackMaker', '.gtz': 'GPSTrackMaker', '.tab': 'MapInfo File', '.mif': 'MapInfo File', '.mid': 'MapInfo File', '.dgn': 'DGN', '.fgb': 'FlatGeobuf'}

def _expand_user(path):
    """Expand paths that use ~."""
    return os.path.expanduser(path)

def _is_url(url):
    """Check to see if *url* has a valid protocol."""
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False

def _read_file(filename, bbox=None, mask=None, columns=None, rows=None, engine=None, **kwargs):
    """
    Returns a GeoDataFrame from a file or URL.

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)
    bbox : tuple | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely
        geometry. With engine="fiona", CRS mis-matches are resolved if given a GeoSeries
        or GeoDataFrame. With engine="pyogrio", bbox must be in the same CRS as the
        dataset. Tuple is (minx, miny, maxx, maxy) to match the bounds property of
        shapely geometry objects. Cannot be used with mask.
    mask : dict | GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the given dict-like geojson
        geometry, GeoSeries, GeoDataFrame or shapely geometry.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox. If multiple geometries are passed, this will
        first union all geometries, which may be computationally expensive.
    columns : list, optional
        List of column names to import from the data source. Column names
        must exactly match the names in the data source. To avoid reading
        any columns (besides the geometry column), pass an empty list-like.
        By default reads all columns.
    rows : int or slice, default None
        Load in specific rows by passing an integer (first `n` rows) or a
        slice() object.
    engine : str,  "pyogrio" or "fiona"
        The underlying library that is used to read the file. Currently, the
        supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
        installed, otherwise tries "fiona". Engine can also be set globally
        with the ``geopandas.options.io_engine`` option.
    **kwargs :
        Keyword args to be passed to the engine, and can be used to write
        to multi-layer data, store data within archives (zip files), etc.
        In case of the "pyogrio" engine, the keyword arguments are passed to
        `pyogrio.write_dataframe`. In case of the "fiona" engine, the keyword
        arguments are passed to fiona.open`. For more information on possible
        keywords, type: ``import pyogrio; help(pyogrio.write_dataframe)``.

    Returns
    -------
    :obj:`geopandas.GeoDataFrame` or :obj:`pandas.DataFrame` :
        If `ignore_geometry=True` a :obj:`pandas.DataFrame` will be returned.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

    When specifying a URL, geopandas will check if the server supports reading
    partial data and in that case pass the URL as is to the underlying engine,
    which will then use the network file system handler of GDAL to read from
    the URL. Otherwise geopandas will download the data from the URL and pass
    all data in-memory to the underlying engine.
    If you need more control over how the URL is read, you can specify the
    GDAL virtual filesystem manually (e.g. ``/vsicurl/https://...``). See the
    GDAL documentation on filesystems for more details
    (https://gdal.org/user/virtual_file_systems.html#vsicurl-http-https-ftp-files-random-access).

    """
    if engine is None:
        engine = "pyogrio" if pyogrio is not None else "fiona"

    if engine == "pyogrio":
        if pyogrio is None:
            raise ImportError("The 'pyogrio' package is required to use the pyogrio engine.")
        return pyogrio.read_dataframe(filename, bbox=bbox, mask=mask, columns=columns, rows=rows, **kwargs)
    elif engine == "fiona":
        if fiona is None:
            raise ImportError("The 'fiona' package is required to use the fiona engine.")
        with fiona.open(filename, **kwargs) as source:
            crs = source.crs
            if bbox is not None:
                if isinstance(bbox, (GeoDataFrame, GeoSeries)):
                    bbox = tuple(bbox.total_bounds)
                elif isinstance(bbox, BaseGeometry):
                    bbox = bbox.bounds
            if mask is not None:
                if isinstance(mask, (GeoDataFrame, GeoSeries)):
                    mask = mask.unary_union
                elif isinstance(mask, dict):
                    mask = shapely.geometry.shape(mask)
            features = list(source.filter(bbox=bbox, mask=mask))
            if columns:
                features = [{k: f['properties'].get(k, None) for k in columns} for f in features]
            if rows is not None:
                if isinstance(rows, int):
                    features = features[:rows]
                elif isinstance(rows, slice):
                    features = features[rows]
            gdf = GeoDataFrame.from_features(features, crs=crs)
            if columns:
                gdf = gdf[columns]
            return gdf
    else:
        raise ValueError("Engine must be either 'pyogrio' or 'fiona'")

def _detect_driver(path):
    """
    Attempt to auto-detect driver based on the extension
    """
    extension = os.path.splitext(path)[-1].lower()
    return _EXTENSION_TO_DRIVER.get(extension)

def _to_file(df, filename, driver=None, schema=None, index=None, mode='w', crs=None, engine=None, metadata=None, **kwargs):
    """
    Write this GeoDataFrame to an OGR data source

    A dictionary of supported OGR providers is available via:

    >>> import pyogrio
    >>> pyogrio.list_drivers()  # doctest: +SKIP

    Parameters
    ----------
    df : GeoDataFrame to be written
    filename : string
        File path or file handle to write to. The path may specify a
        GDAL VSI scheme.
    driver : string, default None
        The OGR format driver used to write the vector file.
        If not specified, it attempts to infer it from the file extension.
        If no extension is specified, it saves ESRI Shapefile to a folder.
    schema : dict, default None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the schema based on each column's dtype.
        Not supported for the "pyogrio" engine.
    index : bool, default None
        If True, write index into one or more columns (for MultiIndex).
        Default None writes the index into one or more columns only if
        the index is named, is a MultiIndex, or has a non-integer data
        type. If False, no index is written.
    mode : string, default 'w'
        The write mode, 'w' to overwrite the existing file and 'a' to append;
        when using the pyogrio engine, you can also pass ``append=True``.
        Not all drivers support appending.
    crs : pyproj.CRS, default None
        If specified, the CRS is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the crs based on crs df attribute.
    engine : str,  "pyogrio" or "fiona"
        The underlying library that is used to read the file. Currently, the
        supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
        installed, otherwise tries "fiona".
    metadata : dict[str, str], default None
        Optional metadata to be stored in the file. Keys and values must be
        strings. Only supported for the "GPKG" driver
        (requires Fiona >= 1.9 or pyogrio >= 0.6).
    **kwargs :
        Keyword args to be passed to the engine, and can be used to write
        to multi-layer data, store data within archives (zip files), etc.

    Notes
    -----
    The format drivers will attempt to detect the encoding of your data, but
    may fail. In this case, the proper encoding can be specified explicitly
    by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.
    """
    if engine is None:
        engine = "pyogrio" if pyogrio is not None else "fiona"

    if driver is None:
        driver = _detect_driver(filename)

    if engine == "pyogrio":
        if pyogrio is None:
            raise ImportError("The 'pyogrio' package is required to use the pyogrio engine.")
        pyogrio.write_dataframe(df, filename, driver=driver, crs=crs, append=mode=='a', metadata=metadata, **kwargs)
    elif engine == "fiona":
        if fiona is None:
            raise ImportError("The 'fiona' package is required to use the fiona engine.")
        if schema is None:
            schema = _geometry_types(df)
        with fiona.open(filename, mode, driver=driver, crs=crs, schema=schema, **kwargs) as colxn:
            colxn.writerecords(df.iterfeatures())
    else:
        raise ValueError("Engine must be either 'pyogrio' or 'fiona'")

def _geometry_types(df):
    """
    Determine the geometry types in the GeoDataFrame for the schema.
    """
    geom_types = set(df.geometry.geom_type)
    if len(geom_types) == 1:
        return geom_types.pop()
    return "Mixed"

def _list_layers(filename) -> pd.DataFrame:
    """List layers available in a file.

    Provides an overview of layers available in a file or URL together with their
    geometry types. When supported by the data source, this includes both spatial and
    non-spatial layers. Non-spatial layers are indicated by the ``"geometry_type"``
    column being ``None``. GeoPandas will not read such layers but they can be read into
    a pd.DataFrame using :func:`pyogrio.read_dataframe`.

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns "name" and "geometry_type" and one row per layer.
    """
    if pyogrio is not None:
        return pyogrio.list_layers(filename)
    elif fiona is not None:
        with fiona.open(filename) as src:
            layers = [{"name": layer, "geometry_type": src.schema["geometry"]} for layer in src.layers]
        return pd.DataFrame(layers)
    else:
        raise ImportError("Either 'pyogrio' or 'fiona' is required to list layers.")
