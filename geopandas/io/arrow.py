import json
import warnings
from packaging.version import Version
import numpy as np
from pandas import DataFrame, Series
import shapely
import geopandas
from geopandas import GeoDataFrame
from geopandas._compat import import_optional_dependency
from geopandas.array import from_shapely, from_wkb
from .file import _expand_user
METADATA_VERSION = '1.0.0'
SUPPORTED_VERSIONS = ['0.1.0', '0.4.0', '1.0.0-beta.1', '1.0.0', '1.1.0']
GEOARROW_ENCODINGS = ['point', 'linestring', 'polygon', 'multipoint', 'multilinestring', 'multipolygon']
SUPPORTED_ENCODINGS = ['WKB'] + GEOARROW_ENCODINGS

def _remove_id_from_member_of_ensembles(json_dict):
    """
    Older PROJ versions will not recognize IDs of datum ensemble members that
    were added in more recent PROJ database versions.

    Cf https://github.com/opengeospatial/geoparquet/discussions/110
    and https://github.com/OSGeo/PROJ/pull/3221

    Mimicking the patch to GDAL from https://github.com/OSGeo/gdal/pull/5872
    """
    if isinstance(json_dict, dict):
        if "members" in json_dict:
            for member in json_dict["members"]:
                if isinstance(member, dict):
                    member.pop("id", None)
        for value in json_dict.values():
            _remove_id_from_member_of_ensembles(value)
    elif isinstance(json_dict, list):
        for item in json_dict:
            _remove_id_from_member_of_ensembles(item)
_geometry_type_names = ['Point', 'LineString', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon', 'GeometryCollection']
_geometry_type_names += [geom_type + ' Z' for geom_type in _geometry_type_names]

def _get_geometry_types(series):
    """
    Get unique geometry types from a GeoSeries.
    """
    return series.geom_type.unique().tolist()

def _create_metadata(df, schema_version=None, geometry_encoding=None, write_covering_bbox=False):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : GeoDataFrame
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', '1.0.0', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.
    write_covering_bbox : bool, default False
        Writes the bounding box column for each row entry with column
        name 'bbox'. Writing a bbox column can be computationally
        expensive, hence is default setting is False.

    Returns
    -------
    dict
    """
    if schema_version is None:
        schema_version = '1.0.0'

    metadata = {
        'version': schema_version,
        'primary_column': df._geometry_column_name,
        'columns': {
            df._geometry_column_name: {
                'encoding': geometry_encoding or 'WKB',
                'geometry_types': _get_geometry_types(df.geometry)
            }
        }
    }

    if df.crs:
        metadata['columns'][df._geometry_column_name]['crs'] = df.crs.to_wkt()

    if write_covering_bbox:
        metadata['columns'][df._geometry_column_name]['bbox'] = df.total_bounds.tolist()

    return metadata

def _encode_metadata(metadata):
    """Encode metadata dict to UTF-8 JSON string

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    UTF-8 encoded JSON string
    """
    return json.dumps(metadata).encode('utf-8')

def _decode_metadata(metadata_str):
    """Decode a UTF-8 encoded JSON string to dict

    Parameters
    ----------
    metadata_str : string (UTF-8 encoded)

    Returns
    -------
    dict
    """
    return json.loads(metadata_str.decode('utf-8'))

def _validate_dataframe(df):
    """Validate that the GeoDataFrame conforms to requirements for writing
    to Parquet format.

    Raises `ValueError` if the GeoDataFrame is not valid.

    copied from `pandas.io.parquet`

    Parameters
    ----------
    df : GeoDataFrame
    """
    if not isinstance(df, GeoDataFrame):
        raise ValueError("'df' must be a GeoDataFrame")

    if df.empty:
        raise ValueError("Cannot write empty GeoDataFrame to Parquet")

    if df.columns.inferred_type not in {"string", "unicode", "empty"}:
        raise ValueError("Cannot write GeoDataFrame with non-string column names to Parquet")

    if df.columns.has_duplicates:
        raise ValueError("Cannot write GeoDataFrame with duplicate column names to Parquet")

def _validate_geo_metadata(metadata):
    """Validate geo metadata.
    Must not be empty, and must contain the structure specified above.

    Raises ValueError if metadata is not valid.

    Parameters
    ----------
    metadata : dict
    """
    if not metadata:
        raise ValueError("Geo metadata cannot be empty")

    required_keys = ['version', 'primary_column', 'columns']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Geo metadata is missing required key: {key}")

    if not isinstance(metadata['columns'], dict):
        raise ValueError("'columns' in geo metadata must be a dictionary")

    primary_column = metadata['primary_column']
    if primary_column not in metadata['columns']:
        raise ValueError(f"Primary column '{primary_column}' not found in columns metadata")

    column_metadata = metadata['columns'][primary_column]
    required_column_keys = ['encoding', 'geometry_types']
    for key in required_column_keys:
        if key not in column_metadata:
            raise ValueError(f"Column metadata for '{primary_column}' is missing required key: {key}")

    if column_metadata['encoding'] not in ['WKB', 'geoarrow']:
        raise ValueError(f"Invalid encoding '{column_metadata['encoding']}' for column '{primary_column}'")

def _geopandas_to_arrow(df, index=None, geometry_encoding='WKB', schema_version=None, write_covering_bbox=None):
    """
    Helper function with main, shared logic for to_parquet/to_feather.
    """
    pass

def _to_parquet(df, path, index=None, compression='snappy', geometry_encoding='WKB', schema_version=None, write_covering_bbox=False, **kwargs):
    """
    Write a GeoDataFrame to the Parquet format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow'.

    This is tracking version 1.0.0 of the GeoParquet specification at:
    https://github.com/opengeospatial/geoparquet. Writing older versions is
    supported using the `schema_version` keyword.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    geometry_encoding : {'WKB', 'geoarrow'}, default 'WKB'
        The encoding to use for the geometry columns. Defaults to "WKB"
        for maximum interoperability. Specify "geoarrow" to use one of the
        native GeoArrow-based single-geometry type encodings.
    schema_version : {'0.1.0', '0.4.0', '1.0.0', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.
    write_covering_bbox : bool, default False
        Writes the bounding box column for each row entry with column
        name 'bbox'. Writing a bbox column can be computationally
        expensive, hence is default setting is False.
    **kwargs
        Additional keyword arguments passed to pyarrow.parquet.write_table().
    """
    pass

def _to_feather(df, path, index=None, compression=None, schema_version=None, **kwargs):
    """
    Write a GeoDataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    This is tracking version 1.0.0 of the GeoParquet specification for
    the metadata at: https://github.com/opengeospatial/geoparquet. Writing
    older versions is supported using the `schema_version` keyword.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'zstd', 'lz4', 'uncompressed'}, optional
        Name of the compression to use. Use ``"uncompressed"`` for no
        compression. By default uses LZ4 if available, otherwise uncompressed.
    schema_version : {'0.1.0', '0.4.0', '1.0.0', None}
        GeoParquet specification version for the metadata; if not provided
        will default to latest supported version.
    kwargs
        Additional keyword arguments passed to pyarrow.feather.write_feather().
    """
    pass

def _arrow_to_geopandas(table, geo_metadata=None):
    """
    Helper function with main, shared logic for read_parquet/read_feather.
    """
    pass

def _get_filesystem_path(path, filesystem=None, storage_options=None):
    """
    Get the filesystem and path for a given filesystem and path.

    If the filesystem is not None then it's just returned as is.
    """
    pass

def _ensure_arrow_fs(filesystem):
    """
    Simplified version of pyarrow.fs._ensure_filesystem. This is only needed
    below because `pyarrow.parquet.read_metadata` does not yet accept a
    filesystem keyword (https://issues.apache.org/jira/browse/ARROW-16719)
    """
    pass

def _read_parquet_schema_and_metadata(path, filesystem):
    """
    Opening the Parquet file/dataset a first time to get the schema and metadata.

    TODO: we should look into how we can reuse opened dataset for reading the
    actual data, to avoid discovering the dataset twice (problem right now is
    that the ParquetDataset interface doesn't allow passing the filters on read)

    """
    pass

def _read_parquet(path, columns=None, storage_options=None, bbox=None, **kwargs):
    """
    Load a Parquet object from the file path, returning a GeoDataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_parquet` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned GeoDataFrame.

    Supports versions 0.1.0, 0.4.0 and 1.0.0 of the GeoParquet
    specification at: https://github.com/opengeospatial/geoparquet

    If 'crs' key is not present in the GeoParquet metadata associated with the
    Parquet object, it will default to "OGC:CRS84" according to the specification.

    Requires 'pyarrow'.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host,
        port, username, password, etc. For HTTP(S) URLs the key-value pairs are
        forwarded to urllib as header options. For other URLs (e.g. starting with
        "s3://", and "gcs://") the key-value pairs are forwarded to fsspec. Please
        see fsspec and urllib for more details.

        When no storage options are provided and a filesystem is implemented by
        both ``pyarrow.fs`` and ``fsspec`` (e.g. "s3://") then the ``pyarrow.fs``
        filesystem is preferred. Provide the instantiated fsspec filesystem using
        the ``filesystem`` keyword if you wish to use its implementation.
    bbox : tuple, optional
        Bounding box to be used to filter selection from geoparquet data. This
        is only usable if the data was saved with the bbox covering metadata.
        Input is of the tuple format (xmin, ymin, xmax, ymax).

    **kwargs
        Any additional kwargs passed to :func:`pyarrow.parquet.read_table`.

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    >>> df = geopandas.read_parquet("data.parquet")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = geopandas.read_parquet(
    ...     "data.parquet",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """
    pass

def _read_feather(path, columns=None, **kwargs):
    """
    Load a Feather object from the file path, returning a GeoDataFrame.

    You can read a subset of columns in the file using the ``columns`` parameter.
    However, the structure of the returned GeoDataFrame will depend on which
    columns you read:

    * if no geometry columns are read, this will raise a ``ValueError`` - you
      should use the pandas `read_feather` method instead.
    * if the primary geometry column saved to this file is not included in
      columns, the first available geometry column will be set as the geometry
      column of the returned GeoDataFrame.

    Supports versions 0.1.0, 0.4.0 and 1.0.0 of the GeoParquet
    specification at: https://github.com/opengeospatial/geoparquet

    If 'crs' key is not present in the Feather metadata associated with the
    Parquet object, it will default to "OGC:CRS84" according to the specification.

    Requires 'pyarrow' >= 0.17.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    columns : list-like of strings, default=None
        If not None, only these columns will be read from the file.  If
        the primary geometry column is not included, the first secondary
        geometry read from the file will be set as the geometry column
        of the returned GeoDataFrame.  If no geometry columns are present,
        a ``ValueError`` will be raised.
    **kwargs
        Any additional kwargs passed to pyarrow.feather.read_table().

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    >>> df = geopandas.read_feather("data.feather")  # doctest: +SKIP

    Specifying columns to read:

    >>> df = geopandas.read_feather(
    ...     "data.feather",
    ...     columns=["geometry", "pop_est"]
    ... )  # doctest: +SKIP
    """
    pass
