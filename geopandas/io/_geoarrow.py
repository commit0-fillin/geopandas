import json
from packaging.version import Version
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
import shapely
from shapely import GeometryType
from geopandas import GeoDataFrame
from geopandas._compat import SHAPELY_GE_204
from geopandas.array import from_shapely, from_wkb
GEOARROW_ENCODINGS = ['point', 'linestring', 'polygon', 'multipoint', 'multilinestring', 'multipolygon']

class ArrowTable:
    """
    Wrapper class for Arrow data.

    This class implements the `Arrow PyCapsule Protocol`_ (i.e. having an
    ``__arrow_c_stream__`` method). This object can then be consumed by
    your Arrow implementation of choice that supports this protocol.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Example
    -------
    >>> import pyarrow as pa
    >>> pa.table(gdf.to_arrow())  # doctest: +SKIP

    """

    def __init__(self, pa_table):
        self._pa_table = pa_table

    def __arrow_c_stream__(self, requested_schema=None):
        return self._pa_table.__arrow_c_stream__(requested_schema=requested_schema)

class GeoArrowArray:
    """
    Wrapper class for a geometry array as Arrow data.

    This class implements the `Arrow PyCapsule Protocol`_ (i.e. having an
    ``__arrow_c_array/stream__`` method). This object can then be consumed by
    your Arrow implementation of choice that supports this protocol.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

    Example
    -------
    >>> import pyarrow as pa
    >>> pa.array(ser.to_arrow())  # doctest: +SKIP

    """

    def __init__(self, pa_field, pa_array):
        self._pa_array = pa_array
        self._pa_field = pa_field

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise NotImplementedError('Requested schema is not supported for geometry arrays')
        return (self._pa_field.__arrow_c_schema__(), self._pa_array.__arrow_c_array__()[1])

def geopandas_to_arrow(df, index=None, geometry_encoding='WKB', interleaved=True, include_z=None):
    """
    Convert GeoDataFrame to a pyarrow.Table.

    Parameters
    ----------
    df : GeoDataFrame
        The GeoDataFrame to convert.
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    geometry_encoding : {'WKB', 'geoarrow' }, default 'WKB'
        The GeoArrow encoding to use for the data conversion.
    interleaved : bool, default True
        Only relevant for 'geoarrow' encoding. If True, the geometries'
        coordinates are interleaved in a single fixed size list array.
        If False, the coordinates are stored as separate arrays in a
        struct type.
    include_z : bool, default None
        Only relevant for 'geoarrow' encoding (for WKB, the dimensionality
        of the individial geometries is preserved).
        If False, return 2D geometries. If True, include the third dimension
        in the output (if a geometry has no third dimension, the z-coordinates
        will be NaN). By default, will infer the dimensionality from the
        input geometries. Note that this inference can be unreliable with
        empty geometries (for a guaranteed result, it is recommended to
        specify the keyword).

    Returns
    -------
    pyarrow.Table
        The converted Arrow table.
    """
    import pyarrow as pa
    from pyarrow import geoarrow

    if geometry_encoding not in ['WKB', 'geoarrow']:
        raise ValueError("geometry_encoding must be either 'WKB' or 'geoarrow'")

    # Convert non-geometry columns to Arrow arrays
    arrays = []
    names = []
    for column_name, column in df.items():
        if column_name != df._geometry_column_name:
            arrays.append(pa.array(column))
            names.append(column_name)

    # Convert geometry column
    geometry_column = df.geometry.values
    if geometry_encoding == 'WKB':
        geometry_array = pa.array(geometry_column.to_wkb())
    else:  # geoarrow encoding
        geometry_array = geoarrow.array(geometry_column, interleaved=interleaved, include_z=include_z)

    arrays.append(geometry_array)
    names.append(df._geometry_column_name)

    # Handle index
    if index is None:
        index = not isinstance(df.index, pd.RangeIndex)
    if index:
        if isinstance(df.index, pd.MultiIndex):
            for i, name in enumerate(df.index.names):
                arrays.insert(i, pa.array(df.index.get_level_values(i)))
                names.insert(i, name if name is not None else f'level_{i}')
        else:
            arrays.insert(0, pa.array(df.index))
            names.insert(0, df.index.name if df.index.name is not None else 'index')

    # Create Arrow table
    table = pa.Table.from_arrays(arrays, names=names)

    # Add CRS as metadata
    if df.crs:
        table = table.replace_schema_metadata({
            **table.schema.metadata,
            'geo': json.dumps({'crs': df.crs.to_wkt()})
        })

    return table

def arrow_to_geopandas(table, geometry=None):
    """
    Convert Arrow table object to a GeoDataFrame based on GeoArrow extension types.

    Parameters
    ----------
    table : pyarrow.Table
        The Arrow table to convert.
    geometry : str, default None
        The name of the geometry column to set as the active geometry
        column. If None, the first geometry column found will be used.

    Returns
    -------
    GeoDataFrame

    """
    import pyarrow as pa
    from pyarrow import geoarrow
    from geopandas import GeoDataFrame

    # Convert Arrow table to pandas DataFrame
    df = table.to_pandas()

    # Find geometry column
    if geometry is None:
        geometry_columns = [
            name for name, field in zip(table.column_names, table.schema)
            if geoarrow.is_geoarrow(field.type)
        ]
        if not geometry_columns:
            raise ValueError("No geometry column found in the Arrow table.")
        geometry = geometry_columns[0]
    elif geometry not in table.column_names:
        raise ValueError(f"Specified geometry column '{geometry}' not found in the Arrow table.")

    # Convert geometry column to GeometryArray
    geometry_array = arrow_to_geometry_array(table[geometry])

    # Create GeoDataFrame
    gdf = GeoDataFrame(df, geometry=geometry_array, crs=None)

    # Set CRS if available in metadata
    if table.schema.metadata and b'geo' in table.schema.metadata:
        geo_metadata = json.loads(table.schema.metadata[b'geo'])
        if 'crs' in geo_metadata:
            gdf.set_crs(geo_metadata['crs'], inplace=True)

    return gdf

def arrow_to_geometry_array(arr):
    """
    Convert Arrow array object (representing single GeoArrow array) to a
    geopandas GeometryArray.

    Specifically for GeoSeries.from_arrow.
    """
    import pyarrow as pa
    from pyarrow import geoarrow
    from geopandas.array import from_shapely

    if not geoarrow.is_geoarrow(arr.type):
        raise ValueError("Input array is not a GeoArrow array")

    # Convert GeoArrow array to WKB
    wkb_array = geoarrow.to_wkb(arr)

    # Convert WKB to shapely geometries
    shapely_geoms = shapely.from_wkb(wkb_array.to_numpy())

    # Create GeometryArray
    return from_shapely(shapely_geoms)

def construct_shapely_array(arr: pa.Array, extension_name: str):
    """
    Construct a NumPy array of shapely geometries from a pyarrow.Array
    with GeoArrow extension type.

    """
    import pyarrow as pa
    from pyarrow import geoarrow
    import numpy as np
    import shapely

    if not geoarrow.is_geoarrow(arr.type):
        raise ValueError("Input array is not a GeoArrow array")

    # Convert GeoArrow array to WKB
    wkb_array = geoarrow.to_wkb(arr)

    # Convert WKB to shapely geometries
    shapely_geoms = shapely.from_wkb(wkb_array.to_numpy())

    # Create NumPy array of shapely geometries
    return np.array(shapely_geoms, dtype=object)
