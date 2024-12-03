from __future__ import absolute_import
import json
import os
import pathlib
from itertools import product
from packaging.version import Version
import numpy as np
from pandas import DataFrame
from pandas import read_parquet as pd_read_parquet
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_feather, read_file, read_parquet
from geopandas._compat import HAS_PYPROJ
from geopandas.array import to_wkb
from geopandas.io.arrow import METADATA_VERSION, SUPPORTED_VERSIONS, _convert_bbox_to_parquet_filter, _create_metadata, _decode_metadata, _encode_metadata, _geopandas_to_arrow, _get_filesystem_path, _remove_id_from_member_of_ensembles, _validate_dataframe, _validate_geo_metadata
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
from pandas.testing import assert_frame_equal
DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / 'data'
pyarrow = pytest.importorskip('pyarrow')
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import feather

@pytest.mark.parametrize('test_dataset', ['naturalearth_lowres', 'naturalearth_cities', 'nybb_filename'])
def test_roundtrip(tmpdir, file_format, test_dataset, request):
    """Writing to parquet should not raise errors, and should not alter original
    GeoDataFrame
    """
    gdf = request.getfixturevalue(test_dataset)
    gdf_copy = gdf.copy()
    
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath)
    
    assert_geodataframe_equal(gdf, read_gdf)
    assert_geodataframe_equal(gdf, gdf_copy)

def test_index(tmpdir, file_format, naturalearth_lowres):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """
    gdf = naturalearth_lowres
    gdf = gdf.set_index('name')
    filepath = str(tmpdir.join(f"test.{file_format}"))

    # Test with index=True
    if file_format == 'parquet':
        gdf.to_parquet(filepath, index=True)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath, index=True)
        read_gdf = read_feather(filepath)
    
    assert_geodataframe_equal(gdf, read_gdf)
    assert gdf.index.equals(read_gdf.index)

    # Test with index=False
    if file_format == 'parquet':
        gdf.to_parquet(filepath, index=False)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath, index=False)
        read_gdf = read_feather(filepath)
    
    assert_geodataframe_equal(gdf.reset_index(drop=True), read_gdf)
    assert read_gdf.index.equals(pd.RangeIndex(len(gdf)))

def test_column_order(tmpdir, file_format, naturalearth_lowres):
    """The order of columns should be preserved in the output."""
    gdf = naturalearth_lowres
    new_column_order = list(reversed(gdf.columns))
    gdf = gdf.reindex(columns=new_column_order)
    
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath)
    
    assert list(gdf.columns) == list(read_gdf.columns)

@pytest.mark.parametrize('compression', ['snappy', 'gzip', 'brotli', None])
def test_parquet_compression(compression, tmpdir, naturalearth_lowres):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """
    gdf = naturalearth_lowres
    filepath = str(tmpdir.join(f"test_{compression}.parquet"))
    
    gdf.to_parquet(filepath, compression=compression)
    read_gdf = read_parquet(filepath)
    
    assert_geodataframe_equal(gdf, read_gdf)

@pytest.mark.skipif(Version(pyarrow.__version__) < Version('0.17.0'), reason='Feather only supported for pyarrow >= 0.17')
@pytest.mark.parametrize('compression', ['uncompressed', 'lz4', 'zstd'])
def test_feather_compression(compression, tmpdir, naturalearth_lowres):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """
    gdf = naturalearth_lowres
    filepath = str(tmpdir.join(f"test_{compression}.feather"))
    
    gdf.to_feather(filepath, compression=compression)
    read_gdf = read_feather(filepath)
    
    assert_geodataframe_equal(gdf, read_gdf)

def test_parquet_multiple_geom_cols(tmpdir, file_format, naturalearth_lowres):
    """If multiple geometry columns are present when written to parquet,
    they should all be returned as such when read from parquet.
    """
    gdf = naturalearth_lowres
    gdf['geometry2'] = gdf.geometry.centroid
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath)
    
    assert isinstance(read_gdf['geometry'], geopandas.GeoSeries)
    assert isinstance(read_gdf['geometry2'], geopandas.GeoSeries)
    assert_geodataframe_equal(gdf, read_gdf)

def test_parquet_missing_metadata(tmpdir, naturalearth_lowres):
    """Missing geo metadata, such as from a parquet file created
    from a pandas DataFrame, will raise a ValueError.
    """
    df = pd.DataFrame(naturalearth_lowres.drop(columns='geometry'))
    filepath = str(tmpdir.join("test.parquet"))
    df.to_parquet(filepath)
    
    with pytest.raises(ValueError, match="Missing geo metadata in Parquet/Feather file"):
        read_parquet(filepath)

def test_parquet_missing_metadata2(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pyarrow Table (which will also not contain pandas metadata),
    will raise a ValueError.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pa.table({'a': [1, 2, 3]})
    filepath = str(tmpdir.join("test.parquet"))
    pq.write_table(table, filepath)
    
    with pytest.raises(ValueError, match="Missing geo metadata in Parquet/Feather file"):
        read_parquet(filepath)

@pytest.mark.parametrize('geo_meta,error', [({'geo': b''}, 'Missing or malformed geo metadata in Parquet/Feather file'), ({'geo': _encode_metadata({})}, 'Missing or malformed geo metadata in Parquet/Feather file'), ({'geo': _encode_metadata({'foo': 'bar'})}, "'geo' metadata in Parquet/Feather file is missing required key")])
def test_parquet_invalid_metadata(tmpdir, geo_meta, error, naturalearth_lowres):
    """Has geo metadata with missing required fields will raise a ValueError.

    This requires writing the parquet file directly below, so that we can
    control the metadata that is written for this test.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    gdf = naturalearth_lowres
    table = pa.Table.from_pandas(gdf)
    filepath = str(tmpdir.join("test.parquet"))
    
    pq.write_table(table, filepath, metadata=geo_meta)
    
    with pytest.raises(ValueError, match=error):
        read_parquet(filepath)

def test_subset_columns(tmpdir, file_format, naturalearth_lowres):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """
    gdf = naturalearth_lowres
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath, columns=['name', 'geometry'])
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath, columns=['name', 'geometry'])
    
    assert list(read_gdf.columns) == ['name', 'geometry']
    assert isinstance(read_gdf['geometry'], geopandas.GeoSeries)
    assert_geodataframe_equal(gdf[['name', 'geometry']], read_gdf)

def test_promote_secondary_geometry(tmpdir, file_format, naturalearth_lowres):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present.
    """
    gdf = naturalearth_lowres
    gdf['geometry2'] = gdf.geometry.centroid
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath, columns=['name', 'geometry2'])
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath, columns=['name', 'geometry2'])
    
    assert read_gdf.geometry.name == 'geometry2'
    assert isinstance(read_gdf.geometry, geopandas.GeoSeries)
    assert_geodataframe_equal(gdf[['name', 'geometry2']].set_geometry('geometry2'), read_gdf)

def test_columns_no_geometry(tmpdir, file_format, naturalearth_lowres):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""
    gdf = naturalearth_lowres
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        with pytest.raises(ValueError, match="No geometry columns found"):
            read_parquet(filepath, columns=['name', 'pop_est'])
    else:  # feather
        gdf.to_feather(filepath)
        with pytest.raises(ValueError, match="No geometry columns found"):
            read_feather(filepath, columns=['name', 'pop_est'])

def test_missing_crs(tmpdir, file_format, naturalearth_lowres):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """
    gdf = naturalearth_lowres.copy()
    gdf.crs = None
    filepath = str(tmpdir.join(f"test.{file_format}"))
    
    if file_format == 'parquet':
        gdf.to_parquet(filepath)
        read_gdf = read_parquet(filepath)
    else:  # feather
        gdf.to_feather(filepath)
        read_gdf = read_feather(filepath)
    
    assert read_gdf.crs is None
    assert_geodataframe_equal(gdf, read_gdf)

@pytest.mark.parametrize('version', ['0.1.0', '0.4.0', '1.0.0-beta.1'])
def test_read_versioned_file(version):
    """
    Verify that files for different metadata spec versions can be read
    created for each supported version:

    # small dummy test dataset (not naturalearth_lowres, as this can change over time)
    from shapely.geometry import box, MultiPolygon
    df = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5,5)],
        crs="EPSG:4326",
    )
    df.to_feather(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.feather')
    df.to_parquet(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.parquet')
    """
    feather_path = DATA_PATH / 'arrow' / f'test_data_v{version}.feather'
    parquet_path = DATA_PATH / 'arrow' / f'test_data_v{version}.parquet'
    
    feather_gdf = read_feather(feather_path)
    parquet_gdf = read_parquet(parquet_path)
    
    assert isinstance(feather_gdf, geopandas.GeoDataFrame)
    assert isinstance(parquet_gdf, geopandas.GeoDataFrame)
    assert_geodataframe_equal(feather_gdf, parquet_gdf)
    
    expected_columns = ['col_str', 'col_int', 'col_float', 'geometry']
    assert list(feather_gdf.columns) == expected_columns
    assert list(parquet_gdf.columns) == expected_columns
    
    assert feather_gdf.crs == "EPSG:4326"
    assert parquet_gdf.crs == "EPSG:4326"

def test_read_gdal_files():
    """
    Verify that files written by GDAL can be read by geopandas.
    This test uses pre-generated files for GDAL versions 3.5.0 and 3.9.0.
    """
    gdal350_parquet = DATA_PATH / 'arrow' / 'test_data_gdal350.parquet'
    gdal350_arrow = DATA_PATH / 'arrow' / 'test_data_gdal350.arrow'
    gdal390_parquet = DATA_PATH / 'arrow' / 'test_data_gdal390.parquet'
    
    gdfs = {
        'GDAL 3.5.0 Parquet': read_parquet(gdal350_parquet),
        'GDAL 3.5.0 Arrow': read_feather(gdal350_arrow),
        'GDAL 3.9.0 Parquet': read_parquet(gdal390_parquet)
    }
    
    expected_columns = ['col_str', 'col_int', 'col_float', 'geometry']
    expected_data = {
        'col_str': ['a', 'b'],
        'col_int': [1, 2],
        'col_float': [0.1, 0.2],
    }
    
    for name, gdf in gdfs.items():
        assert isinstance(gdf, geopandas.GeoDataFrame), f"{name} should be a GeoDataFrame"
        assert list(gdf.columns) == expected_columns, f"{name} columns do not match expected"
        assert gdf.crs == "EPSG:4326", f"{name} CRS should be EPSG:4326"
        assert len(gdf) == 2, f"{name} should have 2 rows"
        
        for col, values in expected_data.items():
            assert gdf[col].tolist() == values, f"{name} {col} data does not match expected"
        
        assert all(isinstance(geom, (shapely.geometry.MultiPolygon, shapely.geometry.Polygon)) for geom in gdf.geometry), \
            f"{name} geometry types are not as expected"
        
        assert isinstance(gdf.geometry.iloc[0], shapely.geometry.MultiPolygon), f"{name} first geometry should be MultiPolygon"
        assert isinstance(gdf.geometry.iloc[1], shapely.geometry.Polygon), f"{name} second geometry should be Polygon"
        assert gdf.geometry.iloc[0].bounds == (0, 0, 3, 3), f"{name} first geometry bounds are incorrect"
        assert gdf.geometry.iloc[1].bounds == (4, 4, 5, 5), f"{name} second geometry bounds are incorrect"
    
    # Check if GDAL 3.9 version has bbox column
    assert 'bbox' in gdfs['GDAL 3.9.0 Parquet']._metadata['columns']['geometry'], "GDAL 3.9.0 Parquet should have bbox metadata"
