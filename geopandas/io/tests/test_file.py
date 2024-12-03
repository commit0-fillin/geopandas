import datetime
import io
import json
import os
import pathlib
import shutil
import tempfile
from collections import OrderedDict
from packaging.version import Version
import numpy as np
import pandas as pd
import pytz
from pandas.api.types import is_datetime64_any_dtype
from shapely.geometry import Point, Polygon, box, mapping
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import HAS_PYPROJ, PANDAS_GE_20, PANDAS_GE_30
from geopandas.io.file import _EXTENSION_TO_DRIVER, _detect_driver
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_series_equal
try:
    import pyogrio
    PYOGRIO_GE_090 = Version(Version(pyogrio.__version__).base_version) >= Version('0.9.0')
except ImportError:
    pyogrio = False
    PYOGRIO_GE_090 = False
try:
    import fiona
    FIONA_GE_19 = Version(Version(fiona.__version__).base_version) >= Version('1.9.0')
except ImportError:
    fiona = False
    FIONA_GE_19 = False
PYOGRIO_MARK = pytest.mark.skipif(not pyogrio, reason='pyogrio not installed')
FIONA_MARK = pytest.mark.skipif(not fiona, reason='fiona not installed')
_CRS = 'epsg:4326'
pytestmark = pytest.mark.filterwarnings('ignore:Value:RuntimeWarning:pyogrio')
driver_ext_pairs = [('ESRI Shapefile', '.shp'), ('GeoJSON', '.geojson'), ('GPKG', '.gpkg'), (None, '.shp'), (None, ''), (None, '.geojson'), (None, '.gpkg')]

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file(tmpdir, df_nybb, df_null, driver, ext, engine):
    """Test to_file and from_file"""
    filename = str(tmpdir.join(f"test{ext}"))
    df_nybb.to_file(filename, driver=driver, engine=engine)
    
    # Read the file back
    df = read_file(filename, engine=engine)
    
    # Check that the read DataFrame is equal to the original
    assert_geodataframe_equal(df, df_nybb)
    
    # Test with df_null (contains null geometries)
    filename_null = str(tmpdir.join(f"test_null{ext}"))
    df_null.to_file(filename_null, driver=driver, engine=engine)
    
    # Read the null geometry file back
    df_null_read = read_file(filename_null, engine=engine)
    
    # Check that the read DataFrame is equal to the original
    assert_geodataframe_equal(df_null_read, df_null)

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_pathlib(tmpdir, df_nybb, driver, ext, engine):
    """Test to_file and from_file with pathlib.Path"""
    filename = tmpdir.join(f"test{ext}")
    path = Path(filename)
    df_nybb.to_file(path, driver=driver, engine=engine)
    
    # Read the file back using pathlib.Path
    df = read_file(path, engine=engine)
    
    # Check that the read DataFrame is equal to the original
    assert_geodataframe_equal(df, df_nybb)

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_bool(tmpdir, driver, ext, engine):
    """Test error raise when writing with a boolean column (GH #437)."""
    df = GeoDataFrame(
        {'a': [1, 2, 3], 'b': [True, False, True]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)]
    )
    
    filename = str(tmpdir.join(f"test{ext}"))
    
    if driver == 'ESRI Shapefile':
        with pytest.raises(ValueError, match="ESRI Shapefile does not support boolean"):
            df.to_file(filename, driver=driver, engine=engine)
    else:
        df.to_file(filename, driver=driver, engine=engine)
        df_read = read_file(filename, engine=engine)
        assert_geodataframe_equal(df_read, df)
TEST_DATE = datetime.datetime(2021, 11, 21, 1, 7, 43, 17500)
eastern = pytz.timezone('America/New_York')
datetime_type_tests = (TEST_DATE, eastern.localize(TEST_DATE))

@pytest.mark.filterwarnings('ignore:Non-conformant content for record 1 in column b:RuntimeWarning')
@pytest.mark.parametrize('time', datetime_type_tests, ids=('naive_datetime', 'datetime_with_timezone'))
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_datetime(tmpdir, driver, ext, time, engine):
    """Test writing a data file with the datetime column type"""
    pass
dt_exts = ['gpkg', 'geojson']

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_with_point_z(tmpdir, ext, driver, engine):
    """Test that 3D geometries are retained in writes (GH #612)."""
    pass

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_with_poly_z(tmpdir, ext, driver, engine):
    """Test that 3D geometries are retained in writes (GH #612)."""
    pass

def test_to_file_types(tmpdir, df_points, engine):
    """Test various integer type columns (GH#93)"""
    pass

def test_to_file_schema(tmpdir, df_nybb, engine):
    """
    Ensure that the file is written according to the schema
    if it is specified

    """
    pass

@pytest.mark.skipif(not HAS_PYPROJ, reason='pyproj not installed')
def test_to_file_crs(tmpdir, engine, nybb_filename):
    """
    Ensure that the file is written according to the crs
    if it is specified
    """
    pass

def test_to_file_column_len(tmpdir, df_points, engine):
    """
    Ensure that a warning about truncation is given when a geodataframe with
    column names longer than 10 characters is saved to shapefile
    """
    pass

@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_append_file(tmpdir, df_nybb, df_null, driver, ext, engine):
    """Test to_file with append mode and from_file"""
    pass

@pytest.mark.filterwarnings("ignore:'crs' was not provided:UserWarning:pyogrio")
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_empty_crs(tmpdir, driver, ext, engine):
    """Test handling of undefined CRS with GPKG driver (GH #1975)."""
    pass
NYBB_CRS = 'epsg:2263'

class FileNumber(object):

    def __init__(self, tmpdir, base, ext):
        self.tmpdir = str(tmpdir)
        self.base = base
        self.ext = ext
        self.fileno = 0

    def __repr__(self):
        filename = '{0}{1:02d}.{2}'.format(self.base, self.fileno, self.ext)
        return os.path.join(self.tmpdir, filename)

    def __next__(self):
        self.fileno += 1
        return repr(self)
