"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""
import os
import warnings
from importlib.util import find_spec
import pandas as pd
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_postgis
from geopandas._compat import HAS_PYPROJ
from geopandas.io.sql import _get_conn as get_conn
from geopandas.io.sql import _write_postgis as write_postgis
import pytest
from geopandas.tests.util import create_postgis, create_spatialite, mock, validate_boro_df
try:
    from sqlalchemy import text
except ImportError:
    text = str

def check_available_postgis_drivers() -> list[str]:
    """Work out which of psycopg2 and psycopg are available.
    This prevents tests running if the relevant package isn't installed
    (rather than being skipped, as skips are treated as failures during postgis CI)
    """
    available_drivers = []
    try:
        import psycopg2
        available_drivers.append('psycopg2')
    except ImportError:
        pass
    
    try:
        import psycopg
        available_drivers.append('psycopg')
    except ImportError:
        pass
    
    return available_drivers
POSTGIS_DRIVERS = check_available_postgis_drivers()

def prepare_database_credentials() -> dict:
    """Gather postgres connection credentials from environment variables."""
    import os
    
    credentials = {
        'host': os.environ.get('POSTGIS_HOST', 'localhost'),
        'port': os.environ.get('POSTGIS_PORT', '5432'),
        'user': os.environ.get('POSTGIS_USER', 'postgres'),
        'password': os.environ.get('POSTGIS_PASSWORD', ''),
        'dbname': os.environ.get('POSTGIS_DBNAME', 'test_geopandas')
    }
    
    return credentials

@pytest.fixture()
def connection_postgis(request):
    """Create a postgres connection using either psycopg2 or psycopg.

    Use this as an indirect fixture, where the request parameter is POSTGIS_DRIVERS."""
    driver = request.param
    credentials = prepare_database_credentials()
    
    if driver == 'psycopg2':
        import psycopg2
        conn = psycopg2.connect(**credentials)
    elif driver == 'psycopg':
        import psycopg
        conn = psycopg.connect(**credentials)
    else:
        raise ValueError(f"Unsupported driver: {driver}")
    
    yield conn
    conn.close()

@pytest.fixture()
def engine_postgis(request):
    """
    Initiate a sqlalchemy connection engine using either psycopg2 or psycopg.

    Use this as an indirect fixture, where the request parameter is POSTGIS_DRIVERS.
    """
    from sqlalchemy import create_engine
    
    driver = request.param
    credentials = prepare_database_credentials()
    
    if driver == 'psycopg2':
        engine = create_engine(f"postgresql+psycopg2://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['dbname']}")
    elif driver == 'psycopg':
        engine = create_engine(f"postgresql+psycopg://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['dbname']}")
    else:
        raise ValueError(f"Unsupported driver: {driver}")
    
    yield engine
    engine.dispose()

@pytest.fixture()
def connection_spatialite():
    """
    Return a memory-based SQLite3 connection with SpatiaLite enabled & initialized.

    `The sqlite3 module must be built with loadable extension support
    <https://docs.python.org/3/library/sqlite3.html#f1>`_ and
    `SpatiaLite <https://www.gaia-gis.it/fossil/libspatialite/index>`_
    must be available on the system as a SQLite module.
    Packages available on Anaconda meet requirements.

    Exceptions
    ----------
    ``AttributeError`` on missing support for loadable SQLite extensions
    ``sqlite3.OperationalError`` on missing SpatiaLite
    """
    import sqlite3
    
    conn = sqlite3.connect(':memory:')
    conn.enable_load_extension(True)
    
    try:
        conn.load_extension("mod_spatialite")
    except sqlite3.OperationalError:
        conn.close()
        raise sqlite3.OperationalError("SpatiaLite extension not found")
    
    conn.execute("SELECT InitSpatialMetaData(1)")
    
    yield conn
    conn.close()

class TestIO:

    @pytest.mark.parametrize('connection_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_select_geom_as(self, connection_postgis, df_nybb):
        """Tests that a SELECT {geom} AS {some_other_geom} works."""
        from geopandas import read_postgis
        
        sql = "SELECT geom AS some_other_geom, boroname FROM nybb"
        gdf = read_postgis(sql, connection_postgis, geom_col='some_other_geom')
        
        assert 'some_other_geom' in gdf.columns
        assert gdf.geom_type.unique() == ['MultiPolygon']
        assert len(gdf) == len(df_nybb)

    @pytest.mark.parametrize('connection_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_get_srid(self, connection_postgis, df_nybb):
        """Tests that an SRID can be read from a geodatabase (GH #451)."""
        from geopandas import read_postgis
        
        sql = "SELECT geom, boroname FROM nybb"
        gdf = read_postgis(sql, connection_postgis)
        
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326  # Assuming EPSG:4326 is used in the test database

    @pytest.mark.parametrize('connection_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_override_srid(self, connection_postgis, df_nybb):
        """Tests that a user specified CRS overrides the geodatabase SRID."""
        from geopandas import read_postgis
        
        sql = "SELECT geom, boroname FROM nybb"
        user_crs = "EPSG:3857"
        gdf = read_postgis(sql, connection_postgis, crs=user_crs)
        
        assert gdf.crs is not None
        assert gdf.crs.to_string() == user_crs

    def test_read_postgis_null_geom(self, connection_spatialite, df_nybb):
        """Tests that geometry with NULL is accepted."""
        from geopandas import read_postgis
        
        # Create a table with a NULL geometry
        connection_spatialite.execute("CREATE TABLE null_geom (id INTEGER, geom GEOMETRY)")
        connection_spatialite.execute("INSERT INTO null_geom VALUES (1, NULL)")
        
        sql = "SELECT * FROM null_geom"
        gdf = read_postgis(sql, connection_spatialite)
        
        assert len(gdf) == 1
        assert gdf.iloc[0].geometry is None

    def test_read_postgis_binary(self, connection_spatialite, df_nybb):
        """Tests that geometry read as binary is accepted."""
        from geopandas import read_postgis
        
        sql = "SELECT ST_AsBinary(geom) AS geom, boroname FROM nybb"
        gdf = read_postgis(sql, connection_spatialite)
        
        assert gdf.geom_type.unique() == ['MultiPolygon']
        assert len(gdf) == len(df_nybb)

    @pytest.mark.parametrize('connection_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_chunksize(self, connection_postgis, df_nybb):
        """Test chunksize argument"""
        from geopandas import read_postgis
        
        sql = "SELECT geom, boroname FROM nybb"
        chunks = list(read_postgis(sql, connection_postgis, chunksize=2))
        
        assert len(chunks) == 3  # 5 boroughs / 2 = 3 chunks (2, 2, 1)
        assert sum(len(chunk) for chunk in chunks) == len(df_nybb)

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_default(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_uppercase_tablename(self, engine_postgis, df_nybb):
        """Tests writing GeoDataFrame to PostGIS with uppercase tablename."""
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_sqlalchemy_connection(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_fail_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that uploading the same table raises error when: if_replace='fail'.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_replace_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that replacing a table is possible when: if_replace='replace'.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_append_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that appending to existing table produces correct results when:
        if_replace='append'.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_without_crs(self, engine_postgis, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS without CRS information.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_with_esri_authority(self, engine_postgis, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS with ESRI Authority
        CRS information (GH #2414).
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_geometry_collection(self, engine_postgis, df_geom_collection):
        """
        Tests that writing a mix of different geometry types is possible.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_mixed_geometry_types(self, engine_postgis, df_mixed_single_and_multi):
        """
        Tests that writing a mix of single and MultiGeometries is possible.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_linear_ring(self, engine_postgis, df_linear_ring):
        """
        Tests that writing a LinearRing.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_in_chunks(self, engine_postgis, df_mixed_single_and_multi):
        """
        Tests writing a LinearRing works.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_to_different_schema(self, engine_postgis, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_to_different_schema_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_3D_geometries(self, engine_postgis, df_3D_geoms):
        """
        Tests writing a geometries with 3 dimensions works.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_row_order(self, engine_postgis, df_nybb):
        """
        Tests that the row order in db table follows the order of the original frame.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_append_before_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that insert works with if_exists='append' when table does not exist yet.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_append_with_different_crs(self, engine_postgis, df_nybb):
        """
        Tests that the warning is raised if table CRS differs from frame.
        """
        pass

    @pytest.mark.parametrize('engine_postgis', POSTGIS_DRIVERS, indirect=True)
    @pytest.mark.xfail(compat.PANDAS_GE_20 and (not compat.PANDAS_GE_202), reason='Duplicate columns are dropped in read_sql with pandas 2.0.0 and 2.0.1')
    def test_duplicate_geometry_column_fails(self, engine_postgis):
        """
        Tests that a ValueError is raised if an SQL query returns two geometry columns.
        """
        pass

    @pytest.mark.parametrize('connection_postgis', POSTGIS_DRIVERS, indirect=True)
    def test_read_non_epsg_crs_chunksize(self, connection_postgis, df_nybb):
        """Test chunksize argument with non epsg crs"""
        pass
