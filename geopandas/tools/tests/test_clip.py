"""Tests for the clip module."""
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection, LinearRing, LineString, MultiPoint, Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas._compat import HAS_PYPROJ
from geopandas.tools.clip import _mask_is_list_like_rectangle
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_index_equal
mask_variants_single_rectangle = ['single_rectangle_gdf', 'single_rectangle_gdf_list_bounds', 'single_rectangle_gdf_tuple_bounds', 'single_rectangle_gdf_array_bounds']
mask_variants_large_rectangle = ['larger_single_rectangle_gdf', 'larger_single_rectangle_gdf_bounds']

@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    from shapely.geometry import Point
    import geopandas as gpd
    
    points = [Point(0, 0), Point(1, 1), Point(2, 2)]
    return gpd.GeoDataFrame(geometry=points)

@pytest.fixture
def point_gdf2():
    """Create a point GeoDataFrame."""
    from shapely.geometry import Point
    import geopandas as gpd
    import pandas as pd
    
    points = [Point(0, 0), Point(1, 1), Point(2, 2)]
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    return gpd.GeoDataFrame(df, geometry=points)

@pytest.fixture
def pointsoutside_nooverlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are outside the single rectangle's."""
    from shapely.geometry import Point
    import geopandas as gpd
    
    points = [Point(-2, -2), Point(-1, -1), Point(-3, -3)]
    return gpd.GeoDataFrame(geometry=points)

@pytest.fixture
def pointsoutside_overlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are overlapping the single rectangle's."""
    from shapely.geometry import Point
    import geopandas as gpd
    
    points = [Point(-1, -1), Point(3, 3), Point(1, 3)]
    return gpd.GeoDataFrame(geometry=points)

@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    from shapely.geometry import box
    import geopandas as gpd
    
    rectangle = box(0, 0, 2, 2)
    return gpd.GeoDataFrame(geometry=[rectangle])

@pytest.fixture
def single_rectangle_gdf_tuple_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    pass

@pytest.fixture
def single_rectangle_gdf_list_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    pass

@pytest.fixture
def single_rectangle_gdf_array_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    pass

@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
    The smaller single rectangle is used to test the edge case where slivers
    are returned when you clip polygons. This fixture is larger which
    eliminates the slivers in the clip return.
    """
    pass

@pytest.fixture
def larger_single_rectangle_gdf_bounds(larger_single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    pass

@pytest.fixture
def buffered_locations(point_gdf):
    """Buffer points to create a multi-polygon."""
    pass

@pytest.fixture
def donut_geometry(buffered_locations, single_rectangle_gdf):
    """Make a geometry with a hole in the middle (a donut)."""
    pass

@pytest.fixture
def two_line_gdf():
    """Create Line Objects For Testing"""
    pass

@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    pass

@pytest.fixture
def multi_line(two_line_gdf):
    """Create a multi-line GeoDataFrame.
    This GDF has one multiline and one regular line."""
    pass

@pytest.fixture
def multi_point(point_gdf):
    """Create a multi-point GeoDataFrame."""
    pass

@pytest.fixture
def mixed_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    pass

@pytest.fixture
def geomcol_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    pass

@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    pass

def test_not_gdf(single_rectangle_gdf):
    """Non-GeoDataFrame inputs raise attribute errors."""
    pass

def test_non_overlapping_geoms():
    """Test that a bounding box returns empty if the extents don't overlap"""
    pass

@pytest.mark.parametrize('mask_fixture_name', mask_variants_single_rectangle)
class TestClipWithSingleRectangleGdf:

    def test_returns_gdf(self, point_gdf, mask):
        """Test that function returns a GeoDataFrame (or GDF-like) object."""
        pass

    def test_returns_series(self, point_gdf, mask):
        """Test that function returns a GeoSeries if GeoSeries is passed."""
        pass

    def test_clip_points(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        pass

    def test_clip_points_geom_col_rename(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        pass

    def test_clip_poly(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        pass

    def test_clip_poly_geom_col_rename(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        pass

    def test_clip_poly_series(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        pass

    def test_clip_multipoly_keep_geom_type(self, multi_poly_gdf, mask):
        """Test a multi poly object where the return includes a sliver.
        Also the bounds of the object should == the bounds of the clip object
        if they fully overlap (as they do in these fixtures)."""
        pass

    def test_clip_multiline(self, multi_line, mask):
        """Test that clipping a multiline feature with a poly returns expected
        output."""
        pass

    def test_clip_multipoint(self, multi_point, mask):
        """Clipping a multipoint feature with a polygon works as expected.
        should return a geodataframe with a single multi point feature"""
        pass

    def test_clip_lines(self, two_line_gdf, mask):
        """Test what happens when you give the clip_extent a line GDF."""
        pass

    def test_mixed_geom(self, mixed_gdf, mask):
        """Test clipping a mixed GeoDataFrame"""
        pass

    def test_mixed_series(self, mixed_gdf, mask):
        """Test clipping a mixed GeoSeries"""
        pass

    def test_clip_with_line_extra_geom(self, sliver_line, mask):
        """When the output of a clipped line returns a geom collection,
        and keep_geom_type is True, no geometry collections should be returned."""
        pass

    def test_clip_no_box_overlap(self, pointsoutside_nooverlap_gdf, mask):
        """Test clip when intersection is empty and boxes do not overlap."""
        pass

    def test_clip_box_overlap(self, pointsoutside_overlap_gdf, mask):
        """Test clip when intersection is empty and boxes do overlap."""
        pass

    def test_warning_extra_geoms_mixed(self, mixed_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a mixed GDF"""
        pass

    def test_warning_geomcoll(self, geomcol_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a GDF with GeometryCollection"""
        pass

def test_clip_line_keep_slivers(sliver_line, single_rectangle_gdf):
    """Test the correct output if a point is returned
    from a line only geometry type."""
    pass

def test_clip_multipoly_keep_slivers(multi_poly_gdf, single_rectangle_gdf):
    """Test a multi poly object where the return includes a sliver.
    Also the bounds of the object should == the bounds of the clip object
    if they fully overlap (as they do in these fixtures)."""
    pass

def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    pass

def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon."""
    pass

@pytest.mark.parametrize('mask_fixture_name', mask_variants_large_rectangle)
def test_clip_single_multipoly_no_extra_geoms(buffered_locations, mask_fixture_name, request):
    """When clipping a multi-polygon feature, no additional geom types
    should be returned."""
    pass

@pytest.mark.filterwarnings('ignore:All-NaN slice encountered')
@pytest.mark.parametrize('mask', [Polygon(), (np.nan,) * 4, (np.nan, 0, np.nan, 1), GeoSeries([Polygon(), Polygon()], crs='EPSG:3857'), GeoSeries([Polygon(), Polygon()], crs='EPSG:3857').to_frame(), GeoSeries([], crs='EPSG:3857'), GeoSeries([], crs='EPSG:3857').to_frame()])
def test_clip_empty_mask(buffered_locations, mask):
    """Test that clipping with empty mask returns an empty result."""
    pass

def test_clip_sorting(point_gdf2):
    """Test the sorting kwarg in clip"""
    pass
