from shapely.geometry import LineString, MultiPoint, Point, Polygon
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest

class TestTools:
    
    def test_collect_points(self):
        points = GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
        result = collect(points)
        assert isinstance(result, MultiPoint)
        assert len(result.geoms) == 3

    def test_collect_mixed_geometries(self):
        geometries = GeoSeries([Point(0, 0), LineString([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 1), (1, 0)])])
        result = collect(geometries)
        assert isinstance(result, Polygon)
        assert len(result.geoms) == 3

    def test_collect_single_geometry(self):
        single = GeoSeries([Point(0, 0)])
        result = collect(single)
        assert isinstance(result, Point)

    def test_collect_empty_series(self):
        empty = GeoSeries([])
        result = collect(empty)
        assert result is None

    def test_collect_with_multi_flag(self):
        points = GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
        result = collect(points, multi=True)
        assert isinstance(result, MultiPoint)
        assert len(result.geoms) == 3

    def test_collect_linestrings(self):
        lines = GeoSeries([LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])])
        result = collect(lines)
        assert isinstance(result, LineString)
        assert len(result.coords) == 3

    @pytest.mark.parametrize("geom_type", [Point, LineString, Polygon])
    def test_collect_homogeneous_geometries(self, geom_type):
        if geom_type == Point:
            geometries = GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
        elif geom_type == LineString:
            geometries = GeoSeries([LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])])
        else:  # Polygon
            geometries = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)]), Polygon([(1, 1), (2, 2), (2, 1)])])
        
        result = collect(geometries)
        assert isinstance(result, geom_type)
