import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
import geopandas
import pytest
from pandas.testing import assert_series_equal
from geopandas.tools.hilbert_curve import (_hilbert_distance, 
                                           _continuous_to_discrete_coords, 
                                           _continuous_to_discrete)

def test_hilbert_distance():
    geoms = geopandas.GeoSeries([Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)])
    total_bounds = (0, 0, 1, 1)
    result = _hilbert_distance(geoms.geometry, total_bounds, level=2)
    expected = np.array([0, 3, 1, 2])
    np.testing.assert_array_equal(result, expected)

def test_continuous_to_discrete_coords():
    bounds = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    total_bounds = (0, 0, 2, 2)
    level = 2
    result = _continuous_to_discrete_coords(bounds, level, total_bounds)
    expected = np.array([[1, 1], [3, 3]])
    np.testing.assert_array_equal(result, expected)

def test_continuous_to_discrete():
    vals = np.array([0, 0.5, 1])
    val_range = (0, 1)
    n = 4
    result = _continuous_to_discrete(vals, val_range, n)
    expected = np.array([0, 1, 3])
    np.testing.assert_array_equal(result, expected)
