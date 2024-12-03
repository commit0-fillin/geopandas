import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection, Point, Polygon, box
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file, sjoin, sjoin_nearest
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

class TestSpatialJoin:
    pass

class TestIndexNames:
    pass

@pytest.mark.usefixtures('_setup_class_nybb_filename')
class TestSpatialJoinNYBB:

    @pytest.mark.parametrize('predicate', ['intersects', 'within', 'contains'])
    def test_sjoin_no_valid_geoms(self, predicate):
        """Tests a completely empty GeoDataFrame."""
        empty_df = GeoDataFrame(geometry=[])
        result = sjoin(empty_df, self.polydf, predicate=predicate)
        assert isinstance(result, GeoDataFrame)
        assert len(result) == 0
        assert result.crs == self.polydf.crs

class TestNearest:
    pass
