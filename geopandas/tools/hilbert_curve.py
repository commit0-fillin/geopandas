import numpy as np

def _hilbert_distance(geoms, total_bounds=None, level=16):
    """
    Calculate the distance along a Hilbert curve.

    The distances are calculated for the midpoints of the geometries in the
    GeoDataFrame.

    Parameters
    ----------
    geoms : GeometryArray
    total_bounds : 4-element array
        Total bounds of geometries - array
    level : int (1 - 16), default 16
        Determines the precision of the curve (points on the curve will
        have coordinates in the range [0, 2^level - 1]).

    Returns
    -------
    np.ndarray
        Array containing distances along the Hilbert curve

    """
    if total_bounds is None:
        total_bounds = geoms.total_bounds

    bounds = geoms.bounds
    discrete_coords = _continuous_to_discrete_coords(bounds, level, total_bounds)

    x, y = discrete_coords[:, 0], discrete_coords[:, 1]
    h = np.zeros(x.shape, dtype=np.uint64)

    for i in range(level):
        h = (h << 2) | ((x & (1 << (level - 1 - i))) << 1) | (y & (1 << (level - 1 - i)))

    return h

def _continuous_to_discrete_coords(bounds, level, total_bounds):
    """
    Calculates mid points & ranges of geoms and returns
    as discrete coords

    Parameters
    ----------

    bounds : Bounds of each geometry - array

    p : The number of iterations used in constructing the Hilbert curve

    total_bounds : Total bounds of geometries - array

    Returns
    -------
    Discrete two-dimensional numpy array
    Two-dimensional array Array of hilbert distances for each geom

    """
    minx, miny, maxx, maxy = total_bounds
    x_range = (minx, maxx)
    y_range = (miny, maxy)

    x_mid = (bounds[:, 0] + bounds[:, 2]) / 2
    y_mid = (bounds[:, 1] + bounds[:, 3]) / 2

    n = 2**level

    x_discrete = _continuous_to_discrete(x_mid, x_range, n)
    y_discrete = _continuous_to_discrete(y_mid, y_range, n)

    return np.column_stack((x_discrete, y_discrete))

def _continuous_to_discrete(vals, val_range, n):
    """
    Convert a continuous one-dimensional array to discrete integer values
    based their ranges

    Parameters
    ----------
    vals : Array of continuous values

    val_range : Tuple containing range of continuous values

    n : Number of discrete values

    Returns
    -------
    One-dimensional array of discrete ints

    """
    min_val, max_val = val_range
    scaled = (vals - min_val) / (max_val - min_val)
    return np.clip((scaled * (n - 1)).astype(int), 0, n - 1)
MAX_LEVEL = 16
