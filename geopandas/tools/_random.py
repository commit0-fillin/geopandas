from warnings import warn
import numpy
from shapely.geometry import MultiPoint
from geopandas.array import from_shapely, points_from_xy
from geopandas.geoseries import GeoSeries

def uniform(geom, size, rng=None):
    """
    Sample uniformly at random from a geometry.

    For polygons, this samples uniformly within the area of the polygon. For lines,
    this samples uniformly along the length of the linestring. For multi-part
    geometries, the weights of each part are selected according to their relevant
    attribute (area for Polygons, length for LineStrings), and then points are
    sampled from each part uniformly.

    Any other geometry type (e.g. Point, GeometryCollection) are ignored, and an
    empty MultiPoint geometry is returned.

    Parameters
    ----------
    geom : any shapely.geometry.BaseGeometry type
        the shape that describes the area in which to sample.

    size : integer
        an integer denoting how many points to sample

    rng : numpy.random.Generator, optional
        A random number generator to use. If None, the default numpy RNG will be used.

    Returns
    -------
    shapely.MultiPoint geometry containing the sampled points

    Examples
    --------
    >>> from shapely.geometry import box
    >>> square = box(0,0,1,1)
    >>> uniform(square, size=102) # doctest: +SKIP
    """
    if rng is None:
        rng = numpy.random.default_rng()

    if isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        return _uniform_polygon(geom, size, rng)
    elif isinstance(geom, (shapely.geometry.LineString, shapely.geometry.MultiLineString)):
        return _uniform_line(geom, size, rng)
    else:
        return shapely.geometry.MultiPoint()

def _uniform_line(geom, size, generator):
    """
    Sample points from an input shapely linestring
    """
    if isinstance(geom, shapely.geometry.LineString):
        length = geom.length
        distances = generator.random(size) * length
        return shapely.geometry.MultiPoint([geom.interpolate(distance) for distance in distances])
    elif isinstance(geom, shapely.geometry.MultiLineString):
        lengths = numpy.array([line.length for line in geom.geoms])
        total_length = numpy.sum(lengths)
        weights = lengths / total_length
        choices = generator.choice(len(geom.geoms), size=size, p=weights)
        points = []
        for i, line in enumerate(geom.geoms):
            line_size = numpy.sum(choices == i)
            if line_size > 0:
                points.extend(_uniform_line(line, line_size, generator).geoms)
        return shapely.geometry.MultiPoint(points)

def _uniform_polygon(geom, size, generator):
    """
    Sample uniformly from within a polygon using batched sampling.
    """
    if isinstance(geom, shapely.geometry.Polygon):
        minx, miny, maxx, maxy = geom.bounds
        points = []
        while len(points) < size:
            batch_size = size - len(points)
            x = generator.uniform(minx, maxx, batch_size)
            y = generator.uniform(miny, maxy, batch_size)
            candidates = shapely.geometry.MultiPoint(list(zip(x, y)))
            valid_points = candidates[geom.contains(candidates)]
            points.extend(valid_points.geoms)
        return shapely.geometry.MultiPoint(points[:size])
    elif isinstance(geom, shapely.geometry.MultiPolygon):
        areas = numpy.array([poly.area for poly in geom.geoms])
        total_area = numpy.sum(areas)
        weights = areas / total_area
        choices = generator.choice(len(geom.geoms), size=size, p=weights)
        points = []
        for i, poly in enumerate(geom.geoms):
            poly_size = numpy.sum(choices == i)
            if poly_size > 0:
                points.extend(_uniform_polygon(poly, poly_size, generator).geoms)
        return shapely.geometry.MultiPoint(points)
