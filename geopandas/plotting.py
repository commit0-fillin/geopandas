import warnings
from packaging.version import Version
import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.plotting import PlotAccessor
import geopandas
from ._decorator import doc

def _sanitize_geoms(geoms, prefix='Multi'):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out.  Maintains 1:1 matching of geometry to value.

    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    components = []
    component_index = []

    for idx, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        if geom.type.startswith(prefix):
            for part in geom.geoms:
                components.append(part)
                component_index.append(idx)
        else:
            components.append(geom)
            component_index.append(idx)

    return components, np.array(component_index)

def _expand_kwargs(kwargs, multiindex):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    for key, value in kwargs.items():
        if isinstance(value, (list, np.ndarray, pd.Series)):
            if len(value) != len(multiindex):
                kwargs[key] = np.take(value, multiindex)
        elif not isinstance(value, (str, int, float, bool)):
            kwargs[key] = [value] * len(multiindex)

def _PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a Polygon geometry

    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes)::

        b = shapely.geometry.Point(0, 0).buffer(1.0)
        patch = _PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
        ax.add_patch(patch)

    GeoPandas originally relied on the descartes package by Sean Gillies
    (BSD license, https://pypi.org/project/descartes) for PolygonPatch, but
    this dependency was removed in favor of the below matplotlib code.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.
        vertices = np.concatenate(
            [np.asarray(polygon.exterior.coords)[:, :2]]
            + [np.asarray(r.coords)[:, :2] for r in polygon.interiors])
        codes = np.concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors])
        return Path(vertices, codes)

    path = pathify(polygon)
    return PathPatch(path, **kwargs)

def _plot_polygon_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, autolim=True, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs.
    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons
    facecolor : single color or sequence of `N` colors
        Color to fill the polygons. Cannot be used together with `values`.
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    geoms, multiindex = _sanitize_geoms(geoms)
    _expand_kwargs(kwargs, multiindex)

    patches = [_PolygonPatch(poly) for poly in geoms]
    collection = PatchCollection(patches, **kwargs)

    if values is not None:
        values = np.take(values, multiindex)
        collection.set_array(values)
        collection.set_cmap(cmap)
        collection.set_norm(Normalize(vmin=vmin, vmax=vmax))
    elif color is not None:
        collection.set_facecolor(color)
        collection.set_edgecolor(color)

    ax.add_collection(collection, autolim=autolim)

    return collection

def _plot_linestring_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, autolim=True, **kwargs):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` LineStrings and/or MultiLineStrings (can be
            mixed)
    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
    color : single color or sequence of `N` colors
        Cannot be used together with `values`.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    geoms, multiindex = _sanitize_geoms(geoms)
    _expand_kwargs(kwargs, multiindex)

    segments = [np.array(linestring.coords) for linestring in geoms]
    collection = LineCollection(segments, **kwargs)

    if values is not None:
        values = np.take(values, multiindex)
        collection.set_array(values)
        collection.set_cmap(cmap)
        collection.set_norm(Normalize(vmin=vmin, vmax=vmax))
    elif color is not None:
        collection.set_color(color)

    ax.add_collection(collection, autolim=autolim)

    return collection

def _plot_point_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, marker='o', markersize=None, **kwargs):
    """
    Plots a collection of Point and MultiPoint geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : sequence of `N` Points or MultiPoints

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    geoms, multiindex = _sanitize_geoms(geoms)
    _expand_kwargs(kwargs, multiindex)

    x = [p.x for p in geoms]
    y = [p.y for p in geoms]

    if values is not None:
        values = np.take(values, multiindex)

    if markersize is not None:
        if isinstance(markersize, (int, float)):
            markersize = [markersize] * len(geoms)
        else:
            markersize = np.take(markersize, multiindex)

    collection = ax.scatter(x, y, c=values, s=markersize, marker=marker, cmap=cmap, vmin=vmin, vmax=vmax, color=color, **kwargs)

    return collection

def plot_series(s, cmap=None, color=None, ax=None, figsize=None, aspect='auto', autolim=True, **style_kwds):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString, Point and MultiPoint
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str, np.array, pd.Series, List (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(s_y * pi/180) with s_y the y coordinate of the middle of
        the GeoSeries (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    geom_types = s.geom_type.unique()
    
    for geom_type in geom_types:
        if geom_type.startswith('Multi'):
            geom_type = geom_type[5:]
        
        geoms = s[s.geom_type.isin([geom_type, f'Multi{geom_type}'])]
        
        if geom_type == 'Polygon':
            _plot_polygon_collection(ax, geoms, color=color, cmap=cmap, autolim=autolim, **style_kwds)
        elif geom_type == 'LineString':
            _plot_linestring_collection(ax, geoms, color=color, cmap=cmap, autolim=autolim, **style_kwds)
        elif geom_type == 'Point':
            _plot_point_collection(ax, geoms, color=color, cmap=cmap, autolim=autolim, **style_kwds)
    
    if aspect == 'auto':
        if s.crs and s.crs.is_geographic:
            bounds = s.total_bounds
            y_mean = np.mean(bounds[1::2])
            ax.set_aspect(1 / np.cos(np.deg2rad(y_mean)))
        else:
            ax.set_aspect('equal')
    elif aspect is not None:
        ax.set_aspect(aspect)
    
    return ax

def plot_dataframe(df, column=None, cmap=None, color=None, ax=None, cax=None, categorical=False, legend=False, scheme=None, k=5, vmin=None, vmax=None, markersize=None, figsize=None, legend_kwds=None, categories=None, classification_kwds=None, missing_kwds=None, aspect='auto', autolim=True, **style_kwds):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as
        dataframe. Values are used to color the plot. Ignored if `color` is
        also set.
    kind: str
        The kind of plots to produce. The default is to create a map ("geo").
        Other supported kinds of plots from pandas:

        - 'line' : line plot
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : BoxPlot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot
        - 'hexbin' : hexbin plot.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str, np.array, pd.Series (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    cax : matplotlib.pyplot Artist (default None)
        axes on which to draw the legend in case of color map.
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns, this
        will be set to True.
    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    k : int (default 5)
        Number of classes (ignore if scheme is None)
    vmin : None or float (default None)
        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.
    markersize : str or float or sequence (default None)
        Only applies to point geometries within a frame.
        If a str, will use the values in the column of the frame specified
        by markersize to set the size of markers. Otherwise can be a value
        to apply to all points, or a sequence of the same length as the
        number of points.
    figsize : tuple of integers (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        axes is given explicitly, figsize is ignored.
    legend_kwds : dict (default None)
        Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or
        :func:`matplotlib.pyplot.colorbar`.
        Additional accepted keywords when `scheme` is specified:

        fmt : string
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`).
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    missing_kwds : dict (default None)
        Keyword arguments specifying color options (as style_kwds)
        to be passed on to geometries with missing values in addition to
        or overwriting other style kwds. If None, geometries with missing
        values are not plotted.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(df_y * pi/180) with df_y the y coordinate of the middle of
        the GeoDataFrame (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(geodatasets.get_path("nybb"))
    >>> df.head()  # doctest: +SKIP
       BoroCode  ...                                           geometry
    0         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
    1         4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
    2         3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
    3         1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
    4         2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...

    >>> df.plot("BoroName", cmap="Set1")  # doctest: +SKIP

    See the User Guide page :doc:`../../user_guide/mapping` for details.

    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in the GeoDataFrame")
        values = df[column]
        if categorical or values.dtype == object:
            categorical = True
            if categories is None:
                categories = values.unique()
            values = pd.Categorical(values, categories=categories)
    else:
        values = None

    if scheme is not None:
        if not categorical:
            if classification_kwds is None:
                classification_kwds = {}
            if 'k' not in classification_kwds:
                classification_kwds['k'] = k

            try:
                import mapclassify
                binning = mapclassify.classify(
                    np.asarray(values),
                    scheme,
                    **classification_kwds
                )
                values = binning.yb
                categorical = True
                categories = binning.bins
                k = len(categories)
            except ImportError:
                warnings.warn("mapclassify not available. Ignoring the 'scheme' keyword")

    if categorical:
        if cmap is None:
            cmap = plt.get_cmap('tab10')
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        categories = list(categories)
        categories.sort()
        color_mapping = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}
        colors = [color_mapping.get(value, (0, 0, 0, 0)) for value in values]
    elif cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin or values.min(), vmax=vmax or values.max())
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        colors = scalar_map.to_rgba(values)
    elif color is not None:
        colors = color
    else:
        colors = None

    geom_types = df.geometry.geom_type.unique()

    for geom_type in geom_types:
        if geom_type.startswith('Multi'):
            geom_type = geom_type[5:]
        
        geoms = df[df.geometry.geom_type.isin([geom_type, f'Multi{geom_type}'])].geometry
        
        if geom_type == 'Polygon':
            collection = _plot_polygon_collection(ax, geoms, colors, **style_kwds)
        elif geom_type == 'LineString':
            collection = _plot_linestring_collection(ax, geoms, colors, **style_kwds)
        elif geom_type == 'Point':
            collection = _plot_point_collection(ax, geoms, colors, markersize=markersize, **style_kwds)
    
    if legend and (categorical or cmap is not None):
        if categorical:
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_mapping[cat]) for cat in categories]
            ax.legend(legend_elements, categories, **legend_kwds or {})
        else:
            plt.colorbar(scalar_map, ax=ax, cax=cax, **legend_kwds or {})

    if aspect == 'auto':
        if df.crs and df.crs.is_geographic:
            bounds = df.total_bounds
            y_mean = np.mean(bounds[1::2])
            ax.set_aspect(1 / np.cos(np.deg2rad(y_mean)))
        else:
            ax.set_aspect('equal')
    elif aspect is not None:
        ax.set_aspect(aspect)

    if autolim:
        ax.autoscale_view()
    
    return ax

@doc(plot_dataframe)
class GeoplotAccessor(PlotAccessor):
    _pandas_kinds = PlotAccessor._all_kinds

    def __call__(self, *args, **kwargs):
        data = self._parent.copy()
        kind = kwargs.pop('kind', 'geo')
        if kind == 'geo':
            return plot_dataframe(data, *args, **kwargs)
        if kind in self._pandas_kinds:
            return PlotAccessor(data)(kind=kind, **kwargs)
        else:
            raise ValueError(f'{kind} is not a valid plot kind')
