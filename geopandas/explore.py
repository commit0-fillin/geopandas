import warnings
from packaging.version import Version
from statistics import mean
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from shapely.geometry import LineString
import geopandas
_MAP_KWARGS = ['location', 'prefer_canvas', 'no_touch', 'disable_3d', 'png_enabled', 'zoom_control', 'crs', 'zoom_start', 'left', 'top', 'position', 'min_zoom', 'max_zoom', 'min_lat', 'max_lat', 'min_lon', 'max_lon', 'max_bounds']

def _explore(df, column=None, cmap=None, color=None, m=None, tiles='OpenStreetMap', attr=None, tooltip=True, popup=False, highlight=True, categorical=False, legend=True, scheme=None, k=5, vmin=None, vmax=None, width='100%', height='100%', categories=None, classification_kwds=None, control_scale=True, marker_type=None, marker_kwds={}, style_kwds={}, highlight_kwds={}, missing_kwds={}, tooltip_kwds={}, popup_kwds={}, legend_kwds={}, map_kwds={}, **kwargs):
    """Interactive map based on GeoPandas and folium/leaflet.js

    Generate an interactive leaflet map based on :class:`~geopandas.GeoDataFrame`

    Parameters
    ----------
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, :class:`numpy.array`,
        or :class:`pandas.Series` to be plotted. If :class:`numpy.array` or
        :class:`pandas.Series` are used then it must have same length as dataframe.
    cmap : str, matplotlib.Colormap, branca.colormap or function (default None)
        The name of a colormap recognized by ``matplotlib``, a list-like of colors,
        :class:`matplotlib.colors.Colormap`, a :class:`branca.colormap.ColorMap` or
        function that returns a named color or hex based on the column
        value, e.g.::

            def my_colormap(value):  # scalar value defined in 'column'
                if value > 1:
                    return "green"
                return "red"

    color : str, array-like (default None)
        Named color or a list-like of colors (named or hex).
    m : folium.Map (default None)
        Existing map instance on which to draw the plot.
    tiles : str, xyzservices.TileProvider (default 'OpenStreetMap Mapnik')
        Map tileset to use. Can choose from the list supported by folium, query a
        :class:`xyzservices.TileProvider` by a name from ``xyzservices.providers``,
        pass :class:`xyzservices.TileProvider` object or pass custom XYZ URL.
        The current list of built-in providers (when ``xyzservices`` is not available):

        ``["OpenStreetMap", "CartoDB positron", “CartoDB dark_matter"]``

        You can pass a custom tileset to Folium by passing a Leaflet-style URL
        to the tiles parameter: ``http://{s}.yourtiles.com/{z}/{x}/{y}.png``.
        Be sure to check their terms and conditions and to provide attribution with
        the ``attr`` keyword.
    attr : str (default None)
        Map tile attribution; only required if passing custom tile URL.
    tooltip : bool, str, int, list (default True)
        Display GeoDataFrame attributes when hovering over the object.
        ``True`` includes all columns. ``False`` removes tooltip. Pass string or list of
        strings to specify a column(s). Integer specifies first n columns to be
        included. Defaults to ``True``.
    popup : bool, str, int, list (default False)
        Input GeoDataFrame attributes for object displayed when clicking.
        ``True`` includes all columns. ``False`` removes popup. Pass string or list of
        strings to specify a column(s). Integer specifies first n columns to be
        included. Defaults to ``False``.
    highlight : bool (default True)
        Enable highlight functionality when hovering over a geometry.
    categorical : bool (default False)
        If ``False``, ``cmap`` will reflect numerical values of the
        column being plotted. For non-numerical columns, this
        will be set to True.
    legend : bool (default True)
        Plot a legend in choropleth plots.
        Ignored if no ``column`` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires ``mapclassify`` >= 2.4.0).
        A :func:`mapclassify.classify` will be used
        under the hood. Supported are all schemes provided by ``mapclassify`` (e.g.
        ``'BoxPlot'``, ``'EqualInterval'``, ``'FisherJenks'``, ``'FisherJenksSampled'``,
        ``'HeadTailBreaks'``, ``'JenksCaspall'``, ``'JenksCaspallForced'``,
        ``'JenksCaspallSampled'``, ``'MaxP'``, ``'MaximumBreaks'``,
        ``'NaturalBreaks'``, ``'Quantiles'``, ``'Percentiles'``, ``'StdMean'``,
        ``'UserDefined'``). Arguments can be passed in ``classification_kwds``.
    k : int (default 5)
        Number of classes
    vmin : None or float (default None)
        Minimum value of ``cmap``. If ``None``, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of ``cmap``. If ``None``, the maximum data value
        in the column to be plotted is used.
    width : pixel int or percentage string (default: '100%')
        Width of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, height is ignored.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    control_scale : bool, (default True)
        Whether to add a control scale on the map.
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed string options are ('marker', 'circle', 'circle_marker'). Defaults to
        folium.CircleMarker.
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected ``marker_type``, e.g.:

        radius : float (default 2 for ``circle_marker`` and 50 for ``circle``))
            Radius of the circle, in meters (for ``circle``) or pixels
            (for ``circle_marker``).
        fill : bool (default True)
            Whether to fill the ``circle`` or ``circle_marker`` with color.
        icon : folium.map.Icon
            the :class:`folium.map.Icon` object to use to render the marker.
        draggable : bool (default False)
            Set to True to be able to drag the marker around the map.

    style_kwds : dict (default {})
        Additional style to be passed to folium ``style_function``:

        stroke : bool (default True)
            Whether to draw stroke along the path. Set it to ``False`` to
            disable borders on polygons or circles.
        color : str
            Stroke color
        weight : int
            Stroke width in pixels
        opacity : float (default 1.0)
            Stroke opacity
        fill : boolean (default True)
            Whether to fill the path with color. Set it to ``False`` to
            disable filling on polygons or circles.
        fillColor : str
            Fill color. Defaults to the value of the color option
        fillOpacity : float (default 0.5)
            Fill opacity.
        style_function : callable
            Function mapping a GeoJson Feature to a style ``dict``.

            * Style properties :func:`folium.vector_layers.path_options`
            * GeoJson features :class:`GeoDataFrame.__geo_interface__`

            e.g.::

                lambda x: {"color":"red" if x["properties"]["gdp_md_est"]<10**6
                                             else "blue"}

        Plus all supported by :func:`folium.vector_layers.path_options`. See the
        documentation of :class:`folium.features.GeoJson` for details.

    highlight_kwds : dict (default {})
        Style to be passed to folium highlight_function. Uses the same keywords
        as ``style_kwds``. When empty, defaults to ``{"fillOpacity": 0.75}``.
    tooltip_kwds : dict (default {})
        Additional keywords to be passed to :class:`folium.features.GeoJsonTooltip`,
        e.g. ``aliases``, ``labels``, or ``sticky``.
    popup_kwds : dict (default {})
        Additional keywords to be passed to :class:`folium.features.GeoJsonPopup`,
        e.g. ``aliases`` or ``labels``.
    legend_kwds : dict (default {})
        Additional keywords to be passed to the legend.

        Currently supported customisation:

        caption : string
            Custom caption of the legend. Defaults to the column name.

        Additional accepted keywords when ``scheme`` is specified:

        colorbar : bool (default True)
            An option to control the style of the legend. If True, continuous
            colorbar will be used. If False, categorical legend will be used for bins.
        scale : bool (default True)
            Scale bins along the colorbar axis according to the bin edges (True)
            or use the equal length for each bin (False)
        fmt : string (default "{:.2f}")
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``. Applies
            if ``colorbar=False``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`). Applies if ``colorbar=False``.
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
            Applies if ``colorbar=False``.
        max_labels : int, default 10
            Maximum number of colorbar tick labels (requires branca>=0.5.0)
    map_kwds : dict (default {})
        Additional keywords to be passed to folium :class:`~folium.folium.Map`,
        e.g. ``dragging``, or ``scrollWheelZoom``.


    **kwargs : dict
        Additional options to be passed on to the folium object.

    Returns
    -------
    m : folium.folium.Map
        folium :class:`~folium.folium.Map` instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> df.head(2)  # doctest: +SKIP
       ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...

    [2 rows x 87 columns]

    >>> df.explore("Pop2012", cmap="Blues")  # doctest: +SKIP
    """
    pass

def _tooltip_popup(type, fields, gdf, **kwds):
    """get tooltip or popup"""
    import folium

    if fields is True:
        fields = gdf.columns.tolist()
    elif isinstance(fields, int):
        fields = gdf.columns[:fields].tolist()
    elif isinstance(fields, str):
        fields = [fields]

    if type == "tooltip":
        return folium.GeoJsonTooltip(fields=fields, aliases=fields, **kwds)
    elif type == "popup":
        return folium.GeoJsonPopup(fields=fields, aliases=fields, **kwds)
    else:
        raise ValueError("Type must be either 'tooltip' or 'popup'")

def _categorical_legend(m, title, categories, colors):
    """
    Add categorical legend to a map

    The implementation is using the code originally written by Michel Metran
    (@michelmetran) and released on GitHub
    (https://github.com/michelmetran/package_folium) under MIT license.

    Copyright (c) 2020 Michel Metran

    Parameters
    ----------
    m : folium.Map
        Existing map instance on which to draw the plot
    title : str
        title of the legend (e.g. column name)
    categories : list-like
        list of categories
    colors : list-like
        list of colors (in the same order as categories)
    """
    from branca.element import Template, MacroElement

    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; 
        right: 50px; 
        width: 120px; 
        height: 90px; 
        z-index:9999; 
        font-size:14px;
        ">
        <p><a style="color: #0000ff" href=""><b>{{this.title}}</b></a></p>
        {% for category, color in this.color_dict.items() %}
        <p><i class="fa fa-circle fa-1x"
                style="color:{{color}}"></i> {{category}}
        </p>
        {% endfor %}
    </div>
    <div style="
        position: fixed; 
        bottom: 50px; 
        left: 50px; 
        width: 150px; 
        height: 30px; 
        z-index:9998; 
        font-size:14px;
        background-color: #ffffff;
        opacity: 0.7;
        ">&copy; <a href="https://github.com/michelmetran">Michel Metran</a>
    </div>
    {% endmacro %}
    """

    color_dict = dict(zip(categories, colors))
    macro = MacroElement()
    macro._template = Template(template)
    macro.title = title
    macro.color_dict = color_dict
    m.get_root().add_child(macro)

def _explore_geoseries(s, color=None, m=None, tiles='OpenStreetMap', attr=None, highlight=True, width='100%', height='100%', control_scale=True, marker_type=None, marker_kwds={}, style_kwds={}, highlight_kwds={}, map_kwds={}, **kwargs):
    """Interactive map based on GeoPandas and folium/leaflet.js

    Generate an interactive leaflet map based on :class:`~geopandas.GeoSeries`

    Parameters
    ----------
    color : str, array-like (default None)
        Named color or a list-like of colors (named or hex).
    m : folium.Map (default None)
        Existing map instance on which to draw the plot.
    tiles : str, xyzservices.TileProvider (default 'OpenStreetMap Mapnik')
        Map tileset to use. Can choose from the list supported by folium, query a
        :class:`xyzservices.TileProvider` by a name from ``xyzservices.providers``,
        pass :class:`xyzservices.TileProvider` object or pass custom XYZ URL.
        The current list of built-in providers (when ``xyzservices`` is not available):

        ``["OpenStreetMap", "CartoDB positron", “CartoDB dark_matter"]``

        You can pass a custom tileset to Folium by passing a Leaflet-style URL
        to the tiles parameter: ``http://{s}.yourtiles.com/{z}/{x}/{y}.png``.
        Be sure to check their terms and conditions and to provide attribution with
        the ``attr`` keyword.
    attr : str (default None)
        Map tile attribution; only required if passing custom tile URL.
    highlight : bool (default True)
        Enable highlight functionality when hovering over a geometry.
    width : pixel int or percentage string (default: '100%')
        Width of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium :class:`~folium.folium.Map`. If the argument
        m is given explicitly, height is ignored.
    control_scale : bool, (default True)
        Whether to add a control scale on the map.
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed string options are ('marker', 'circle', 'circle_marker'). Defaults to
        folium.Marker.
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected ``marker_type``, e.g.:

        radius : float
            Radius of the circle, in meters (for ``'circle'``) or pixels
            (for ``circle_marker``).
        icon : folium.map.Icon
            the :class:`folium.map.Icon` object to use to render the marker.
        draggable : bool (default False)
            Set to True to be able to drag the marker around the map.

    style_kwds : dict (default {})
        Additional style to be passed to folium ``style_function``:

        stroke : bool (default True)
            Whether to draw stroke along the path. Set it to ``False`` to
            disable borders on polygons or circles.
        color : str
            Stroke color
        weight : int
            Stroke width in pixels
        opacity : float (default 1.0)
            Stroke opacity
        fill : boolean (default True)
            Whether to fill the path with color. Set it to ``False`` to
            disable filling on polygons or circles.
        fillColor : str
            Fill color. Defaults to the value of the color option
        fillOpacity : float (default 0.5)
            Fill opacity.
        style_function : callable
            Function mapping a GeoJson Feature to a style ``dict``.

            * Style properties :func:`folium.vector_layers.path_options`
            * GeoJson features :class:`GeoSeries.__geo_interface__`

            e.g.::

                lambda x: {"color":"red" if x["properties"]["gdp_md_est"]<10**6
                                             else "blue"}


        Plus all supported by :func:`folium.vector_layers.path_options`. See the
        documentation of :class:`folium.features.GeoJson` for details.

    highlight_kwds : dict (default {})
        Style to be passed to folium highlight_function. Uses the same keywords
        as ``style_kwds``. When empty, defaults to ``{"fillOpacity": 0.75}``.
    map_kwds : dict (default {})
        Additional keywords to be passed to folium :class:`~folium.folium.Map`,
        e.g. ``dragging``, or ``scrollWheelZoom``.

    **kwargs : dict
        Additional options to be passed on to the folium.

    Returns
    -------
    m : folium.folium.Map
        folium :class:`~folium.folium.Map` instance

    """
    pass
