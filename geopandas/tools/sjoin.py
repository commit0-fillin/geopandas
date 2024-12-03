import warnings
from functools import partial
from typing import Optional
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas._compat import PANDAS_GE_30
from geopandas.array import _check_crs, _crs_mismatch_warn

def sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', distance=None, on_attribute=None, **kwargs):
    """Spatial join of two GeoDataFrames.

    See the User Guide page :doc:`../../user_guide/mergingdata` for details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    predicate : string, default 'intersects'
        Binary predicate. Valid values are determined by the spatial index used.
        You can check the valid values in left_df or right_df as
        ``left_df.sindex.valid_query_predicates`` or
        ``right_df.sindex.valid_query_predicates``
        Replaces deprecated ``op`` parameter.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance : number or array_like, optional
        Distance(s) around each input geometry within which to query the tree
        for the 'dwithin' predicate. If array_like, must be
        one-dimesional with length equal to length of left GeoDataFrame.
        Required if ``predicate='dwithin'``.
    on_attribute : string, list or tuple
        Column name(s) to join on as an additional join restriction on top
        of the spatial predicate. These must be found in both DataFrames.
        If set, observations are joined only if the predicate applies
        and values in specified columns match.

    Examples
    --------
    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)

    >>> chicago.head()  # doctest: +SKIP
        ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin(groceries, chicago)
    >>> groceries_w_communities.head()  # doctest: +SKIP
       OBJECTID       community                           geometry
    0        16          UPTOWN  MULTIPOINT ((-87.65661 41.97321))
    1        18     MORGAN PARK  MULTIPOINT ((-87.68136 41.69713))
    2        22  NEAR WEST SIDE  MULTIPOINT ((-87.63918 41.86847))
    3        23  NEAR WEST SIDE  MULTIPOINT ((-87.65495 41.87783))
    4        27         CHATHAM  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 95 columns]

    See also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    # Perform basic checks on input data
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute)

    # Check CRS
    if not _check_crs(left_df.crs, right_df.crs):
        _crs_mismatch_warn(left_df.crs, right_df.crs, stacklevel=3)

    # Perform spatial join
    indices = _geom_predicate_query(left_df, right_df, predicate, distance, on_attribute)

    # Join the DataFrames
    joined = _frame_join(left_df, right_df, indices, None, how, lsuffix, rsuffix, predicate, on_attribute)

    return joined

def _basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute=None):
    """Checks the validity of join input parameters.

    `how` must be one of the valid options.
    `'index_'` concatenated with `lsuffix` or `rsuffix` must not already
    exist as columns in the left or right data frames.

    Parameters
    ------------
    left_df : GeoDataFrame
    right_df : GeoData Frame
    how : str, one of 'left', 'right', 'inner'
        join type
    lsuffix : str
        left index suffix
    rsuffix : str
        right index suffix
    on_attribute : list, default None
        list of column names to merge on along with geometry
    """
    import geopandas

    if not isinstance(left_df, geopandas.GeoDataFrame):
        raise ValueError("'left_df' should be GeoDataFrame")

    if not isinstance(right_df, geopandas.GeoDataFrame):
        raise ValueError("'right_df' should be GeoDataFrame")

    allowed_hows = ['left', 'right', 'inner']
    if how not in allowed_hows:
        raise ValueError(f"'how' was '{how}' but is expected to be in {allowed_hows}")

    index_left = 'index_' + lsuffix
    index_right = 'index_' + rsuffix

    if index_left in left_df.columns:
        raise ValueError(f"'{index_left}' column already exists in left GeoDataFrame")

    if index_right in right_df.columns:
        raise ValueError(f"'{index_right}' column already exists in right GeoDataFrame")

    if on_attribute:
        if not isinstance(on_attribute, (list, tuple)):
            on_attribute = [on_attribute]
        for attr in on_attribute:
            if attr not in left_df.columns:
                raise ValueError(f"'{attr}' column not found in left GeoDataFrame")
            if attr not in right_df.columns:
                raise ValueError(f"'{attr}' column not found in right GeoDataFrame")

def _geom_predicate_query(left_df, right_df, predicate, distance, on_attribute=None):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    predicate : string
        Binary predicate to query.
    distance : number or array_like, optional
        Distance(s) for 'dwithin' predicate.
    on_attribute: list, default None
        list of column names to merge on along with geometry

    Returns
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    import pandas as pd

    if predicate == 'dwithin' and distance is None:
        raise ValueError("Distance is required for 'dwithin' predicate")

    if not right_df.sindex:
        right_df.sindex

    if predicate == 'dwithin':
        matches = right_df.sindex.query(left_df.geometry, predicate=predicate, distance=distance)
    else:
        matches = right_df.sindex.query(left_df.geometry, predicate=predicate)

    left_index = left_df.index.take(matches[0])
    right_index = right_df.index.take(matches[1])

    if on_attribute:
        if not isinstance(on_attribute, (list, tuple)):
            on_attribute = [on_attribute]
        
        left_df_subset = left_df.loc[left_index, on_attribute]
        right_df_subset = right_df.loc[right_index, on_attribute]
        
        attribute_match = (left_df_subset == right_df_subset).all(axis=1)
        left_index = left_index[attribute_match]
        right_index = right_index[attribute_match]

    return pd.DataFrame({'_key_left': left_index, '_key_right': right_index})

def _reset_index_with_suffix(df, suffix, other):
    """
    Equivalent of df.reset_index(), but with adding 'suffix' to auto-generated
    column names.
    """
    import pandas as pd

    if isinstance(df.index, pd.MultiIndex):
        new_columns = [f'level_{i}_{suffix}' if name is None else f'{name}_{suffix}'
                       for i, name in enumerate(df.index.names)]
        df = df.reset_index()
        df.columns = new_columns + list(df.columns[len(new_columns):])
    else:
        index_name = df.index.name
        if index_name is None:
            index_name = 'index'
        new_name = f'{index_name}_{suffix}'
        if new_name in other.columns:
            i = 0
            while f'{new_name}_{i}' in other.columns:
                i += 1
            new_name = f'{new_name}_{i}'
        df = df.reset_index(names=new_name)
    return df

def _process_column_names_with_suffix(left: pd.Index, right: pd.Index, suffixes, left_df, right_df):
    """
    Add suffixes to overlapping labels (ignoring the geometry column).

    This is based on pandas' merge logic at https://github.com/pandas-dev/pandas/blob/
    a0779adb183345a8eb4be58b3ad00c223da58768/pandas/core/reshape/merge.py#L2300-L2370
    """
    def _get_new_name(name, suffix, column_names):
        new_name = f'{name}{suffix}'
        if new_name in column_names:
            i = 0
            while f'{new_name}_{i}' in column_names:
                i += 1
            return f'{new_name}_{i}'
        return new_name

    to_rename = {
        'left': {},
        'right': {}
    }

    if not left.equals(right):
        overlap = set(left) & set(right)
        overlap = overlap - {'geometry'}

        if overlap:
            for name in overlap:
                left_suffix, right_suffix = suffixes

                left_new_name = _get_new_name(name, left_suffix, left)
                right_new_name = _get_new_name(name, right_suffix, right)

                to_rename['left'][name] = left_new_name
                to_rename['right'][name] = right_new_name

    return to_rename

def _restore_index(joined, index_names, index_names_original):
    """
    Set back the the original index columns, and restoring their name as `None`
    if they didn't have a name originally.
    """
    if index_names:
        joined = joined.set_index(index_names)
        for i, name in enumerate(index_names_original):
            if name is None:
                joined.index.names = list(joined.index.names[:i]) + [None] + list(joined.index.names[i+1:])
    return joined

def _adjust_indexers(indices, distances, original_length, how, predicate):
    """
    The left/right indexers from the query represents an inner join.
    For a left or right join, we need to adjust them to include the rows
    that would not be present in an inner join.
    """
    import numpy as np

    if how == 'inner':
        return indices, distances

    if how == 'left':
        left_index = np.arange(original_length)
        right_index = indices['_key_right'].values
        mask = np.isin(left_index, indices['_key_left'].values)
        left_index_adjusted = np.where(mask, indices['_key_left'].values, left_index)
        right_index_adjusted = np.where(mask, right_index, -1)
    elif how == 'right':
        left_index = indices['_key_left'].values
        right_index = np.arange(original_length)
        mask = np.isin(right_index, indices['_key_right'].values)
        left_index_adjusted = np.where(mask, left_index, -1)
        right_index_adjusted = np.where(mask, indices['_key_right'].values, right_index)

    if distances is not None and predicate == 'dwithin':
        distances_adjusted = np.where(mask, distances, np.nan)
        return pd.DataFrame({'_key_left': left_index_adjusted, '_key_right': right_index_adjusted}), distances_adjusted
    else:
        return pd.DataFrame({'_key_left': left_index_adjusted, '_key_right': right_index_adjusted}), None

def _frame_join(left_df, right_df, indices, distances, how, lsuffix, rsuffix, predicate, on_attribute=None):
    """Join the GeoDataFrames at the DataFrame level.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    indices : tuple of ndarray
        Indices returned by the geometric join. Tuple with with integer
        indices representing the matches from `left_df` and `right_df`
        respectively.
    distances : ndarray, optional
        Passed trough and adapted based on the indices, if needed.
    how : string
        The type of join to use on the DataFrame level.
    lsuffix : string
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string
        Suffix to apply to overlapping column names (right GeoDataFrame).
    on_attribute: list, default None
        list of column names to merge on along with geometry

    Returns
    -------
    GeoDataFrame
        Joined GeoDataFrame.
    """
    import geopandas as gpd

    left_df = left_df.copy()
    right_df = right_df.copy()

    # Adjust indices for left/right joins
    indices, distances = _adjust_indexers(indices, distances, len(left_df), how, predicate)

    # Reset index on both dataframes
    left_df = _reset_index_with_suffix(left_df, lsuffix, right_df)
    right_df = _reset_index_with_suffix(right_df, rsuffix, left_df)

    # Rename overlapping columns
    to_rename = _process_column_names_with_suffix(
        left_df.columns, right_df.columns, (lsuffix, rsuffix), left_df, right_df
    )
    left_df = left_df.rename(columns=to_rename['left'])
    right_df = right_df.rename(columns=to_rename['right'])

    # Perform the join
    if how == 'left':
        joined = left_df.merge(right_df.loc[indices['_key_right'].replace(-1, np.nan)], left_index=True, right_index=True, how='left', suffixes=('', rsuffix))
    elif how == 'right':
        joined = left_df.loc[indices['_key_left'].replace(-1, np.nan)].merge(right_df, left_index=True, right_index=True, how='right', suffixes=(lsuffix, ''))
    else:  # inner
        joined = left_df.loc[indices['_key_left']].merge(right_df.loc[indices['_key_right']], left_index=True, right_index=True, suffixes=(lsuffix, rsuffix))

    # Add distance column if applicable
    if distances is not None and predicate == 'dwithin':
        joined['distance'] = distances

    # Restore original index
    joined = _restore_index(joined, left_df.index.names, left_df.index.names)

    # Ensure the result is a GeoDataFrame
    if not isinstance(joined, gpd.GeoDataFrame):
        joined = gpd.GeoDataFrame(joined, geometry=joined.geometry, crs=left_df.crs)

    return joined

def _filter_shared_attribute(left_df, right_df, l_idx, r_idx, attribute):
    """
    Returns the indices for the left and right dataframe that share the same entry
    in the attribute column. Also returns a Boolean `shared_attribute_rows` for rows
    with the same entry.
    """
    import pandas as pd

    if not isinstance(attribute, (list, tuple)):
        attribute = [attribute]

    left_values = left_df.loc[l_idx, attribute]
    right_values = right_df.loc[r_idx, attribute]

    if len(attribute) == 1:
        shared_attribute_rows = left_values.values == right_values.values
    else:
        shared_attribute_rows = (left_values == right_values).all(axis=1)

    return l_idx[shared_attribute_rows], r_idx[shared_attribute_rows], shared_attribute_rows

def sjoin_nearest(left_df: GeoDataFrame, right_df: GeoDataFrame, how: str='inner', max_distance: Optional[float]=None, lsuffix: str='left', rsuffix: str='right', distance_col: Optional[str]=None, exclusive: bool=False) -> GeoDataFrame:
    """Spatial join of two GeoDataFrames based on the distance between their geometries.

    Results will include multiple output records for a single input record
    where there are multiple equidistant nearest or intersected neighbors.

    Distance is calculated in CRS units and can be returned using the
    `distance_col` parameter.

    See the User Guide page
    https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
    for more details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    max_distance : float, default None
        Maximum distance within which to query for nearest geometry.
        Must be greater than 0.
        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance_col : string, default None
        If set, save the distances computed between matching geometries under a
        column of this name in the joined GeoDataFrame.
    exclusive : bool, default False
        If True, the nearest geometries that are equal to the input geometry
        will not be returned, default False.

    Examples
    --------
    >>> import geodatasets
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... )
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... ).to_crs(groceries.crs)

    >>> chicago.head()  # doctest: +SKIP
       ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
       OBJECTID     Ycoord  ...  Category                           geometry
    0        16  41.973266  ...       NaN  MULTIPOINT ((-87.65661 41.97321))
    1        18  41.696367  ...       NaN  MULTIPOINT ((-87.68136 41.69713))
    2        22  41.868634  ...       NaN  MULTIPOINT ((-87.63918 41.86847))
    3        23  41.877590  ...       new  MULTIPOINT ((-87.65495 41.87783))
    4        27  41.737696  ...       NaN  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago)
    >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                   Chain    community                                geometry
    0     VIET HOA PLAZA       UPTOWN   MULTIPOINT ((1168268.672 1933554.35))
    1  COUNTY FAIR FOODS  MORGAN PARK  MULTIPOINT ((1162302.618 1832900.224))


    To include the distances:

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago, distance_col="distances")
    >>> groceries_w_communities[["Chain", "community", "distances"]].head(2)
                   Chain    community  distances
    0     VIET HOA PLAZA       UPTOWN        0.0
    1  COUNTY FAIR FOODS  MORGAN PARK        0.0

    In the following example, we get multiple groceries for Uptown because all
    results are equidistant (in this case zero because they intersect).
    In fact, we get 4 results in total:

    >>> chicago_w_groceries = geopandas.sjoin_nearest(groceries, chicago, distance_col="distances", how="right")
    >>> uptown_results = chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
    >>> uptown_results[["Chain", "community"]]
                Chain community
    30  VIET HOA PLAZA    UPTOWN
    30      JEWEL OSCO    UPTOWN
    30          TARGET    UPTOWN
    30       Mariano's    UPTOWN

    See also
    --------
    sjoin : binary predicate joins
    GeoDataFrame.sjoin_nearest : equivalent method

    Notes
    -----
    Since this join relies on distances, results will be inaccurate
    if your geometries are in a geographic CRS.

    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    # Perform basic checks
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)

    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be greater than 0")

    # Check CRS
    if not _check_crs(left_df.crs, right_df.crs):
        _crs_mismatch_warn(left_df.crs, right_df.crs, stacklevel=3)

    # Perform nearest neighbor query
    if not right_df.sindex:
        right_df.sindex

    nearest = right_df.sindex.nearest(left_df.geometry, max_distance=max_distance, return_distance=True, exclusive=exclusive)

    left_index = left_df.index.take(nearest[0])
    right_index = right_df.index.take(nearest[1])
    distances = nearest[2]

    # Create DataFrame with matching indices
    indices = pd.DataFrame({'_key_left': left_index, '_key_right': right_index})

    # Join the DataFrames
    joined = _frame_join(left_df, right_df, indices, distances, how, lsuffix, rsuffix, 'nearest')

    # Add distance column if requested
    if distance_col is not None:
        joined[distance_col] = distances

    return joined
