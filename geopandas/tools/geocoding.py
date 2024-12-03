import time
from collections import defaultdict
import pandas as pd
from shapely.geometry import Point
import geopandas

def _get_throttle_time(provider):
    """
    Amount of time to wait between requests to a geocoding API, for providers
    that specify rate limits in their terms of service.
    """
    throttle_times = {
        'nominatim': 1.0,  # 1 request per second
        'googlev3': 0.2,   # 50 requests per second
        'bing': 0.5,       # 2 requests per second
        'photon': 0.1,     # 10 requests per second (default)
    }
    return throttle_times.get(provider, 0.1)  # Default to 0.1 if provider not listed

def geocode(strings, provider=None, **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : str or geopy.geocoder
        Specifies geocoding service to use. If none is provided,
        will use 'photon' (see the Photon's terms of service at:
        https://photon.komoot.io).

        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.Photon) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Geocoding requires geopy. Install it using 'pip install geopy'. See also
    https://github.com/geopy/geopy

    Examples
    --------
    >>> df = geopandas.tools.geocode(  # doctest: +SKIP
    ...         ["boston, ma", "1600 pennsylvania ave. washington, dc"]
    ...     )
    >>> df  # doctest: +SKIP
                        geometry                                            address
    0  POINT (-71.05863 42.35899)                          Boston, MA, United States
    1  POINT (-77.03651 38.89766)  1600 Pennsylvania Ave NW, Washington, DC 20006...
    """
    from geopy.geocoders import get_geocoder_for_service
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    if provider is None:
        provider = 'photon'

    if isinstance(provider, str):
        geocoder = get_geocoder_for_service(provider)(**kwargs)
    else:
        geocoder = provider

    results = {}
    for i, string in enumerate(strings):
        try:
            result = geocoder.geocode(string)
            if result is not None:
                results[i] = (string, (result.latitude, result.longitude))
        except (GeocoderTimedOut, GeocoderServiceError):
            continue
        time.sleep(_get_throttle_time(provider))

    return _prepare_geocode_result(results)

def reverse_geocode(points, provider=None, **kwargs):
    """
    Reverse geocode a set of points and get a GeoDataFrame of the resulting
    addresses.

    The points

    Parameters
    ----------
    points : list or Series of Shapely Point objects.
        x coordinate is longitude
        y coordinate is latitude
    provider : str or geopy.geocoder (opt)
        Specifies geocoding service to use. If none is provided,
        will use 'photon' (see the Photon's terms of service at:
        https://photon.komoot.io).

        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.Photon) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Reverse geocoding requires geopy. Install it using 'pip install geopy'.
    See also https://github.com/geopy/geopy

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> df = geopandas.tools.reverse_geocode(  # doctest: +SKIP
    ...     [Point(-71.0594869, 42.3584697), Point(-77.0365305, 38.8977332)]
    ... )
    >>> df  # doctest: +SKIP
                         geometry                                            address
    0  POINT (-71.05941 42.35837)       29 Court Sq, Boston, MA 02108, United States
    1  POINT (-77.03641 38.89766)  1600 Pennsylvania Ave NW, Washington, DC 20006...
    """
    from geopy.geocoders import get_geocoder_for_service
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    if provider is None:
        provider = 'photon'

    if isinstance(provider, str):
        geocoder = get_geocoder_for_service(provider)(**kwargs)
    else:
        geocoder = provider

    results = {}
    for i, point in enumerate(points):
        try:
            result = geocoder.reverse((point.y, point.x))
            if result is not None:
                results[i] = (result.address, (point.y, point.x))
        except (GeocoderTimedOut, GeocoderServiceError):
            continue
        time.sleep(_get_throttle_time(provider))

    return _prepare_geocode_result(results)

def _prepare_geocode_result(results):
    """
    Helper function for the geocode function

    Takes a dict where keys are index entries, values are tuples containing:
    (address, (lat, lon))

    """
    if not results:
        return geopandas.GeoDataFrame()

    df = pd.DataFrame(
        {
            "geometry": [Point(lon, lat) for _, (lat, lon) in results.values()],
            "address": [address for address, _ in results.values()],
        },
        index=results.keys(),
    )
    return geopandas.GeoDataFrame(df, crs="EPSG:4326")
