import importlib
import platform
import sys
import os

def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information
    """
    return {
        "python": sys.version,
        "python-bits": f"{sys.maxsize.bit_length() + 1}",
        "OS": platform.platform(),
        "OS-release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "LC_ALL": ".".join(platform.lang.getlocale()),
        "LANG": os.environ.get("LANG", "None"),
    }

def _get_C_info():
    """Information on system PROJ, GDAL, GEOS
    Returns
    -------
    c_info: dict
        system PROJ information
    """
    import pyproj
    import fiona
    from shapely import geos_version_string

    return {
        "PROJ": pyproj.proj_version_str,
        "GDAL": fiona.env.get_gdal_release_name(),
        "GEOS": geos_version_string,
    }

def _get_deps_info():
    """Overview of the installed version of main dependencies

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "geopandas",
        "pandas",
        "fiona",
        "numpy",
        "shapely",
        "pyproj",
        "rtree",
        "pytest",
        "matplotlib",
    ]

    def get_version(module):
        try:
            return importlib.import_module(module).__version__
        except (ImportError, AttributeError):
            return None

    return {d: get_version(d) for d in deps}

def show_versions():
    """
    Print system information and installed module versions.

    Examples
    --------

    ::

        $ python -c "import geopandas; geopandas.show_versions()"
    """
    sys_info = _get_sys_info()
    c_info = _get_C_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, v in sys_info.items():
        print(f"{k}: {v}")

    print("\nC libraries:")
    for k, v in c_info.items():
        print(f"{k}: {v}")

    print("\nPython dependencies:")
    for k, v in deps_info.items():
        print(f"{k}: {v}")
