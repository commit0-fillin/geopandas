"""
Script to create the data and write legacy storage (pickle) files.

Based on pandas' generate_legacy_storage_files.py script.

To use this script, create an environment for which you want to
generate pickles, activate the environment, and run this script as:

$ python geopandas/geopandas/io/tests/generate_legacy_storage_files.py     geopandas/geopandas/io/tests/data/pickle/ pickle

This script generates a storage file for the current arch, system,

The idea here is you are using the *current* version of the
generate_legacy_storage_files with an *older* version of geopandas to
generate a pickle file. We will then check this file into a current
branch, and test using test_pickle.py. This will load the *older*
pickles and test versus the current data that is generated
(with master). These are then compared.

"""
import os
import pickle
import platform
import sys
import pandas as pd
from shapely.geometry import Point
import geopandas

def create_pickle_data():
    """create the pickle data"""
    # Create a sample GeoDataFrame
    geometry = [Point(x, y) for x, y in zip(range(3), range(3))]
    df = geopandas.GeoDataFrame(
        {'id': [1, 2, 3], 'name': ['A', 'B', 'C']},
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    # Create a GeoSeries
    gs = geopandas.GeoSeries(geometry, crs="EPSG:4326")
    
    # Return a dictionary of objects to pickle
    return {
        'gdf': df,
        'gs': gs,
        'geometry_list': geometry,
        'pandas_dataframe': df.drop(columns=['geometry'])
    }
def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python generate_legacy_storage_files.py <output_dir> <storage_format>")
        sys.exit(1)

    output_dir = sys.argv[1]
    storage_format = sys.argv[2]

    if storage_format != 'pickle':
        print("Only 'pickle' storage format is currently supported.")
        sys.exit(1)

    # Create the pickle data
    data = create_pickle_data()

    # Generate the filename
    version = geopandas.__version__
    pf = platform.platform().split('-')[0].lower()
    arch = platform.machine().lower()
    filename = f"geopandas-{version}-{pf}-{arch}.{storage_format}"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the pickle file
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)  # Use protocol 4 for Python 3.4+
    
    print(f"Created {path}")

if __name__ == '__main__':
    main()
