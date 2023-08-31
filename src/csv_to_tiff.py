
import xarray as xr
import numpy as np
import os
import pandas as pd
from osgeo import gdal
from constants import PRECIPITATION, LATITUDE, LONGITUDE, DATE, NO_DATA_VALUE

def write_geotiff(fileName, raster, swne, noDataValue = None, dtype = None):
    '''
    Writes the raster in data['raster'] to a GeoTiff file.
    Args:
        fileName (str):    Name of the output file.
        raster (np.array): The raster. (0, 0) is the top-left (north-west) pixel.
                           The shape of the raster is (latitude, longitude).
        swne (tuple):      Coordinates of bottom-left and top-right pixels.
                           The convention is that these give the **centers** of
                           the pixels.
        noDataValue:       The no data value.
        dtype (GDAL type): GDAL data type (e.g.)
    '''
    
    # We cannot set the default dtype in the function signature since we cannot 
    # assume gdal to exist. So we set it here.
    if dtype is None:
        dtype = gdal.GDT_Float32

    nJ, nI = raster.shape

    latMin, lonMin, latMax, lonMax = swne
    dLat = (latMax - latMin) / (nJ - 1)
    dLon = (lonMax - lonMin) / (nI - 1)
    nwCornerLat = latMax + dLat / 2.
    nwCornerLon = lonMin - dLon / 2.
    gdalGeoRef = (nwCornerLon, dLon, 0, nwCornerLat, 0, -dLat) # Note the negative dLat increment.
    if noDataValue is not None:
        raster = np.where(np.isnan(raster), noDataValue, raster)

    # Use GDAL to create the GeoTiff
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(fileName, nI, nJ, 1, dtype)
    ds.SetGeoTransform(gdalGeoRef)
    rasterBand = ds.GetRasterBand(1)
    if noDataValue is not None:
        rasterBand.SetNoDataValue(noDataValue)
    rasterBand.WriteArray(raster)
    ds = None

def write_metadata(file_name_geotiff, 
                  timestamp, 
                  data_layer, 
                  noDataValue = None, 
                  geospatialProjection = 'EPSG:4326', 
                  interpolation = 'near', 
                  ignoreTile = False, 
                  dimensions = [],
                  dir_to_tiff = './'):
    
    """
    The wgen will always have the simulaion, so simulation will always be a dimension, that way the list dimensions 
    has at least one element.
    """
    
    meta_string_json = \
    '{\n' \
    '    "timestamp": "'+str(timestamp)+'",\n' \
    '    "datalayer_id": ["'+str(data_layer)+'"],\n' \
    '    "pairsdatatype": "2draster",\n' \
    '    "geospatialprojection": "'+str(geospatialProjection)+'",\n' \
    '    "datainterpolation": ["'+str(interpolation)+'"],\n' \
    '    "pairsdimension": '+str([d['name'] for d in dimensions])+',\n' \
    '    "dimension_value": '+str([d['value'] for d in dimensions])+'\n' \
    '}'
    
      
    # Write to meta file
    meta_file_name = file_name_geotiff + '.meta.json'
    meta_file_fp = os.path.join(dir_to_tiff, meta_file_name)
    
    with open(meta_file_fp, 'w') as fout:
        fout.write(meta_string_json)
        
#TESTED
def simulations_to_tiff(df, data_layer, noDataValue=NO_DATA_VALUE, 
                        key_variable=PRECIPITATION, 
                        dir_path=None)->None:
    
    dimensions = [{"name": "simulation", "value": "0"}]

    simulations = list(df.n_simu.unique())
    df = df.reset_index()
    dates = list(df.Date.unique())

    for simulation in simulations:
        df_sub = df[df['n_simu'] == simulation]
        dimensions[0]["value"] = str(simulation)

        for date in dates:
            df_subb = df_sub[df_sub[DATE] == date]
            date_file_str = pd.Timestamp(date).strftime("%Y_%m_%dT%H_%M_%S")

            file_name = f'/IBMWG_simulations-'+str(simulation).zfill(2)+'_date-'+date_file_str+'.tiff'
            if dir_path:
                file_name = dir_path+file_name
            #print('Generating "{}"'.format(file_name))

            df_subb = df_subb.set_index([LATITUDE, LONGITUDE])[[key_variable]]
            ds = xr.Dataset.from_dataframe(df_subb)

            lats = ds.variables[LATITUDE][:]
            lons = ds.variables[LONGITUDE][:]

            raster = ds[key_variable].values
            lat = raster[0]
            lon = raster[1]
            swne_center = (lats[-1], lons[0], lats[0], lons[-1])

            #write geotiff
            write_geotiff(file_name, raster, swne_center, noDataValue=noDataValue, dtype=gdal.gdalconst.GDT_Float32)

            #write metadata
            timestamp_for_metafile = pd.Timestamp(date).isoformat()
            write_metadata(file_name, 
                            timestamp=timestamp_for_metafile,
                            data_layer=data_layer, 
                            dimensions=dimensions,
                            )