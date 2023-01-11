
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:12:19 2022

@author: jschwenk

Clips the Livneh forcing data (which has already been clipped to CO River Basin)
to the CO River Basin polygons (VotE-extracted). 
"""
import xarray as xr
import rasterio as rio
import geopandas as gpd
import rasterstats as rstats
import pandas as pd
from osgeo import gdal
import subprocess
from pyproj import CRS
import os

# Stolen from https://gis.stackexchange.com/questions/363120/computing-annual-spatial-zonal-statistics-of-a-netcdf-file-for-polygons-in-sha
# load and read shp-file with geopandas
shp_fo = r"C:\Users\375237\Desktop\CRB-human-impacts\Data\forcings\stations_calibrated_CRB_updated.shp"
shp_df = gpd.read_file(shp_fo)

# # load and read netCDF-file to dataset and get datarray for variable
# nc_fo = r"C:\Users\375237\Desktop\CRB-human-impacts\Data\forcings\prec_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc"
# nc_ds = xr.open_dataset(nc_fo)
# nc_var = nc_ds['Prec']

# # extract forcing data for each polygon
# """ Make a mask of each watershed geometry that has the same extents/resolution
# as the netCDF files. """


# gdobj = gdal.Open(nc_fo)
# gt = gdobj.GetGeoTransform()
# xmax = gt[0] + gdobj.RasterXSize * gt[1]
# ymin = gt[3] + gdobj.RasterYSize * gt[-1]

# # Loop through each gage/watershed
# for _, row in shp_df.iterrows():
#     path_out_mask = os.path.join(r"C:\Users\375237\Desktop\CRB-human-impacts\Data\forcings\station_masks", row['BASIN_NAME'] + '.tif') 
#     path_out_temp = r"C:\Users\375237\Desktop\CRB-human-impacts\Data\temp.gpkg"
#     if os.path.isfile(path_out_temp):
#         os.remove(path_out_temp)
#     if os.path.isfile(path_out_mask):
#         os.remove(path_out_mask)
#     gdf_temp = gpd.GeoDataFrame(geometry=[row['geometry']], crs=CRS.from_epsg(4326))
#     gdf_temp.to_file(path_out_temp, driver='GPKG')
    
#     callstring = ['gdal_rasterize',
#                 '-burn', str(1),
#                 '-at',
#                 '-tap',
#                 '-tr', str(gt[1]), str(gt[-1]), 
#                 '-te', str(gt[0]), str(ymin), str(xmax), str(gt[3]),  
#                 '-ot', 'UInt32',
#                  '-of', "GTiff",
#                   path_out_temp,
#                   path_out_mask
#                   ] 
                      
#     proc = subprocess.Popen(callstring, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#     stdout,stderr=proc.communicate()

""" Compute the time series """
import numpy as np
path_out = r"C:\Users\375237\Desktop\CRB-human-impacts\Data\per_station"
# f_paths = {'precip' : r"prec_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc",
#            'tmax' : r"tmax_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc",
#            'tmin' : r"tmin_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc",
#            'wind' : r"wind_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc",}
f_paths = {'precip' : r"prec_COL_livneh_NAmerExt_15Oct2014_195001_201312.nc",}

for f in f_paths:
    print(f)

    this_path_out = os.path.join(path_out, f + '_sum.csv')
    if os.path.isfile(this_path_out):
        continue
    
    this_df = pd.DataFrame()
    nc_ds = xr.open_dataset(f_paths[f])
    nc_var = nc_ds[list(nc_ds)[0]]
    np_var = nc_var.to_numpy()
    # For some reason all the data are duplicated in time; we take only the
    # first half. I checked and the arrays seem to be exactly the same.
    dates = sorted(set(nc_ds['time'].values))
    np_var = np_var[:int(np_var.shape[0]/2),:,:]

    these_timeseries = {}
    these_timeseries['dates'] = dates
    for _, row in shp_df.iterrows():
        print(row['BASIN_NAME'])
        path_mask = os.path.join(r'C:\Users\375237\Desktop\CRB-human-impacts\Data\forcings\station_masks', row['BASIN_NAME'] + '.tif')
        Imask = gdal.Open(path_mask).ReadAsArray()
        masked = np_var * Imask
        print(masked)
        these_timeseries[row['BASIN_NAME']] = np.nansum(masked, axis=(1,2))
    this_df = pd.DataFrame.from_dict(these_timeseries)
    this_df.to_csv(this_path_out, index=False)