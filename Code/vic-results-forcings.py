# import pandas as pd
# import os
# import xarray as xr

# files, lat, long = [], [], []
# for f in os.listdir("C:/Users/375237/Desktop/CRB-human-impacts/Data/baseline_calib_long/"):
#     if f.split('_')[0] == 'wbal':
#         files.append(f)
#         lat.append(f.split('_')[1])
#         long.append(f.split('_')[2])

# lat = [*set(lat)]
# long = [*set(long)]

# colnames = ["year","month","day","prec","evap","runoff","baseflow","wdew","snowcanopy","swe","soilm1","soilm2","soilm3",
#             "evapc","transv","evapb","subc","subs","sdepth","snowm","snowcover"]




# for f in files:
#     # if counter%10 == 0:
#     #     # nc_xr = xr.concat(xr_list, dim='date')
#     #     nc_xr = xr.merge(xr_list)
#     #     xr_list = [nc_xr]
#     if counter == 500:
#         break
#     file_name = f'../vic/results/baseline_calib_long/{f}'
#     df = pd.read_csv(file_name, delimiter='\t', names=colnames, header=None)
#     lat = f.split('_')[1]
#     long = f.split('_')[2]
#     print(lat, long, '\n')
#     df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
#     # df.drop(['year', 'month', 'day'], axis=1, inplace=True)
#     df['soil'] = df['soilm1'] + df['soilm2'] + df['soilm3']
#     df = df[['evap', 'swe', 'soil', 'sdepth', 'date']]
#     df['lat'] = lat
#     df['long'] = long
#     df.set_index(['lat', 'long', 'date'], inplace=True)
#     evap_list.append(df[['evap']].to_xarray())
#     swe_list.append(df[['swe']].to_xarray())
#     soil_list.append(df[['soil']].to_xarray())
#     sdepth_list.append(df[['sdepth']].to_xarray())
#     counter += 1

# nc_xr = xr.concat(xr_list, dim='date')
# nc_xr = xr.merge(evap_list)
# nc_xr.to_netcdf('extracted-livneh-evap.nc')
# nc_xr = xr.merge(swe_list)
# nc_xr.to_netcdf('extracted-livneh-swe.nc')
# nc_xr = xr.merge(soil_list)
# nc_xr.to_netcdf('extracted-livneh-soil.nc')
# nc_xr = xr.merge(sdepth_list)
# nc_xr.to_netcdf('extracted-livneh-sdepth.nc')

# for f in files:
#     # if counter%10 == 0:
#     #     # nc_xr = xr.concat(xr_list, dim='date')
#     #     nc_xr = xr.merge(xr_list)
#     #     xr_list = [nc_xr]
#     if counter == 500:
#         break
#     file_name = f'../vic/results/baseline_calib_long/{f}'
#     df = pd.read_csv(file_name, delimiter='\t', names=colnames, header=None)
#     lat = f.split('_')[1]
#     long = f.split('_')[2]
#     print(lat, long, '\n')
#     df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
#     # df.drop(['year', 'month', 'day'], axis=1, inplace=True)
#     df['soil'] = df['soilm1'] + df['soilm2'] + df['soilm3']
#     df = df[['evap', 'swe', 'soil', 'sdepth', 'date']]
#     df['lat'] = lat
#     df['long'] = long
#     # df.set_index(['lat', 'long', 'date'], inplace=True)
#     # evap_list.append(df[['evap']].to_xarray())
#     # swe_list.append(df[['swe']].to_xarray())
#     # soil_list.append(df[['soil']].to_xarray())
#     # sdepth_list.append(df[['sdepth']].to_xarray())
#     counter += 1

#     for _, row in shp_df.iterrows():
#         print(row['BASIN_NAME'])
#         path_mask = os.path.join(r'C:\Users\375237\Desktop\CRB-human-impacts\Data\forcings\station_masks', row['BASIN_NAME'] + '.tif')
#         Imask = gdal.Open(path_mask)

#         masked = np_var * Imask
#         print(Imask, Imask.RasterCount, Imask.RasterXSize, Imask.RasterYSize, Imask.GetMetadata())
#         these_timeseries[row['BASIN_NAME']] = np.nanmean(masked, axis=(1,2))
#     this_df = pd.DataFrame.from_dict(these_timeseries)
#     this_df.to_csv(path_out, index=False)


import pandas as pd
import os
import xarray as xr
import numpy as np
import geopandas as gpd

# load data
files = []
for f in os.listdir("../vic/results/baseline_calib_long/"):
# for f in os.listdir("C:/Users/375237/Desktop/CRB-human-impacts/Data/baseline_calib_long/"):
    if f.split('_')[0] == 'wbal':
        files.append(f)

colnames = ["year","month","day","prec","evap","runoff","baseflow","wdew","snowcanopy","swe","soilm1","soilm2","soilm3",
            "evapc","transv","evapb","subc","subs","sdepth","snowm","snowcover"]

shp_fo = "stations_calibrated_CRB_updated.shp"
# shp_fo = "C:/Users/375237/Desktop/CRB-human-impacts/Data/forcings/stations_calibrated_CRB_updated.shp"
shp_df = gpd.read_file(shp_fo)

# create dataframe of zeros
sample_f = f'../vic/results/baseline_calib_long/{files[0]}'
# sample_f = f'C:/Users/375237/Desktop/CRB-human-impacts/Data/baseline_calib_long/{files[0]}'
sample_df = pd.read_csv(sample_f, delimiter='\t', names=colnames, header=None)
dates = pd.to_datetime(sample_df[['year', 'month', 'day']])
cols = ['GRFRW','GRGRW','GRNGU','LSNSC','YRNMC','WRNWU','GRAGR','COGSC','GRBMC','GRGJC','CRNCU','DRNCU','SJNBU',
           'CRLFA','CRGCA','CRHDA','CRDDA','CRPDA','CAIDA','SRAGA','GRHSA','GRNDA','CRAMA','SJARN','TRTPC']
swe = pd.DataFrame(columns=cols, index=dates)
soil = pd.DataFrame(columns=cols, index=dates)
evap = pd.DataFrame(columns=cols, index=dates)
sdepth = pd.DataFrame(columns=cols, index=dates)
baseflow = pd.DataFrame(columns=cols, index=dates)
runoff = pd.DataFrame(columns=cols, index=dates)
swe.fillna(0, inplace=True)
soil.fillna(0, inplace=True)
evap.fillna(0, inplace=True)
sdepth.fillna(0, inplace=True)
baseflow.fillna(0, inplace=True)
runoff.fillna(0, inplace=True)

# add info from each file to dataframe
count = {'GRFRW': 0,'GRGRW': 0,'GRNGU': 0,'LSNSC': 0,'YRNMC': 0,'WRNWU': 0,'GRAGR': 0,'COGSC': 0,'GRBMC': 0,'GRGJC': 0,'CRNCU': 0,'DRNCU': 0,'SJNBU': 0,
         'CRLFA': 0,'CRGCA': 0,'CRHDA': 0,'CRDDA': 0,'CRPDA': 0,'CAIDA': 0,'SRAGA': 0,'GRHSA': 0,'GRNDA': 0,'CRAMA': 0,'SJARN': 0,'TRTPC': 0} # for computing mean

counter = 0
n_files = len(files)
for f in files:
  counter += 1
  lat = f.split('_')[1]
  long = f.split('_')[2]
  print(long, lat, counter, '/', n_files)
  coords = gpd.points_from_xy([long], [lat], crs="EPSG:4326")
  for _, row in shp_df.iterrows():
    if row.geometry.contains(coords):
      file_name = f'../vic/results/baseline_calib_long/{f}'
    #   file_name = f'C:/Users/375237/Desktop/CRB-human-impacts/Data/baseline_calib_long/{f}'
      df = pd.read_csv(file_name, delimiter='\t', names=colnames, header=None)
      df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
      df['soil'] = df['soilm1'] + df['soilm2'] + df['soilm3']
      df.set_index(df['date'], inplace=True)
      swe[row.BASIN_NAME] += df['swe']
      soil[row.BASIN_NAME] += df['soil']
      evap[row.BASIN_NAME] += df['evap']
      sdepth[row.BASIN_NAME] += df['sdepth']
      baseflow[row.BASIN_NAME] += df['baseflow']
      runoff[row.BASIN_NAME] += df['runoff']
      count[row.BASIN_NAME] += 1 # for computing mean

swe.to_csv('swe_livneh_sum.csv')
soil.to_csv('soil_livneh_sum.csv')
evap.to_csv('evap_livneh_sum.csv')
sdepth.to_csv('sdepth_livneh_sum.csv')
baseflow.to_csv('baseflow_livneh_sum.csv')
runoff.to_csv('runoff_livneh_sum.csv')

# update values to reflect mean
for key in count.keys():
  swe[key] /= count[key]
  soil[key] /= count[key]
  evap[key] /= count[key]
  sdepth[key] /= count[key]
  baseflow[key] /= count[key]
  runoff[key] /= count[key]

# save to csv
swe.to_csv('swe_livneh_mean.csv')
soil.to_csv('soil_livneh_mean.csv')
evap.to_csv('evap_livneh_mean.csv')
sdepth.to_csv('sdepth_livneh_mean.csv')
baseflow.to_csv('baseflow_livneh_mean.csv')
runoff.to_csv('runoff_livneh_mean.csv')