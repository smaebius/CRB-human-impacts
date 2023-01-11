import pandas as pd

## load vote usgs streamflow data
taylor = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/taylor_park.csv", index_col="date")
blue_mesa = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/blue_mesa.csv", index_col="date")
fontenelle = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/fontenelle.csv", index_col="date")
flaming_gorge = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/flaming_gorge.csv", index_col="date")
navajo = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/navajo.csv", index_col="date")
lake_powell = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/lake_powell.csv", index_col="date")

taylor.index = pd.to_datetime(taylor.index)
blue_mesa.index = pd.to_datetime(blue_mesa.index)
fontenelle.index = pd.to_datetime(fontenelle.index)
flaming_gorge.index = pd.to_datetime(flaming_gorge.index)
navajo.index = pd.to_datetime(navajo.index)
lake_powell.index = pd.to_datetime(lake_powell.index)


## load forcings data and combine with streamflow
# The corresponding basins are TRTPC (Taylor Park), GRBMC (Blue Mesa), GRFRW (Fontenelle), GRNGU (Flaming Gorge),
# SJARN (Navajo), and CRLFA (Lake Powell)
def load_forcings(file_name, col_name, taylor, blue_mesa, fontenelle, flaming_gorge,
                    navajo, lake_powell):
  df_forcing = pd.read_csv(file_name,
                           usecols=['dates', 'TRTPC', 'GRFRW', 'GRBMC', 'GRNGU', 'SJARN', 'CRLFA'],
                           index_col='dates')
  df_forcing.index = pd.to_datetime(df_forcing.index)

  taylor = taylor.join(df_forcing[['TRTPC']]).rename(columns={"TRTPC": col_name})
  blue_mesa = blue_mesa.join(df_forcing[['GRBMC']]).rename(columns={"GRBMC": col_name})
  fontenelle = fontenelle.join(df_forcing[['GRFRW']]).rename(columns={"GRFRW": col_name})
  flaming_gorge = flaming_gorge.join(df_forcing[['GRNGU']]).rename(columns={"GRNGU": col_name})
  navajo = navajo.join(df_forcing[['SJARN']]).rename(columns={"SJARN": col_name})
  lake_powell = lake_powell.join(df_forcing[['CRLFA']]).rename(columns={"CRLFA": col_name})

  return taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell

# # evapotraspiration
# taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('/content/drive/MyDrive/CRB-analysis/et_mean.csv',
#                                                               'et', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# mean precipitation
taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings("C:/Users/375237/Desktop/CRB-human-impacts/Data/per_station/precip_mean.csv",
                                                              'prec_mean', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# total precipitation
taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings("C:/Users/375237/Desktop/CRB-human-impacts/Data/per_station/precip_sum.csv",
                                                              'prec_sum', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# # mean soil moisture
# taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('/content/drive/MyDrive/CRB-analysis/soilmoist_mean.csv',
#                                                               'soilmoist', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# # mean snow water equivalent
# taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('/content/drive/MyDrive/CRB-analysis/swe_mean.csv',
#                                                               'swe_mean', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# # total snow water equivalent
# taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('/content/drive/MyDrive/CRB-analysis/swe_sum.csv',
#                                                               'swe_sum', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# mean max temperature
taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('C:/Users/375237/Desktop/CRB-human-impacts/Data/per_station/tmax_mean.csv',
                                                              'tmax', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# mean min temperature
taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('C:/Users/375237/Desktop/CRB-human-impacts/Data/per_station/tmin_mean.csv',
                                                              'tmin', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

# mean wind speed
taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell = load_forcings('C:/Users/375237/Desktop/CRB-human-impacts/Data/per_station/wind_mean.csv',
                                                              'wind', taylor, blue_mesa, fontenelle, flaming_gorge, navajo, lake_powell)

## append naturalized streamflow data
df = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/NaturalMonthly.csv", index_col='datetime', thousands=',')
df.rename(columns={"09109000": "Nat-09109000", "09124700": "Nat-09124700", "09211200": "Nat-09211200",
                   "09234500": "Nat-09234500", "09355500": "Nat-09355500", "09380000": "Nat-09380000"}, inplace=True)
df.index = pd.DatetimeIndex(df.index)
df = df.resample('D').bfill() # upsample to daily resolution

taylor = taylor.join(df['Nat-09124700'])
blue_mesa = blue_mesa.join(df['Nat-09109000'])
fontenelle = fontenelle.join(df['Nat-09211200'])
flaming_gorge = flaming_gorge.join(df['Nat-09234500'])
navajo = navajo.join(df['Nat-09355500'])
lake_powell = lake_powell.join(df['Nat-09380000'])

taylor.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/taylor-combined.csv")
blue_mesa.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/blue_mesa-combined.csv")
fontenelle.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/fontenelle-combined.csv")
flaming_gorge.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/flaming_gorge-combined.csv")
navajo.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/navajo-combined.csv")
lake_powell.to_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/lake_powell-combined.csv")