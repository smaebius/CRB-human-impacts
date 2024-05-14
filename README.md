## Machine learning partitioning approach to improve 🌊 streamflow 🌊 estimates in basins with human and climate alterations in the Colorado River Basin

### Data

- `VotE`: a data fusion platform providing access to streamflow data allowing for quick analysis, mapping, and modeling of Earth's rivers
  - [Source](https://github.com/VeinsOfTheEarth/VotE)
- `Streamflow`: target streamflow in cubic m/s
  - [USGS Source](https://waterdata.usgs.gov/nwis/rt)
- `Meteorological Forcings`: daily inputs including evapotranspiration, total precipitation, soil moisture, SWE, average and maximum temperature, and solar radiation variables
  - [ERA5 Hourly Data Source](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)

### Project Code
Jupyter notebooks containing code used in this study:
- `S0-Load_Data`: Code used to download streamflow data and forcings from `VotE`
- `S1-Load_CA_Data`: Code used to download geospatial datasets from Google Earth Engine
- `S2-Join_CA_Data`: Code used to combine geospatial datasets with the streamflow model data
- `S3-Partitions`: Code used to partition datasets into Experiments A, B, C, D, and E.
