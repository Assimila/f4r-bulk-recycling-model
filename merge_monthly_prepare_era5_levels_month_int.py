# ## ERA5 level data readin and monthly average for rho calcs
# ### Read in ERA5 and resample to monthly

import numpy as np
import xarray as xr
import pandas as pd
import scipy
import sys 
import warnings
import matplotlib.pyplot as plt
from shapely.geometry import mapping
import cartopy.crs as ccrs
import cartopy.feature
warnings.filterwarnings('ignore')
import time as timer
start_all = timer.time()

dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/2026_mergeds/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

## 1994 and 1995 is being skipped until they can be downloaded properly

#years = [1990, 1991, 1992, 1993, 
#         1996, 1997, 1998, 1999, 2000, 2001, 2002, 
#         2003, 2004, 2005, 2006, 2007, 2008, 2009, 
#         2010, 2011, 2012, 2013, 2014, 2015, 2016,
#         2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

years = [2020]

for YR in years:
    print(YR)
    #For selection and plotting
    time_bnds = (str(YR)+'-01-01',str(YR)+'-12-31')
    lon_bnds, lat_bnds = (8, 32), (12,-15)
    lon_bnds_f, lat_bnds_f = (8, 32), (-15,12) 
    p_bnds = (30000,100000)
    
    # **Read in ERA5 data on pressure levels (hourly timesteps in fortnightly files)**
    # - *fortnightly files currently run from 1994-2024*
    # - resampled to monthly MS timestep
    # - shum multiplied by 1000 to convert from kg/kg --> g/kg
    # - pressure levels are divided by 100 to convert from Pa to hPa (only for fortnightly files)
    # - sort data by descending pressure levels (only for fortnightly files)
    # 
    # **Input file units:**
    # - plev - pa
    # - q - kg/kg
    # - u - m/s
    # - v - m/s
    
    from functools import partial
    def _preprocess_pres(x, lon_bnds, lat_bnds, p_bnds):
        return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds),
                     plev=slice(*p_bnds),drop=True)
    partial_func_pres = partial(_preprocess_pres, lon_bnds=lon_bnds, lat_bnds=lat_bnds, p_bnds=p_bnds)
    
    #Reading in pressure level variables from ERA5
    ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(YR)+"*.nc",
                                    drop_variables=['r','t','w'],
                                    preprocess=partial_func_pres,parallel=False).resample(time='MS').mean(dim='time').load()
    ds_era_pres = ds_era_pres.rename({'plev':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
    ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
    ds_era_pres['level'] = ds_era_pres['level']/100.0  
    ds_era_pres = ds_era_pres.sortby('level', ascending=False) 
    
    # **Write out one monthly pressure level dataset for recyling code called ds**
    # - sort everything so latitude is south to north
    # - transpose dimensions so they run (lon,lat,level,time) as in recycling code
    # - save input ds to file
    
    ds_era_pres = ds_era_pres.sortby('lat', ascending=True)
    ds_era_pres = ds_era_pres.sel(lat=slice(*lat_bnds_f),lon=slice(*lon_bnds_f))
    ds_era_pres = ds_era_pres.transpose("lon", "lat", "level", "time",missing_dims='ignore')
    
    ds_era_pres.to_netcdf(datao+"merge_erads_L_M_"+str(YR)+".nc", mode='w', format='NETCDF4', engine='netcdf4')
    end = timer.time()
    length = end - start_all
    print("Level dataset output took ", length, "seconds")
    
    
    