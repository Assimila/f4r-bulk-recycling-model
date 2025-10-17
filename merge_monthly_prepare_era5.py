# %%
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

# %%
dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

for YR in range(1993,1993):
    print('Year: ',YR)
    # %%
    #For selection and plotting
    time_bnds = (str(YR)+'-01-01',str(YR)+'-12-31')
    lon_bnds, lat_bnds = (7, 35), (13,-16)
    lon_bnds_f, lat_bnds_f = (9, 34), (-15,11) 
    p_bnds = (30000,100000)
    
    # %% [markdown]
    # **Read in ERA5 data on pressure levels (hourly timesteps in fortnightly files)**
    # - *fortnightly files currently run from 1994-2024*
    # - resampled to monthly MS timestep
    # - shum multiplied by 1000 to convert from kg/kg --> g/kg
    # - pressure levels are divided by 100 to convert from Pa to hPa (only for fortnightly files)
    # - sort data by descending pressure levels (only for fortnightly files)
    # 
    # 
    # **Input file units:**
    # - plev - pa
    # - q - kg/kg
    # - u - m/s
    # - v - m/s
    
    # %%
    start = timer.time()
    from functools import partial
    def _preprocess_pres(x, lon_bnds, lat_bnds, p_bnds):
        return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds),
                     plev=slice(*p_bnds),drop=True)
    partial_func_pres = partial(_preprocess_pres, lon_bnds=lon_bnds, lat_bnds=lat_bnds, p_bnds=p_bnds)
    
    #Reading in pressure level variables from ERA5
    ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(YR)+"*.nc",
                                    drop_variables=['r','t','w'],
                                    preprocess=partial_func_pres,parallel=True).resample(time='MS').mean(dim='time').load()
    ds_era_pres = ds_era_pres.rename({'plev':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
    ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
    ds_era_pres['level'] = ds_era_pres['level']/100.0  
    ds_era_pres = ds_era_pres.sortby('level', ascending=False) 
    end = timer.time()
    length = end - start
    print("ERA5 pressure level data read in took ", length, "seconds")
    
    # %% [markdown]
    # **Read in ERA5 land data (hourly in monthly files)**
    # - selecting hour 23 (0-23) of Prec and Evap because of how ERA5 Land variables are accumulated (https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790 - https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-accumulationsAccumulations)
    # - prec is multiplied by 1000 to convert from m to mm
    # - evap is multiplied by -1000 to convert from m to mm and upward fluxes in land model are considered negative
    # - Prec, Evap, and Psfc are then resampled to MS monthly and also interpolated to coarser pressure level grid
    # 
    # **Input file units:**
    # - tp - m
    # - e - m (-)
    # - sp - pa
    
    # %%
    start = timer.time()
    from functools import partial
    def _preprocess_land(x, lon_bnds, lat_bnds):
        x = x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),drop=True)
        return x
    partial_func_land = partial(_preprocess_land, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    
    #Reading in surface variables from ERA5 Land
    ds_era_land = xr.open_mfdataset(dataf+"era5_land/era5_land_variables_central_africa_"+str(YR)+"*.nc",
                                    drop_variables=['expver','number','pev','ssr','t2m'],
                                    preprocess=partial_func_land,parallel=True).load()
    ds_era_land = ds_era_land.rename({'valid_time':'time','latitude':'lat',
                                      'longitude':'lon','tp':'Prec','e':'Evap','sp':'Psfc'})
    ds_era_land = ds_era_land.interp(lat=ds_era_pres['lat'],lon=ds_era_pres['lon'],method='linear',kwargs={"fill_value": "extrapolate"})
    Prec = ds_era_land['Prec'].where(ds_era_land['time.hour']==23,drop=True)*1000.0
    Evap_land = ds_era_land['Evap'].where(ds_era_land['time.hour']==23,drop=True)*-1000.0
    Prec = Prec.resample(time='MS').mean(dim='time') 
    Evap_land = Evap_land.resample(time='MS').mean(dim='time')
    Psfc_land = ds_era_land['Psfc'].resample(time='MS').mean(dim='time')/100.0
    ds_era_land.close()
    end = timer.time()
    length = end - start
    print("ERA5 land data read in took ", length, "seconds")
    
    # %% [markdown]
    # Read in ERA5 surface vars for merging - setting non land evaporation to zero. Merging surface pressure level and land surface pressure to output. 
    
    # %%
    start = timer.time()
    
    #Reading in surface variables from ERA5 surface files
    ds_era_psfc = xr.open_mfdataset(dataf+"era5/era5_surface/era5_surface_pressure_central_africa_"+str(YR)+"*.nc",
                                    drop_variables=['expver','number'],
                                    preprocess=partial_func_land,parallel=True).load()
    ds_era_psfc = ds_era_psfc.rename({'valid_time':'time','latitude':'lat',
                                      'longitude':'lon','sp':'Psfc'})
    ds_era_psfc = ds_era_psfc.interp(lat=ds_era_pres['lat'],lon=ds_era_pres['lon'],method='linear',kwargs={"fill_value": "extrapolate"})
    Psfc_surface = ds_era_psfc['Psfc'].resample(time='MS').mean(dim='time')/100.0
    ds_era_psfc.close()
    
    Evap = Evap_land.fillna(0.0)
    Psfc = Psfc_land.combine_first(Psfc_surface)
    
    length = end - start
    print("ERA5 surface data read in took ", length, "seconds")
    
    # %% [markdown]
    # **Merging all input datasets into one dataset for recyling code called ds**
    # - close both input datasets
    # - sort everything so latitude is south to north
    # - transpose dimensions so they run (lon,lat,level,time) as in recycling code
    # - save input ds to file
    
    # %%
    start = timer.time()
    ds = xr.merge([ds_era_pres,Prec,Evap,Psfc]) 
    ds_era_pres.close()
    ds = ds.sortby('lat', ascending=True)
    ds = ds.sel(lat=slice(*lat_bnds_f),lon=slice(*lon_bnds_f))
    ds = ds.transpose("lon", "lat", "level", "time",missing_dims='ignore')
    
    ds.to_netcdf(datao+"merge_erads_"+str(YR)+".nc", mode='w', format='NETCDF4', engine='netcdf4')
    end = timer.time()
    length = end - start
    print("Merging and dataset output took ", length, "seconds")
    
    
    