# ## ERA5 level data readin and integrate flux before averaging for rho calcs
# ### Read in ERA5 and calculate hourly integrated flux before resampling to monthly

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
import bulk_recycling_model.numerical_integration
warnings.filterwarnings('ignore')
import time as timer
start_all = timer.time()

dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

## 1994 and 1995 is being skipped until they can be downloaded properly

#years = [1990, 1991, 1992, 1993, 
#         1996, 1997, 1998, 1999, 2000, 2001, 2002, 
#         2003, 2004, 2005, 2006, 2007, 2008, 2009, 
#         2010, 2011, 
#years = [2012, 2013, 2014, 2015, 2016,
#years = [2017, 2018, 2019, 2020, 
years = [2021, 2022, 2023, 2024]

for YR in years:
    print(YR)
    ds_list = []
    #For selection and plotting
    time_bnds = (str(YR)+'-01-01',str(YR)+'-12-31')
    lon_bnds, lat_bnds = (8, 32), (12,-15)
    p_bnds = (30000,100000)


    # **Read in ERA5 surface pressure to calculate integrated moisture flux on hourly timestep**
    from functools import partial
    def _preprocess_land(x, lon_bnds, lat_bnds):
        x = x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),drop=True)
        return x
    partial_func_land = partial(_preprocess_land, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    
    surf_M = {}
    for M in np.arange(1,13):
        #Reading in surface variables from ERA5 surface files
        ds_era_psfc = xr.open_mfdataset(dataf+"era5/era5_surface/era5_surface_pressure_central_africa_"+str(YR)+"-"+str("{:02d}".format(M))+".nc",
                                        drop_variables=['expver','number'],
                                        preprocess=partial_func_land,parallel=True).load()
        ds_era_psfc = ds_era_psfc.rename({'valid_time':'time','latitude':'lat',
                                          'longitude':'lon','sp':'Psfc'})
        Psfc = ds_era_psfc['Psfc']/100.0
        Psfc = Psfc.sortby('lat', ascending=True) 
        #print(Psfc.time)
        surf_M[M]=Psfc
        ds_era_psfc.close()
    
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
    
    levs_M = {}
    for M in np.arange(1,13):
        print(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(YR)+"-"+str("{:02d}".format(M))+"*.nc")
        #Reading in pressure level variables from ERA5
        ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(YR)+"-"+str("{:02d}".format(M))+"*.nc",
                                        drop_variables=['r','t','w'],
                                        preprocess=partial_func_pres,parallel=True).load()
                                        
        ds_era_pres = ds_era_pres.rename({'plev':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
        ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
        ds_era_pres['level'] = ds_era_pres['level']/100.0  
        ds_era_pres = ds_era_pres.sortby('level', ascending=False) 
        ds_era_pres = ds_era_pres.sortby('lat', ascending=True)
        #print(ds_era_pres)
        levs_M[M]=ds_era_pres
        ds_era_pres.close() 
    
        # **Write out one monthly pressure level dataset for recyling code called ds**
        # - calculate integrated moisture flux using surface pressure
        # Integrate 10^-3 Shum Uwnd dp
        # Because the integration limits are from high pressure to low pressure, we need to invert the sign.
        integrand = -1 * 1e-3 * levs_M[M]["Shum"] * levs_M[M]["Uwnd"]
        levs_M[M]['Fx'] = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, surf_M[M])
        # Units: mb x m/s
        
        # Integrate 10^-3 Shum Vwnd dp
        # Because the integration limits are from high pressure to low pressure, we need to invert the sign.
        integrand = -1 * 1e-3 * levs_M[M]["Shum"] * levs_M[M]["Vwnd"]
        levs_M[M]['Fy'] = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, surf_M[M])
        # Units: mb x m/s
    
        # **Write out one monthly pressure level dataset for recyling code called ds**
        # - resample to monthly timestep
        # - transpose dimensions so they run (lon,lat,level,time) as in recycling code
        # - save input ds to file
    
        levs_M[M] = levs_M[M].resample(time='MS').mean(dim='time')
        levs_M[M] = levs_M[M].transpose("lon", "lat", "level", "time",missing_dims='ignore')
        ds_list.append(levs_M[M])
        print("done ",M)
        
    ds_era_pres_out = xr.concat(ds_list,dim='time')
    print(ds_era_pres_out)
    
    ds_era_pres_out.to_netcdf(datao+"merge_erads_L_HI_"+str(YR)+".nc", mode='w', format='NETCDF4', engine='netcdf4')
    end = timer.time()
    length = end - start_all
    print("Merging and dataset output took ", length, "seconds")
    
    
    