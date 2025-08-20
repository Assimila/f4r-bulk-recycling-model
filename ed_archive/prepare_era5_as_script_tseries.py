# %% [markdown]
# Extract data from a ERA5 NetCDF files

# %%
import numpy as np
import xarray as xr
import pandas as pd
import scipy
import sys 

# %%
dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/" 
#datao ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/era/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

#For selection and plotting
time_bnds = ('1990-01-01','2000-12-31')
lon_bnds, lat_bnds = (15, 30), (5,-10)
p_bnds = (1000,300)
#Reading in surface variables from ERA5 Land
ds_era_land = xr.open_mfdataset("/Volumes/ESA_F4R/era/era5_land/era5_land_variables_central_africa_*.nc",
                                drop_variables=['expver','number','pev','ssr','t2m'])
ds_era_land = ds_era_land.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),valid_time=slice(*time_bnds))
ds_era_land = ds_era_land.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','tp':'Prec','e':'Evap','sp':'Psfc'})
#Convert m to mm/day for tp and e
ds_era_land['Prec'] = 1000.0*ds_era_land['Prec'].resample(time='D').sum()
ds_era_land['Evap'] = 1000.0*ds_era_land['Evap'].resample(time='D').sum()
#Convert Pa to hPa
ds_era_land['Psfc'] = 0.01*ds_era_land['Psfc'].resample(time='D').mean()

#Reading in pressure level variables from ERA5
ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/daily/era5_pressure_level_variables_central_africa_*.nc",
                                drop_variables=['r','t','w'])
ds_era_pres = ds_era_pres.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),
                              valid_time=slice(*time_bnds),pressure_level=slice(*p_bnds))
ds_era_pres = ds_era_pres.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','pressure_level':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
ds = xr.merge([ds_era_pres,ds_era_land]) 
ds_era_pres.close()
ds_era_land.close()
print(ds)
print('datasets merged')
ds.load()
ds.to_netcdf(datao+"erads.nc")