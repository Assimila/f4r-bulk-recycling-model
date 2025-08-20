# %% [markdown]
# Extract data from a ERA5 NetCDF files

# %%
import numpy as np
import xarray as xr
import pandas as pd
import scipy
import sys 
import warnings
import glob
warnings.filterwarnings('ignore')
import time

# %%
dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/" 
#datao ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/era/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

#For selection and plotting
Y1, Y2 = 1994,1994
m1 = '01'
time_bnds = (str(Y1)+'-01-01',str(Y2)+'-01-31')
lon_bnds, lat_bnds = (15, 30), (5,-10)
#p_bnds = (1000,300) #for daily folder files
p_bnds = (30000,100000)

# %%
start = time.time()
from functools import partial
def _preprocess_pres(x, lon_bnds, lat_bnds, p_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds),
                 plev=slice(*p_bnds),drop=True)
partial_func_pres = partial(_preprocess_pres, lon_bnds=lon_bnds, lat_bnds=lat_bnds, p_bnds=p_bnds)

#Reading in pressure level variables from ERA5
ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(Y1)+"-"+m1+"*.nc",
                                drop_variables=['r','t','w'],
                                preprocess=partial_func_pres,parallel=True).resample(time='D').mean(dim='time').load()
print('opened press files', ds_era_pres['plev'])
ds_era_pres = ds_era_pres.sel(time=slice(*time_bnds),drop=True)
ds_era_pres = ds_era_pres.rename({'plev':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
ds_era_pres['level'] = ds_era_pres['level']/1000.0 #for half monthly files only 
ds_era_pres = ds_era_pres.sortby('level', ascending=False) #for half monthly files only
print('DONE read in pres',ds_era_pres)
end = time.time()
length = end - start
print("It took", length, "seconds!")
print('---------------------------------------------------------------------')

#---------------------------------------------------
# %%

start = time.time()
from functools import partial
def _preprocess_land(x, lon_bnds, lat_bnds):
    x = x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),drop=True)
    return x
partial_func_land = partial(_preprocess_land, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

#Reading in surface variables from ERA5 Land
ds_era_land = xr.open_mfdataset("/Volumes/ESA_F4R/era/era5_land/era5_land_variables_central_africa_"+str(Y1)+"-"+m1+".nc",
                                drop_variables=['expver','number','pev','ssr','t2m'],
                                preprocess=partial_func_land,parallel=True).load()
ds_era_land = ds_era_land.sel(valid_time=slice(*time_bnds),drop=True)
ds_era_land = ds_era_land.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','tp':'Prec','e':'Evap','sp':'Psfc'})
ds_era_land = ds_era_land.interp(lat=ds_era_pres['lat'],lon=ds_era_pres['lon'],method='linear',kwargs={"fill_value": "extrapolate"})
print('era after int before resample',ds_era_land)
Psfc = ds_era_land['Psfc'].resample(time='D').mean(dim='time')
print('*********PSFC',Psfc)
ds_era_land = ds_era_land.resample(time='D').sum(dim='time')
ds_era_land['Psfc'] = Psfc
print('DONE read in land',ds_era_land)
end = time.time()
length = end - start
print("It took", length, "seconds!")
print('---------------------------------------------------------------------')

#---------------------------------------------------
# %%
start = time.time()
ds = xr.merge([ds_era_pres,ds_era_land]) 
ds_era_pres.close()
ds_era_land.close()
ds = ds.sortby('lat', ascending=True)
ds = ds.transpose("lon", "lat", "level", "time",missing_dims='ignore')
print('Done data read in', ds)
end = time.time()
length = end - start
print("It took", length, "seconds!")
start = time.time()
ds.to_netcdf(datao+"erads.nc", mode='w', format='NETCDF4', engine='netcdf4')
print('done saving file')
end = time.time()
length = end - start
print("It took", length, "seconds!")