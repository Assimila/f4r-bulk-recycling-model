# ## Surface Evap and Masked Surface Evap
# ### Read in ERA5 and create a surface evap + land masked surface evap array
# ** **not using ERA5 Land in this code**

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

dataf ="/Volumes/blue_wd/ESA_F4R/ERA5_monthly/" 
datao ="/Volumes/blue_wd/ESA_F4R/2026_mergeds/"
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

#For selection and plotting
lon_bnds, lat_bnds = (8, 32), (12,-15)
lon_bnds_f, lat_bnds_f = (8, 32), (-15,12) 

# **Read in ERA5 land-sea mask to use to create a land only version of surface evaporation**
# Values greater than 0 are land values which can include water bodies on land such as rivers

mask = xr.open_dataset("/Users/ellendyer/Documents/GitHub/f4r-bulk-recycling-model/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc")['lsm']
mask = mask.rename({'latitude':'lat','longitude':'lon'})
mask = mask.squeeze(['time'], drop=True)
mask = mask.where(mask.lon<24.0,1)
water = mask.where(mask==0.0,drop=True)


# Read in ERA5 surface vars for merging - setting non land evaporation to zero. 

from functools import partial
def _preprocess_land(x, lon_bnds, lat_bnds):
    x = x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),drop=True)
    return x
partial_func_land = partial(_preprocess_land, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

#Reading in surface variables from ERA5 surface files
ds_era_psfc = xr.open_dataset(dataf+"era5_sp.nc",
                                drop_variables=['expver','number']).load()
ds_era_psfc = ds_era_psfc.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','sp':'Psfc'})
Psfc_surface = ds_era_psfc['Psfc']/100.0
ds_era_psfc.close()

ds_era_evap = xr.open_dataset(dataf+"era5_e_tp.nc",
                                drop_variables=['expver','number']).load()
ds_era_evap = ds_era_evap.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','e':'Evap','tp':'Prec'})
print(ds_era_evap)
Evap_surface = ds_era_evap['Evap']*-1000.0
Prec = ds_era_evap['Prec']*1000.0
ds_era_evap.close()

Evap_all = Evap_surface
mask_surf = mask.interp(lat=Evap_surface['lat'],lon=Evap_surface['lon'],method='linear',kwargs={"fill_value": "extrapolate"})
Evap_land = Evap_surface.where(mask_surf!=0.0,0,0)

Evap_all = Evap_all.rename('Evap_all')
Evap_land = Evap_land.rename('Evap_land')

Evap_land['time'] = Psfc_surface['time']
Evap_all['time'] = Psfc_surface['time']
Prec['time'] = Psfc_surface['time']

print(Evap_all)
print(Evap_land)
print(Prec)

# **Merging all input datasets into one dataset for recyling code called ds**
# - close both input datasets
# - sort everything so latitude is south to north
# - transpose dimensions so they run (lon,lat,level,time) as in recycling code
# - save input ds to file

ds = xr.merge([Evap_land,Evap_all,Psfc_surface,Prec]) 
print(ds)
ds = ds.sortby('lat', ascending=True)
ds = ds.sel(lat=slice(*lat_bnds_f),lon=slice(*lon_bnds_f))

ds = ds.transpose("lon", "lat", "level", "time",missing_dims='ignore')

ds.to_netcdf(datao+"merge_era_surface.nc", mode='w', format='NETCDF4', engine='netcdf4')
end = timer.time()
length = end - start_all
print("Merging and dataset output took ", length, "seconds")


    