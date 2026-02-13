# ## DRC shapefile recycling
# - running with merged data (zero evap over ocean) and DRC shapefile
# - running with rotation options and plotting seasonal averages for all rotations

import numpy as np
import xarray as xr
import pandas as pd
import scipy
import sys 
import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
from shapely.geometry import mapping
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import time as timer
start_all = timer.time()

# ### Read in surface and pressure level files
# **Surface file options:**
# - S_SE (surface vars including only surface evaporation)
# - S_LSE (surface vars including land and surface evaporation)
# 
# **Pressure level file options:**
# - L_M (pressure level vars resampled to monthly timestep)
# - L_HI (pressure level vars with integrated moist flux calcualted hourly and then resampled to monthly timestep)

dataf ="/Volumes/ESA_F4R/ed_prepare/2026_mergeds/" 
datao ="/Volumes/ESA_F4R/ed_prepare/2026_rho/drc/" 
datap ="/Volumes/ESA_F4R/ed_prepare/2026_plots/drc/" 
datas ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/Shapefiles/"
shp_cod = gpd.read_file(datas+"geoBoundaries-COD-ADM0.shp")

S_NAME = "S_SE" # S_SE or S_LSE 
L_NAME = "L_M" # L_M or L_HI

years = [1990, 1991, 1992, 1993, 
         1996, 1997, 1998, 1999, 2000, 2001, 2002, 
         2003, 2004, 2005, 2006, 2007, 2008, 2009, 
         2010, 2012, 2013, 2014, 2015,
         2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

for YR in years:
        print(YR)
        
        # #### **Read in pre-processed files that have the following conversions:**
        # 
        # **Read in ERA5 data on pressure levels**
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
        # 
        # ** ***Might include Fx and Fy if L_HI is being read in***
        # 
        # **Read in ERA5 land and surface data**
        # - Land: selecting hour 23 (0-23) of Prec and Evap because of how ERA5 Land variables are accumulated (https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790 - https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-accumulationsAccumulations)
        # - Surface: evap is accumulated per hour so a daily sum is calculated
        # - prec is multiplied by 1000 to convert from m to mm
        # - evap is multiplied by -1000 to convert from m to mm and upward fluxes in land model are considered negative
        # - Prec, Evap, and Psfc are then resampled to MS monthly and also interpolated to coarser pressure level grid
        # 
        # **Input file units:**
        # - tp - m (no longer reading in)
        # - e - m (-)
        # - sp - pa
        # 
        # **Merging all input datasets into one dataset for recyling code called ds**
        # - close both input datasets
        # - sort everything so latitude is south to north
        # - transpose dimensions so they run (lon,lat,level,time) as in recycling code
        # - save input ds to file
        # 
        # **Integrate zonal and meridional moisture flux**
        # - **Must check if there are any nans in input arrays** - there can be none because we are using nans as an indicator in the modified definitions
        
        dsS = xr.open_dataset(dataf+"merge_erads_"+S_NAME+"_"+str(YR)+".nc")
        dsL = xr.open_dataset(dataf+"merge_erads_"+L_NAME+"_"+str(YR)+".nc")
        ds = xr.merge([dsS,dsL])
        
        #Arrange data and prepare it to be clipped by the shapefile
        ds = ds.transpose("time","level","lat","lon",missing_dims='ignore')
        #ds = ds.rio.set_spatial_dims(x_dim="lon",y_dim="lat")
        ds = ds.rename({'lon': 'x','lat': 'y'})
        ds.rio.write_crs("epsg:4326", inplace=True)
        ds['Evap_drc'] = ds['Evap_all'].rio.clip(shp_cod.geometry.apply(mapping),shp_cod.crs,drop=False)
        ds = ds.rename({'x': 'lon','y': 'lat'})
        ds['Evap_drc'] = ds['Evap_drc'].fillna(0.0)
        
        #Points that are 0.0 are inside the region, points on the boundary will be
        ds = ds.transpose("lon","lat","level","time",missing_dims='ignore')
        
        ds = ds.sel(lon=slice(11,32),lat=slice(-14,6))
        
        print(np.isnan(ds['Evap_drc'].values).any())
        print(ds['Evap_drc'].isnull().count().values)
        
        if L_NAME=='L_HI':
            Fx = ds['Fx']
            Fy = ds['Fy']
        else:
            #Prepping datasets near surface for recycling
            import bulk_recycling_model.numerical_integration
            
            # Integrate 10^-3 Shum Uwnd dp
            # Because the integration limits are from high pressure to low pressure, we need to invert the sign.
            integrand = -1 * 1e-3 * ds["Shum"] * ds["Uwnd"]
            Fx = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, ds["Psfc"])
            # Units: mb x m/s
            
            # Integrate 10^-3 Shum Vwnd dp
            # Because the integration limits are from high pressure to low pressure, we need to invert the sign.
            integrand = -1 * 1e-3 * ds["Shum"] * ds["Vwnd"]
            Fy = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, ds["Psfc"])
            # Units: mb x m/s
        
        # **Prepare scaled data for recycling code**
        # - Evaporation and moisture fluxes
        # Prepare and scale the data
        from bulk_recycling_model import preprocess
        from bulk_recycling_model import ED_preprocess
        from bulk_recycling_model.axis import Axis
        from bulk_recycling_model.scaling import Scaling, UnitSystem
        
        # degrees
        L = ds.coords["lon"].max().item() - ds.coords["lon"].min().item()
        # convert to meters
        L = L * 111e3 * np.cos(np.deg2rad(ds.coords["lat"].mean().item()))
        dx = L / ds.sizes["lon"]
        
        # lon axis
        lon_axis = Axis(
            ds.coords["lon"].min().item(),
            ds.coords["lon"].diff("lon").mean().item(),
            ds.sizes["lon"],
        )
        
        # degrees
        H = ds.coords["lat"].values[-1] - ds.coords["lat"].values[0]
        # convert to meters
        H = H * 111e3
        dy = H / ds.sizes["lat"]
        
        # lat axis
        lat_axis = Axis(
            ds.coords["lat"].min().item(),
            ds.coords["lat"].diff("lat").mean().item(),
            ds.sizes["lat"],
        )
        
        print(f"{L = :.2e} m")
        print(f"{dx = :.2e} m")
        print(f"{H = :.2e} m")
        print(f"{dy = :.2e} m")
        
        # make a scaling object to convert between unit systems
        scaling = Scaling(H)
        
        dx = scaling.distance.convert(dx, UnitSystem.SI, UnitSystem.scaled)
        dy = scaling.distance.convert(dy, UnitSystem.SI, UnitSystem.scaled)
        print(f"{dx = :.2e} scaled")
        print(f"{dy = :.2e} scaled")
        
        # convert Fx and Fy to scaled units
        Fx = scaling.water_vapor_flux.convert(Fx.values, UnitSystem.natural, UnitSystem.scaled)
        Fy = scaling.water_vapor_flux.convert(Fy.values, UnitSystem.natural, UnitSystem.scaled)
        
        # convert E to scaled units
        # Do this for both the total E and the local regionally clipped E]
        #print('pre-scaled',ds['Evap'])
        E_total = scaling.evaporation.convert(ds["Evap_all"].values, UnitSystem.natural, UnitSystem.scaled)
        E_local = scaling.evaporation.convert(ds["Evap_drc"].values, UnitSystem.natural, UnitSystem.scaled)
        
        # **Run recycling model for each timestep**
        # - Create recycling output array based on the shape of one of the surface input files: evap 
        # - Translate evap and fluxes to secondary grid
        # - Calculate modeled precipitation
        # - Plot scaled input variables (evap and fluxes)
        # - Run through each timestep in the input files and calculate recycling ratio at each timestep across domain
        # - Plot rho and convergence metric for each timestep
        
        import matplotlib.pyplot as plt
        import logging
        logging.basicConfig()
        logging.getLogger("bulk_recycling_model").setLevel(logging.INFO)
        from bulk_recycling_model import plotting
        from bulk_recycling_model.main import run_4_orientations
        
        #Make the rho array the same shape as the total E - will clip the external points at the end
        rho_ar = np.empty((4,np.shape(E_total)[0]-1,np.shape(E_total)[1]-1,np.shape(E_total)[2]))
        #Entering preprocessing and time step loop
        #Run model and plot
        for i,time in enumerate(ds.time):
             
            # preprocess E onto the secondary grid
            Ei_total = ED_preprocess.prepare_E(E_total[:,:,i])
            Ei_local = ED_preprocess.prepare_E(E_local[:,:,i])
            
            # preprocess water vapor fluxes onto the secondary grid
            Fxi_left = preprocess.prepare_Fx_left(Fx[:,:,i])
            Fxi_right = preprocess.prepare_Fx_right(Fx[:,:,i])
            Fyi_bottom = preprocess.prepare_Fy_bottom(Fy[:,:,i])
            Fyi_top = preprocess.prepare_Fy_top(Fy[:,:,i])
            
            # compute P
            Pi = preprocess.calculate_precipitation(Fxi_left, Fxi_right, Fyi_bottom, Fyi_top, Ei_total, dx, dy)
        
            # Run the model
            status = run_4_orientations(
                Fxi_left,
                Fxi_right,
                Fyi_bottom,
                Fyi_top,
                Ei_local,
                Pi,
                dx,
                dy,
                R=0.2,
                R_1=0.2,
                max_iter=500,
                tol=1e-3,
            )
                
            #Print timestep and status (converged or not) and add rho to recycling ration array
            rot = 2
            print(i,time.values)
            print('Rotation is: ',rot+1)
            print(status[rot]['k'])
            rho_ar[0,:,:,i] = status[0]["rho"]
            rho_ar[1,:,:,i] = status[1]["rho"]
            rho_ar[2,:,:,i] = status[2]["rho"]
            rho_ar[3,:,:,i] = status[3]["rho"]
        
        # **Create and save rho xarray file**
        # 
        # - Create an xarray to store all of the calculated recycling ratios that is organised in an easy to plot/interpret format
        # - Count number of values in array over 1 - replace all of these with 1
        # - Count number of negative rho values - replace all of these with zero
        # - Save to file
        
        lon_ar = np.linspace(start=ds.coords["lon"].min().values+lon_axis.step/2,
                             stop=ds.coords["lon"].max().values-lon_axis.step/2,
                             num=lon_axis.n_points-1)
        lat_ar = np.linspace(start=ds.coords["lat"].min().values+lat_axis.step/2,
                             stop=ds.coords["lat"].max().values-lat_axis.step/2,
                             num=lat_axis.n_points-1)
        rho_xarr = xr.Dataset(
            data_vars=dict(rho=(["rot","lon","lat","time"],rho_ar)),
            coords=dict(
                rot=(["rot"],[1,2,3,4]),
                lon=(["lon"], lon_ar),
                lat=(["lat"], lat_ar),
                time=(["time"],ds.time.data)
            ),
            attrs=dict(
                description="Recycling ratio",
                units="%",
            ),
        ) 
        rho_xarr = rho_xarr.transpose("rot","time","lat","lon")
        rho_xarr = rho_xarr.rio.set_spatial_dims(x_dim="lon",y_dim="lat")
        rho_xarr.rio.write_crs("epsg:4326", inplace=True)
        rho_xarr = rho_xarr.rio.clip(shp_cod.geometry.apply(mapping),shp_cod.crs,drop=False)
        rho_xarr.to_netcdf(datao+L_NAME+"_"+S_NAME+"_drc_rot_rho_era5_"+str(YR)+".nc")
                
        rho_xarr.close()
        
        
        
        