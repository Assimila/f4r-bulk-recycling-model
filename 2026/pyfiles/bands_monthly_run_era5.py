# ## Latitude bands recycling
# - running with merged data (zero evap over ocean) over three different band options
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

## 1994, 1995, 2010, 2016 are being skipped until they can be downloaded properly

#years = [1990, 1991, 1992, 1993, 
#         1996, 1997, 1998, 1999, 2000, 2001, 2002, 
#         2003, 2004, 2005, 2006, 2007, 2008, 2009, 
#years = [2010, 2012, 2013, 2014, 2015,
#         2017, 2018, 2019, 2020, 
years = [2021, 2022, 2023, 2024]

band = {'N':[5,12,10,31],'EQ':[-5,5,8,29],'S':[-15,-5,12,31]}

for B in band:
    print("Band: ",B)
    for YR in years:
        print(YR)
        dataf ="/Volumes/ESA_F4R/ed_prepare/2026_mergeds/" 
        datao ="/Volumes/ESA_F4R/ed_prepare/2026_rho/bands_rho/" 
        datap ="/Volumes/ESA_F4R/ed_prepare/2026_plots/bands_rho/" 
        datas ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/Shapefiles/"
        shp_cod = gpd.read_file(datas+"congo_basin_evergreen.shp")
        
        S_NAME = "S_SE" # S_SE or S_LSE 
        L_NAME = "L_HI" # L_M or L_HI
        
        # ### Band definitions used
        # - North: 5-12**N** / 10-31**E**
        # - Equatorial: 5**S**-5**N** / 8-29**E**
        # - South: 15-5**S** / 12-31**E**
        
        
        dsS = xr.open_dataset(dataf+"merge_erads_"+S_NAME+"_"+str(YR)+".nc")
        dsL = xr.open_dataset(dataf+"merge_erads_"+L_NAME+"_"+str(YR)+".nc")
        ds = xr.merge([dsS,dsL])
        
        #Clip out specific band
        ds = ds.transpose("time","level","lat","lon",missing_dims='ignore')
        ds = ds.sel(lon=slice(band[B][2],band[B][3]),lat=slice(band[B][0],band[B][1]))
                  
        ds = ds.transpose("lon","lat","level","time",missing_dims='ignore')
        
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
        # 
        
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
        E_local = scaling.evaporation.convert(ds["Evap_land"].values, UnitSystem.natural, UnitSystem.scaled)
        
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
                R=0.05,
                R_1=0.05,
                max_iter=500,
                tol=1e-3,
            )
                
            #Print timestep and status (converged or not) and add rho to recycling ration array
            rot = 0
            print(i,time.values)
            print('Rotation is: ',rot)
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
        rho_xarr.to_netcdf(datao+L_NAME+"_"+S_NAME+"_band_"+B+"_rot_rho_era5_"+str(YR)+".nc")
        
        rho_xarr.close()
            
            
        
        
        