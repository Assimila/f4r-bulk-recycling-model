# %% [markdown]
# ## Latitude bands recycling
# - running with merged data (zero evap over ocean) over three different band options
# - running with rotation options and plotting seasonal averages for all rotations

# %%
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

# %%
for B in ['N','EQ','S']:
    print('running band:', B)
    years = np.arange(2009,2010)
    for YR in years:
        print('running year:', YR)
        dataf ="/Volumes/ESA_F4R/era/" 
        datao ="/Volumes/ESA_F4R/ed_prepare/" 
        datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"
        datas ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/Shapefiles/"
        shp_cod = gpd.read_file(datas+"congo_basin_evergreen.shp")
        
        # %% [markdown]
        # ### Band definitions used
        # - North: 5-12**N** / 10-31**E**
        # - Equatorial: 5**S**-5**N** / 8-29**E**
        # - South: 15-5**S** / 12-31**E**
        
        # %%
        band = {'N':[5,12,10,31],'EQ':[-5,5,8,29],'S':[-15,-5,12,31]}
        
        # %% [markdown]
        # #### **Read in pre-processed files that have the following conversions:**
        # 
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
        
        # %% [markdown]
        # **Merging all input datasets into one dataset for recyling code called ds**
        # - close both input datasets
        # - sort everything so latitude is south to north
        # - transpose dimensions so they run (lon,lat,level,time) as in recycling code
        # - save input ds to file
        
        # %% [markdown]
        # **Integrate zonal and meridional moisture flux**
        # - ***To do:*** do this with hourly data to test difference in rho
        # - **Must check if there are any nans in input arrays** - there can be none because we are using nans as an indicator in the modified definitions
        
        # %%
        ds = xr.open_dataset(datao+"merge_ds/merge_erads_"+str(YR)+".nc")
        
        #Running the code with an irregular boundary means that there will be nans once the data is clipped
        #For this to work properly it is important that there are no nans in the input data which would 
        #actually be missing data where recycling will be calculated
        print(np.isnan(ds['Evap'].values).any())
        print(ds['Evap'].isnull().count().values)
        print('Number of nans in Evap: ', ds['Evap'].where(np.isnan(ds['Evap'])==True,drop=True).count().values)
        print('Number of nans in Prec: ', ds['Prec'].where(np.isnan(ds['Prec'])==True,drop=True).count().values)
        print('Number of nans in Psfc: ', ds['Psfc'].where(np.isnan(ds['Psfc'])==True,drop=True).count().values)
        print('Number of nans in Uwnd: ', ds['Uwnd'].where(np.isnan(ds['Uwnd'])==True,drop=True).count().values)
        print('Number of nans in Vwnd: ', ds['Vwnd'].where(np.isnan(ds['Vwnd'])==True,drop=True).count().values)
        print('Number of nans in Shum: ', ds['Shum'].where(np.isnan(ds['Shum'])==True,drop=True).count().values)
        
        #Clip out specific band
        ds = ds.transpose("time","level","lat","lon",missing_dims='ignore')
        ds = ds.sel(lon=slice(band[B][2],band[B][3]),lat=slice(band[B][0],band[B][1]))
        ds['Evap_local'] = ds['Evap'].fillna(0.0)
        
        ds = ds.transpose("lon","lat","level","time",missing_dims='ignore')
        print(np.isnan(ds['Evap_local'].values).any())
        print(ds['Evap_local'].isnull().count().values)
        
        # %%
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
        
        # %% [markdown]
        # **Prepare scaled data for recycling code**
        # - Evaporation and moisture fluxes
        # 
        
        # %%
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
        E_total = scaling.evaporation.convert(ds["Evap"].values, UnitSystem.natural, UnitSystem.scaled)
        E_local = scaling.evaporation.convert(ds["Evap_local"].values, UnitSystem.natural, UnitSystem.scaled)
        
        # %% [markdown]
        # **Plot the scaled E array**
        # 
        
        # %% [markdown]
        # **Run recycling model for each timestep**
        # - Create recycling output array based on the shape of one of the surface input files: evap 
        # - Translate evap and fluxes to secondary grid
        # - Calculate modeled precipitation
        # - Plot scaled input variables (evap and fluxes)
        # - Run through each timestep in the input files and calculate recycling ratio at each timestep across domain
        # - Plot rho and convergence metric for each timestep
        
        # %%
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
        #    for k, run_status in status.items():
        #    print(f"Rotation {k * 90} degrees")
        #    print(f"    success = {run_status["success"]}")
        #    print(f"    number of iterations = {run_status["k"]}")
        #    rho = [s["rho"] for s in status.values()]
        #    # compare the solutions pairwise
        #    for i in range(4):
        #        for j in range(i + 1, 4):
        #            diff = np.abs(rho[i] - rho[j]).max()
        #            print(f"Max abs diff between k={i} and k={j}: {diff}")
                
            #Print timestep and status (converged or not) and add rho to recycling ration array
            rot = 2
            print(i,time.values)
            rho_ar[0,:,:,i] = status[0]["rho"]
            rho_ar[1,:,:,i] = status[1]["rho"]
            rho_ar[2,:,:,i] = status[2]["rho"]
            rho_ar[3,:,:,i] = status[3]["rho"]
        
        # %% [markdown]
        # **Create and save rho xarray file**
        # 
        # - Create an xarray to store all of the calculated recycling ratios that is organised in an easy to plot/interpret format
        # - Count number of values in array over 1 - replace all of these with 1
        # - Count number of negative rho values - replace all of these with zero
        # - Save to file
        
        # %%
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
        rho_xarr.to_netcdf(datao+"bands_rho/band_"+B+"_rot_rho_era5_"+str(YR)+".nc")
            
        rho_xarr.close()
            
            
        
        
        