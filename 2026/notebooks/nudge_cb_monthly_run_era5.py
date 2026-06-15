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

S_NAME = "S_SE" # S_SE or S_LSE 
L_NAME = "L_M" # L_M or L_HI

dataf ="/Volumes/blue_wd/ESA_F4R/2026_mergeds/" 
datao ="/Volumes/blue_wd/ESA_F4R/2026_rho/cb/" 
datap ="/Volumes/blue_wd/ESA_F4R/2026_plots/cb/" 
datas ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/Shapefiles/"
shp_cod = gpd.read_file(datas+"congo_basin_evergreen.shp")

max_iter = 1000
tol = 1e-3

# ### Read in surface and pressure level files
# **Surface file options:**
# - S_SE (surface vars including only surface evaporation)
# - S_LSE (surface vars including land and surface evaporation)
# 
# **Pressure level file options:**
# - L_M (pressure level vars resampled to monthly timestep)
# - L_HI (pressure level vars with integrated moist flux calcualted hourly and then resampled to monthly timestep)

years = [1990, 1991, 1992, 1993, 1994,1995,
         1996, 1997, 1998, 1999, 2000, 2001, 2002, 
         2003, 2004, 2005, 2006, 2007, 2008, 2009, 
         2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
         2020,2021, 2022, 2023, 2024]

for YR in years:
    print("Year: ", YR)
    
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
    ds['Evap_cb'] = ds['Evap_all'].rio.clip(shp_cod.geometry.apply(mapping),shp_cod.crs,drop=False)
    ds = ds.rename({'x': 'lon','y': 'lat'})
    ds['Evap_cb'] = ds['Evap_cb'].fillna(0.0)
    
    #Points that are 0.0 are inside the region, points on the boundary will be
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
    
    # make a scaling object to convert between unit systems
    scaling = Scaling(H)
    
    dx = scaling.distance.convert(dx, UnitSystem.SI, UnitSystem.scaled)
    dy = scaling.distance.convert(dy, UnitSystem.SI, UnitSystem.scaled)
    
    # convert Fx and Fy to scaled units
    Fx = scaling.water_vapor_flux.convert(Fx.values, UnitSystem.natural, UnitSystem.scaled)
    Fy = scaling.water_vapor_flux.convert(Fy.values, UnitSystem.natural, UnitSystem.scaled)
    
    # convert E to scaled units
    # Do this for both the total E and the local regionally clipped E]
    E_total = scaling.evaporation.convert(ds["Evap_all"].values, UnitSystem.natural, UnitSystem.scaled)
    E_local = scaling.evaporation.convert(ds["Evap_cb"].values, UnitSystem.natural, UnitSystem.scaled)
    
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
    #logging.getLogger("bulk_recycling_model").setLevel(logging.INFO)
    logging.disable(logging.CRITICAL)
    from bulk_recycling_model import plotting
    from bulk_recycling_model.main import run_4_orientations, run_rotated
    from bulk_recycling_model import coefficients
    
    #Make the rho array the same shape as the total E - will clip the external points at the end
    rho_ar = np.empty((4,np.shape(E_total)[0]-1,np.shape(E_total)[1]-1,np.shape(E_total)[2]))
    #Entering preprocessing and time step loop
    #Run model and plot
    count_success_pre_nudge = 0
    count_success_post_nudge = 0
    count_fail_pre_nudge = 0
    count_fail_post_nudge = 0
    too_many_pixels = 0
    no_hot_pixel = 0
    for i,time in enumerate(ds.time):
         
        # preprocess E onto the secondary grid
        Ei_total = preprocess.prepare_E(E_total[:,:,i])
        Ei_local = preprocess.prepare_E(E_local[:,:,i])
        
        #print('post-processed land min',Ei_local.min())
        #print('post-processed total min',Ei_total.min())
        
        # preprocess water vapor fluxes onto the secondary grid
        Fxi_left = preprocess.prepare_Fx_left(Fx[:,:,i])
        #print("***Fxi_left**")
        
        
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
            max_iter=max_iter,
            tol=tol,
        )
        #Print timestep and status (converged or not) and add rho to recycling ration array
        print(i,time.values)
        for ROT in np.arange(0,4):
            print(ROT,status[ROT]['success'],'**pre-nudge')    
            if status[ROT]['success']==True:
                rho_ar[ROT,:,:,i] = status[ROT]["rho"]
                count_success_pre_nudge = count_success_pre_nudge + 1
            else:
                rho_ar[ROT,:,:,i] = np.nan
                count_fail_pre_nudge = count_fail_pre_nudge + 1
                
                #----Pre-nudge------------------------
                
                coeffs = coefficients.Coefficients(Fxi_left, Fxi_right, Fyi_bottom, Fyi_top, Ei_local, Pi, dx,dy)
                instability_heuristic = coeffs.rotated_instability_heuristic(k=ROT)
    
                #identify hot pixel
                from bulk_recycling_model.numerical_stability_ED import identify_hot_pixel, identify_hot_pixel_thresh, nudge_hot_pixel
                #printing example of the hottest pixels
                p = 4
                s_p = 1
                thresh=1.75
                s_thresh = 1.2
                #tune these nudging values as needed
                offset = 2.0
                kernel_size = 15
                #hot_ind = identify_hot_pixel(p,coeffs.rotated_instability_heuristic(k=ROT))
                hot_ind = identify_hot_pixel_thresh(thresh,coeffs.rotated_instability_heuristic(k=ROT))
                if len(hot_ind)>p:
                    hot_ind=hot_ind
                if len(hot_ind)==0:
                    hot_ind = identify_hot_pixel_thresh(s_thresh,coeffs.rotated_instability_heuristic(k=ROT)) 
                    if len(hot_ind)>0:
                        hot_ind = identify_hot_pixel(s_p,coeffs.rotated_instability_heuristic(k=ROT)) 
                print('*There are ',len(hot_ind),' hot pixels*')
                if len(hot_ind)>0 and len(hot_ind)<=p:
                    i_hot, j_hot = np.unravel_index(np.argmax(instability_heuristic, axis=None), instability_heuristic.shape)
                    print(
                        f"Hottest pixel identified at (i={i_hot}, j={j_hot}) "
                        f"with instability heuristic = {coeffs.rotated_instability_heuristic(k=ROT)[i_hot, j_hot]}"
                    )
                    print("E =", Ei_local[i_hot, j_hot])
                    print("P =", Pi[i_hot, j_hot])
                    
                    #nudge hot pixel
                    #print('post-processed land min',Ei_local.min())
                    #print('post-processed total min',Ei_total.min())
                    E_local_nudged = nudge_hot_pixel(Ei_local, hot_ind, offset=offset, kernel_size=kernel_size)
                    E_total_nudged = nudge_hot_pixel(Ei_total, hot_ind, offset=offset, kernel_size=kernel_size)
                    P_nudged = preprocess.calculate_precipitation(Fxi_left, Fxi_right, Fyi_bottom, Fyi_top, E_total_nudged, dx, dy)
                    coeffs_nudged = coefficients.Coefficients(Fxi_left, Fxi_right, Fyi_bottom, Fyi_top, E_local_nudged, P_nudged, dx, dy)
                    
                    #----Post-nudge------------------------
                    
                    print(
                        f"Hottest pixel post-nudge at (i={i_hot}, j={j_hot}) "
                        f"with instability heuristic = {coeffs_nudged.rotated_instability_heuristic(k=ROT)[i_hot, j_hot]}"
                        )
                    print("E =", E_local_nudged[i_hot, j_hot])
                    print("P =", P_nudged[i_hot, j_hot])
        
                    # Run the model again
                    status_or = run_rotated(
                        Fxi_left,
                        Fxi_right,
                        Fyi_bottom,
                        Fyi_top,
                        E_local_nudged,
                        P_nudged,
                        dx,
                        dy,
                        R=0.2,
                        R_1=0.2,
                        max_iter=max_iter,
                        tol=tol,
                        rotation=ROT,
                    )
                    #Print timestep and status (converged or not) and add rho to recycling ration array
                    print(i,time.values)
                    print(ROT,status_or['success'],'**post-nudge')    
                    if status_or['success']==True:
                        rho_ar[ROT,:,:,i] = status_or["rho"]
                        count_success_post_nudge = count_success_post_nudge + 1
                    else:
                        rho_ar[ROT,:,:,i] = np.nan 
                        count_fail_post_nudge = count_fail_post_nudge + 1
                    
                        lon_ar = np.linspace(start=ds.coords["lon"].min().values+lon_axis.step/2,
                                 stop=ds.coords["lon"].max().values-lon_axis.step/2,
                                 num=lon_axis.n_points-1)
                        lat_ar = np.linspace(start=ds.coords["lat"].min().values+lat_axis.step/2,
                                             stop=ds.coords["lat"].max().values-lat_axis.step/2,
                                             num=lat_axis.n_points-1)
                        rho_xarr = xr.Dataset(
                            data_vars=dict(rho=(["lon","lat"],status_or["rho"])),
                            coords=dict(
                                lon=(["lon"], lon_ar),
                                lat=(["lat"], lat_ar)
                            ),
                            attrs=dict(
                                description="Recycling ratio",
                                units="%",
                            ),
                        ) 
                        rho_xarr = rho_xarr.transpose("lat","lon")
                        
                else:
                    print('No hot pixels above threshold or there are too many hot pixels')
                    if len(hot_ind)==0: 
                      no_hot_pixel = no_hot_pixel + 1 
                    if len(hot_ind)>p:
                      too_many_pixels = too_many_pixels + 1  
                        
    print(' ***********Year***: ',YR)  
    print('count_success_pre_nudge: ', count_success_pre_nudge)
    print('count_fail_pre_nudge: ', count_fail_pre_nudge)
    print('count_success_post_nudge: ', count_success_post_nudge)
    print('count_fail_post_nudge: ', count_fail_post_nudge)
    print('count_fail_no_hot_pixel: ', no_hot_pixel)
    print('count_fail_too_many_pixels: ', too_many_pixels)
                        
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
    rho_xarr.rio.write_crs("epsg:4326", inplace=True)
    rho_xarr = rho_xarr.rename({'lon': 'x','lat': 'y'})
    rho_xarr = rho_xarr.rio.clip(shp_cod.geometry.apply(mapping),shp_cod.crs,drop=False)    
    rho_xarr = rho_xarr.rename({'x': 'lon','y': 'lat'})
                                                                                            
    rho_xarr.to_netcdf(datao+L_NAME+"_"+S_NAME+"_cb_rot_rho_era5_"+str(YR)+".nc",engine=    "scipy")
                                                                                            
    rho_xarr.close()                                                                        
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            
                                                                                            