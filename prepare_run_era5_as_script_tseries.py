# %% [markdown]
# Extract data from a ERA5 NetCDF files

# %%
import numpy as np
import xarray as xr
import pandas as pd
import scipy
import sys 
import warnings
warnings.filterwarnings('ignore')
import time as timer
start_all = timer.time()

# %%
dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/" 
#datao ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/era/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/era/"

#%%
#For selection and plotting
Y1, Y2 = 1995,1995
#m1 = '03'
time_bnds = (str(Y1)+'-01-01',str(Y2)+'-12-31')
lon_bnds, lat_bnds = (15, 30), (5,-10)
#p_bnds = (1000,300) #for daily folder files
p_bnds = (30000,100000)

# %%
start = timer.time()
from functools import partial
def _preprocess_pres(x, lon_bnds, lat_bnds, p_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds),
                 plev=slice(*p_bnds),drop=True)
partial_func_pres = partial(_preprocess_pres, lon_bnds=lon_bnds, lat_bnds=lat_bnds, p_bnds=p_bnds)

#Reading in pressure level variables from ERA5
ds_era_pres = xr.open_mfdataset(dataf+"era5/pressure_levels/era5_pressure_level_variables_central_africa_"+str(Y1)+"*.nc",
                                drop_variables=['r','t','w'],
                                preprocess=partial_func_pres,parallel=True).resample(time='D').mean(dim='time').load()
print('opened press files', ds_era_pres['plev'])
ds_era_pres = ds_era_pres.sel(time=slice(*time_bnds),drop=True)
ds_era_pres = ds_era_pres.rename({'plev':'level','q':'Shum','u':'Uwnd','v':'Vwnd'})
ds_era_pres['Shum'] = 1000.0*ds_era_pres['Shum']
ds_era_pres['level'] = ds_era_pres['level']/100.0 #for half monthly files only 
ds_era_pres = ds_era_pres.sortby('level', ascending=False) #for half monthly files only
print('DONE read in pres',ds_era_pres)
end = timer.time()
length = end - start
print("It took", length, "seconds!")
print('---------------------------------------------------------------------')

#---------------------------------------------------
# %%

start = timer.time()
from functools import partial
def _preprocess_land(x, lon_bnds, lat_bnds):
    x = x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds),drop=True)
    return x
partial_func_land = partial(_preprocess_land, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

#Reading in surface variables from ERA5 Land
ds_era_land = xr.open_mfdataset("/Volumes/ESA_F4R/era/era5_land/era5_land_variables_central_africa_"+str(Y1)+"*.nc",
                                drop_variables=['expver','number','pev','ssr','t2m'],
                                preprocess=partial_func_land,parallel=True).load()
ds_era_land = ds_era_land.sel(valid_time=slice(*time_bnds),drop=True)
ds_era_land = ds_era_land.rename({'valid_time':'time','latitude':'lat',
                                  'longitude':'lon','tp':'Prec','e':'Evap','sp':'Psfc'})
ds_era_land = ds_era_land.interp(lat=ds_era_pres['lat'],lon=ds_era_pres['lon'],method='linear',kwargs={"fill_value": "extrapolate"})
print('era after int before resample',ds_era_land)
Psfc = ds_era_land['Psfc'].resample(time='D').mean(dim='time')/100.0
print('*********PSFC',Psfc)
ds_era_land = ds_era_land.resample(time='D').sum(dim='time')
ds_era_land['Psfc'] = Psfc
ds_era_land['Prec'] = ds_era_land['Prec']*1000.0 
ds_era_land['Evap'] = ds_era_land['Evap']*-1000.0 
print('DONE read in land',ds_era_land)
end = timer.time()
length = end - start
print("It took", length, "seconds!")
print('---------------------------------------------------------------------')

#---------------------------------------------------
# %%
start = timer.time()
ds = xr.merge([ds_era_pres,ds_era_land]) 
ds_era_pres.close()
ds_era_land.close()
ds = ds.sortby('lat', ascending=True)
ds = ds.transpose("lon", "lat", "level", "time",missing_dims='ignore')
print('Done data read in', ds)
ds = ds.resample(time='MS').mean(dim='time')
ds.to_netcdf(datao+"erads.nc", mode='w', format='NETCDF4', engine='netcdf4')
end = timer.time()
length = end - start
print("It took", length, "seconds!")

# %%
print('prepping datasets near surface for recycling')
import bulk_recycling_model.numerical_integration

# Integrate 10^-3 Shum Uwnd dp
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
integrand = -1 * 1e-3 * ds["Shum"] * ds["Uwnd"]
Fx = bulk_recycling_model.numerical_integration.integrate_no_extrapolation(integrand, ds["Psfc"])
# Units: mb x m/s

# Integrate 10^-3 Shum Vwnd dp
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
integrand = -1 * 1e-3 * ds["Shum"] * ds["Vwnd"]
Fy = bulk_recycling_model.numerical_integration.integrate_no_extrapolation(integrand, ds["Psfc"])
# Units: mb x m/s

# %% [markdown]
# Prepare and scale the data

# %%
from bulk_recycling_model import preprocess
from bulk_recycling_model.axis import Axis
from bulk_recycling_model.scaling import Scaling, UnitSystem

print("prepping data for recycling - scaling and flux calcs etc")

# %%
# degrees
L = ds.coords["lon"].max().item() - ds.coords["lon"].min().item()
# convert to meters
L = L * 111e3 * np.cos(np.deg2rad(ds.coords["lat"].mean().item()))
dx = L / ds.sizes["lon"]

# %%
# lon axis
lon_axis = Axis(
    ds.coords["lon"].min().item(),
    ds.coords["lon"].diff("lon").mean().item(),
    ds.sizes["lon"],
)

# %%
# degrees
H = ds.coords["lat"].values[-1] - ds.coords["lat"].values[0]
# convert to meters
H = H * 111e3
dy = H / ds.sizes["lat"]

# %%
# lat axis
lat_axis = Axis(
    ds.coords["lat"].min().item(),
    ds.coords["lat"].diff("lat").mean().item(),
    ds.sizes["lat"],
)

# %%
print(f"{L = :.2e} m")
print(f"{dx = :.2e} m")
print(f"{H = :.2e} m")
print(f"{dy = :.2e} m")

# %%
# make a scaling object to convert between unit systems
scaling = Scaling(H)

# %%
dx = scaling.distance.convert(dx, UnitSystem.SI, UnitSystem.scaled)
dy = scaling.distance.convert(dy, UnitSystem.SI, UnitSystem.scaled)
print(f"{dx = :.2e} scaled")
print(f"{dy = :.2e} scaled")

# %%
# convert Fx and Fy to scaled units
Fx = scaling.water_vapor_flux.convert(Fx.values, UnitSystem.natural, UnitSystem.scaled)
Fy = scaling.water_vapor_flux.convert(Fy.values, UnitSystem.natural, UnitSystem.scaled)

# %%
# convert E to scaled units
#print('pre-scaled',ds['Evap'])
E = scaling.evaporation.convert(ds["Evap"].values, UnitSystem.natural, UnitSystem.scaled)
#print('scaled',E)
#print('Fx',Fx)

rho_ar = np.empty((np.shape(E)[0]-1,np.shape(E)[1]-1,np.shape(E)[2]))
# Entering preprocessing and time step loop
print("run model and plot")
for i,time in enumerate(ds.time):
     
    # %%
    # preprocess E onto the secondary grid
    Ei = preprocess.prepare_E(E[:,:,i])
    
    # %%
    # preprocess water vapor fluxes onto the secondary grid
    Fxi_left = preprocess.prepare_Fx_left(Fx[:,:,i])
    Fxi_right = preprocess.prepare_Fx_right(Fx[:,:,i])
    Fyi_bottom = preprocess.prepare_Fy_bottom(Fy[:,:,i])
    Fyi_top = preprocess.prepare_Fy_top(Fy[:,:,i])
    #print(Fxi_left)
    #print(Fxi_right)
    
    # %%
    # compute P
    Pi = preprocess.calculate_precipitation(Fxi_left, Fxi_right, Fyi_bottom, Fyi_top, Ei, dx, dy)
    
    # %% [markdown]
    # Run the model
    
    # %%
    import matplotlib.pyplot as plt
    
    # %%
    import logging
    
    logging.basicConfig()
    logging.getLogger("bulk_recycling_model").setLevel(logging.INFO)
    
    # %%
    from bulk_recycling_model import plotting
    #from bulk_recycling_model.main_ncltest import run
    from bulk_recycling_model.main import run
    
    # %%
    class Callback:
        def __init__(self):
            self.n_calls = 0  # number of times callback has executed
    
        def __call__(self, rho: np.ndarray, k: int):
            fig, ax = plt.subplots()
            collection = plotting.pcolormesh(ax, rho, lon_axis, lat_axis, vmin=0.0, vmax=0.8)
            fig.colorbar(collection)
            fig.suptitle(f"$\\rho^k$ @ k={k:04d}")
            plt.savefig(datap+"{self.n_calls:05d}_{k:04d}.png")
            plt.close(fig)
            self.n_calls += 1
    
    # %%
    #! rm plots/*.png
    
    # %%
    status = run(
        Fxi_left,
        Fxi_right,
        Fyi_bottom,
        Fyi_top,
        Ei,
        Pi,
        dx,
        dy,
        max_iter=500,
        tol=1e-3,
        #callback=Callback(),
    )
#    if status["success"]==False:
#        print("Failed convergence")
#        print(i,time.values)
#        print("failed rho: ",status["rho"])
#        # plot the convergence
#        deltas = status["deltas"]
#        fig, ax = plt.subplots()
#        ax.plot(deltas)
#        ax.set_title("Convergence")
#        ax.set_xlabel("Iteration")
#        plt.show()
#        #plt.close()
#        sys.exit()   
#    assert status["success"]
    print(i,time.values)
    print(status['k'])
    rho_ar[:,:,i] = status["rho"]
    #print("rho: ",status["rho"])

    # %%
    # plot each timestep 
    fig, ax = plt.subplots()
    collection = plotting.pcolormesh(ax, status["rho"], lon_axis, lat_axis, vmin=0.0, vmax=0.8)
    fig.colorbar(collection)
    fig.suptitle(str(time.values)+" $\\rho$")
    #plt.savefig(datap+"rho_"+str(time.values)+".png")
    #plt.show()
    plt.close()

    
    # %%
    # plot the convergence
    deltas = status["deltas"]
    fig, ax = plt.subplots()
    ax.plot(deltas)
    ax.set_title("Convergence")
    ax.set_xlabel("Iteration")
    #plt.show()
    plt.close()
# %%
lon_ar = np.linspace(start=ds.coords["lon"].min().values+lon_axis.step/2,
                     stop=ds.coords["lon"].max().values-lon_axis.step/2,
                     num=lon_axis.n_points-1)
lat_ar = np.linspace(start=ds.coords["lat"].min().values+lat_axis.step/2,
                     stop=ds.coords["lat"].max().values-lat_axis.step/2,
                     num=lat_axis.n_points-1)
rho_xarr = xr.DataArray(
    data=rho_ar,
    dims=["lon", "lat", "time"],
    coords=dict(
        lon=(["lon"], lon_ar),
        lat=(["lat"], lat_ar),
        time=(["time"],ds.time.data)
    ),
    attrs=dict(
        description="Recycling ratio",
        units="%",
    ),
) 
rho_xarr = rho_xarr.transpose("time","lat","lon")
#print(rho_xarr.max())
#print(rho_xarr.min())
#print(rho_xarr.where(rho_xarr.values>1.0).count())
rho_xarr.to_netcdf(datao+"rho_era5.nc")

mam_rho = rho_xarr.sel(time=rho_xarr.time.dt.month.isin([3,4,5]))
print(mam_rho)    

end_all = timer.time()
length = end_all - start_all
print("It took", length, "seconds!")

fig, ax = plt.subplots()
collection = mam_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.75,levels=12,ax=ax)
fig.suptitle("MAM $\\rho$")
#plt.savefig(datap+"rho_MAM"+str(year1)+"_"+str(year2)+".png")
plt.show()
    
#son_rho = rho_xarr.sel(time=rho_xarr.time.dt.month.isin([9,10,11]))
#print(son_rho)    
#fig, ax = plt.subplots()
#collection = son_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.75,levels=12,ax=ax)
#fig.suptitle("SON $\\rho$")
##plt.savefig(datap+"rho_SON"+str(year1)+"_"+str(year2)+".png")
#plt.show()
#
#jja_rho = rho_xarr.sel(time=rho_xarr.time.dt.month.isin([6,7,8]))
#print(jja_rho)    
#fig, ax = plt.subplots()
#collection = jja_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.75,levels=12,ax=ax)
#fig.suptitle("JJA $\\rho$")
##plt.savefig(datap+"rho_JJA"+str(year1)+"_"+str(year2)+".png")
#plt.show()
## %%