# %% [markdown]
# Extract data from a NCEP NetCDF files

# %%
import numpy as np
import xarray as xr
import scipy
import sys 

# %%
def latlonlev(ds,latmax,latmin,lonmax,lonmin):
    """
    Normalize all the dimensions of datasets to merge into one dataset
    
    Arguments:
        dataset and regional box

    Returns:
       dataset 
    """

    if ds.coords["level"].isnull().all():
        print("Fixing broken pressure level data")
        levels = [1000.0, 950.0, 900.0, 850.0, 700.0, 500.0, 300.0, 200.0]
        ds.coords["level"] = xr.DataArray(levels, dims=["level"], coords={"level": levels}, attrs={"units": "hPa"})
    print(ds.coords['level'])
    
    # make sure lat runs from south to north
    if not ds["lat"].to_index().is_monotonic_increasing:
        print("flipping lat")
        ds = ds.sortby("lat", ascending=True)

    # make sure lon runs from west to east
    if not ds["lon"].to_index().is_monotonic_increasing:
        print("flipping lon")
        ds = ds.sortby("lon", ascending=True)

    # get a spatial subset 
    ds = ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))

    # make sure that the order of the dimensions is (lon, lat, ...) for all variables
    ds = ds.transpose("lon", "lat", "level", "time",missing_dims='ignore')
    ds = ds.isel(time=slice(0,60), drop=True)
    #ds = ds.isel(time=60, drop=True)
    print(ds)
    #sys.exit()
        
    return ds
             
# %%
datao ="/Users/ellendyer/Desktop/" 
datap ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/dao/"
#For selection and plotting

ds = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/dao.80_93.nc")
ds = latlonlev(ds,latmin=-6,latmax=6,lonmin=11,lonmax=31)
ds.to_netcdf(datao+"daods.nc")

# %%
print('prepping datasets near surface for recycling')
import bulk_recycling_model.numerical_integration

# Integrate 10^-3 Shum Uwnd dp
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
integrand = -1 * 1e-3 * ds["Shum"] * ds["Uwnd"]
if not xr.ufuncs.isfinite(ds["Shum"]).all():
    print("Shum not finite")
if not xr.ufuncs.isfinite(ds["Uwnd"]).all():
    print("Uwnd not finite")
if not xr.ufuncs.isfinite(ds["Psfc"]).all():
    print("Psfc not finite")
Fx = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, ds["Psfc"])
# Units: mb x m/s

# Integrate 10^-3 Shum Vwnd dp
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
integrand = -1 * 1e-3 * ds["Shum"] * ds["Vwnd"]
Fy = bulk_recycling_model.numerical_integration.integrate_with_extrapolation(integrand, ds["Psfc"])
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
        max_iter=1000,
        tol=1e-3,
        #callback=Callback(),
    )
    if status["success"]==False:
        # plot the convergence
        deltas = status["deltas"]
        fig, ax = plt.subplots()
        ax.plot(deltas)
        ax.set_title("Convergence")
        ax.set_xlabel("Iteration")
        plt.show()
        #plt.close()
        sys.exit()   
    assert status["success"]
    print(i,time.values)
    rho_ar[:,:,i] = status["rho"]

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
print(rho_xarr)
rho_xarr.to_netcdf(datao+"rho_dao.nc")

sys.exit()
mam_rho = rho_xarr.sel(time=rho_xarr.time.dt.month.isin([3,4,5]))
print(mam_rho)    
fig, ax = plt.subplots()
collection = mam_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.75,levels=6,ax=ax)
fig.suptitle("MAM $\\rho$")
plt.savefig(datap+"rho_MAM"+str(year1)+"_"+str(year2)+".png")
plt.show()
    
son_rho = rho_xarr.sel(time=rho_xarr.time.dt.month.isin([9,10,11]))
print(son_rho)    
fig, ax = plt.subplots()
collection = son_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.75,levels=6,ax=ax)
fig.suptitle("SON $\\rho$")
plt.savefig(datap+"rho_SON"+str(year1)+"_"+str(year2)+".png")
plt.show()
# %%
