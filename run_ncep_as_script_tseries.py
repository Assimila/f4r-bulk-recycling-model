# %% [markdown]
# Extract data from a NCEP NetCDF files

# %%
import numpy as np
import xarray as xr
import scipy
import sys 

# %%
def latlonlev(dataset_list,year1,year2,latmax,latmin,lonmax,lonmin,pmin,pmax,deg):
    """
    Normalize all the dimensions of datasets to merge into one dataset
    
    Arguments:
        dataset_list: a list of datasets with one variable each

    Returns:
       list of datasets 
    """
    dataset_out = []
    for d in dataset_list:

        d = d.sel(time=slice(str(year1)+"-01-01", str(year2)+"-12-31"),drop=True)

        # make sure lat runs from south to north
        if not d["lat"].to_index().is_monotonic_increasing:
            print(d.data_vars," flipping lat")
            d = d.sortby("lat", ascending=True)
        d = d.sel(lat=slice(latmin, latmax))
        d = d.interp(lat=np.arange(latmin, latmax, deg),method='linear',kwargs={"fill_value": "extrapolate"})
        
        # make sure lon runs from west to east
        if not d["lon"].to_index().is_monotonic_increasing:
            print(d.data_vars," flipping lon")
            d = d.sortby("lon", ascending=True)
        # Convert from 0-360 longitudes
        if d["lon"].max() > 185.:
           print(d.data_vars,"  shifting lon")
           d['lon'] = d['lon']-180.0
        d = d.sel(lon=slice(lonmin, lonmax))
        d = d.interp(lon=np.arange(lonmin, lonmax, deg),method='linear',kwargs={"fill_value": "extrapolate"})

        try:
            d = d.sortby("level", ascending=False)
            print(d.data_vars,"  sorting pressure levels")
            d = d.sel(level=slice(pmax, pmin))
        except:
            pass

        # make sure that the order of the dimensions is (lon, lat, ...) for all variables
        d = d.transpose("lon", "lat", "level", "time",missing_dims='ignore')
        
        dataset_out.append(d)
    print("dataset_out", dataset_out)
        
    return dataset_out
             
          

# %%
ds_prate = (86400.0)*xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/prate.mon.mean.nc")
ds_pres = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/pres.mon.mean.nc")
ds_lhtfl = (86400/2.5e6)*xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/lhtfl.mon.mean.nc")
ds_shum = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/shum.mon.mean.nc")
ds_uwnd  = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/uwnd.mon.mean.nc")
ds_vwnd = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/vwnd.mon.mean.nc")
ds_list_in = [ds_prate.rename({"prate": "Prec"}), ds_pres.rename({"pres": "Psfc"}), ds_lhtfl.rename({"lhtfl": "Evap"}), 
               ds_shum.rename({"shum": "Shum"}), ds_uwnd.rename({"uwnd": "Uwnd"}), ds_vwnd.rename({"vwnd": "Vwnd"})] 
ds_list_out = latlonlev(ds_list_in,year1=1968,year2=2000,latmin=-10,latmax=10,lonmin=11,lonmax=31,pmax=1050,pmin=300,deg=2.5)
ds = xr.merge(ds_list_out) 
print('datasets merged')

# %%
#ds = ds.sel(lat=slice(-12, 5), lon=slice(15, 25))
# %%
# grab the first time step
#ds = ds.sel(time='1990-04-01', drop=True)
ds = ds.sel(time=slice('1968-01-01','2001-12-31'), drop=True)
print(ds)

# %%
def nan_trapz(a: np.ndarray, x: np.ndarray):
    mask = ~np.isnan(a)  # real values
    return np.trapezoid(y=a[mask], x=x[mask])


def integrator(a: np.ndarray, axis: int, x: np.ndarray) -> np.ndarray:
    """
    Apply the trapezium rule for 1D integration, dropping NaNs.

    Integrate y dx,
    where y is the dependent variable (given by an axis of a),
    and x is the independent variable (sample_points).

    Arguments:
        a: ND array to integrate
        axis: axis of a to integrate over
        x: 1D array of sample points

    Returns:
        N-1 dimensional array
    """
    return np.apply_along_axis(
        func1d=nan_trapz,
        axis=axis,
        arr=a,
        x=x,
    )

def surf_ext(ps,data):
    print(ps)
    print(data.level.max())

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
print('pre-scaled',ds['Evap'])
E = scaling.evaporation.convert(ds["Evap"].values, UnitSystem.natural, UnitSystem.scaled)
print('scaled',E)
print('Fx',Fx)

# Entering preprocessing and time step loop
for i,time in enumerate(ds.time):
    print(i,time.values)    
     
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
    print("run model and plot")
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
            plt.savefig(f"/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/ncep/{self.n_calls:05d}_{k:04d}.png")
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
        tol=1e-3,
        #callback=Callback(),
    )
    assert status["success"]
    print(time.values)

    # %%
    # plot each timestep 
    
    rho = status["rho"]
    print(rho)
    print(lat_axis.min)
    print(lat_axis)
    print(lon_axis.min)
    print(lon_axis)
    sys.exit()
    fig, ax = plt.subplots()
    collection = plotting.pcolormesh(ax, rho, lon_axis, lat_axis, vmin=0.0, vmax=0.8)
    fig.colorbar(collection)
    fig.suptitle(str(time.values)+" $\\rho$")
    plt.savefig(f"/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/ncep/rho_"+str(time.values)+".png")
    #plt.show()
    
#    # %%
#    # plot the convergence
#    deltas = status["deltas"]
#    fig, ax = plt.subplots()
#    ax.plot(deltas)
#    ax.set_title("Convergence")
#    ax.set_xlabel("Iteration")
#    plt.show()
#    # %%
    
    
    
    