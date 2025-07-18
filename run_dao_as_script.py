# %% [markdown]
# Extract data from a DAO NetCDF file

# %%
import numpy as np
import xarray as xr
import sys

# %%
ds = xr.open_dataset("/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/dao.80_93.nc")

# %%
if ds.coords["level"].isnull().all():
    print("Fixing broken pressure level data")
    levels = [1000.0, 950.0, 900.0, 850.0, 700.0, 500.0, 300.0, 200.0]
    ds.coords["level"] = xr.DataArray(levels, dims=["level"], coords={"level": levels}, attrs={"units": "hPa"})
print(ds.coords['level'])

# %%
# make sure lat runs from south to north
if not ds["lat"].to_index().is_monotonic_increasing:
    print("flipping lat")
    ds = ds.sortby("lat", ascending=True)

# %%
# make sure lon runs from west to east
if not ds["lon"].to_index().is_monotonic_increasing:
    print("flipping lon")
    ds = ds.sortby("lon", ascending=True)

# %%
# get a spatial subset 4°N–16°S, 50°–76°W
ds = ds.sel(lat=slice(-16, 4), lon=slice(-76, -50))

# %%
# make sure that the order of the dimensions is (lon, lat, ...) for all variables
ds = ds.transpose("lon", "lat", ...)

# %%
# grab the first time step
# should be Jan 1980
ds = ds.isel(time=0, drop=True)

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

# %%
# Integrate 10^-3 Shum Uwnd dp
# The input dataset has NaNs where pressure levels correspond to heights below ground level.
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
da = -1 * 1e-3 * ds["Shum"] * ds["Uwnd"]
Fx = da.reduce(integrator, dim="level", x=ds.coords["level"].values)
# Units: mb x m/s

# %%
# Integrate 10^-3 Shum Vwnd dp
# The input dataset has NaNs where pressure levels correspond to heights below ground level.
# Because the integration limits are from high pressure to low pressure, we need to invert the sign.
da = -1 * 1e-3 * ds["Shum"] * ds["Vwnd"]
Fy = da.reduce(integrator, dim="level", x=ds.coords["level"].values)
# Units: mb x m/s

# %% [markdown]
# Prepare and scale the data

# %%
from bulk_recycling_model import preprocess
from bulk_recycling_model.axis import Axis
from bulk_recycling_model.scaling import Scaling, UnitSystem

# %% [markdown]
# 

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
# preprocess water vapor fluxes onto the secondary grid
Fx_left = preprocess.prepare_Fx_left(Fx)
Fx_right = preprocess.prepare_Fx_right(Fx)
Fy_bottom = preprocess.prepare_Fy_bottom(Fy)
Fy_top = preprocess.prepare_Fy_top(Fy)

# %%
# convert E to scaled units
E = scaling.evaporation.convert(ds["Evap"].values, UnitSystem.natural, UnitSystem.scaled)

# %%
# preprocess E onto the secondary grid
E = preprocess.prepare_E(E)

# %%
# compute P
P = preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy)

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
        plt.savefig(f"/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/plots/dao/{self.n_calls:05d}_{k:04d}.png")
        plt.close(fig)
        self.n_calls += 1

# %%
#! rm plots/*.png

# %%
status = run(
    Fx_left,
    Fx_right,
    Fy_bottom,
    Fy_top,
    E,
    P,
    dx,
    dy,
    tol=1e-2,
    callback=Callback(),
)
assert status["success"]

# %% [markdown]
# Animate the plots using ffmpeg
# 
# ```bash
# ffmpeg -framerate 30 -pattern_type glob -i "*.png" -vf "tpad=stop_mode=clone:stop_duration=3" -c:v libx264 -pix_fmt yuv420p mov.mp4
# ```

# %%
# plot the final solution
rho = status["rho"]
fig, ax = plt.subplots()
collection = plotting.pcolormesh(ax, rho, lon_axis, lat_axis, vmin=0.0, vmax=0.8)
fig.colorbar(collection)
fig.suptitle("$\\rho$")

# %%
# plot the convergence
deltas = status["deltas"]
fig, ax = plt.subplots()
ax.plot(deltas)
ax.set_title("Convergence")
ax.set_xlabel("Iteration")

# %%



