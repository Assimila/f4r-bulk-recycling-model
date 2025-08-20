import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys

ds = xr.open_dataset('./wpfiles/recycl_ncep_new_500.nc').load()

ds = ds['rec']
#ds = ds.isel(time=0, drop=True)

#print(ds.where((ds.values>1.0) & (ds.values<2.0)).count())
#print(ds.where(ds.values<-1.0).count())

print(ds.max())
ds = ds.where(np.abs(ds.values)<2.0)
print(ds)

#ds.plot.contourf(vmin=0.0,vmax=0.75,levels=6)
#plt.title('timestep')
##plt.savefig('djf_ncep.png')
#plt.show()
#plt.clf()


ds = ds.groupby('time.season').mean('time')

ds.sel(season='DJF').plot.contourf(vmin=0.0,vmax=0.03,levels=6)
plt.title('DJF NCEP')
#plt.savefig('djf_ncep.png')
plt.show()
plt.clf()

#ds.sel(season='MAM').plot.contourf()
#plt.title('MAM NCEP')
##plt.savefig('mam_ncep.png')
#plt.show()
#plt.clf()

fig, ax = plt.subplots()
collection = ds.sel(season='MAM').plot.contourf(vmin=0.0,vmax=0.75,levels=12,ax=ax)
fig.suptitle("MAM NCEP")
plt.show()

ds.sel(season='JJA').plot.contourf()
plt.title('JJA NCEP')
#plt.savefig('jja_ncep.png')
plt.show()
plt.clf()

ds.sel(season='SON').plot.contourf()
plt.title('SON NCEP')
#plt.savefig('son_ncep.png')
plt.show()
plt.clf()
