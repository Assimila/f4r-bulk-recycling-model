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

dataf ="/Volumes/ESA_F4R/era/" 
datao ="/Volumes/ESA_F4R/ed_prepare/bands_rho/" 
datap ="/Volumes/ESA_F4R/ed_prepare/bands_rho/plots/" 
datas ="/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/Shapefiles/"
shp_cod = gpd.read_file(datas+"congo_basin_evergreen.shp")
band = {'N':[5,12,10,31],'EQ':[-5,5,8,29],'S':[-15,-5,12,31]}
# %%
for B in ['N','EQ','S']:
    print('running band:', B)
    years = np.arange(2009,2010)
    for YR in years:
        print('running year:', YR)
        rho_xarr = xr.open_dataset(datao+"band_"+B+"_rot_rho_era5_"+str(YR)+".nc")
       
        #Filtering out outliers for plotting 
        for r in np.arange(1,5):
            print("*** Rotation is: ",r)
            try:
                print('Number of rhos over 1: ', rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values>1.0).count().values)
                print('Number of negative rhos: ', rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values<0.0).count().values)
                rho_xarr['rho'][r-1,:,:,:] = rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values<=1.0,1.0)
                rho_xarr['rho'][r-1,:,:,:] = rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values>0.0,0.0)
                print('Number of rhos over 1: ', rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values>1.0).count().values)
                print('Number of negative rhos: ', rho_xarr['rho'][r-1,:,:,:].where(rho_xarr['rho'][r-1,:,:,:].values<0.0).count().values)
                print('-------------------------')
            except:
                print('No outliers')
                print(rho_xarr)
        end_all = timer.time()
        length = end_all - start_all
        print("Running the whole prep and recycling code took ", length, "seconds")
        
        # **Plotting**
        # Create seasonal arrays and plot these
        for r in np.arange(1,5):
            rho_plot = rho_xarr['rho'][r-1,:,:,:] 
            mam_rho = rho_plot.sel(time=rho_xarr.time.dt.month.isin([3,4,5]))
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.OCEAN)
            cmap=plt.cm.viridis
            cmap.set_extremes(over='lightgrey')
            collection = mam_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.6,
                                                            levels=13,ax=ax,extend='max',cmap=cmap,
                                                            cbar_kwargs={"location":"bottom"} )
            ax.set_extent([band[B][2]-1,band[B][3]+1,band[B][0]-1,band[B][1]+1])
            ax.set_title(" $\\rho$"+" MAM "+str(YR)+"\n band-"+B+" (rot "+str(r-1)+")")
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='dotted')
            gl.top_labels = False
            gl.right_labels = False
            plt.savefig(datap+"rho_MAM_"+str(YR)+"_"+B+"rot"+str(r-1)+".png")
            #plt.show()
                
            son_rho = rho_plot.sel(time=rho_xarr.time.dt.month.isin([9,10,11]))
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.OCEAN)
            cmap=plt.cm.viridis
            cmap.set_extremes(over='lightgrey')
            collection = son_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.6,
                                                            levels=13,ax=ax,extend='max',cmap=cmap,
                                                            cbar_kwargs={"location":"bottom"} )
            ax.set_extent([band[B][2]-1,band[B][3]+1,band[B][0]-1,band[B][1]+1])
            ax.set_title(" $\\rho$"+" SON "+str(YR)+"\n band-"+B+" (rot "+str(r-1)+")")
            ax.set_title(" SON "+str(YR)+" $\\rho$"+"\n Band-"+B+" (rot "+str(r-1)+")")
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='dotted')
            gl.top_labels = False
            gl.right_labels = False
            plt.savefig(datap+"rho_SON_"+str(YR)+"_"+B+"rot"+str(r-1)+".png")
            #plt.show()
            
            djf_rho = rho_plot.sel(time=rho_xarr.time.dt.month.isin([12,1,2]))
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.OCEAN)
            cmap=plt.cm.viridis
            cmap.set_extremes(over='lightgrey')
            collection = djf_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.6,
                                                            levels=13,ax=ax,extend='max',cmap=cmap,
                                                            cbar_kwargs={"location":"bottom"} )
            ax.set_extent([band[B][2]-1,band[B][3]+1,band[B][0]-1,band[B][1]+1])
            ax.set_title(" $\\rho$"+" DJF "+str(YR)+"\n band-"+B+" (rot "+str(r-1)+")")
            ax.set_title(" DJF "+str(YR)+" $\\rho$"+"\n Band-"+B+" (rot "+str(r-1)+")")
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='dotted')
            gl.top_labels = False
            gl.right_labels = False
            plt.savefig(datap+"rho_DJF_"+str(YR)+"_"+B+"rot"+str(r-1)+".png")
            #plt.show()

            jja_rho = rho_plot.sel(time=rho_xarr.time.dt.month.isin([6,7,8]))
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.RIVERS)
            ax.add_feature(cartopy.feature.OCEAN)
            cmap=plt.cm.viridis
            cmap.set_extremes(over='lightgrey')
            collection = jja_rho.mean("time").plot.contourf(vmin=0.0,vmax=0.6,
                                                            levels=13,ax=ax,extend='max',cmap=cmap,
                                                            cbar_kwargs={"location":"bottom"} )
            ax.set_extent([band[B][2]-1,band[B][3]+1,band[B][0]-1,band[B][1]+1])
            ax.set_title(" $\\rho$"+" JJA "+str(YR)+"\n band-"+B+" (rot "+str(r-1)+")")
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='dotted')
            gl.top_labels = False
            gl.right_labels = False
            plt.savefig(datap+"rho_JJA_"+str(YR)+"_"+B+"rot"+str(r-1)+".png")
            #plt.show()

        rho_xarr.close()
            
            
        
        
        