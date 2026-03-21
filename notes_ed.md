### January 19th 2026

- noted that the issue might be that we are using ERA5Land data and ERA5 pressure level data which aren't connected and are actually on a different grid -- re-running a few codes to see if using the surface data from ERA5 rather than ERA5Land makes a difference in the equatorial band run

- data in **1994 and 1995** are a bit broken - it has the wrong number of timesteps but all other years seem to be ok so will just remove those from the runs for now until can re-download

- code for alternative surface data is:

> merge_monthly_prepare_era5_land_and_surface_e.ipynb

> merge_monthly_prepare_era5_surface_e_only.ipynb

- new code from this year is in 2026 - won't use anything from pre-2026 unless made an updated and checked copy which will be in 2026


***currently testing all the code for 2010:***

- EQ and S bands are totally fixed by using surface evap and psfc rather than ERA5Land evap and psfc
- the N band still has some months (July and August) which don't converge for two rotations (1 and 4) but do converge for two rotations (2 and 3)

- **still can't get the hourly integration code to run which is odd because it ran in the past**

- am trying to break up the code into month chunks to loop through (fortnight for levels and monthly for pressure) in current notebook:

> merge_monthly_prepare_era5_levels_hourly_int.ipynb

**this works now - looping by month and then concatenating a full year**

### January 20th 2026

- managed to get hourly into to work 

- have now tested all the combinations and most of the problems are solved > it seems that EQ and S problems are really helped by the proper local/total evap masking and the N problems are solved by using surface variables rather than land variables

- **today need to run the Congo Basin and DRC runs to share with Wilfried**

- done DRC
- done CB

- **Need to run all years for the input**

> need to note in future if there are values over 1 or if there are many under 0 values - currently it looks like there are only negative values rather than large values...


### January 21st 2026

- running L_M S_SE recycling and **2016** said it was not finite

### January 22nd 2026

- will need to try to download **1994, 1995, 2011, 2016, 2020** to do the full timeseries (and 2021 for the L_HI)
- for now will proceed with multiplot

### January 28th 2026

- Have run all the bands for rho (still haven't redownloaded the files with issues)
- Multiplot is taking way too long - may just need to do it manually?
- Running N band for S_SE and L_HI to test whether there are less parts that don't converge >> finished this and now just need to get multiplot to work..

### 19th Feb 2026

- 1994 and 1995 are sorted
- 2011, 2016, and 2020 do need to be redownloaded still (tested with merge L_HI)

### 21st Feb 2026

- 2011 done
- downloading 2016

### 22 Feb 2026

## Running:
1990-2015 of:
> Bands - L_M S_SE R=0.2 iter=1000

> DRC - L_M S_SE R=0.2 iter=1000 

> CB - L_M S_SE R=0.2 iter=1000 

### 23 Feb 2026

2016 still has integrated values that are not finite

### 21 March 2026 (with Wilfried)

testing for EQ band

1990 bands
- 04, 09

1990 lowres
- 08, 09

1990 git lowres
- 02, 04, 07, 08, 09