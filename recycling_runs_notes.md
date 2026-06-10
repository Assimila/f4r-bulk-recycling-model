# Notes on recycling runs
Last edited on: 8 June 2026



### Negative Precipitation

>1990-01-01 ERA5 precipitation

![alt text](readme_figures/1990-01-01-ERA5.png)

**the red points in this plot are where precip is less than 0.001 mm/day

>1990-01-01 S_SE L_M (no spatial regridding)

![alt text](readme_figures/1990-01-01_S_SE_L_M_noregrid.png)

**while there are more negative points in L_M rainfall there is also more of a positive bias in rainfall south of these negative areas

- Running with hourly integrated and then monthly averaged moisture flux reduces the spatial area of negative rainfall

>1990-01-01 S_SE L_HI (no spatial regridding)

![alt text](readme_figures/1990-01-01_S_SE_L_HI_noregrid.png)

>1990-01-01 S_SE L_HI (linear regridding to 0.5 degrees)

![alt text](readme_figures/1990-01-01_S_SE_L_HI_0.5deg_regrid.png)

>1990-01-01 S_SE L_HI (linear regridding to 1.0 degrees)

![alt text](readme_figures/1990-01-01_S_SE_L_HI_1deg_regrid.png)


### Convergence Issues

Conditions:
- S_NAME = "S_SE" # S_SE or S_LSE 
- L_NAME = "L_HI" # L_M or L_HI
- grid = 0.5
- max_iter = 1000
- tol = 1e-3
- #tune these nudging values as needed
- offset = 2.0
- kernel_size = 15
- #threshold
- p = 4 #max number of very unstable points
- s_p = 4 #number of less stable points
- thresh=1.75 #extreme threshold
- s_thresh = 1.75 #less extreme threshold

Results:
- count_success_pre_nudge:  1365
- count_fail_pre_nudge:  315
- count_success_post_nudge:  122
- count_fail_post_nudge:  139
- count_fail_no_hot_pixel:  14
- count_fail_too_many_pixels:  40