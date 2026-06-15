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

### Band runs on 15th June 2026

Run settings:
- S_SE (all ERA5 and no ERA5Land)
- L_HI (vertical flux integrated hourly)
- linear regridding to 0.5 degrees
- maximum number of iterations: 1000
- convergence threshold: 1e-3

Nudge settings:
- threshold for instability heuristic: 1.75
- maximum number of unstable points per month: 4
- offset for evaporation at unstable point: 2
- kernel size to distribute balanced adjustment in evaporation: 15

> Equatorial Band convergence results

count_success_pre_nudge:  1634 \
count_fail_pre_nudge:  46 \
count_success_post_nudge:  25 \
count_fail_post_nudge:  19 \
count_fail_no_hot_pixel:  0 \
count_fail_too_many_pixels:  2 

> Southern Band convergence results

count_success_pre_nudge:  1562 \
count_fail_pre_nudge:  118 \
count_success_post_nudge:  52 \
count_fail_post_nudge:  41 \
count_fail_no_hot_pixel:  16 \
count_fail_too_many_pixels:  9

> Northern Band convergence results

count_success_pre_nudge:  1566 \
count_fail_pre_nudge:  114 \
count_success_post_nudge:  43 \
count_fail_post_nudge:  52 \
count_fail_no_hot_pixel:  13 \
count_fail_too_many_pixels:  6 

> Congo Basin convergence results:

count_success_pre_nudge:  1608 \
count_fail_pre_nudge:  72 \
count_success_post_nudge:  27 \
count_fail_post_nudge:  42 \
count_fail_no_hot_pixel:  1 \
count_fail_too_many_pixels:  2

> DRC convergence results:

count_success_pre_nudge:  1606 \
count_fail_pre_nudge:  74 \
count_success_post_nudge:  36 \
count_fail_post_nudge:  30 \
count_fail_no_hot_pixel:  2 \
count_fail_too_many_pixels:  6