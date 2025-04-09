# Forests For Resilience - Bulk Recycling Model

This repository is a python implementation of a bulk model of precipitation recycling.
It is part of the European Space Agency (ESA) project Forests For Resilience.

## Introduction

Here we implement the numerical model defined in the appendix of [[2]](#ref-2).

## Concepts

Recycling: how much evaporation in a continental region contributes to the precipitation in the same region.

Recycling ratio $r$: the fraction of local-origin precipitation to the total precipitation in the region.
- $r=1$ implies that all precipitation in the region comes from local evaporation.
- $r=0$ implies that all precipitation in the region comes from non-local sources.

Water vapor content / depth $w$: quantity of precipitable water in the atmosphere above a point on the surface.
This is the integral of the specific humidity $q$ over the vertical coordinate $z$.

Specific humidity $q$: ration of mass of water vapor per unit mass of air.

Total water vapor content is decomposed into advective (non-local) $w_a$ and evaporative $w_m$ (local) components:

$$w = w_a + w_m$$

Precipitation $P$: the amount of water that falls from the atmosphere to the surface in a given time interval.

Precipitation is also decomposed into advective (non-local) $P_a$ and evaporative $P_m$ (local) components:

$$P = P_a + P_m$$

Velocity $V = (u, v)$: movement of the air in the atmosphere.

Water vapor flux $F = (F^{(x)}, F^{(y)})$: movement of water vapor in the atmosphere.

### Auxiliary variable

Local recycling ratio $\rho$: scalar field defined by equation (2.8) of [[1]](#ref-1).

$$\rho(x, y) = \frac{P_m(x, y)}{P(x, y)}$$

Note that $P_m(x, y)$ is the contribution of evaporation from the
total area of the domain to precipitation at this specific
point.

Not the contribution of evaporation from the point (x, y) to
precipitation at the same point.

Thus, the quantity $\rho$ represents the regional contribution to local precipitation.

## References

<a id="ref-1"></a>1. Burde, G. I., 2006: Bulk Recycling Models with Incomplete Vertical Mixing.
Part I: Conceptual Framework and Models. 
J. Climate, 19, 1461–1472, https://doi.org/10.1175/JCLI3687.1. 

<a id="ref-2"></a>2. Burde, G. I., C. Gandush, and Y. Bayarjargal, 2006: Bulk Recycling Models with Incomplete Vertical Mixing.
Part II: Precipitation Recycling in the Amazon Basin.
J. Climate, 19, 1473–1489, https://doi.org/10.1175/JCLI3688.1. 

<a id="ref-3"></a>3. Atmospheric water cycle in West Equatorial Africa in present and perturbed future climate.
Pokam, W.

<a id="ref-4"></a>4. Ncep_recycl.ncl
NCAR Command Language (NCL) implementation.
Pokam, W.
