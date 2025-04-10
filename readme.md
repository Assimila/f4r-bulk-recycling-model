# Forests For Resilience - Bulk Recycling Model

This repository is a python implementation of a bulk model of precipitation recycling.
It is part of the European Space Agency (ESA) project Forests For Resilience.

## Introduction

Here we implement the numerical model defined in the appendix of [[2]](#ref-2). 
This is an equation of state for the conservation of atmospheric water vapor,
with decomposition into advective (non-local) and evaporative (local) components via an auxiliary variable $\rho$.
The model is solved numerically using a finite-difference method on a staggered grid.

## Concepts

Recycling: how much evaporation in a continental region contributes to the precipitation in the same region.

Recycling ratio $r$: the fraction of local-origin precipitation to the total precipitation in the region.
- $r=1$ implies that all precipitation in the region comes from local evaporation.
- $r=0$ implies that all precipitation in the region comes from non-local sources.

Water vapor content / depth $w$: quantity of precipitable water in the atmosphere above a point on the surface.
This is the integral of the specific humidity $q$ over the vertical coordinate $z$.

Specific humidity $q$: ratio of mass of water vapor per unit mass of air.

Total water vapor content is decomposed into advective (non-local) $w_a$ and evaporative $w_m$ (local) components:

$$
w = w_a + w_m
$$

Precipitation $P$: the amount of water that falls from the atmosphere to the surface in a given time interval.

Precipitation is also decomposed into advective (non-local) $P_a$ and evaporative $P_m$ (local) components:

$$
P = P_a + P_m
$$

The basic assumption of bulk models is the condition of a well-mixed atmosphere:

$$
\frac{P_m}{P} = \frac{w_m}{w}
$$

Velocity $\vec{V} = (u, v)$: movement of the air in the atmosphere.

Water vapor flux $\vec{F} = (F^{(x)}, F^{(y)}) = (uw, vw)$: movement of water vapor in the atmosphere.

### Auxiliary variable

Local recycling ratio $\rho$: scalar field defined by equation (2.8) of [[1]](#ref-1).

$$
\rho(x, y) = \frac{P_m(x, y)}{P(x, y)}
$$

Note that $P_m(x, y)$ is the contribution of evaporation
from the **total area of the domain**
to precipitation at this point.
**Not** the contribution of evaporation from the point (x, y)
to precipitation at the same point.

Thus, the quantity $\rho$ represents the regional contribution to local precipitation.

### Conservation equations

The model is essentially a conservation of atmospheric water vapor

```math
\begin{align*}
\nabla \cdot \vec{F} &= E - P \\
\frac{\partial F^{(x)}}{\partial x} + \frac{\partial F^{(y)}}{\partial y} &= E - P \\
\end{align*}
```

and writing the same conservation law for water vapor of local origin:

```math
\begin{align*}
\nabla \cdot \vec{F_m} &= E - P_m \\
\frac{\partial(\rho F^{(x)})}{\partial x} + \frac{\partial(\rho F^{(y)})}{\partial y} &= E - \rho P \\
\end{align*}
```

These are the primary equations (2.1) and (2.2) of [[2]](#ref-2).

## Iterative scheme

We solve for the auxiliary variable $\rho$ using an iterative scheme.
Iterations are indicated by the superscript $k$.
Equation (A.2) of [[2]](#ref-2) is modified:

- the left hand side according to the 4 cases of (A.4)
- the right hand side $\rho \rightarrow \rho^{k+1}$

Solutions are given in the form of (A.5)

$$
\rho^{k+1}_{i+1/2, j+1/2} = \frac{A_1}{A_0}
$$

### Case 1 - wind from the south-west

defined by the conditions

```math
\begin{gather*}
F^{(x)}_{i, j+1/2} > 0 \\
F^{(y)}_{i+1/2, j} > 0
\end{gather*}
```

following from (A.2)

```math
\begin{gather*}
\frac{1}{2} ( \rho^{k+1}_{i+1/2, j+1/2} + \rho^{k}_{i+3/2, j+1/2} ) F_{i+1, j+1/2}^{(x)} \Delta y
- \frac{1}{2} ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i-1/2, j+1/2} ) F_{i, j+1/2}^{(x)} \Delta y \\
+ \frac{1}{2} ( \rho^{k+1}_{i+1/2, j+1/2} + \rho^{k}_{i+1/2, j+3/2} ) F_{i+1/2, j+1}^{(y)} \Delta x
- \frac{1}{2} ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i+1/2, j-1/2} ) F_{i+1/2, j}^{(y)} \Delta x \\
= E_{i+1/2, j+1/2} \Delta x \Delta y 
- \rho^{k+1}_{i+1/2, j+1/2} P_{i+1/2, j+1/2} \Delta x \Delta y
\end{gather*}
```

```math
\begin{gather*}
\rho^{k+1}_{i+1/2, j+1/2} (
    2 P_{i+1/2, j+1/2} \Delta x \Delta y
    + F_{i+1, j+1/2}^{(x)} \Delta y
    + F_{i+1/2, j+1}^{(y)} \Delta x
) \\
= 2 E_{i+1/2, j+1/2} \Delta x \Delta y
- \rho^{k}_{i+3/2, j+1/2} F_{i+1, j+1/2}^{(x)} \Delta y
- \rho^{k}_{i+1/2, j+3/2} F_{i+1/2, j+1}^{(y)} \Delta x \\
+ ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i-1/2, j+1/2} ) F_{i, j+1/2}^{(x)} \Delta y \\
+ ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i+1/2, j-1/2} ) F_{i+1/2, j}^{(y)} \Delta x
\end{gather*}
```

```math
A_0 = 2 P_{i+1/2, j+1/2} \Delta x \Delta y
+ F_{i+1, j+1/2}^{(x)} \Delta y
+ F_{i+1/2, j+1}^{(y)} \Delta x
```

```math
\begin{gather*}
A_1 = 2 E_{i+1/2, j+1/2} \Delta x \Delta y
- \rho^{k}_{i+3/2, j+1/2} F_{i+1, j+1/2}^{(x)} \Delta y
- \rho^{k}_{i+1/2, j+3/2} F_{i+1/2, j+1}^{(y)} \Delta x \\
+ ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i-1/2, j+1/2} ) F_{i, j+1/2}^{(x)} \Delta y \\
+ ( \rho^{k}_{i+1/2, j+1/2} + \rho^{k+1}_{i+1/2, j-1/2} ) F_{i+1/2, j}^{(y)} \Delta x
\end{gather*}
```

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
