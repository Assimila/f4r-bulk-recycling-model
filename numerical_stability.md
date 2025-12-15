# Numerical stability

We have observed that in some cases this model does not converge to a well-behaved solution.
But instead diverges (runs away) to very large values of $\rho$.

To understand why this happens, we need to look at the coefficients of the model equations.

$$
\rho^{k+1} = \frac{A_1}{A_0}
$$

where $A_0$ is defined by the four cases of flux direction (SW, NW, NE, SE) as per [[2]](#ref-2) equation (A.4).

and $A_0$ is decomposed into terms:

$$
A_1 = \alpha_1 
    + \alpha_C \rho^k_C 
    + \alpha_U \rho^k_U 
    + \alpha_R \rho^k_R
    + \alpha_D \rho^{k+1}_D
    + \alpha_L \rho^{k+1}_L
$$

If $| A_1 / A_0 | \gg 1$ then the iterative scheme will diverge outside of the physical bounds of $\rho \in [0, 1]$.

## Term $A_0$

$A_0$ is defined by the four cases of flux direction (SW, NW, NE, SE) as per [[2]](#ref-2) equation (A.4).

Looking at SW case (the other cases are similar):

$$
A_0 = 2 P \Delta x \Delta y + F^{(x)}_R \Delta y + F^{(y)}_U \Delta x
$$

where 

- $P$ is the precipitation rate
- $F^{(x)}_R$ is the x-flux of atmospheric water vapour on the **right** edge of the grid cell
- $F^{(y)}_U$ is the y-flux of atmospheric water vapour on the **upper** edge of the grid cell

I think it is clear from all four cases that $A_0$ should be positive, assuming a physically meaningful precipitation rate $P$,
and additive flux terms matching predominant flux directions.

### Negative $P$

$P$ is a derived quantity in this model.
See [[2]](#ref-2), §2a.

> Since the derivation of Eq. (2.2) implies the validity
of (2.1), the biases in the data inevitably result in
errors in the recycling ratio values.
> The procedure for
determining the recycling ratio will be more consistent if one introduces adjustments into the data to make Eq.
(2.1) valid.
> [...]
> we determine the precipitation flux P at the grid
points from the data using (2.1) and then solve Eq. (2.2)
with the calculated P values. 
> The level of the biases in
the data can be estimated by comparing the flux P calculated
as the analysis residuals with the precipitation
data Pd.

We have observed that in some cases the calculated $P$ is negative.
See also [[2]](#ref-2), §2d. and Fig. 7.

Clearly, a negative precipitation rate is unphysical, and should be a concern.
However, it is not _necessarily_ fatal to the model stability.

The problematic regime is when $P$ is negative enough to push $A_0$ close to zero.

### Divergent fluxes

The other possibility is that the flux terms,
which are assumed to be positive terms in $A_0$,
are in fact negative.

This case arises because flux direction for a cell (SW, NW, NE, SE) is determined by the sign of $F^{(x)}_L$ and $F^{(y)}_D$, where

- $F^{(x)}_L$ is the x-flux of atmospheric water vapour on the **left** edge of the grid cell
- $F^{(y)}_D$ is the y-flux of atmospheric water vapour on the **bottom** edge of the grid cell

whilst the flux terms in $A_0$ may include contributions from the **right** and **upper** edges of the grid cell.
These terms may be in opposition.

## Term $A_1$

### Term $\alpha_1$

The term $\alpha_1$ is a constant, independent of fluxes

$$
\alpha_1 = 2 E \Delta x \Delta y
$$

In the limit of no fluxes $F$, $P = E$, and so

$$
\rho^{k+1} 
= \frac{\alpha_1}{A_0} 
= \frac{2 E \Delta x \Delta y}{2 P \Delta x \Delta y} 
= 1
$$

Note that if starting from an initial guess of $\rho^0=0$,
with relaxation parameter $R$,
the first iteration is

$$
\rho^1 \sim R \frac{\alpha_1}{A_0}
$$

### Term $\alpha_C$

This term is the coefficient of self-coupling, proportional to $\rho^k$.

We have numerical instability if $| \alpha_C / A_0 | \gg 1$.

A similar analysis can be done for the other coupling terms $\alpha_U$, $\alpha_R$, $\alpha_D$, and $\alpha_L$.

## Instability condition

We propose a heuristic for numerical instability:

$$
\max ( \, | \alpha_X / A_0 | \, ) \gg 1
$$

for $X \in \{1, C, U, R, D, L\}$.

This is implemented as `Coefficients.instability_heuristic`.

## References

<a id="ref-1"></a>1. Burde, G. I., 2006: Bulk Recycling Models with Incomplete Vertical Mixing.
Part I: Conceptual Framework and Models. 
J. Climate, 19, 1461–1472, https://doi.org/10.1175/JCLI3687.1. 

<a id="ref-2"></a>2. Burde, G. I., C. Gandush, and Y. Bayarjargal, 2006: Bulk Recycling Models with Incomplete Vertical Mixing.
Part II: Precipitation Recycling in the Amazon Basin.
J. Climate, 19, 1473–1489, https://doi.org/10.1175/JCLI3688.1. 
