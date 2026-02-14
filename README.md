# Klausmeier–Gray–Scott Reaction–Diffusion Project

## Overview

This project investigates the dimensionless Klausmeier–Gray–Scott reaction–diffusion system describing vegetation–water interactions in semi-arid ecosystems. We examine self-organization mechanisms leading to spatial vegetation patterns (spots, stripes, gaps) and tipping-point transitions between vegetated and desert states.

In the project we will have combined:

* analytical stability analysis (stationary states, Jacobian, Turing instability),
* numerical bifurcation experiments,
* 2D finite-difference simulations with implicit time stepping.

The implementation follows a structured, modular design to ensure clarity, reproducibility and computational efficiency.

---

## Mathematical Model

We study the dimensionless system:

$$\partial_t u = a - u - uv^2 + d_1 \Delta u$$
$$\partial_t v = uv^2 - mv + d_2 \Delta v$$

with Dirichlet boundary conditions and finite-difference spatial discretization.

---

### Design Principles

The code is divided into logically separated functions and modules, so if the structure is kept unchanged, in notebooks the experiments can be remade if the user give the correct input for neccessary components.

---

## AI Usage Statement

Artificial Intelligence tools were used as supportive development aids in the following way:

* debugging numerical and structural errors,
* identifying potential improvements for code, such as function lu_factor, whose correctness and usefulness were independently verified.
* suggesting code splitting for better integrity,

All mathematical derivations, stability analyses and final numerical results were checked, validated and tested by the author.

---

## References

* Klausmeier, C. A. (1999). Regular and irregular patterns in semiarid vegetation. Science.
* Wang, X., et al. (2021). Bifurcation and pattern formation in diffusive Klausmeier-Gray-Scott model. JMAA.
* Suzuki, K. (2011). Mechanism Generating Spatial Patterns in Reaction-Diffusion Systems. *Interdisciplinary Information Sciences.*
