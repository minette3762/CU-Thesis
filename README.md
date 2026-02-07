## Two-Dimensional Simulation of Contaminant Transport and Treatments in Groundwater System

This work addresses the critical issue of groundwater contamination. It aims to create a realistic two-dimensional model to simulate how pollutants move through groundwater systems over time. The model specifically accounts for transient groundwater flow in heterogeneous aquifer. The study also evaluates the effectiveness of two common clean-up techniques: **Pump and Treat** (P&T) and **Permeable Reactive Barriers** (PRB).
{: style="text-align: justify"}

<div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="../../assets/prbc0.png" alt="Initial Condition" style="max-width: 300px; width: 100%;">
  </div>
  <div style="text-align: center;">
    <img src="../../assets/prbc2000.png" alt="After Treatment" style="max-width: 300px; width: 100%;">
  </div>
</div>
<div style="text-align: center; font-size: 0.9em; color: #555; margin-top: 0.5em;">
  <em>Comparison of contaminant concentrations at the initial state (left) and after 2000 days of remediation using the Permeable Reactive Barrier (PRB) technique (right).</em>
</div>

## Numerical Methods
- **Spatial Discretization**  
The **Discontinuous Galerkin** (DG) method is used. This method is particularly effective for problems where the flow of water (advection) is the dominant process and variations in material properties well.
{: style="text-align: justify"}

- **Time Discretization**  
The second-order, two-stage **Diagonally Implicit Runge-Kutta** (DIRK) method is used to model. This implicit method provides stability, allowing for larger time steps in the simulation, which is efficient for long-term predictions.
{: style="text-align: justify"}

- **Implementation**
The model was developed using the [DUNE](https://dune-project.org/) (Distributed and Unified Numerics Environment) C++ library.
{: style="text-align: justify"}


