# Sam's SunnyContributed Scripts

All of these scripts should be considered works-in-progress unless otherwise noted.
If you're interested in any of these, take a look at the explainers (linked below) or reach out to me (Sam Quinn) on Sunny's Slack workspace.

Some of the links below go to scripts rather than explainers; this means that the corresponding explainer isn't written yet. Stay tuned!

## [Visualize the eigenmodes of a Linear Spin Wave system (GUI)](docs/eigenmode_viewer_examples.md)
Linear spin wave theory can be considered as a fully classical theory with normal modes.
This script visualizes the dipole-sector oscillations of the classical trajectories for a selectable point in the LSWT band structure.

## [Explainer: (Quadratic) Casimirs](docs/quadratic_casimirs.md)
This explainer demonstrates how to derive and verify a uniquely normalized quadratic casimir element for a Lie algebra.
This is important for comparing theories with different numbers of boson flavors (e.g. SU(N) vs dipole).

## [Finding a few of the lowest-energy modes of a Heisenberg Pyrochlore Spin Glass using Sunny and ARPACK](docs/arnoldi_spin_glass_example.md)
Since the spin wave hamiltonian produced by Sunny is sparse, we can use Arnoldi iteration on it.
This script packages this concept into a Sunny-like `intensity_formula_arnoldi(...)` interface.
The linked example applies the method to identify some zero modes of the J1 AFM pyrochlore spin glass.

## [Explainer: Principled Upscaling of the Diffuse Scattering of a Cooperative Paramagnet](docs/cooperative_chain.md)
The momentum-space resolution of the diffuse scattering function $S(Q)$ computed by classical simulations is limited by the size of the simulation domain.
However, when only short-range correlations are present, it's possible to (correctly) upscale to arbitrary momentum-space resolution as a post-processing step, using [this script](realspace/classical.jl).
This explainer demonstrates how to use this functionality as well as the limits of validity.

## [Compute Spectral Response Functions using Sunny](susceptibility/susceptibility.jl)
Usually, Sunny focuses on generating neutron scattering intensity data.
By a simple calculation, it's possible to instead extract linear response theoretic quantities, e.g. the complex generalized susceptibility.
This script provides this functionality.

## [Compare Linear and Nonlinear AC Susceptibility (GUI)](susceptibility/AC_field_viewer.jl)
The linear response theoretic AC susceptibility computed by Linear Spin Wave Theory predicts the long-time response of a system's magnetization to a small applied AC magnetic field.
Sunny supports simulating a system with a time-varying external field.
By combining these, we can benchmark whether a given system is exhibiting a linear or nonlinear response, based on whether it agrees with the linear response.

## [View time-domain correlations computed by Sunny (GUI)](susceptibility/correlation_viewer.jl)
Usually, Sunny generates frequency-domain correlation data.
By a simple inverse fourier transform, the time-domain correlations can be extracted.
This script extracts and displays time-domain correlations, leveraging existing Sunny infrastructure.

## [Online/Streaming calculation of correlations](susceptibility/online_correlations.jl)
Sunny's `SampledCorrelations` are designed to efficiently sample the thermal equilibrium average correlations of a system.
This script implements an alternative approach to computing correlations.
Benchmarks not yet written.

## [Inverse modelling toolkit](inverse-toolkit/)
Tools for using Sunny to perform inverse modelling tasks.
Currently under development (Nov 15) and currently under-documented.

## [Detailed Balance Factors (Desmos widget)](https://www.desmos.com/calculator/e4xnsq6hf3)
Which functions $g$ obey $g(x) = e^x g(-x)$?
This widget parameterizes them (two real parameters) to fifth order in $x$.

## Miscellaneous support functions
### [`plot_band_intensities`](eigenmodes/support.jl)
Plots a band structure (with delta function broadening) using color to display the scattering intensity.

### [`plot_spin_data`](eigenmodes/support.jl)
Variant of Sunny's `plot_spins` which accepts a Makie Observable `spin_data` argument to allow for Makie-style updating of the plot.
Effectively makes `plot_spins` a Makie component.

### [`detail_sys`](susceptibility/support.jl)
A more detailed display for Sunny's `System` type, which describes the interactions.
Try `detail_sys(sys)` or `display(sys.interactions_union)` or `display(sys.interactions_union[1])`.




