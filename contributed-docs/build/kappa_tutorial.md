# Enforcing the quantum sum rule with moment renormalization

**AUTHOR**: David Dahlbom (dahlbomda@ornl.gov), **DATE**: August 29, 2024 (Sunny 0.7.0)

One goal of the Sunny project is to extend classical techniques to incorporate
a greater number of quantum effects. The generalization of the Landau-Lifshitz
(LL) equations from SU(2) to SU($N$) coherent states is the cornerstone of
this approach [1], but Sunny includes a number of other "classical-to-quantum"
corrections. For example, in the zero-temperature limit, there is a well-known
correspondence between Linear Spin Wave Theory (LSWT) and the quantization of
the normal modes of the linearized LL equations. This allows the dynamical
spin structure factor (DSSF) that would be calculated with LSWT,
$\mathcal{S}_{\mathrm{Q}}(\mathbf{q}, \omega)$, to be recovered from a DSSF
that has been calculated classically, $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$.
This is achieved by applying a classical-to-quantum correspondence
factor to $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$:

```math
\mathcal{S}_{\mathrm{Q}}(\mathbf{q}, \omega)=\frac{\hbar\omega}{k_{\mathrm{B}} T} \left[1+ n_{\mathrm{B}}(\omega/T) \right] \mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega),
```

Sunny automatically applies this correction when you call
`intensity_static` on a `SampledCorrelations` and provide a temperature.
This will be demonstrated in the code
example below.

The quantum structure factor satisfies a familiar "zeroth-moment" sum rule,

```math
\int\int d\mathbf{q}d\omega\mathcal{S}_{\mathrm{Q}}(\mathbf{q}, \omega) = N_S S(S+1),
```
where $N_S$ is the number of sites. An immediate consequence of the
correspondence is that the "corrected" classical structure factor satisfies
the same sum rule:

```math
\int\int d\mathbf{q}d\omega \frac{\hbar\omega}{k_{\mathrm{B}} T} \left[1+ n_{\mathrm{B}}(\omega/T) \right] \mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega) = N_S S(S+1)
```

Note, however, that this correspondence depends on a harmonic oscillator
approximation and only applies near $T=0$. This is reflected in the fact that
the correction factor,

```math
\frac{\hbar\omega}{k_{\mathrm{B}} T} \left[1+ n_{\mathrm{B}}(\omega/T) \right],
```
approaches unity for all $\omega$ whenever $T$ grows large. In particular,
this means that the corrected classical $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$
will no longer satisfy the quantum sum rule at elevated
temperatures. It will instead approach the "classical sum rule":
```math
\lim_{T\rightarrow\infty}\int\int d\mathbf{q}d\omega \frac{\hbar\omega}{k_{\mathrm{B}} T} \left[1+ n_{\mathrm{B}}(\omega/T) \right] \mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega) = N_S S^2

```

A simple approach to maintaining a classical-to-quantum correspondence at
elevated temperatures is to renormalize the classical magnetic moments so that
the quantum sum rule is satisfied. The renormalization factor can
be determinied analytically in the infinite temperature limit [2]. For an
arbitrary temperature, however, it must be determined empirically [3]. While
determining an appropriate rescaling factor can be computationally expensive,
Sunny makes it straightforward to evaluate spectral sums and apply moment
renormalization, as shown below. One approach to determining the rescaling
factors themselves is demonstrated in the sample code
[here](https://github.com/SunnySuite/2023-Dahlbom-Quantum_to_classical_crossover).

## Evaluating spectral sums in Sunny

We'll begin by building a spin system representing the effective Spin-1
compound FeI₂. The functions for doing this are imported from
`kappa_supplementals.jl`, available for download
[here](https://github.com/SunnySuite/SunnyContributed/blob/main/contributed-docs/src/kappa_supplementals.jl).

````julia
using Sunny, LinearAlgebra
include(joinpath(@__DIR__, "kappa_supplementals.jl"))

dims = (8, 8, 4)
seed = 102
units = Units(:meV, :angstrom)
sys, cryst = FeI2_sys_and_cryst(dims; seed);
````

We will next estimate $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$ using
classical dynamics. (For more details on setting up such a calculation, see
the tutorials in the official Sunny documentation.)

````julia
# Parameters for generating equilbrium samples.
dt_therm = 0.004                            # Step size for Langevin integrator
dur_therm = 10.0                            # Safe thermalization time
damping = 0.1                               # Phenomenological coupling to thermal bath
kT = 0.3 * units.K
langevin = Langevin(dt_therm; damping, kT)  # Langevin integrator

# Parameters for sampling correlations.
dt = 0.025                     # Integrator step size for dissipationless trajectories
nsamples = 3                   # Number of dynamical trajectories to collect for estimating S(𝐪,ω)
energies = range(0, 10, 200);  # Energies to resolve, in meV, when calculating the dynamics
````

Since FeI₂ is a Spin-1 material, we will use the SU($2S+1=3$) formalism and
require a set of $N^2-1=8$ observables to calculate the total spectral weight.

````julia
Sx, Sy, Sz = spin_matrices(1)  # Spin-1 representation of spin operators
observables = [

    # Dipoles
    Sx,
    Sy,
    Sz,

    # Quadrupoles
    -(Sx*Sz + Sz*Sx),
    -(Sy*Sz + Sz*Sy),
    Sx^2 - Sy^2,
    Sx*Sy + Sy*Sx,
    √3 * Sz^2 - I*2/√3,
];
````

At the moment, Sunny does not expose a high-level interface for working with
custom observables such as these. As a workaround, we will write low-level
code that accesses the internal Sunny datastructure `Sunny.MeasureSpec`. The
notation `Sunny.*` indicates that this functionality is not part of the public
interface and is subject to change in future Sunny versions.

Building a `Sunny.MeasureSpec` requires the following:

1. An array containing a set of $N$ observables for each site of the system.
2. A vector of tuples $(n, m)$, with $1 <= n, m <= N$. Each tuple specifies a pair of observables. Together these determine which correlations will be calculated.
3. A function for reducing these correlation pairs into a final intensity.
4. A list of form factors.

The `Sunny.MeasureSpec` below sums over the autocorrelations of each observable and
disables form factor corrections.

````julia
observable_field = fill(Hermitian(zeros(ComplexF64, 3, 3)), length(observables), size(sys.coherents)...);
for site in Sunny.eachsite(sys), μ in axes(observables, 1)
    observable_field[μ, site] = Hermitian(observables[μ])
end
corr_pairs = [(i, i) for i in 1:length(observables)]
combiner(_, data) = real(sum(data))
measure = Sunny.MeasureSpec(observable_field, corr_pairs, combiner, [one(FormFactor)]);
````

Finally, we can construct a `SampledCorrelations` and perform the calculations.

````julia
sc = SampledCorrelations(sys; dt, energies, measure)

# Thermalize and add several samples
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    # Decorrelate sample
    for _ in 1:2_000
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end
````

`sc` now contains an estimate of $\mathcal{S}_{\mathrm{cl}}(\mathbf{q},
\omega)$. We next wish to evaluate the total spectral weight. We are working
on a finite lattice and using discretized dynamics, so the integral will
reduce to a sum. Since we'll be evaluating this sum repeatedly, we'll define a
function to do this. (Note that this function only works on Bravais lattices --
To evaluate spectral sums on a decorated lattice, the sum rule needs to be
evaluated on each sublattice individually!)

````julia
function total_spectral_weight(sc::SampledCorrelations; kT = nothing)
    # Retrieve all available discrete wave vectors in the equivalent of
    # one Brillouin zone.
    qs = Sunny.available_wave_vectors(sc)[:]

    # Calculate the intensities. Note that we must include negative energies to
    # evaluate the total spectral weight.
    is = intensities(sc, qs; energies=:available_with_negative, kT)

    return sum(is.data * sc.Δω)
end;
````

Now evaluate the total spectral weight without temperature corrections.

````julia
total_spectral_weight(sc) / (prod(sys.dims))
````

````
1.3333333333333335
````

The result is 4/3, which is the expected "classical" sum rule. This reference can be
established by evaluating $\sum_{\alpha}\langle Z\vert T^{\alpha} \vert Z \rangle^2$
for any SU(3) coherent state $Z$,
with $T^{\alpha}$ a complete set of generators of SU(3) -- for example, the
`observables` above. However, the quantum sum rule is in fact 16/3, as can
be determined by directly calculating $\sum_{\alpha}(T^{\alpha})^2$.

Now let's try again, this time applying the classical-to-quantum
correspondence factor by providing the simulation temperature.

````julia
total_spectral_weight(sc; kT) / prod(sys.dims)
````

````
5.480822137387131
````

This is relatively close to 16/3. So, at low temperatures, application of
the classical-to-quantum correspondence factor yields results that
(approximately) satisfy the quantum sum rule.

This will no longer hold at higher temperatures. Let's repeat the above
experiment with a simulation temperature above $T_N=3.05$.

````julia
sys, cryst = FeI2_sys_and_cryst(dims; seed)
kT = 3.5 * units.K
langevin = Langevin(dt_therm; damping, kT)
sc = SampledCorrelations(sys; dt, energies, measure)

# Thermalize
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    # Decorrelate sample
    for _ in 1:2_000
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end
````

Evaluating the sum without the classical-to-quantum correction factor will
again give 4/3, as you can easily verify. Let's examine the result with
the correction:

````julia
total_spectral_weight(sc; kT) / prod(sys.dims)
````

````
2.9491843014932657
````

While this is larger than the classical value of 4/3, it is still
substantially short of the quantum value of 16/3.

## Implementing moment renormalization

One way to enforce the quantum sum rule is by simply renormalizing the
magnetic moments. In Sunny, this can be achieved by
calling `set_spin_rescaling!(sys, κ)`, where κ is the desired renormalization.
Let's repeat the calculation above at the same temperature, this
time setting $κ=1.25$.

````julia
sys, cryst = FeI2_sys_and_cryst(dims; seed)
sc = SampledCorrelations(sys; dt, energies, measure)
κ = 1.25

# Thermalize
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    # Generate new equilibrium sample.
    for _ in 1:2_000
        step!(sys, langevin)
    end

    # Renormalize magnetic moments before collecting a time-evolved sample.
    set_spin_rescaling!(sys, κ)

    # Generate a trajectory and calculate correlations.
    add_sample!(sc, sys)

    # Turn off κ renormalization before generating a new equilibrium sample.
    set_spin_rescaling!(sys, 1.0)
end
````

Finally, we evaluate the sum.

````julia
total_spectral_weight(sc; kT) / prod(sys.dims)
````

````
5.0315750139022875
````

The result is something slightly greater than 5, substantially closer to the
expected quantum sum rule. We can now adjust $\kappa$ and iterate
until we reach a value sufficiently close to 16/3.  In general, this should
be done while collecting substantially more statistics.

Note that $\kappa (T)$ needs to be determined empirically for each model.
A detailed example, demonstrating the calculations used in [3],
is available [here](https://github.com/SunnySuite/2023-Dahlbom-Quantum_to_classical_crossover).

## References
[1] - [H. Zhang, C. D. Batista, "Classical spin dynamics based on SU(N) coherent states," PRB (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.104409)

[2] - [T. Huberman, D. A. Tennant, R. A. Cowley, R. Coldea and C. D. Frost, "A study of the quantum classical crossover in the spin dynamics of the 2D S = 5/2 antiferromagnet Rb2MnF4: neutron scattering, computer simulations and analytic theories" (2008)](https://iopscience.iop.org/article/10.1088/1742-5468/2008/05/P05017/meta)

[3] - [D. Dahlbom, D. Brooks, M. S. Wilson, S. Chi, A. I. Kolesnikov, M. B. Stone, H. Cao, Y.-W. Li, K. Barros, M. Mourigal, C. D. Batista, X. Bai, "Quantum to classical crossover in generalized spin systems," arXiv:2310.19905 (2023)](https://arxiv.org/abs/2310.19905)

