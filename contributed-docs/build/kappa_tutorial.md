# Enforcing the quantum sum rule with moment renormalization

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

Sunny automatically applies this correction when you provide an
`intensity_formula` with a temperature. This will be demonstrated in the code
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

We'll begin by building a spin system representing the effective Spin-1 compound FeI2.

````julia
using Sunny, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "kappa_supplementals.jl"))

dims = (8, 8, 4)
seed = 101
sys, cryst = FeI2_sys_and_cryst(dims; seed);
````

We will next estimate $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$ using
classical dynamics. (For more details on setting up such a calculation, see
the tutorials in the official Sunny documentation.)

````julia
# Parameters for generating equilbrium samples.
Δt_therm = 0.004                      # Step size for Langevin integrator
dur_therm = 10.0                      # Safe thermalization time
λ = 0.1                               # Phenomenological coupling to thermal bath
kT = 0.1 * Sunny.meV_per_K            # Simulation temperature
langevin = Langevin(Δt_therm; λ, kT)  # Langevin integrator

# Parameters for sampling correlations.
Δt = 0.025          # Integrator step size for dissipationless trajectories
ωmax = 10.0         # Maximum energy to resolve
nω = 200            # Number of energy bins
nsamples = 3        # Number of dynamical trajectories to collect for estimating S(𝐪,ω)

# Since FeI2 is a Spin-1 material, we'll need a complete set of observables for SU($2S+1=3$)
# with which to calculate correlations.
Sx, Sy, Sz = spin_matrices(1)  # Spin-1 representation of spin operators
observables = [
    Sx,
    Sy,
    Sz,
    -(Sx*Sz + Sz*Sx),
    -(Sy*Sz + Sz*Sy),
    Sx^2 - Sy^2,
    Sx*Sy + Sy*Sx,
    √3 * Sz^2 - I*2/√3,
]

# Build the `SampledCorrelations` object to hold calculation results.
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)

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

`sc` now contains an estimate of $\mathcal{S}_{\mathrm{cl}}(\mathbf{q}, \omega)$.
We next wish to evaluate the total spectral weight. We are working
on a finite lattice and using discretized dynamics, so the integral will
reduce to a sum. Since we'll be evaluating this sum repeatedly, we'll define a
function.

````julia
function total_spectral_weight(sc::SampledCorrelations; kT = Inf)
    # Retrieve all available discrete wave vectors in the first Brillouin zone.
    qs = available_wave_vectors(sc)

    # Set up a formula to tell Sunny how to calculate intensities. For
    # evaluating the sum rule, we want to take the trace over S(𝐪,ω).
    # Setting `kT` to something other than `Inf` will result in the application
    # temperature corrections.
    formula = intensity_formula(sc, :trace; kT)

    # Calculate the intensities. Note that we must include negative energies to
    # evaluate the total spectral weight.
    is = intensities_interpolated(sc, qs, formula; interpolation=:round, negative_energies=true)

    return sum(is)
end;
````

Now evaluate the total spectral weight without temperature corrections.

````julia
total_spectral_weight(sc; kT=Inf) / prod(sys.latsize)
````

````
1.3333333333333337
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
total_spectral_weight(sc; kT) / prod(sys.latsize)
````

````
5.345455394762397
````

This is in fact very close to 16/3. So, at low temperatures, application of
the classical-to-quantum correspondence factor yields results that
(approximately) satisfy the quantum sum rule.

This will no longer hold at higher temperatures. Let's repeat the above
experiment with a simulation temperature above $T_N=3.05$.

````julia
sys, cryst = FeI2_sys_and_cryst(dims; seed)
kT = 3.5 * Sunny.meV_per_K
langevin = Langevin(Δt_therm; λ, kT)
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)

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
total_spectral_weight(sc; kT) / prod(sys.latsize)
````

````
2.9003231784920787
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
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)
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

Finally, we evaluate the sum:

````julia
total_spectral_weight(sc; kT) / prod(sys.latsize)
````

````
5.16048201770543
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

