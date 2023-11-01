# # Maintaining a quantum-to-classical correspondence at eleveated temperatures 
#
# One goal of the Sunny project is to expand the scope of classical simulations
# to encompass a greater number of quantum effects. The extension of the
# Landau-Lifshitz (LL) equations from SU(2) to SU($N$) coherent states is the
# cornerstone of this approach [1], but Sunny includes a number of other
# "classical-to-quantum" corrections. For example, in the zero-temperature
# limit, there is a well-known correspondence between Linear Spin Wave Theory
# (LSWT) and the quantization of the normal modes of the linearized LL
# equations. This allows LSWT to be recovered from an estimate of $(\mathcal{S}_{\rm cl}\mathbf{q}, \omega)$ 
# that has been calculated classically. This is achieved by applying a
# classical-to-quantum correspondence factor to the classical 
# $(\mathcal{S}_{\rm cl}\mathbf{q}, \omega)$ as follows,
#
# ```math
# \mathcal{S}_{\rm Q}(\mathbf{q}, \omega)=\frac{\hbar\omega}{k_{\rm B} T} \left[1+ n_{\rm B}(\omega/T) \right] \mathcal{S}_{\rm cl}}(\mathbf{q}, \omega),   
# ```
#
# Sunny automatically applies this correction when you provide an
# `intensity_formula` with a temperature.
#
# One consequence of this of this is that the "corrected" $\mathcal{S}_{\rm cl}(\mathbf{q}, \omega)$ 
# satisfies the quantum sum rule:
#
# ```math
# \int\intd\mathbf{q}d\omega\mathcal{S}_{\rm Q}(\mathbf{q}, \omega) = N_S S(S+1)
# ```
# where $N_S$ is the number of sites.
#
# However, this correspondence depends on a harmonic oscillator approximation
# and hence only applies near $T=0$. This is reflected in the fact that the
# correction factor,
#
# ```math
# \frac{\hbar\omega}{k_{\rm B} T} \left[1+ n_{\rm B}(\omega/T) \right],
# ```
# approaches unity for all $\omega$ whenever $T$ is large. In particular, this
# means that the corrected classical $(\mathcal{S}_{\rm cl}\mathbf{q}, \omega)$ will no longer satisify the
# quantum sum rule at elevated temperatures. It will instead approach the
# "classical sum rule": ∫∫d𝐪ωS(𝐪,ω)=NS²``.
# ```math
# \int\intd\mathbf{q}d\omega\mathcal{S}_{\rm cl}(\mathbf{q}, \omega) = N_S S^2
# 
# ```
#
# A simple approach to maintaining a classical-to-quantum correspondence at
# elevated temperatures is to renormalize the classical magnetic moments in a
# temperature-dependent fashion to ensure satisfaction of the quantum sum rule
# [2]. A Sunny makes it straightforward to apply such a renormalization, as will
# be demonstrated below. 
#
# ## Checking ``∫∫d𝐪ωS(𝐪,ω)=NS²`` in Sunny
#
# We'll begin by building a spin system representing the compound FeI2.

using Sunny, LinearAlgebra
include(joinpath(@__DIR__, "kappa_supplementals.jl")) 

dims = (8, 8, 4)
seed = 101
sys, cryst = FeI2_sys_and_cryst(dims; seed) # This function is available in the `kappa_supplement.jl`

# We will next estimate S(𝐪,ω) using classical dynamics. For more details on
# setting up such a calculation, see the tutorials in the Sunny documentation.

## Set a parameters for generating equilbrium samples.
Δt_therm = 0.004                      # Step size for Langevin integrator
dur_therm = 10.0                      # Safe thermalization time
λ = 0.1                               # Phenomenological coupling to thermal bath
kT = 0.1 * Sunny.meV_per_K            # Simulation temperature
langevin = Langevin(Δt_therm; λ, kT)  # Langevin integrator

## Now establish parameters for sampling correlations. 
Δt = 0.025          # Integrator step size for dissipationless trajectories
ωmax = 10.0         # Maximum energy to resolve
nω = 200            # Number of energy bins
nsamples = 3        # Number of dynamical trajectories to collect for estimating S(𝐪,ω)

## In particular, we'll need a complete set of observables for SU(3).
Sx, Sy, Sz = Sunny.spin_matrices(; N=3) # Spin-1 representation of spin operators
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

## And finally build the `SampledCorrelations` object to hold calculation results.
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)

## Thermalize and add several samples
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    ## Decorrelate sample
    for _ in 1:2000
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end

# `sc` now contains an estimate of `S(𝐪,ω)`. We next wish to evaluate
# ∫∫d𝐪ωS(𝐪,ω). Since we are working on a finite lattice and using discretized
# dynamics, the integral will reduce to a sum. Since we'll be evaluating this
# sum a lot, we'll define a function.

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
end

# Now evaluate the total spectral weight without temperature corrections.

total_spectral_weight(sc; kT=Inf) / prod(sys.latsize) 

# The result is 4/3, which is the expected "classical" sum rule, which can be
# established by evaluating ``∑ₐ<Z|Tᵃ|Z>²`` for any SU(3) coherent state ``Z``,
# with ``Tᵃ`` a complete set of generators of SU(3) -- for example, the
# `observables` above. However, the quantum sum rule is in fact 16/3, as can
# be determined by directly calculating ``∑ₐ(Tᵃ)²``.
#
# Now apply the classical-to-quantum correspondence factor. 

total_spectral_weight(sc; kT) / prod(sys.latsize) 

# This result is in fact very close to 16/3. So, at low temperatures,
# application of the classical-to-quantum correspondence factor results in
# satisfaction of the quantum sum rule as anticipated. This will not hold,
# however, at higher temperatures. Let's repeat the above experiment choosing a
# temperature above T_N=3.05.

sys, cryst = FeI2_sys_and_cryst(dims; seed) 
kT = 3.5 * Sunny.meV_per_K
langevin = Langevin(Δt_therm; λ, kT)
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)

## Thermalize
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    ## Decorrelate sample
    for _ in 1:2000
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end

# Evaluating the sum without the classical-to-quantum correction factor
# will again result in 4/3, as you can easily verify. Let's examine
# the result with this correction:

total_spectral_weight(sc; kT) / prod(sys.latsize) 

# The result is something close to 3, which certainly does not satisfy the
# quantum sum rule.
#
# ## Implementing moment renormalization 
#
# We can attempt to enforce satisfaction of the quantum sum rule by simply
# renormalizing the magnetic moments by an appropriate factor. In Sunny,
# this can be achieved by calling `set_spin_renormalization!(sys, κ)`,
# where κ is the desired renormalization. Let's repeat the calculation
# above one more time at the same temperature, this time setting ``κ=1.25``.

sys, cryst = FeI2_sys_and_cryst(dims; seed) 
sc = dynamical_correlations(sys; Δt, nω, ωmax, observables)
κ = 1.25

## Thermalize
for _ in 1:5_000
    step!(sys, langevin)
end

for _ in 1:nsamples
    ## Generate new equilibrium sample.
    for _ in 1:2000
        step!(sys, langevin)
    end

    ## Renormalize magnetic moments before collecting a time-evolved sample.
    set_spin_rescaling!(sys, κ)

    ## Generate a trajectory and calculate correlations .
    add_sample!(sc, sys)

    ## Turn off κ renormalization before generating a new equilibrium sample.
    set_spin_rescaling!(sys, 1.0)
end

# Finally, we evaluate the sume
total_spectral_weight(sc; kT) / prod(sys.latsize) 

# The result is something slightly greater than 5, substantially closer to the
# expected quantum sum rule. 
#
# In practice, ``κ(kT)`` needs to be determined empirically for each model. For
# a detailed example of how this may be done in practice, see the sample code
# [here](https://github.com/SunnySuite/2023-Dahlbom-Quantum_to_classical_crossover),
# which gives a complete account of the calculations used in [2].

# ## References
# [1] - Hao's paper
# [2] - New Manuscript
# [3] - Old paper