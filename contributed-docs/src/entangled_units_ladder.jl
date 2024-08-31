# # Entangled Units Formalism
# **AUTHOR** David Dahlbom (dahlbomda@ornl.gov), **DATE**: August 30, 2024

# Traditional "large-_S_" classical methods start by modeling a spin system as a
# set of two-level quantum systems (dipoles) on each site of a lattice. This is
# the so-called product state assumption. In classical dynamics, these different
# two-level systems _interact_ with each other, but they are never _entangled_.
# The classical dynamics evolves these product states into other product states,
# meaning that one can always think about the state of a system as a set of
# individual spins (dipoles or, equivalently, two-level quantum systems)
# existing on each site of the lattice.
#
# The SU(_N_) formalism extends this picture in a straightforward way. Instead
# of having a two-level system on each site, one has an _N_-level system. The
# product state assumption remains. This allows the local behavior to be richer
# than that of a simple dipole. These local _N_-level systems interact with each
# other in the classical picture, but they are never entangled. The treatment of
# each individual _N_-level system is quantum mechanical, however. Thus, if one
# puts two-spins inside each one of these sites, the entanglement between the
# two spins will be faithfully represented and evolved.
#
# This tutorial gives a simple illustration of this idea using Sunny's
# experimental entangled unit formalism. The model we will consider is the the
# strong-run _S_=1/2 ladder.

# ## Making an `EntangledSystem`
#
# The approach to modeling systems with localized entangled is to first build a
# `System` in the standard way, with an individual spin on each site.
# Interactions are specified in the standard way. Note that this system must be
# built in `:SUN` mode, even when _S_=1/2. Another important restriction is that
# any spins which one wishes to entangle must exist within a crystalographic
# unit cell. This may require reshaping from the conventional unit cell. For the 
# spin ladder this presents no difficulties. First specify a the crystal.

using Sunny, GLMakie

latvecs = [
    1 0 0
    0 1 0
    0 0 2
]
positions = [[0, 0, 0], [0, 1/2 + 0.001, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

# Then specify a system and the two exchange interactions, J (rungs)
# and J′ (lengthwise bonds). 

J = 1
J′ = 0.2J
sys = System(crystal, [1 => Moment(s=1/2, g=2)], :SUN; dims=(2, 1, 1))
set_exchange!(sys, J′, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))

# Examine the behavior of this system when we randomize the spins and minimize
# the energy.

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# ## Spin Wave Calculations

# Note that the spins form an antiferromagnetic, $q=(π,π)$. Because this model is
# Heisenberg, the Hamiltonian has an SU(2) symmetry, and any such ground state
# breaks this symmetry. This will lead to a Goldstone mode.

swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
qs = q_space_path(crystal, [[0, 1, 0], [1/2, 1, 0], [1, 1, 0]], 200)
energies = range(0, 2.5, 200)
res = intensities(swt, qs; energies, kernel=gaussian(; fwhm=0.2))
plot_intensities(res)

# This result is incorrect for such a small J′. Instead, the ground state on each
# bond should be a singlet, i.e. a non-magnetic ground state. Correspondingly, the 
# excitations should be gapped singlet-triplet excitations. This can be reproduced 
# using the entangled units formalism. An `EntangledSystem` is constructed from
# an ordinary `System` by providing a list of sites "to entangle" within each unit cell.
# This time we will build a system with only one rung, since the ordering wave vector
# of the singlet ground state is q=0.

## TODO: Public reshape?
J = 1
J′ = 0.3J
sys = System(crystal, [1 => Moment(s=1/2, g=2)], :SUN)
set_exchange!(sys, J′, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))

esys = Sunny.EntangledSystem(sys, [(1, 2)])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

# Corresponding to the fact that the new ground state is a singlet state, the
# dipoles now have magnitude 0 (at least to numerical precision). Next calculate
# the excitations using linear spin wave theory.

eswt = Sunny.EntangledSpinWaveTheory(esys; measure=ssf_trace(sys)) ## TODO: ssf functions for esys
res = intensities(eswt, qs; energies, kernel=gaussian(; fwhm=0.2))

fig = Figure()
ax = plot_intensities!(fig[1,1], res)
fig

# This reproduces the expected gapped, triplon mode. Note that this mode is only
# visible when looking at the antisymmetric channel. Since we placed our second
# atom at a position of 1/2 along the b-axis, the excitations are visible along
# [H, 1, 0]. 

# ## Finite-T Classical Calculations
#
# As with ordinary `Systems`s, excitations may also be calculated at finite
# temperature using classical dynamics using the SU(_N_) generalization of the
# Landau-Lifshitz equations. Now we need to construct a system with enough sites
# along the a-axis so that we have sufficient momentum resolution to resolve the
# dispersion. 

sys = System(crystal, [1 => Moment(s=1/2, g=2)], :SUN; dims=(10, 1, 1))
set_exchange!(sys, J′, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))

esys = Sunny.EntangledSystem(sys, [(1, 2)])

# Construct a Langevin integrator and thermalize the system.

damping = 0.2
kT = 0.1J
integrator = Langevin(; damping, kT)
suggest_timestep(esys.sys, integrator; tol=1e-2)
dt = integrator.dt = 0.13

for _ in 1:500
    step!(esys, integrator)
end

# Next construct an `SampledCorrelations` and collect the correlations of
# sampled trajectories.

sc = SampledCorrelations(esys; energies, dt, measure=ssf_trace(sys))

for _ in 1:10
    for _ in 1:300
        step!(esys, integrator)
    end
    add_sample!(sc, esys)
end;

# Finally we retrieve intensity information along the same path as above. For
# comparison, also look along the zero-channel, where the intensities will look
# much different.

qs_0 = q_space_path(crystal, [[0, 0, 0], [1/2, 0, 0], [1, 0, 0]], 200)
qs_π = q_space_path(crystal, [[0, 1, 0], [1/2, 1, 0], [1, 1, 0]], 200)

res_0 = intensities(sc, qs_0; energies=:available, kT)
res_π = intensities(sc, qs_π; energies=:available, kT)

fig = Figure(size=(800, 400))
plot_intensities!(fig[1,1], res_0; axisopts=(; title="Symmetric Channel"))
plot_intensities!(fig[1,2], res_π; axisopts=(; title="Anti-symmetric Channel"))
fig

# Note that the classical dynamics reproduces the same dispersion as spin wave
# theory. We can now easily examine the behavior of the system at much higher
# temperatures, where the classical theory can be expected to be even more
# accurate.

sc = SampledCorrelations(esys; energies, dt, measure=ssf_trace(sys))

integrator.kT = 20.0J
for _ in 1:500
    step!(esys, integrator)
end

for _ in 1:10
    for _ in 1:300
        step!(esys, integrator)
    end
    add_sample!(sc, esys)
end

res_0 = intensities(sc, qs_0; energies=:available, kT)
res_π = intensities(sc, qs_π; energies=:available, kT)

fig = Figure(size=(800, 400))
plot_intensities!(fig[1,1], res_0; axisopts=(; title="Symmetric Channel"))
plot_intensities!(fig[1,2], res_π; axisopts=(; title="Anti-symmetric Channel"))
fig

# A pseudogap persists even well above the ordering temperature, in agreement
# with exact solutions. This is in contrast with traditional Landau-Lifshitz
# dynamics, which remains gapless throughout all temperatures.

# ## References