# # 0. Introduction to the SU(_N_) formalism with Ba₂FeSi₂O₇
#
# Traditional classical and semiclassical approaches to spin systems start by
# placing an SU(2) coherent state on each site. An SU(2) coherent state may
# simply be thought of as a dipole, or as a state of a 2-level quantum system,
# that is, as a linear combination of combination of $\vert\uparrow\rangle$ and
# $\vert\downarrow\rangle$ states.

# The idea behind the SU(_N_) generalization of this approach is to instead put
# an SU(_N_) coherent state on each site. This is equivalent to having a state
# from an _N_-level system on each site, for example, some linear combination of
# $\vert\frac{N-1}{2}\rangle$, $\vert\frac{N-2}{2}\rangle$, $\ldots$, $\vert\frac{-(N-1)}{2}\rangle$.
# The expectation of any relevant operator (including the dipole operators) can
# always be evaluated by taking an expectation value in this coherent state, as
# we will demonstrate concretely below.

# The chief advantage of this approach is that enables a richer description of
# the local physics. In particular, non-magnetic states can be represented
# directly and the physics of single-ion anisotropies can be modeled more
# faithfully. A useful showcase for this formalism the the square lattice
# antiferromagnet Ba₂FeSi₂O₇. This is is a quasi-2D system with effective $S=2$
# and strong easy-plane anisotropy. BFSO (as we will refer to it) has been
# studied using the SU(_N_) formalism in a number of recent studies, in
# particular the following:

# - S.-H. Do et al., "Decay and renormalization of a longitudinal mode...," [Nature Communications **12** (2021)](https://doi.org/10.1038/s41467-021-25591-7).
# - M. Lee et al., "Field-induced spin level crossings...," [PRB **107** (2023)](https://doi.org/10.1103/PhysRevB.107.144427).
# - S.-H. Do et al., "Understanding temperature-dependent SU(3) spin dynamics...," [npj quantum materials **5** (2023)](https://doi.org/10.1038/s41535-022-00526-7).

# # 1. Anisotropies and large spins 
#
# Before specifying the complete Hamiltonian, we'll consider a cartoon picture
# of the single-ion physics. The Hamiltonian for a single $S=2$ spin with single
# ion anisotropy is simply $\mathcal{H}_{\mathrm{SI}} = D(\hat{S}^z)^2$, where
# $\hat{S}^z$ is in the $S=2$ representation. We can use Sunny to represent this
# as a matrix.

## Import relevant libraries 
using Sunny, GLMakie, LinearAlgebra, Statistics

S = spin_matrices(2)  # Returns a vector of Sx, Sy, Sz
Sx, Sy, Sz = S        # Julia's "unpacking" syntax
## EXERCISE: Write the single-ion anisotropy (with D=1) and call it H_SI
## EXERCISE: How would you add a Zeeman term?

H_SI = Sz^2  # + h*μB*g*Sz

# The result is a diagonal matrix. Ordering of the basis elements is simply
# $\vert 2\rangle$, $\vert 1\rangle$, $\vert 0\rangle$, $\vert -1\rangle$, and
# $\vert -2\rangle$. Clearly the ground state is the $\vert 0 \rangle$, which
# is a non-magnetic state as we can quickly verify. First we'll write a complex
# vector to represent the ground state:

Z = [0., 0, 1 + 0im, 0, 0]

# We can now use this to evaluate the expectation values of the dipole operators.
expectation(op, Z) = real(Z' * op * Z)
sx = expectation(Sx, Z) 
sy = expectation(Sy, Z) 
sz = expectation(Sz, Z) 

# This is obviously a non-magnetic state that cannot be represented as a
# classical dipole of fixed length $S$. The SU(_N_) formalism provides a way for
# modeling states like these and calculating their dynamics. To see this in
# action, we'll make a spin system with only this single-ion anisotropy and no
# other interactions. We'll start by constructing a primitive tetragonal
# lattice. 

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
positions = [[0, 0, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

# Next we'll create a spin system. This is just as in previous examples, only we
# will now set the mode to `:SUN`.

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=2)], :dipole; dims, seed=1)
set_onsite_coupling!(sys, Sz^2, 1) # Set the anisotropy term

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


sys.dipoles[1,1,1,1]
sys.coherents[1,1,1,1] / exp(im*angle(sys.coherents[1,1,1,1][3]))


# # 2. BFSO Hamiltonian specification
a = 8.3194
c = 5.336
latvecs = lattice_vectors(a, a, c, 90, 90, 90)
positions = [[0, 0, 0]]
spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])
view_crystal(crystal)

# We use this `Crystal` to specify a `System`.
units = Units(:meV, :angstrom)

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=1.93)], :SUN; dims)

A = 1.16units.K
C = -1.74units.K
D = 28.65units.K

Sx, Sy, Sz = spin_matrices(2)
H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
set_onsite_coupling!(sys, H_SI, 1)

print_site(crystal, 1)
print_stevens_expansion(H_SI)

O = stevens_matrices(2)
O[2,-2]
O[2,-1]
O[2,0]
O[2,1]
O[2,2]

view_crystal(crystal)

bond1 = Bond(1, 2, [0, 0, 0])
bond2 = Bond(1, 1, [1, 0, 0])
bond3 = Bond(1, 1, [0, 0, 1])

J = 1.028 * units.K 
J′ = 0.1J
set_exchange!(sys, J, bond1)
set_exchange!(sys, J′, bond2)
set_exchange!(sys, J′, bond3)

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

function BFSO(dims; mode=:SUN, seed=1)
    units = Units(:meV, :angstrom)

    a = 8.3194
    c = 5.336
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
    crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])

    sys = System(crystal, dims, [SpinInfo(1; S=2, g=1.93)], mode; seed)

    A = 1.16units.K
    C = -1.74units.K
    D = 28.65units.K
    Sx, Sy, Sz = spin_matrices(2)
    H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
    set_onsite_coupling!(sys, H_SI, 1)

    Sx, Sy, Sz = spin_matrices(2)
    H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
    set_onsite_coupling!(sys, H_SI, 1)

    bond1 = Bond(1, 2, [0, 0, 0])
    bond2 = Bond(1, 1, [1, 0, 0])
    bond3 = Bond(1, 1, [0, 0, 1])

    J = 1.028units.K
    J′ = 0.1J
    set_exchange!(sys, J, bond1)
    set_exchange!(sys, J′, bond2)
    set_exchange!(sys, J′, bond3)


    return sys
end

# # 3. M vs. H
# Let's now proceed to generate a figure of $M$ vs $H$ for a range of field
# values. First we'll define a function to calculate the magnetization per site.

sys = BFSO(dims)
randomize_spins!(sys)
minimize_energy!(sys)

function magnetization(sys)
    return norm(mean(magnetic_moment(sys, site) for site in eachsite(sys)))
end

magnetization(sys)

# Helper function to calculate the relevant order parameter, the staggered
# magnetization in the plane. 

function order_parameter(sys)
    xy1 = [1/√2, 1/√2, 0]   # Unit vector in the (1, -1, 0) direction
    xy2 = [-1/√2, 1/√2, 0]  # Unit vector in the (1, 1, 0) direction
    M_xy1 = 0.0
    M_xy2 = 0.0
    for site in eachsite(sys)
        sublattice = (-1)^(site[4]) * (-1)^(site[3])
        M_xy1 = sublattice * (magnetic_moment(sys, site) ⋅ xy1)
        M_xy2 = sublattice * (magnetic_moment(sys, site) ⋅ xy2)
    end
    return max(abs(M_xy1), abs(M_xy2))
end

order_parameter(sys)

# Then we'll simply generate a list of applied field values and iteratively
# apply those fields, reoptimizing the spin configuration each time.

units = Units(:meV, :angstrom)
Hs = range(0.0, 55.0, 50)
Ms = Float64[]
OPs = Float64[]
for H in Hs
    set_field!(sys, (0, 0, H*units.T))
    minimize_energy!(sys)
    push!(Ms, magnetization(sys))
    push!(OPs, order_parameter(sys))
end

fig = Figure(size=(1200,400))
lines(fig[1,1], Hs, Ms; axis=(xlabel="H", ylabel="M"), color=:red)
lines(fig[1,2], Hs, OPs; axis=(xlabel="H", ylabel="Staggered XY Magnetization"), color=:red)
fig


# # 5. Spin waves
#
# A conceptually useful way to think of linear spin wave theory is as the
# quantization of classical dynamics linearized about the ground state. An
# important point is that for an S=1/2 spin, it is not possible to have
# longitudinal oscillations classically -- the classical magnitude has a fixed
# value of S. Similarly, at the linear level, there are no longitudinal
# oscillations in a traditional SWT calculation -- one has to incorporate 1/S
# corrections to recover such behavior. In the SU(_N_) generalization, the
# "spin" has additional degrees of freedom, corresponding to, for example,
# higher-order moments like quadrupoles and octupoles. As a consequence, 
# the dipole does not have a fixed magnitude. We can illustrate this aspect of
# the SU(_N_) classical dynamics with a simple animation.

set_external_field!(sys, (0, 0, 0))
minimize_energy!(sys)
plot_spins(sys)

# We'll next extend the length of the local dipoles by applying a
# sublattice-dependent local field.

xy = [√2/2, √2/2, 0]  # Unit vector in the (1, 1, 0) direction
for site in eachsite(sys)
    sublattice = (-1)^(site.I[4]) * (-1)^(site.I[3])  
    M_xy = set_external_field_at!(sys, 50*sublattice * xy, site) 
end

minimize_energy!(sys)
plot_spins(sys)

# We'll remove the magnetic fields and then run a classical trajectory using the
# generalized Landau-Lifshitz equations. This will allow us to see the
# longitudinal oscillations.

dt = 0.1
set_external_field!(sys, (0, 0, 0))
integrator = ImplicitMidpoint(dt)
suggest_timestep(sys, integrator; tol=1e-2)
integrator.dt = 0.01

fig = plot_spins(sys)

for _ in 1:500
    for _ in 1:5
        step!(sys, integrator)
    end
    notify(fig)
    sleep(1/60)
end

# This is an important observation: when we go to the SU(_N_) formalism,
# longitudinal oscillations become something possible at a classical level as a
# consequence of the local physics rather than true many-body quantum effects.
# When we quantize the result below using the SU(_N_) approach (a kind of
# multiflavor boson theory), we _will_ be able to capture this longitudinal
# oscillation at the linear level, that is, without loop expansions. 
#
# We now move onto our spin wave calculation by making a new BFSO system
# representing a single magnetic unit cell. We'll do this both using SU(_N_)
# mode, as well as dipole mode. We'll start with a small system to make
# optimization easy.

sys_sun = BFSO((2, 2, 2); mode=:SUN)
sys_dip = BFSO((2, 2, 2); mode=:dipole)

randomize_spins!(sys_sun)
minimize_energy!(sys_sun)
plot_spins(sys_sun)

# We'll set the ground state for the `:dipole` system to the corresponding
# degenerate ground state so our paths through reciprocal space correspond as
# well.

for site in eachsite(sys_dip)
    set_dipole!(sys_dip, sys_sun.dipoles[site], site)
end
minimize_energy!(sys_dip)
plot_spins(sys_dip)

# Now we'll reduce to a single magnetic unit cell.

print_wrapped_intensities(sys_dip)
suggest_magnetic_supercell([[0, 0, 1/2]])
sys_dip = reshape_supercell(sys_dip, [1 0 0; 0 1 0; 0 0 2])
sys_sun = reshape_supercell(sys_sun, [1 0 0; 0 1 0; 0 0 2])

# Finally, we'll create `SpinWaveTheory`s for both systems.

swt_dip = SpinWaveTheory(sys_dip; measure=ssf_perp(sys_dip))
swt_sun = SpinWaveTheory(sys_sun; measure=ssf_perp(sys_sun))

# We're now in a position to extract dispersions and intensities. First
# define a path in reciprocal space that we wish to examine.

points_rlu = [[0, 0, 1/2], [1, 0, 1/2], [2, 0, 1/2], [3, 0, 1/2]]
density = 300
path = q_space_path(sys.crystal, points_rlu, density);

# Next specify how we would like Sunny to calculate the intensities, and then
# calculate both the dispersion curves as well as intensities with artificial
# broadening. 

res_bands_dip = intensities_bands(swt_dip, path)
res_bands_sun = intensities_bands(swt_sun, path)

energies = range(0, 3.5, 400) 
res_dip = intensities(swt_dip, path; energies, kernel=gaussian(fwhm=0.1))
res_sun = intensities(swt_sun, path; energies, kernel=gaussian(fwhm=0.1))

fig = Figure()
plot_intensities!(fig[1,1], res_bands_dip; units=Units(:K, :angstrom), ylims=(0, 3.5))
plot_intensities!(fig[1,2], res_bands_sun; units=Units(:K, :angstrom), ylims=(0, 3.5))
plot_intensities!(fig[2,1], res_dip; units=Units(:K, :angstrom))
plot_intensities!(fig[2,2], res_sun; units=Units(:K, :angstrom))
fig


# # 6. S(q,ω) with classical dynamics
#
# We noted above that the longitudinal mode should actually decay, an effect
# that can only be captured when going beyond linear SWT by adding 1-loop
# corrections. While this is a planned future for Sunny, we note for now that
# some of these effects can be capture in finite-temperature simulations using
# the classical dynamics. Intuitively, this is possible because the classical
# dynamics is never linearized, unlike LSWT, so "magnon-magnon" interactions are
# included up to arbitrary order.
#
# In this next section, we'll calculate 𝒮(q,ω) using the generalized classical
# dynamics, examining the exact same path through reciprocal space, only this
# time we'll perform the simulation at T > 0. To start with, we'll make another
# BFSO system. This time, however, we'll need a large unit cell, rather than a
# single unit cell. 

sys = repeat_periodically(sys_sun, (10, 10, 1))
minimize_energy!(sys)
plot_spins(sys)

# Next we'll make a `Langevin` integrator to thermalize and decorrelate the system.

kT = 0.1 * meV_per_K
integrator = Langevin(; kT, damping=0.1)
suggest_timestep(sys, integrator; tol=1e-2)
integrator.dt = dt = 0.04

# Now we'll create a `SampledCorrelations` objects to collect information about
# trajectory correlations.

energies = range(0.0, 3.5units.meV, 200)
dt = 0.04
sc = SampledCorrelations(sys; dt, energies, measure=ssf_perp(sys))

nsamples = 10
for _ in 1:nsamples
    ## Thermalize the system
    for _ in 1:500
        step!(sys, integrator)
    end

    ## Add a trajectory
    @time add_sample!(sc, sys)
end

res_cold = intensities(sc, path; energies, kT)

begin
    fig = Figure(size=(800, 300))
    plot_intensities!(fig[1,1], res_cold; saturation=0.8, title="Classical")
    plot_intensities!(fig[1,2], res_sun; units=Units(:K, :angstrom), title="LSWT")
    fig
end

# Now let's repeat the procedure above at several different temperatures.

kTs_K = [2, 4, 8] .* (1.38/5.2)
kTs = kTs_K * meV_per_K
scs = []
for kT in kTs
    sc = SampledCorrelations(sys; dt, energies, measure=ssf_perp(sys))
    integrator.kT = kT

    ## Collect correlations from trajectories
    for _ in 1:nsamples
        ## Thermalize/decorrelate the system
        for _ in 1:500
            step!(sys, integrator)
        end

        ## Add a trajectory
        @time add_sample!(sc, sys)
    end
    
    push!(scs, sc)
end

fig = Figure(size=(1200, 300))
for (n, sc) in enumerate(scs)
    is = intensities(sc, path; energies, kT=kTs[n])
    plot_intensities!(fig[1,n], is)
end
fig

# Notice that the longitudinal mode, which decays when 1-loop corrections are
# applied, is extremely delicate in the classical simulations, dropping in energy
# and intensity quite rapidly as the temperature is increased.