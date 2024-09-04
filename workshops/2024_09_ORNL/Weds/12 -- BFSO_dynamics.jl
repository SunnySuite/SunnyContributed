# # Magnetic excitations of BFSO
#
# A conceptually useful way to think of linear spin wave theory is as the
# quantization of classical dynamics linearized about the ground state. An
# important point is that for an S=1/2 spin, it is not possible to have
# longitudinal oscillations classically -- the classical magnitude has a fixed
# value of S. Similarly, at the linear level, there are no longitudinal
# oscillations in a traditional SWT calculation -- one has to incorporate 1/S
# corrections to recover such behavior. In the SU(_N_) generalization, the
# "spin" has additional degrees of freedom, corresponding to, for example,
# higher-order moments like quadrupoles and octupoles. As a consequence, it does
# the dipole does not have a fixed magnitude. We can illustrate this aspect of
# the SU(_N_) classical dynamics with a simple animation. We'll start by
# importing the relevant libraries and redefining a function to generate BFSO
# systems, which we first wrote in example 09.
using Sunny, GLMakie, LinearAlgebra, Statistics

function BFSO(dims; mode=:SUN, seed=1)
    a = 8.3194
    c = 5.336
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
    crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])

    sys = System(crystal, [1 => Moment(s=2, g=1.93)], mode; dims, seed)

    A = 1.16
    C = -1.74
    D = 28.65

    Sx, Sy, Sz = spin_matrices(2)
    H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
    set_onsite_coupling!(sys, H_SI, 1)

    bond1 = Bond(1, 2, [0, 0, 0])  
    bond2 = Bond(1, 1, [1, 0, 0]) 
    bond3 = Bond(1, 1, [0, 0, 1])

    J = 1.028
    J′ = 0.1J
    set_exchange!(sys, J, bond1)
    set_exchange!(sys, J′, bond2)
    set_exchange!(sys, J′, bond3)

    return sys
end

units = Units(:K, :angstrom)
sys = BFSO((6, 6, 2))

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# We'll next extend the length of the local dipoles by applying a
# sublattice-dependent local field.

xy = [√2/2, √2/2, 0]  # Unit vector in the (1, 1, 0) direction
for site in eachsite(sys)
    sublattice = (-1)^(site.I[4]) * (-1)^(site.I[3])  
    M_xy = set_field_at!(sys, 50*sublattice * xy * units.T, site) 
end

minimize_energy!(sys)
plot_spins(sys)

# We'll remove the magnetic fields and then run a classical trajectory using the
# generalized Landau-Lifshitz equations. This will allow us to see the
# longitudinal oscillations.

set_field!(sys, (0, 0, 0))
integrator = ImplicitMidpoint(0.1)
suggest_timestep(sys, integrator; tol=1e-2)
integrator.dt = 0.01

fig = plot_spins(sys; colorfn=i->sys.dipoles[i][2])

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
plot_spins(sys_dip)

# Finally, we'll create `SpinWaveTheory`s for both systems.

swt_dip = SpinWaveTheory(sys_dip; measure=ssf_perp(sys_dip))
swt_sun = SpinWaveTheory(sys_sun; measure=ssf_perp(sys_sun))

# We're now in a position to extract dispersions and intensities. First
# define a path in reciprocal space that we wish to examine.

points_rlu = [[0, 0, 1/2], [1, 0, 1/2], [2, 0, 1/2], [3, 0, 1/2]]
qpts = q_space_path(sys.crystal, points_rlu, 400)

## EXERCISE: After completing this section, repeat the same steps using a different path
## through reciprocal space:
## points_rlu_alt = [[0, 0, 1/2], [1/2, 1/2, 1/2],[1, 1, 1/2],[3/2, 3/2, 1/2]]

# Next specify how we would like Sunny to calculate the intensities, and then
# calculate both the dispersion curves as well as intensities with artificial
# broadening. 

bands_dip = intensities_bands(swt_dip, qpts)
bands_sun = intensities_bands(swt_sun, qpts)

fwhm = 1.0 
energies = range(0, 40, 400) 
broadened_dip = intensities(swt_dip, qpts; energies, kernel=gaussian(; fwhm))
broadened_sun = intensities(swt_sun, qpts; energies, kernel=gaussian(; fwhm))

fig = Figure()
plot_intensities!(fig[1,1], bands_dip; ylims=(0, 80.0))
plot_intensities!(fig[1,2], bands_sun; ylims=(0, 80.0))
plot_intensities!(fig[2,1], broadened_dip)
plot_intensities!(fig[2,2], broadened_sun)
fig

## EXERCISE: Change the upper bound on the `ylims` of the dispersions plot to 80.0. What do you see?


# # 6. S(q,ω) with classical dynamics
#
# We noted above that the longitudinal mode should actually decay, an effect
# that can only be captured when going beyond linear SWT by adding 1-loop
# corrections. While this is a planned future for Sunny, we note for now that
# some of these effects can be captured in finite-temperature simulations using
# the classical dynamics. Intuitively, this is possible because the classical
# dynamics is never linearized, unlike LSWT, so "magnon-magnon" interactions are
# included up to arbitrary order. How, the substitution of thermal fluctuations
# for quantum fluctuations in somewhat adhoc.
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

kT = 0.1
integrator = Langevin(; kT, damping=0.1)
suggest_timestep(sys, integrator; tol=1e-2)
integrator.dt = dt = 0.0015

# Now we'll create a `SampledCorrelations` objects to collect information about
# trajectory correlations.

energies = range(0, 40, 200)
sc = SampledCorrelations(sys; energies, dt, measure=ssf_perp(sys))

nsamples = 10
for _ in 1:nsamples
    ## Thermalize the system
    for _ in 1:500
        step!(sys, integrator)
    end

    ## Add a trajectory
    @time add_sample!(sc, sys)
end

# The procedure for extracting intensities is broadly similar to the LSWT case.
# We can then reuse the same path we specified above and compare to the LSWT
# result.

res_classical = intensities(sc, qpts; kT, energies=:available)
fig = Figure(size=(900,400))
plot_intensities!(fig[1,1], broadened_sun; colorrange=(0, 100))
plot_intensities!(fig[1,2], res_classical; colorrange=(0, 100))
fig

# Now let's repeat the procedure above at several different temperatures.

kTs = [0.3, 0.4, 0.5]
scs = []
for kT in kTs
    sc = SampledCorrelations(sys; energies, dt, measure=ssf_perp(sys))
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

fig = Figure(size=(1200,400))
for (n, sc) in enumerate(scs)
    res = intensities(sc, qpts; kT=kTs[n], energies=:available)
    plot_intensities!(fig[1,n], res)
end
fig

# Notice that the longitudinal mode, which decays when 1-loop corrections are
# applied, is extremely delicate in the classical simulations, broadening and
# dropping in energy quite rapidly as the temperature is increased.