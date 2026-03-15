using Sunny, LinearAlgebra, GLMakie

################################################################################
# Simple entangled units example 
################################################################################
using Sunny, LinearAlgebra, GLMakie

latvecs = lattice_vectors(1, 1, 2, 90, 90, 90)
positions = [[0, 0, 0], [0, 1/2 - 1e-4, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

# To model a quasi-1D system, symmetry must be broken along the b-axis by making
# bonds 1 -> 2 and 2 -> 1 inequivalent. This was achieved by offsetting the
# position of the second atom slightly from `[0, 0.5, 0]`.

# Specify a system and the two exchange interactions, J (rungs) and J′
# (lengthwise bonds). 

J = 1
J′ = 0.2J
sys = System(crystal, [1 => Moment(s=1/2, g=2)], :SUN; dims=(2, 1, 1))
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))
set_exchange!(sys, J′, Bond(1, 1, [1, 0, 0]));

# This completes specification of the model without consideration for
# entanglement between the sites bonded by rungs. Examine a ground state of
# this system.

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


# ## Spin Wave Calculations

# The last result is the expected classical ordering, with full-length dipoles
# arranged antiferromagnetically. This results in a `q=(π,π)` ordering. Because
# this model is Heisenberg, the Hamiltonian has an SU(2) symmetry, and any such
# ground state breaks this symmetry. This will lead to a Goldstone mode, as can
# readily be seen when we calculate the excitations with spin wave theory.

swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
qs = q_space_path(crystal, [[0, 1, 0], [1/2, 1, 0], [1, 1, 0]], 200)
energies = range(0, 2.5, 200)
res = intensities(swt, qs; energies, kernel=gaussian(; fwhm=0.2))
plot_intensities(res)

# This result is incorrect for such a small J′. Instead, the ground state on
# each bond should be a singlet, i.e. non-magnetic. Correspondingly, the
# excitations should be gapped singlet-triplet excitations. This can be
# reproduced using the entangled units formalism. An `EntangledSystem` is
# constructed from an ordinary `System` by providing a list of sites "to
# entangle" within each unit cell.

esys = Sunny.EntangledSystem(sys, [(1, 2)])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

# The ground state here is a pair of singlets, and the magnitude of the
# dipoles on each site is zero (up to numerical precision). The ordering wave
# vector is now `q=0`, so the system should be reshaped into the magnetic unit
# cell (one bond) before performing spin wave calculations.

esys = reshape_supercell(esys, I(3))
plot_spins(esys)

# Next create a `SpinWaveTheory` and calculate intensities just as would be
# done for an ordinary Sunny `System`.

eswt = SpinWaveTheory(esys; measure=ssf_trace(esys))
res = intensities(eswt, qs; energies, kernel=gaussian(; fwhm=0.2))
plot_intensities(res)


################################################################################
# Shastry-Sutherland Material
################################################################################
include("helper_functions.jl")

crystal_full = Crystal(joinpath(@__DIR__, "BaCe2ZnS5.cif"); symprec=0.01)
view_crystal(crystal_full)

crystal_Ce = subcrystal(crystal_full, "Ce1")
view_crystal(crystal_Ce)

crystal = cerium_crystal()
view_crystal(crystal)

# Set up the system and its interactions. 
units = Units(:meV, :angstrom)
g = [
     1.82  -0.62  0 
     -0.62  1.82  0
     0.0    0.0   2.14
]

J1 = [
    -0.75 -0.75 0
    -0.75 -0.75 0
     0     0    -1.5
]
J2 = [
    0.05 0 0
    0 0.05 0
    0 0 0.05
]
J3 = [
    0 0 0
    0 0 0
    0 0 0.0
]

sys = System(crystal, [1 => Moment(; s=1/2, g=g)], :SUN; dims=(1,1,1))
set_exchange!(sys, J1, Bond(3, 2, [0, 0, 0]))
set_exchange!(sys, J2, Bond(2, 4, [0, 0, 0]))
set_exchange!(sys, J3, Bond(1, 4, [0, 1, 0]))

# Ensure entanglement of dimers is considered.
esys = Sunny.EntangledSystem(sys, [(1,4), (2, 3), (5, 8), (6, 7)])


# H=0.0
# set_field!(esys, H*[1/sqrt(2), 1/sqrt(2), 0]*units.T)
randomize_spins!(esys)
minimize_energy!(esys; g_tol=1e-14, maxiters=10_000)
plot_spins(esys, show_cell=false, arrowscale = 0.8)

formfactors = [1 => FormFactor("Ce2")]
measure = ssf_perp(esys; formfactors)
swt = SpinWaveTheory(esys; measure)

path = q_space_path(crystal, [[-2.5, 0, 0], [-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2.5, 0, 0]], 400)

energies = range(0, 2, 200)
res_disp = intensities_bands(swt, path)
res = intensities(swt, path; energies, kernel=gaussian(; fwhm=0.1))

fig = Figure(size=(500, 600))
plot_intensities!(fig[1,1], res_disp)
plot_intensities!(fig[2,1], res; saturation=0.8)
fig


# Look at a single-energy slice and an integrated energy slab.
grid = q_space_grid(crystal, [1, 0, 0], range(-2.5, 2.5, 100), [0, 1, 0], (-2.5, 2.5))
res = intensities(swt, grid; energies=[0.75], kernel=gaussian(; fwhm=0.1))
plot_intensities(res; colormap=:magma)

energy_slice = range(0.65, 0.85, 5)
res = intensities(swt, grid; energies=energy_slice, kernel=gaussian(; fwhm=0.1))
data_integrated = sum(res.data, dims=(1,))[1,:,:]
heatmap((-2.5, 2.5), (-2.5, 2.5), data_integrated; axis=(aspect=true,), colormap=:magma)
