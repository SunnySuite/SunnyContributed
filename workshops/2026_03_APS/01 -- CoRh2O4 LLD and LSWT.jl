# # 2. Landau-Lifshitz dynamics of CoRh₂O₄
using Sunny, GLMakie

units = Units(:meV, :angstrom)

# Define the lattice

a = 8.5031 # (Å)
latvecs = lattice_vectors(a, a, a, 90, 90, 90)
positions = [[1/8, 1/8, 1/8]]
spacegroup = 227
crystal = Crystal(latvecs, positions, spacegroup)

# Examine interactions that are possible on the crystal
view_crystal(crystal)
print_symmetry_table(crystal, 10.0)

# -- Question: What happens if you remove the `spacegroup` argument?

# Set up the system and all interactions
dims = (3, 3, 3)
sys = System(crystal, [1 => Moment(s=3/2, g=2)], :dipole; dims)
J = 0.63 # (meV)
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
set_field!(sys, [0.01units.T, 0, 0])

# Examine a trajectory
kT = 0.1units.K
dt = 0.01
integrator = Langevin(dt; damping=0.1, kT)

fig = plot_spins(sys; colorfn=i->sys.dipoles[i][2], colorrange=(-1, 1))
for _ in 1:500
    for _ in 1:5
        step!(sys, integrator)
    end
    notify(fig)
    sleep(1/60)
end

# To prepare for calculating correlations, let's start working with a bigger
# system at a higher temperature. 

set_field!(sys, [0, 0, 0])
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; color=[S[3] for S in sys.dipoles])

sys = repeat_periodically(sys, (3, 3, 3))
plot_spins(sys; color=[S[3] for S in sys.dipoles])
energy_per_site(sys)

kT = 16units.K
langevin = Langevin(; damping=0.2, kT)
suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.025;

# Relax the system at a new temperature
energies = [energy_per_site(sys)]
for _ in 1:1000
    step!(sys, langevin)
    push!(energies, energy_per_site(sys))
end

lines(energies, color=:blue, figure=(size=(600,300),), axis=(xlabel="Timesteps", ylabel="Energy (meV)"))

# Reexamine appropriate time step at new temperature.
suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.042;

# Take a look at the thermal state.
S0 = sys.dipoles[1,1,1,1]
plot_spins(sys; color=[S'*S0 for S in sys.dipoles])

# Calculate static correlations, i.e., spatial spin correlations from sampled thermal states.

formfactors = [1 => FormFactor("Co2")]
measure = ssf_perp(sys; formfactors)
sc = SampledCorrelationsStatic(sys; measure)
add_sample!(sc, sys)

# Collect 20 additional samples. Perform 100 Langevin time-steps between
# measurements to approximately decorrelate the sample in thermal equilibrium.

for _ in 1:20
    for _ in 1:100
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end

grid = q_space_grid(crystal, [1, 0, 0], range(-10, 10, 200), [0, 1, 0], (-10, 10))

res = intensities_static(sc, grid)
plot_intensities(res; saturation=1.0, title="Static Intensities at T=16K")

# Question: How to look at a different L?


# We'll next calculate dynamical correlations from trajectories.

dt = 2*langevin.dt
energies = range(0, 6, 50)
sc = SampledCorrelations(sys; dt, energies, measure)

for _ in 1:5
    for _ in 1:100
        step!(sys, langevin)
    end
    add_sample!(sc, sys)
end

# Select points that define a piecewise-linear path through reciprocal space,
# and a sampling density.

qs = [[3/4, 3/4,   0],
      [  0,   0,   0],
      [  0, 1/2, 1/2],
      [1/2,   1,   0],
      [  0,   1,   0],
      [1/4,   1, 1/4],
      [  0,   1,   0],
      [  0,  -4,   0]]
path = q_space_path(crystal, qs, 500)

# Calculate and plot the intensities along this path.

res = intensities(sc, path; energies, langevin.kT)
plot_intensities(res; units, title="Intensities at 16 K")

# -- Question: How does this change when collecting more samples?
# -- Look it up: Sunny function `print_irreducible_bz_paths`

# Let's now go to a lower temperature. We've learned from Hao that LL and LSWT
# are somehow equivalent in the T->0 limit. We'll perform a low-temperature
# calculation and then compare to our first LSWT calculation.

integrator = Langevin(0.025; kT=0.5units.K, damping=0.2)
sc_lo = SampledCorrelations(sys; dt, energies, measure)

# Thermalize the system
for _ in 1:1000
    step!(sys, integrator)
end

for _ in 1:10
    for _ in 1:100
        step!(sys, integrator)
    end
    add_sample!(sc_lo, sys)
end

res_lo = intensities(sc_lo, path; energies, langevin.kT)
plot_intensities(res_lo)

# LSWT Calculation. Begin by making the magnetic unit cell.

sys_swt = System(crystal, [1 => Moment(s=3/2, g=2)], :dipole)
J = 0.63 # (meV)
set_exchange!(sys_swt, J, Bond(2, 3, [0, 0, 0]))

randomize_spins!(sys_swt)
minimize_energy!(sys_swt)
plot_spins(sys_swt)

# Reshape into the primitive cell
shape = primitive_cell(crystal)

sys_prim = reshape_supercell(sys_swt, shape)
plot_spins(sys_prim)
@assert energy_per_site(sys_prim) ≈ -2J*(3/2)^2

measure_swt = ssf_perp(sys_prim; formfactors)
swt = SpinWaveTheory(sys_prim; measure=measure_swt)
res_disp = intensities_bands(swt, path)
plot_intensities(res_disp; ylims=(0, 4))

# -- Question: Where can you find the data for the dispersions and intensities?

kernel = gaussian(; fwhm=0.25units.meV)
res_swt = intensities(swt, path; energies, kernel)

# Compare with LL calculation
fig = Figure(size=(1200, 500))
plot_intensities!(fig[1,1], res_lo;  title="LL")
plot_intensities!(fig[1,2], res_swt; title="LSWT")
fig

# -- Question: What happens if we manually set the ground state to something
# different from a Neel order in the LSWT calculation?

# Frequently one does not have a single crystal of a sample, but one can produce
# a powder. Sunny provides a simple interface for calculating the "powder
# averaged" spectrum, both for LL and LSWT calculations.

radii = range(0, 3.5, 200) # (1/Å)
res = powder_average(crystal, radii, 350) do qs
    intensities(sc_lo, qs; energies, langevin.kT)
end
plot_intensities(res; units, title="Powder Average at 16 K")

# -- Question: How do you think we redo this with the LSWT calculation?
