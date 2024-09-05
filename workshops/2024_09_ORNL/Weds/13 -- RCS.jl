using Sunny, GLMakie, LinearAlgebra

# Instantiate our simple Hamiltonian on a tetragonal lattice.
# We'll use all three Sunny "modes" and compare the results.

latvecs = lattice_vectors(1, 1, 1.1, 90, 90, 90)
positions = [[0, 0, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)


# Using this crystal, make three systems, one in each of Sunny's modes.

sys_S = System(crystal, [1 => Moment(s=1, g=1)], :dipole_large_S)
sys_SUN = System(crystal, [1 => Moment(s=1, g=1)], :SUN)
sys_dip = System(crystal, [1 => Moment(s=1, g=1)], :dipole; dims=(2, 2, 2))

# Set the exchange for each of the systems.

J = 1
set_exchange!(sys_S, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_SUN, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_dip, J, Bond(1, 1, [1, 0, 0]))

# Set the anisotropy for each of the systems.

D = 0.2
S = spin_matrices(1)
S_symbolic = spin_matrices(Inf)
set_onsite_coupling!(sys_dip, D*S[3]^2, 1)
set_onsite_coupling!(sys_S, D*S_symbolic[3]^2, 1)
set_onsite_coupling!(sys_SUN, D*S[3]^2, 1)

# Apply an external field so canted out of plane.

B = (0, 0, -0.1)
set_field!(sys_dip, B)
set_field!(sys_S, B)
set_field!(sys_SUN, B)

# Reshape into magnetic unit cell for (π, π, π) ordering

sys_dip = reshape_supercell(sys_dip, [1 1 1; -1 1 0; 0 0 1])
sys_S = reshape_supercell(sys_S, [1 1 1; -1 1 0; 0 0 1])
sys_SUN = reshape_supercell(sys_SUN, [1 1 1; -1 1 0; 0 0 1])

# Find ground states

randomize_spins!(sys_dip)
minimize_energy!(sys_dip)
randomize_spins!(sys_S)
minimize_energy!(sys_S)
randomize_spins!(sys_SUN)
minimize_energy!(sys_SUN)

# Make SpinWaveTheorys and calculation dispersions.

swt_dip = SpinWaveTheory(sys_dip; measure=ssf_trace(sys_dip))
swt_S = SpinWaveTheory(sys_S; measure=ssf_trace(sys_S))
swt_SUN = SpinWaveTheory(sys_SUN; measure=ssf_trace(sys_SUN))

qpts = q_space_path(crystal, [[0, 0, 0], [1/2, 1/2, 0], [1, 1, 0]], 200)

res_dip = intensities_bands(swt_dip, qpts)
res_S = intensities_bands(swt_S, qpts)
res_SUN = intensities_bands(swt_SUN, qpts)

# Plot results

fig = Figure(size=(1200,400))
ymax = 6.0
plot_intensities!(fig[1,1], res_S; ylims=(0, 12), title="Large-S")
plot_intensities!(fig[1,2], res_SUN; ylims=(0, 12), title="SU(N)")
plot_intensities!(fig[1,3], res_dip; ylims=(0, 12), title="RCS")
fig