using Sunny, GLMakie, LinearAlgebra, FFTW, Statistics

################################################################################
# Single-ion anisotropies in SU(N) mode
################################################################################

S = spin_matrices(2)  
Sx, Sy, Sz = S        

H_SI = Sz^2  
eigen(H_SI)

Z = [0., 0, 1 + 0im, 0, 0]

# We can now use this to evaluate the expectation values of the dipole operators.
expectation(op, Z) = real(Z' * op * Z)
sx = expectation(Sx, Z) 
sy = expectation(Sy, Z) 
sz = expectation(Sz, Z) 

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
positions = [[0, 0, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=2)], :SUN; dims, seed=1)
set_onsite_coupling!(sys, Sz^2, 1) # Set the anisotropy term

randomize_spins!(sys)
minimize_energy!(sys; g_tol=1e-16)
plot_spins(sys)

sys.dipoles[1,1,1,1]
sys.coherents[1,1,1,1]

################################################################################
# BFSO 
################################################################################

a = 8.3194
c = 5.336
latvecs = lattice_vectors(a, a, c, 90, 90, 90)
positions = [[0, 0, 0]]
spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])

view_crystal(crystal)

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=1.93)], :SUN; dims)

A = 1.16 * meV_per_K
C = -1.74 * meV_per_K
D = 28.65 * meV_per_K

Sx, Sy, Sz = spin_matrices(2)
H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
set_onsite_coupling!(sys, H_SI, 1)


J = 1.028 * meV_per_K
J′ = 0.1J
bond1 = Bond(1, 2, [0, 0, 0])
bond2 = Bond(1, 1, [1, 0, 0])
bond3 = Bond(1, 1, [0, 0, 1])
set_exchange!(sys, J, bond1)
set_exchange!(sys, J′, bond2)
set_exchange!(sys, J′, bond3)

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

norm(sys.dipoles[1,1,1,1])

function BFSO(dims; mode=:SUN, seed=1)
    a = 8.3194
    c = 5.336
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
    crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])

    sys = System(crystal, dims, [SpinInfo(1; S=2, g=1.93)], mode; seed)

    A = 1.16 * meV_per_K
    C = -1.74 * meV_per_K
    D = 28.65 * meV_per_K

    Sx, Sy, Sz = spin_matrices(2)
    H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
    set_onsite_coupling!(sys, H_SI, 1)

    bond1 = Bond(1, 2, [0, 0, 0])  
    bond2 = Bond(1, 1, [1, 0, 0]) 
    bond3 = Bond(1, 1, [0, 0, 1])

    J = 1.028 * meV_per_K
    J′ = 0.1J
    set_exchange!(sys, J, bond1)
    set_exchange!(sys, J′, bond2)
    set_exchange!(sys, J′, bond3)

    return sys
end

function magnetization(sys, dir=[0, 0, 1.])
    nsites = prod(size(sys.dipoles))
    M_avg = sum(magnetic_moment(sys, site) for site in eachsite(sys)) / nsites
    return M_avg ⋅ dir
end

function staggered_magnetization(sys)
    xy = [1/√2, 1/√2, 0]  # Unit vector in the (1, 1, 0) direction
    M_xy = 0.0
    for site in eachsite(sys)
        sublattice = (-1)^(site.I[4]) * (-1)^(site.I[3])  
        M_xy = sublattice * (magnetic_moment(sys, site) ⋅ xy)
    end
    return abs(M_xy)
end

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)
staggered_magnetization(sys)
magnetization(sys, [0, 0, 1])

units = Units(:meV, :angstrom)
Hs = range(0.0, 1000.0, 50)
Ms = Float64[]
OPs = Float64[]
for H in Hs
    set_external_field!(sys, (0, 0, H*units.T))
    minimize_energy!(sys)
    push!(Ms, magnetization(sys))
    push!(OPs, staggered_magnetization(sys))
end

fig = Figure(size=(1200,400))
scatter(fig[1,1], Hs, Ms; axis=(xlabel="H", ylabel="M"))
scatter(fig[1,2], Hs, OPs; axis=(xlabel="H", ylabel="Staggered XY Magnetization"))
fig

################################################################################
# Dynamics
################################################################################

set_external_field!(sys, (0, 0, 0))
minimize_energy!(sys)
plot_spins(sys)

# We'll next extend the length of the local dipoles by applying a
# sublattice-dependent local field.

xy = [√2/2, √2/2, 0]  # Unit vector in the (1, 1, 0) direction
for site in eachsite(sys)
    sublattice = (-1)^(site.I[4]) * (-1)^(site.I[3])  
    M_xy = set_field_at!(sys, 50*sublattice * xy, site) 
end

minimize_energy!(sys)
plot_spins(sys)

set_field!(sys, (0, 0, 0))
integrator = ImplicitMidpoint(dt)
suggest_timestep(sys, integrator; tol=1e-2)
integrator.dt = 0.01

fig = plot_spins(sys; colorfn=i->norm(sys.dipoles[i][3]))

for _ in 1:500
    for _ in 1:5
        step!(sys, integrator)
    end
    notify(fig)
    sleep(1/60)
end

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

swt_dip = SpinWaveTheory(sys_dip; measure=ssf_perp(sys_dip))
swt_sun = SpinWaveTheory(sys_sun; measure=ssf_perp(sys_sun))

points_rlu = [[0, 0, 1/2], [1, 0, 1/2], [2, 0, 1/2], [3, 0, 1/2]]
path = q_space_path(sys.crystal, points_rlu, 300)
fwhm = 0.1
energies = range(0, 3.5, 400) 

disp_dip = intensities_bands(swt_dip, path)
disp_sun = intensities_bands(swt_sun, path)

res_dip = intensities(swt_dip, path; energies, kernel=gaussian(; fwhm))
res_sun = intensities(swt_sun, path; energies, kernel=gaussian(; fwhm))

fig = Figure()
plot_intensities!(fig[1,1], disp_dip; title="Dipole")
ylims!(0.0, 3.5)
plot_intensities!(fig[1,2], disp_sun; title="SU(N)", ylims=(0.0, 3.5))
ylims!(0.0, 3.5)
plot_intensities!(fig[2,1], res_dip)
plot_intensities!(fig[2,2], res_sun)
fig