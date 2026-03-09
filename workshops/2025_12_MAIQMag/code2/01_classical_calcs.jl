using Sunny, GLMakie, LinearAlgebra  


# Set up the system exactly as in the spin wave calculation.

units = Units(:meV, :angstrom)
a = 8.5031 # (Å)
latvecs = lattice_vectors(a, a, a, 90, 90, 90)
cryst = Crystal(latvecs, [[1/8, 1/8, 1/8]], 227)
view_crystal(cryst)

sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)
J = 0.63 # (meV)
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; color=[S[3] for S in sys.dipoles])


# Perform the spin wave calculation to keep as a reference.

shape = primitive_cell(cryst)
sys_prim = reshape_supercell(sys, shape)
plot_spins(sys_prim; color=[S[3] for S in sys_prim.dipoles])

formfactors = [1 => FormFactor("Co2")]
measure = ssf_perp(sys_prim; formfactors)
swt = SpinWaveTheory(sys_prim; measure)
kernel = lorentzian(fwhm=0.2)
qs = [[3/4, 3/4,   0],
      [  0,   0,   0],
      [  0, 1/2, 1/2],
      [1/2,   1,   0],
      [  0,   1,   0],
      [1/4,   1, 1/4],
      [  0,   1,   0],
      [  0,  -4,   0]]
path = q_space_path(cryst, qs, 500)
energies = range(0, 6, 300)

res_swt = intensities(swt, path; energies, kernel)
plot_intensities(res_swt; units, title="CoRh₂O₄ LSWT")


# Now we'll do some classical calculations, starting with S(q)

sys_big = repeat_periodically(sys, (10, 10, 10))
plot_spins(sys_big; color=[S[3] for S in sys_big.dipoles])

langevin = Langevin(; damping=0.2, kT=16*units.K)

suggest_timestep(sys_big, langevin; tol=1e-2)
langevin.dt = 0.025;

energies_per_site = [energy_per_site(sys)]
for _ in 1:1000
    step!(sys_big, langevin)
    push!(energies_per_site, energy_per_site(sys_big))
end

im = ImplicitMidpoint()

suggest_timestep(sys_big, im; tol=1e-2)
langevin.dt = 0.042;

lines(energies_per_site, color=:blue, figure=(size=(600,300),), axis=(xlabel="Timesteps", ylabel="Energy (meV)"))

S0 = sys_big.dipoles[1,1,1,1]
plot_spins(sys_big; color=[S'*S0 for S in sys_big.dipoles])

formfactors = [1 => FormFactor("Co2")]
measure = ssf_perp(sys_big; formfactors)
sc = SampledCorrelationsStatic(sys_big; measure)
add_sample!(sc, sys_big)

for _ in 1:20
    for _ in 1:100
        step!(sys_big, langevin)
    end
    add_sample!(sc, sys_big)
end

grid = q_space_grid(cryst, [1, 0, 0], range(-10, 10, 200), [0, 1, 0], (-10, 10))

res_static = intensities_static(sc, grid)
plot_intensities(res_static; saturation=1.0, title="Static Intensities at T=16K")

langevin.kT = 1units.K
for _ in 1:1000
    step!(sys_big, langevin)
end

energies = range(0, 6, 50)
dt = 2*langevin.dt
sc = SampledCorrelations(sys_big; dt, energies, measure)



for _ in 1:5
    for _ in 1:100
        step!(sys_big, langevin)
    end
    add_sample!(sc, sys_big)
end

res_cl = intensities(sc, path; energies, kT=nothing)
res_cl_βω = intensities(sc, path; energies, kT=langevin.kT)

begin
    fig = Figure(size=(1200, 300))
    axis = (; xticklabelrotation=π/3)
    plot_intensities!(fig[1,1], res_cl;    axis, units, title="Intensities at 16 K")
    plot_intensities!(fig[1,2], res_cl_βω; axis, units, title="Intensities at 16 K")
    plot_intensities!(fig[1,3], res_swt;   axis, units, title="Intensities at 16 K")
    fig
end

radii = range(0, 3.5, 200) # (1/Å)
res = powder_average(cryst, radii, 350) do qs
    intensities(sc, qs; energies, langevin.kT)
end
plot_intensities(res; units, title="Powder Average at 16 K")


# Now let's look at a simple parallelism programming pattern and use it to
# calculate several temperatures at once.

using Distributed
addprocs(4)

@everywhere using Sunny

kTs = [4, 8, 16, 32] * units.K

scs = pmap(kTs) do kT

    # Make a system
    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)
    J = 0.63 # (meV)
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    # Enlarge and thermalize
    sys_big = repeat_periodically(sys, (10, 10, 10))
    langevin = Langevin(0.025; damping=0.2, kT=kT)
    for _ in 1:1000
        step!(sys_big, langevin)
    end
    langevin.dt = 0.042;

    # Make a SampledCorrelations
    energies = range(0, 6, 50)
    measure = ssf_perp(sys_big)
    sc = SampledCorrelations(sys_big; dt=2langevin.dt, energies, measure)

    # Collect samples
    for _ in 1:5
        for _ in 1:100
            step!(sys_big, langevin)
        end
        add_sample!(sc, sys_big)
    end

    # Return the calculations to the main process
    sc
end

ress = map(zip(scs, kTs)) do (sc, kT)
    radii = range(0, 3.5, 200) # (1/Å)
    res = powder_average(cryst, radii, 350) do qs
        intensities(sc, qs; energies, langevin.kT)
    end
    res
end

begin
    fig = Figure(size=(1200, 300))
    for (i, res) in enumerate(ress)
        plot_intensities!(fig[1,i], res)
    end
    fig
end