using Sunny, LinearAlgebra, GLMakie


crystal_full = Crystal(joinpath(@__DIR__, "Ba3Mn2O8.cif"); symprec=0.01)
crystal = subcrystal(crystal_full, "Mn1")
view_crystal(crystal)


units = Units(:meV, :angstrom)
sys = System(crystal, [1 => Moment(s=1, g=2)], :SUN)
D = -0.032
J0 = 1.642
J1 = 0.118
J2 = 0.256
J3 = 0.142
J4 = 0.037

set_exchange!(sys, J0, Bond(1, 2, [0, 0, 0]))
set_exchange!(sys, J1, Bond(2, 3, [0, 0, 0]))
set_exchange!(sys, J2, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J3, Bond(1, 2, [1, 0, 0]))
set_exchange!(sys, J4, Bond(4, 5, [0, 1, 0]))
set_onsite_coupling!(sys, S -> D*S[3]^2, 1)

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


esys = Sunny.EntangledSystem(sys, [(1, 2), (3, 4), (5, 6)])
# set_field!(esys, [0, 0, 3*units.T])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

formfactors = [1 => FormFactor("Mn5")]
measure = ssf_perp(esys; formfactors)
swt = SpinWaveTheory(esys; measure)

fwhm = 0.295
points = [
    [0.175, 0.175, 1.5],
    [0.85, 0.85, 1.5],
    [0.85, 0.85, 3],
    [0.0, 0.0, 3],
    [0.0, 0.0, 8],
]
qpts = q_space_path(crystal, points, 400)
energies = range(0.0, 4.0, 400)
res = intensities(swt, qpts; energies, kernel=gaussian(; fwhm))

plot_intensities(res)

# Classical dynamics
sys = System(crystal, [1 => Moment(s=1, g=2)], :SUN; dims=(12, 12, 1))
D = -0.032
J0 = 1.642
J1 = 0.118
J2 = 0.256
J3 = 0.142
J4 = 0.037

set_exchange!(sys, J0, Bond(1, 2, [0, 0, 0]))
set_exchange!(sys, J1, Bond(2, 3, [0, 0, 0]))
set_exchange!(sys, J2, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J3, Bond(1, 2, [1, 0, 0]))
set_exchange!(sys, J4, Bond(4, 5, [0, 1, 0]))
set_onsite_coupling!(sys, S -> D*S[3]^2, 1)

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


esys = Sunny.EntangledSystem(sys, [(1, 2), (3, 4), (5, 6)])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

kT = 0.5
integrator = Langevin(; damping=0.1, kT)
suggest_timestep(esys, integrator; tol=1e-2)
integrator.dt = dt = 0.06835

nsteps = 1000
Es = zeros(nsteps)
for i in 1:nsteps
    step!(esys, integrator)
    Es[i] = energy(esys)
end
lines(Es; axis=(xlabel="Steps", ylabel="E"))

energies = range(0, 3.0, 200)
sc = SampledCorrelations(esys; energies, dt, measure=ssf_perp(esys))

@time for _ in 1:3
    for _ in 1:100
        step!(esys, integrator)
    end
    add_sample!(sc, esys)
end

res = intensities(sc, qpts; energies=:available, kT)
plot_intensities(res)
