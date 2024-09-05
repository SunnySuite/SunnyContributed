using Sunny, LinearAlgebra, GLMakie

# Set up the system

function BaMnO_crystal()
    crystal_full = Crystal(joinpath(@__DIR__, "Ba3Mn2O8.cif"); symprec=0.01)
    subcrystal(crystal_full, "Mn1")
end

function BaMnO(; mode=:SUN, dims=(1,1,1))
    crystal = BaMnO_crystal()

    sys = System(crystal, [1 => Moment(s=1, g=2)], mode; dims)
    D = -0.032 # meV
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

    return sys
end

crystal = BaMnO_crystal()
view_crystal(crystal)

sys = BaMnO(; mode=:SUN)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

esys = Sunny.EntangledSystem(sys, [(1, 2), (3, 4), (5, 6)])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

# EXERCISE: Set the external field and see what happens
# EXERCISE: Visualize the crystal of the reprocessed system system.

points = [
    [0.175, 0.175, 1.5],
    [0.85, 0.85, 1.5],
    [0.85, 0.85, 3],
    [0.0, 0.0, 3],
    [0.0, 0.0, 8],
]
qpts = q_space_path(crystal, points, 400)

formfactors = [1 => FormFactor("Mn5")]
measure = ssf_perp(esys; formfactors)
swt = SpinWaveTheory(esys; measure)

res = intensities(swt, qpts; energies, kernel=gaussian(; fwhm))

plot_intensities(res)
