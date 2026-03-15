using Sunny
using GLMakie

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])

sys_sun     = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :SUN)
sys_rcs     = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :dipole)
sys_large_S = System(cryst, (10, 10, 1), [1 => Moment(s=1, g=-1)], :dipole_uncorrected)

J = 1.0
h = 0.4
D = 0.2

# Exchange
set_exchange!(sys_sun, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_rcs, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_large_S, J, Bond(1, 1, [1, 0, 0]))

# Single-ion anisotropy
Ss = spin_matrices(1)
set_onsite_coupling!(sys_sun, D*Ss[3]^2, 1)
set_onsite_coupling!(sys_rcs, D*Ss[3]^2, 1)

Ss_inf = spin_matrices(Inf)
set_onsite_coupling!(sys_large_S, D*Ss_inf[3]^2, 1)

# External field
set_field!(sys_large_S, [0,0,h])
set_field!(sys_rcs, [0,0,h])
set_field!(sys_sun, [0,0,h])


names = ["Large S (SpinW)", "SU(N)", "RCS"]
syss = [sys_large_S, sys_sun, sys_rcs]

fig = Figure(resolution=(1200,800))
for (n, (sys, name)) in enumerate(zip(syss, names))
    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=10_000)
    sys_min = reshape_supercell(sys, [1 -1 0; 1 1 0; 0 0 1])
    swt = SpinWaveTheory(sys_min; measure=ssf_trace(sys_min))

    path = q_space_path(cryst, [[0,0,0], [1/2, 1/2,0], [1,1,0]], 500)
    disp = intensities_bands(swt, path)

    plot_intensities!(fig[1,n], disp; title=names[n])
    ylims!(0, 9)
end
fig