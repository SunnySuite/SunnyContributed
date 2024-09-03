latvecs = lattice_vectors(1.2, 1, 1.3, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])

units = Sunny.Units.theory
seed = 101
sys_rcs = System(cryst, (10, 10, 1), [SpinInfo(1, S=1, g=1)], :dipole; units, seed)

## Model parameter
J = 1.0
h = 0.5 
D = 0.05

## Set exchange interactions
set_exchange!(sys_rcs, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_rcs, J, Bond(1, 1, [0, 1, 0]))

## Single-ion anisotropy
Ss = spin_operators(sys_rcs, 1)
set_onsite_coupling!(sys_rcs, D*Ss[1]^2, 1)


## External field
set_external_field!(sys_rcs, h*[0,0,3])

##
randomize_spins!(sys_rcs)
minimize_energy!(sys_rcs)
out = minimize_energy!(sys_rcs)
println(out)

Δt = 0.025
kT = 0.2
λ = 0.1
langevin = Langevin(Δt; kT, λ)

for _ in 1:10_000
    step!(sys_rcs, langevin)
end

sc = dynamical_correlations(sys_rcs; Δt=2Δt, nω=100, ωmax=5.0)
add_sample!(sc, sys_rcs)

nsamples = 50
for _ in 1:nsamples
    for _ in 1:1_000
        step!(sys_rcs, langevin)
    end
    add_sample!(sc, sys_rcs)
end

params = slice_2D_binning_parameters(available_energies(sc),[0,0.5,0],[1.0,0.5,0],10,0.1)
is, counts = intensities_binned(sc, params, intensity_formula(sc, :trace; kT=kT))
bcs = axes_bincenters(params)
heatmap(0:1,bcs[4],(-is ./ counts)[:,1,1,:],axis = (;xlabel = "[H,½,0]",ylabel = "Energy"),colorrange = (0,0.01))
ylims!(0,10)

formula = intensity_formula(sc, :trace; kT)
is_all, counts_all = intensities_binned(sc, unit_resolution_binning_parameters(sc), formula)
total_weight = sum(is_all) / *(size(sys_rcs.coherents)...)
println("Total spectral weight (classical): ", total_weight)
#heatmap(1:size(data, 1), available_energies(sc), data; axis=(xticks=xticks,), colorrange=(0.0, 0.5))

