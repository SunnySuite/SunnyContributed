using Sunny, GLMakie, LinearAlgebra, FFTW

# Model
seed = 101
crystal = Crystal(I(3), [[0,0,0]], 1)
units = Sunny.Units.theory
sys = System(crystal, (20,5,1), [SpinInfo(1; S=1, g=2)], :dipole; units, seed)
J = 1.0
h = 0.4
D = 0.1
set_exchange!(sys, J, Bond(1, 1, [1, 0,0]))
set_external_field!(sys, [h, 0, 0])
S = spin_matrices(1)
#set_onsite_coupling!(sys, D*S[3]^2, 1)

# Calculate S(q,ω) 
nsamples = 25
Δt = 0.05
kT = 0.2
λ = 0.1
sc = dynamical_correlations(sys; Δt=2Δt, nω=100, ωmax=5.0)

langevin = Langevin(Δt; kT, λ)
randomize_spins!(sys)
minimize_energy!(sys; maxiters=300)

for _ in 1:50000
    step!(sys, langevin)
end

for _ in 1:nsamples
    for _ in 1:1000
        step!(sys, langevin)
    end
    add_sample!(sc, sys; alg = :window)
end

density = 40
path, xticks = reciprocal_space_path(crystal, [[0.0,0.0,0.0],[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0]], density)
formula = intensity_formula(sc, :trace)
data = intensities_interpolated(sc, path, formula; interpolation=:round)

formula_full = intensity_formula(sc, Sunny.FullTensor(sc.observables; unilateral_to_bilateral = false))
data_full = intensities_interpolated(sc, path, formula_full; interpolation=:round)

fig = Figure()
ax1 = Axis(fig[1,1]; title="No clipping (log)", xticks)
ax2 = Axis(fig[1,2]; title="Yes clipping", xticks)
heatmap!(ax1, 1:size(data, 1), available_energies(sc), log10.(abs.(data)))
heatmap!(ax2, 1:size(data, 1), available_energies(sc), data; colorrange=(0,0.1))

real_sc_data = real.(ifft(sc.data,[4,5,6,7]));

ax3 = Axis(fig[2,1]; title="in-chain neighbors yy (black is self; krby)")
plot!(ax3,real_sc_data[5,1,1,1,1,1,:])
plot!(ax3,real_sc_data[5,1,1,2,1,1,:],color = :red)
plot!(ax3,real_sc_data[5,1,1,3,1,1,:],color = :blue)
plot!(ax3,real_sc_data[5,1,1,4,1,1,:],color = :yellow)

#yy
#plot!(ax4,real_sc_data[5,1,1,1,1,1,:])

ax4 = Axis(fig[2,2]; title="out-of-chain neighbors yy (black is self)")
plot!(ax4,real_sc_data[5,1,1,1,1,1,:])
plot!(ax4,real_sc_data[5,1,1,1,2,1,:],color = :red)
plot!(ax4,real_sc_data[5,1,1,1,3,1,:],color = :blue)
plot!(ax4,real_sc_data[5,1,1,1,4,1,:],color = :yellow)

include("../eigenmodes/support.jl")

sys = System(crystal, (2,1,1), [SpinInfo(1; S=1, g=2)], :dipole; units, seed)
J = 1.0
h = 0.4
D = 0.1
set_exchange!(sys, J, Bond(1, 1, [1, 0,0]))
set_external_field!(sys, [h, 0, 0])
randomize_spins!(sys)
minimize_energy!(sys)
swt = SpinWaveTheory(sys)
formula = intensity_formula(swt, :trace; kernel=Sunny.delta_function_kernel)
dispersion, intensity = intensities_bands(swt, path, formula)


plot_band_intensities!(ax2, dispersion, min.(3.0,intensity))

#ax4 = Axis(fig[2,2]; title="neighbor")
#plot!(ax4,real_sc_data[9,1,1,2,1,1,:])

#plot!(ax3,real_sc_data[5,1,1,1,1,1,:], color = :blue)
#plot!(ax4,real_sc_data[5,1,1,2,1,1,:], color = :blue)

fig
