println()
using Sunny, GLMakie, LinearAlgebra, JLD2


function diamond_model(; J, dims = (10, 10, 10), kwargs...)
    crystal = Crystal(I(3), [[0,0,0]])
    S = 1
    sys = System(crystal, dims, [SpinInfo(1; S, g=2)], :dipole; kwargs...)
    set_exchange!(sys, J, Bond(1, 1, [1,0,0]))
    randomize_spins!(sys)
    return sys
end

seed = 101
J = Sunny.meV_per_K * 7.5413 
sys = diamond_model(; J, seed)

randomize_spins!(sys)
minimize_energy!(sys)

Δt_langevin = 0.07 
kT = Sunny.meV_per_K * 2. # Units of meV
λ  = 0.1
langevin = Langevin(Δt_langevin; kT, λ)

# Thermalize
for _ in 1:4000
    step!(sys, langevin)
end

sc = dynamical_correlations(sys; nω=25, ωmax=5.5, Δt=2Δt_langevin)
nsamples = 10 
for _ in 1:nsamples
    for _ in 1:1000
        step!(sys, langevin)
    end
    add_sample!(sc, sys;alg = :window)
end
density = 100
qs, xticks = reciprocal_space_path(sc.crystal, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0)], density)
data = intensities_interpolated(sc, qs, intensity_formula(sc, :trace; kT); interpolation=:round);
begin
    fig = Figure()
    ax = Axis(fig[1,1]; xticks)
    hm = heatmap!(ax, 1:size(data, 1), available_energies(sc), data)
    Colorbar(fig[1,2], hm)
    fig
end
println(sum(data))

qs_all = available_wave_vectors(sc)
energies_all = available_energies(sc; negative_energies=true)
data_all = intensities_interpolated(sc, qs_all, intensity_formula(sc, :trace); interpolation=:round, negative_energies=true);
sum(data_all / *(size(sys.coherents)...))


################################################################################
# Plot comparisons
################################################################################
#=
data_new = load("/home/uud/Research/Scratch/data_new.jld2", "data")
data_old = load("/home/uud/Research/Scratch/data_old.jld2", "data")

begin
    fig = Figure(resolution=(1200,400))

    ax1a = Axis(fig[1,1][1,1]; title="Old Approach")
    ax1b = fig[1,1][1,2]
    ax2a = Axis(fig[1,2][1,1]; title="Asymmetric Approach")
    ax2b = fig[1,2][1,2]
    ax3a = Axis(fig[1,3][1,1]; title="Δ")
    ax3b = fig[1,3][1,2]

    hm = heatmap!(ax1a, data_old)
    Colorbar(ax1b, hm)
    hm = heatmap!(ax2a, data_new)
    Colorbar(ax2b, hm)
    hm = heatmap!(ax3a, data_old .- data_new)
    Colorbar(ax3b, hm)

    fig
end


=#

################################################################################
# Something with strong goldstone mode 
################################################################################

# view_crystal(cryst, 2.0)

begin
    latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
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

    ## Single-ion anisotropy
    Ss = spin_operators(sys_rcs, 1)
    set_onsite_coupling!(sys_rcs, D*Ss[3]^2, 1)


    ## External field
    set_external_field!(sys_rcs, h*[0,0,3])

    ##
    randomize_spins!(sys_rcs)
    minimize_energy!(sys_rcs)
    out = minimize_energy!(sys_rcs)
    println(out)

    Δt = 0.025
    kT = 0.02
    λ = 0.1
    langevin = Langevin(Δt; kT, λ)

    for _ in 1:10_000
        step!(sys_rcs, langevin)
    end

    scgs = dynamical_correlations(sys_rcs; Δt=2Δt, nω=100, ωmax=5.0)

    nsamples = 50
    for _ in 1:nsamples
        for _ in 1:1_000
            step!(sys_rcs, langevin)
        end
        add_sample!(scgs, sys_rcs; alg = :window)
    end
end

gsspins = plot_spins(sys_rcs)

begin
    density = 100
    qs, xticks = reciprocal_space_path(scgs.crystal, [(0.0, 0.5, 0.0), (0.5, 0.5, 0.0), (1.0, 0.5, 0.0)], density)
    data = intensities_interpolated(scgs, qs, intensity_formula(scgs, :trace; kT); interpolation=:round);
    hm1 = heatmap(1:size(data, 1), available_energies(scgs), data; axis=(xticks=xticks,), colorrange=(0.0, 0.5))
    hm2 = heatmap(1:size(data, 1), 1:size(data, 2), data; axis=(xticks=xticks,))
    hm3 = heatmap(data)

  end

real_sc_data = real.(ifft(scgs.data,[4,5,6,7]));

f = Figure()
ax3 = Axis(f[1,1]; title="in-chain neighbors yy (black is self; krby)")
plot!(ax3,real_sc_data[5,1,1,1,1,1,:])
plot!(ax3,real_sc_data[5,1,1,2,1,1,:],color = :red)
plot!(ax3,real_sc_data[5,1,1,3,1,1,:],color = :blue)
plot!(ax3,real_sc_data[5,1,1,4,1,1,:],color = :yellow)
f


    #=
    fig = Figure(resolution=(1200,400))

    r = 1
    ax1a = Axis(fig[r,1][1,1]; title="Old Approach", xticks=([],[]))
    ax1b = fig[r,1][1,2]
    ax2a = Axis(fig[r,2][1,1]; title="Asymmetric Approach", xticks=([],[]))
    ax2b = fig[r,2][1,2]
    ax3a = Axis(fig[r,3][1,1]; title="Δ", xticks=([],[]))
    ax3b = fig[r,3][1,2]

    hm = heatmap!(ax1a, data_old)
    Colorbar(ax1b, hm)
    hm = heatmap!(ax2a, data_new)
    Colorbar(ax2b, hm)
    hm = heatmap!(ax3a, data_old .- data_new)
    Colorbar(ax3b, hm)

    r = 2
    ax1a = Axis(fig[r,1][1,1])
    ax1b = fig[r,1][1,2]
    ax2a = Axis(fig[r,2][1,1])
    ax2b = fig[r,2][1,2]
    ax3a = Axis(fig[r,3][1,1])
    ax3b = fig[r,3][1,2]

    hm = heatmap!(ax1a, data_old; colorrange=(0, 0.5))
    Colorbar(ax1b, hm)
    hm = heatmap!(ax2a, data_new; colorrange=(0, 0.5))
    Colorbar(ax2b, hm)
    hm = heatmap!(ax3a, data_old .- data_new)
    Colorbar(ax3b, hm)

    fig
end
=#
