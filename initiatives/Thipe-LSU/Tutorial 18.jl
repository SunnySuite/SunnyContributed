using Sunny, CairoMakie, LinearAlgebra, Revise, GLMakie
using Optim , Optimization
includet("DispersionAndIntensitiesDipoleSingleQ.jl")
includet("SWTSingleQcalculations.jl")

# Build a crystal

a = 10.2
b = 5.94
c = 7.81
latvecs = lattice_vectors(a, b, c, 90, 117.7, 90)
positions = [[0, 0, 0],[1/4, 1/4, 0]]
types = ["Cu1","Cu2"]
cryst = Crystal(latvecs,positions,12;types,setting="b1")
GLMakie.activate!()
view_crystal(cryst)
print_symmetry_table(cryst,8)

sys = System(cryst,(1,1,1),[SpinInfo(1,S=1/2,g=2), SpinInfo(3,S=1/2,g=2)], :dipole,seed =0)
J   = -2;
Jp  = -1;
Jab = 0.75;
Ja  = -J/.66 - Jab;
Jip = 0.01;
set_exchange!(sys,J,Bond(1, 3, [0, 0, 0]))
set_exchange!(sys,Jp,Bond(3, 5, [0, 0, 0]))
set_exchange!(sys,Ja,Bond(3, 4, [0, 0, 0]))
set_exchange!(sys,Jab, Bond(1, 2, [0,0,0]))
set_exchange!(sys,Jip,Bond(3, 4, [0, 0, 1]))


n = [0.0,0.0,1]
#Sunny.check_rotational_symmetry(sys; n, θ=0.01)
randomize_spins!(sys)
#       Phi1   Phi2  Phi3  k_x  k_y   k_z
xmin = [-1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6] # Minimum value of x
xmax = [2π 2π 2π 2π 2π 2π 1 1 1]  # Maximum value of x
x0 = [3.2 0.4 3.2 1 1.7 4.1 0.23 0.9 0.799]  # Initial value of x
k = optimagstr(x->gm_planar!(sys,n,x),xmin,xmax,x0)



# Define a path in reciprocal space.

q_points=[[0,0,0],[1,0,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt = SpinWaveTheory(sys)
formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=delta_function_kernel)
disp, intensity = Sunny.intensities_bands_SingleQ(swt, path, formula);

γ = 0.01
energies = collect(0:0.01:5.5)
broadened_formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=lorentzian(γ),formfactors=nothing)
is = Sunny.intensities_broadened_SingleQ(swt, path, energies, broadened_formula);

CairoMakie.activate!
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Spin wave dispersion: ", xticks)
ylims!(ax, 0, 5)
xlims!(ax, 1, size(disp, 1))
for i in axes(disp, 2)
    lines!(ax, 1:length(disp[:,i]), disp[:,i];color="black",colorrange = (0,0.03))
end
fig



radii = 0.01:0.02:3 
output = zeros(Float64, length(radii), length(energies))
for (i, radius) in enumerate(radii)
    n = 300
    qs = reciprocal_space_shell(cryst, radius, n)
    is1 = Sunny.intensities_broadened_SingleQ(swt, qs, energies, broadened_formula);
    is2=dropdims(sum(is1[:,:,:,:],dims=[3,4]),dims=(3,4))
    output[i, :] = sum(is2, dims=1) / size(is2, 1)
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Q (Å⁻¹)", ylabel="ω (meV)",title="Convoluted powder spectra:")
ylims!(ax, 0.0, 5)
heatmap!(ax, radii, energies, output, colormap=:gnuplot2,colorrange=(0.0,1))
fig

####luttinger tisza method

n = [0.0,0.0,1]
randomize_spins!(sys)
k = Sunny.optimize_luttinger_tisza_exchange(sys,[0.21,1.0,0.89])

Sunny.luttinger_tisza_exchange(sys, k)
Sunny.minimize_energy_spiral!(sys,k, n)

q_points=[[0,0,0],[1,0,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt = SpinWaveTheory(sys)
formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=delta_function_kernel)
disp, intensity = Sunny.intensities_bands_SingleQ(swt, path, formula);

####monte carlo

sys = System(cryst,(25,5,10),[SpinInfo(1,S=1/2,g=2), SpinInfo(3,S=1/2,g=2)], :dipole,seed =0)


J   = -2;
Jp  = -1;
Jab = 0.75;
Ja  = -J/.66 - Jab;
Jip = 0.01;
set_exchange!(sys,J,Bond(1, 3, [0, 0, 0]))
set_exchange!(sys,Jp,Bond(3, 5, [0, 0, 0]))
set_exchange!(sys,Ja,Bond(3, 4, [0, 0, 0]))
set_exchange!(sys, Jab, Bond(1, 2, [0,0,0]))
set_exchange!(sys,Jip,Bond(3, 4, [0, 0, 1]))
randomize_spins!(sys)

n_step = 1000
Δt=0.05
kT_Ks = 10 .^ range(log10(100),log10(0.1),100)
for kT_K in kT_Ks
    kT=kT_K*Sunny.meV_per_K
    λ=0.1
    langevin = Langevin(Δt; λ, kT)
    for _ in 1:n_step
        Sunny.step!(sys,langevin)
    end
end


print_wrapped_intensities(sys)

minimize_energy!(sys;maxiters=30000)

print_wrapped_intensities(sys)

