
using Sunny, CairoMakie, LinearAlgebra, Revise, GLMakie
using Optim , Optimization
includet("DispersionAndIntensitiesDipoleSingleQ.jl")
includet("SWTSingleQcalculations.jl")


a = 8.539;
b = 8.539;
c = 5.2414;
latvecs = lattice_vectors(a, b, c, 90, 90, 120)
positions=[[0.24964, 0, 1/2]]
Fe = Crystal(latvecs,positions,150)
GLMakie.activate!()
view_crystal(Fe)
print_symmetry_table(Fe,8)
sys=System(Fe, (1,1,7),[SpinInfo(1,S=5/2,g=2)], :dipole, seed=0)


eD = -1;
eH = +1;
eT = eD*eH;

J1 = 0.85;
J2 = 0.24;
J3 = 0.053;
J4 = 0.017;
J5 = 0.24;
D=0.01
set_exchange!(sys,J1,Bond(3, 2, [1, 1, 0]))
set_exchange!(sys,J2,Bond(1, 3, [0, 0, 0]))
set_exchange!(sys,J4,Bond(1, 1, [0, 0, 1]))
if eT==1
    set_exchange!(sys,J3,Bond(3, 2, [1, 1, 1]))
    set_exchange!(sys,J5,Bond(2, 3, [-1, -1, 1]))
elseif eT==-1
    set_exchange!(sys,J3,Bond(2, 3, [-1, -1, 1]))
    set_exchange!(sys,J5,Bond(3, 2, [1, 1, 1]))
end
S=spin_operators(sys,1)
set_onsite_coupling!(sys,D*S[3]^2,1)
randomize_spins!(sys)
minimize_energy!(sys;maxiters=2000)
plot_spins(sys)


q_points=[[0,1,-1],[0,1,2]]
density = 200
path, xticks = reciprocal_space_path(Fe, q_points, density);
swt = SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
γ = 0.02 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.02:12)  
is = intensities_broadened(swt, path, energies, broadened_formula);

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Convoluted spectra", xticks)
ylims!(ax, 0.0, 6.0)
pl=heatmap!(ax, 1:size(is, 1), energies, is,color="black",colorrange=(0,8))
pl.colormap = Reverse(:Spectral)
fig


#####SQ METHOD ####
sys=System(Fe, (1,1,1),[SpinInfo(1,S=5/2,g=2)], :dipole, seed=0)
eD = -1;
eH = +1;
eT = eD*eH;

J1 = 0.85;
J2 = 0.24;
J3 = 0.053;
J4 = 0.017;
J5 = 0.24;
D=0.01
set_exchange!(sys,J1,Bond(3, 2, [1, 1, 0]))
set_exchange!(sys,J2,Bond(1, 3, [0, 0, 0]))
set_exchange!(sys,J4,Bond(1, 1, [0, 0, 1]))
if eT==1
    set_exchange!(sys,J3,Bond(3, 2, [1, 1, 1]))
    set_exchange!(sys,J5,Bond(2, 3, [-1, -1, 1]))
elseif eT==-1
    set_exchange!(sys,J3,Bond(2, 3, [-1, -1, 1]))
    set_exchange!(sys,J5,Bond(3, 2, [1, 1, 1]))
end

n = [0.0,0.0,1]
k=Sunny.minimize_energy_spiral!(sys, n)

q_points=[[0,1,-1],[0,1,2]]
density = 200
path, xticks = reciprocal_space_path(Fe, q_points, density);
swt = SpinWaveTheory(sys)
formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=delta_function_kernel)
disp, intensity = Sunny.intensities_bands_SingleQ(swt, path, formula);

γ = 0.01
energies = collect(0:0.01:5.5)
broadened_formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=lorentzian(γ),formfactors=nothing)
is = Sunny.intensities_broadened_SingleQ(swt, path, energies, broadened_formula);

begin
    CairoMakie.activate!()
    fig = Figure() 
    ax = Axis(fig[1,1];title="SingleQ convoluted spectra", xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", xticks)
    pl = heatmap!(ax,1:size(is, 1),energies,dropdims(sum(is[:,:,:,1:3],dims=[3,4]),dims=(3,4)))
    pl.colormap = Reverse(:Spectral)
    pl.colorrange = (0,4)
    fig
end