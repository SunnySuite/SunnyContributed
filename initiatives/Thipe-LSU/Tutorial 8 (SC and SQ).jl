using Sunny, Revise,LinearAlgebra,CairoMakie,GLMakie
using Optim, Optimization, OptimizationOptimJL
includet("DispersionAndIntensitiesDipoleSingleQ.jl")
includet("SWTSingleQcalculations.jl")

a=6.0
b=6.0
c=40.0

latvecs = lattice_vectors(a, b, c, 90, 90, 120)
positions=[[1/2, 0, 0]]
Cr = Crystal(latvecs,positions,147)
GLMakie.activate!()
view_crystal(Cr)
print_symmetry_table(Cr,8)
sys=System(Cr, (3,3,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=1.0
set_exchange!(sys,J1,Bond(2, 3, [0, 0, 0]))
q = -[1/3, 1/3, 0]
axis = [0,0,1]
set_spiral_order_on_sublattice!(sys, 1; q, axis, S0=[cos(0),sin(0),0])
set_spiral_order_on_sublattice!(sys, 2; q, axis, S0=[cos(0),sin(0),0])
set_spiral_order_on_sublattice!(sys, 3; q, axis, S0=[cos(2π/3),sin(2π/3),0])
plot_spins(sys)



q_points = [[-1/2,0,0], [0,0,0], [1/2,1/2,0]]
density = 50
path, xticks = reciprocal_space_path(Cr, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Spin wave dispersion", xticks)
ylims!(ax, 0.0, 2.5)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color=intensity[:,i],colorrange)
end
fig

γ = 0.02 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.01:10)  
is = intensities_broadened(swt, path, energies, broadened_formula)
fig=Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Convoluted spectra", xticks)
ylims!(ax, 0.0, 3.0)
pl=heatmap!(ax, 1:size(is, 1), energies, is,colorrange=(0,10))
pl.colormap = Reverse(:Spectral)
fig


##### SQ method #######

sys = System(Cr,(1,1,1),[SpinInfo(1,S=1,g=2)], :dipole,seed =0)
k=[1/3,1/3,0]
n=[0,0,1.0]
bond1 = Bond(2,3,[0,0,0])
J1 = 1.
set_exchange!(sys,J1,bond1)

set_dipole!(sys, [+1, 0, 0], (1, 1, 1, 1))
set_dipole!(sys,[+1, 0, 0], (1, 1, 1, 2))
set_dipole!(sys,[-1/2, -sqrt(3)/2, 0],(1, 1, 1, 3))
plot_spins(sys)

qs = [[-0.5,0,0.],[0.0,0.,0],[0.5,0.5,0]]
density = 250
path, xticks = reciprocal_space_path(Cr, qs, density)
swt = SpinWaveTheory(sys)
formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=delta_function_kernel)
disp, intensity = Sunny.intensities_bands_SingleQ(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Spin wave dispersion", xticks)
ylims!(ax, -1e-1, 2.3)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i];color="black",colorrange=(0,1e-2))
end
fig

γ = 0.02 # width in meV
broadened_formula = Sunny.intensity_formula_SingleQ(swt,k,n, :perp; kernel=lorentzian(γ),formfactors=nothing)
energies = collect(0:0.005:3)  # 0 < ω < 10 (meV).
is = Sunny.intensities_broadened_SingleQ(swt, path, energies, broadened_formula)

begin
    CairoMakie.activate!()
    fig = Figure() 
    ax = Axis(fig[1,1];title="SingleQ convoluted spectra", xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", xticks)
    pl = heatmap!(ax,1:size(is, 1),energies,dropdims(sum(is[:,:,1:2,1:3],dims=[3,4]),dims=(3,4)))
    pl.colormap = Reverse(:Spectral)
    pl.colorrange = (0,1.5)
    fig
end


