
using Sunny, GLMakie, CairoMakie

a=3.0
b=3.0
c=9
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[1, 0, 0]]

Cu = Crystal(latvecs,positions,2)
GLMakie.activate!()
view_crystal(Cu)
print_symmetry_table(Cu,8)
sys=System(Cu, (2,2,1),[SpinInfo(1,S=1/2,g=2)], :dipole, seed=0)


J1=59.65*2 ##(J-Jc/2)/2
J2=-3.75*2##(Jp-Jc/4)/2
J3=1*2 ##Jpp/2
set_exchange!(sys,J1,Bond(1, 1, [1, 0, 0]))
set_exchange!(sys,J1,Bond(1, 1, [0, 1, 0]))
set_exchange!(sys,J2,Bond(1, 1, [1, 1, 0]))
set_exchange!(sys,J2,Bond(1, 1, [-1, 1, 0]))
set_exchange!(sys,J3,Bond(1, 1, [2, 0, 0]))
set_exchange!(sys,J3,Bond(1, 1, [0, 2, 0]))

randomize_spins!(sys)
minimize_energy!(sys;maxiters=2000)
plot_spins(sys)


q_points=[[3/4,1/4,0],[1/2, 1/2, 0],[1/2, 0, 0],[3/4, 1/4, 0],[1,0,0],[1/2 0 0]]
density = 50
path, xticks = reciprocal_space_path(Cu, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);


CairoMakie.activate!()

γ = 20
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.02:350)
is = intensities_broadened(swt, path, energies, broadened_formula);
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Convoluted spectra:", xticks)
ylims!(ax, 0.0, 350.0)
pl=heatmap!(ax, 1:size(is, 1), energies*1.18,is,colorrange=(0,0.01))
pl.colormap = Reverse(:Spectral)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]*1.18; color="black", colorrange)
end
fig

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Intensity of the spin-spin correlation function:", xticks)
ylims!(ax, 0.0, 20)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(intensity[:,i]), intensity[:,i];colorrange)
end
fig
