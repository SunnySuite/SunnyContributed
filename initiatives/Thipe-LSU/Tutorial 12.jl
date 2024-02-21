using Sunny,GLMakie, CairoMakie

a=3
b=3
c=4

latvecs = lattice_vectors(a, b, c, 90, 90, 120) 
positions=[[0, 0, 0]]
Cr = Crystal(latvecs,positions)
GLMakie.activate!()
view_crystal(Cr)
print_symmetry_table(Cr,8)
sys=System(Cr, (3,3,1),[SpinInfo(1,S=3/2,g=2)], :dipole, seed=0)

J1=1
D=0.2
print_symmetry_table(Cr,8)
set_exchange!(sys,J1,Bond(1, 1, [1, 0, 0]))
S=spin_operators(sys,1)
set_onsite_coupling!(sys,D*S[3]^2,1)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

q_points=[[0,0,0],[1, 1, 0]]
density = 50
path, xticks = reciprocal_space_path(Cr, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Spin wave dispersion:", xticks)
ylims!(ax, 0.0, 6)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black",colorrange)
end
fig

γ = 0.02 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.02:10) 
is = intensities_broadened(swt, path, energies, broadened_formula);
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Convoluted spectra:", xticks)
ylims!(ax, 0.0, 6.0)
pl=heatmap!(ax, 1:size(is, 1), energies, is,colorrange=(0,1))
pl.colormap = Reverse(:Spectral)
fig

