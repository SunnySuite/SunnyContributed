using Sunny, GLMakie, CairoMakie    

a=3.0
b=3.0
c=6.0

latvecs = lattice_vectors(a, b, c, 90, 90, 90) 
positions=[[0, 0, 0]]
types=["Cu1"]
Cu = Crystal(latvecs, positions; types)
cryst=subcrystal(Cu,"Cu1")
GLMakie.activate!()
view_crystal(cryst)
sys=System(cryst, (2,2,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=1.0
J2=-0.1
set_exchange!(sys,J1,Bond(1, 1, [1, 0, 0]))
set_exchange!(sys,J2,Bond(1, 1, [1, 1, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


swt=SpinWaveTheory(sys)
q_points = [[0,0,0], [1/2,0,0], [1/2,1/2,0], [0,0,0]]
density = 50
path, xticks = reciprocal_space_path(cryst, q_points, density);
disp = dispersion(swt, path);
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);


CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)",title="Spin wave dispersion:", xticks)
ylims!(ax, 0.0, 5.0)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i];color="black", colorrange)
end
fig

γ = 0.15 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.01:10) 
is1 = intensities_broadened(swt, path, energies, broadened_formula);
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)",title="Convoluted spectra:" , xticks)
ylims!(ax, 0.0, 5.0)
pl=heatmap!(ax, 1:size(is1, 1), energies, is1,colorrange=(0.0,2))
pl.colormap = Reverse(:Spectral)
fig
