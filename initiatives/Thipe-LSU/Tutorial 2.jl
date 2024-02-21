using Sunny, GLMakie, CairoMakie    

a=8.0
b=8.0
c=3.0

latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[0, 0, 0]]
types=["Cu1"]
cryst = Crystal(latvecs, positions; types)
view_crystal(cryst,10)
print_symmetry_table(cryst,8.0)
sys=System(cryst, (1,1,2),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=1

set_exchange!(sys,J1,Bond(1, 1, [0, 0, 1]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


swt=SpinWaveTheory(sys)
η = 0.4 # (meV)
kernel = lorentzian(η)
q_points = [[0,0,0],[0,0,1]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)", title ="Spin wave dispersion:",xticks)
ylims!(ax, 0.0, 2.5)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i];color="black", colorrange)
end
fig

int=log.(intensity)
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)", title="Intensity of the spin-spin correlation function:", xticks)
ylims!(ax, -3, 10)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(int[:,i]), int[:,i];color="black", colorrange)
end
fig
