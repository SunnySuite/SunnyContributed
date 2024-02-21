
using Sunny, GLMakie, CairoMakie    

a=6.0
b=6.0
c=5.0


latvecs = lattice_vectors(a, b, c, 90, 90, 120) 
positions=[[1/2, 0, 0]]
types=["Cu1"]
Cu = Crystal(latvecs,positions,147;types)
GLMakie.activate!()
view_crystal(Cu,5)
sys=System(Cu, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)
print_symmetry_table(Cu,8.0)


J1=-1.0

set_exchange!(sys,J1,Bond(2, 3, [0, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


swt=SpinWaveTheory(sys)
q_points = [[-1/2,0,0], [0,0,0], [1/2,1/2,0]]
density = 50
path, xticks = reciprocal_space_path(Cu, q_points, density);
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);


CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)", title="Spin wave dispersion:", xticks)
ylims!(ax, 0.0, 8.0)
xlims!(ax, 1, size(disp, 1))

for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i];color="black")
end
fig

γ = 0.02 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.01:10)  
is1 = intensities_broadened(swt, path, energies, broadened_formula);

radii = 0.01:0.02:3 
output = zeros(Float64, length(radii), length(energies))
for (i, radius) in enumerate(radii)
    n = 300
    qs = reciprocal_space_shell(Cu, radius, n)
    is1 = intensities_broadened(swt, qs, energies, broadened_formula)
    output[i, :] = sum(is1, dims=1) / size(is1, 1)
end
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Q (Å⁻¹)", ylabel="ω (meV)",title="Convoluted powder spectra:")
ylims!(ax, 0.0, 7.0)
heatmap!(ax, radii, energies, output, colormap=:gnuplot2,colorrange=(0.01,0.5))
fig