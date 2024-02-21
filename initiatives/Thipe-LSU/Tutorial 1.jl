using Sunny, GLMakie, CairoMakie    

a=8.0
b=8.0
c=3.0

latvecs = lattice_vectors(a, b, c, 90, 90, 90) #defining the lattice
positions=[[0, 0, 0]]
types=["Cu1"]
cryst = Crystal(latvecs, positions; types)
GLMakie.activate!()
view_crystal(cryst,10)
print_symmetry_table(cryst,8.0)
sys=System(cryst, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=-1

set_exchange!(sys,J1,Bond(1, 1, [0, 0, 1]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

swt=SpinWaveTheory(sys)
η = 0.4
kernel = lorentzian(η)
q_points = [[0,0,0], [0,0,1]]
density = 50
path, xticks = reciprocal_space_path(cryst, q_points, density);
disp = dispersion(swt, path);
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);


CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Energy (meV)", title="Spin Wave Dispersion",xticks)
ylims!(ax, 0.0, 5)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; colorrange)
end
fig

γ = 0.15 # width in meV
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.01:10)  # 0 < ω < 10 (meV).
is1 = intensities_broadened(swt, path, energies, broadened_formula);
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)",title="Intensity of the spin-spin correlation function:", xticks)
ylims!(ax, 0.0, 2)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(intensity[:,i]), intensity[:,i];color="black", colorrange)
end
fig


radii = 0.01:0.02:3 # (1/Å)##powder averaging
output = zeros(Float64, length(radii), length(energies))
for (i, radius) in enumerate(radii)
    n = 300
    qs = reciprocal_space_shell(cryst, radius, n)
    is1 = intensities_broadened(swt, qs, energies, broadened_formula)
    output[i, :] = sum(is1, dims=1) / size(is1, 1)
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="ω (meV)", title="Convoluted powder spectra")
ylims!(ax, 0.0, 5.0)
heatmap!(ax, radii, energies, output, colormap=:gnuplot2,colorrange=(0.1,1))
fig

