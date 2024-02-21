
using Sunny,GLMakie,CairoMakie


a = 5.2821/sqrt(2);
b = 5.6144/sqrt(2);
c = 7.5283;

latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[0, 0, 0],[0,0,1/2]]
V = Crystal(latvecs,positions,1)
GLMakie.activate!()
view_crystal(V)
print_symmetry_table(V,8)
sys=System(V, (2,2,1),[SpinInfo(1,S=1/2,g=2),SpinInfo(2,S=1/2,g=2)], :dipole, seed=0)



Jab   = 2.6;
Jc    = 3.1;
delta = 0.35;
K1    = 0.90;
K2    = 0.97;
d     = 1.15;

theta = 1/2*atan(2*d/(2*Jc-K1-K2))

Jc1=[-Jc*(1+delta)+K2 0 -d;0 -Jc*(1+delta) 0; +d 0 -Jc*(1+delta)]
Jc2=[-Jc*(1-delta)+K2 0 d;0 -Jc*(1-delta) 0; -d 0 -Jc*(1-delta)]

set_exchange!(sys,Jab,Bond(1, 1, [1, 0, 0]))
set_exchange!(sys,Jab,Bond(2, 2, [1, 0, 0]))
set_exchange!(sys,Jc1,Bond(1, 2, [0, 0, 0]))
set_exchange!(sys,Jc2,Bond(2, 1, [0, 0, 1]))
set_exchange!(sys,Jab,Bond(1, 1, [0, 1, 0]))
set_exchange!(sys,Jab,Bond(2, 2, [0, 1, 0]))


#S=spin_matrices(1/2)
set_onsite_coupling!(sys, S -> -K1*S[1]^2, 1)
set_onsite_coupling!(sys, S -> -K1*S[1]^2, 2)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


q_points=[[0.75,0.75,0],[0.5,0.5,0],[0.5,0.5,1]]
density = 200
path, xticks = reciprocal_space_path(V, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Spin wave dispersion: ")
ylims!(ax, 0, 15)
xlims!(ax, 1, size(disp, 1))
colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black",colorrange)
end
fig

γ = 0.02 
broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))
energies = collect(0:0.02:25) 
is = intensities_broadened(swt, path, energies, broadened_formula);
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Convoluted spectra: ")
ylims!(ax, 0.0, 25.0)
pl=heatmap!(ax, 1:size(is, 1), energies, is,colorrange=(0,1))
pl.colormap = Reverse(:Spectral)
fig