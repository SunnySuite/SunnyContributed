using Sunny,GLMakie,CairoMakie,LinearAlgebra


a = 5.4097;
b = 5.4924;
c = 11.9613;

latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[0, 0, 0]]
cryst= Crystal(latvecs,positions,61;setting="")
GLMakie.activate!()
view_crystal(cryst)
print_symmetry_table(cryst,8)
sys=System(cryst, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)



J = 5.2;
a = 0.10*J;
e = 1;
E = 21.5;
X = 1.0;

Jmat=[J X 0;X J 0;0 0 J-a]
Amat=[e 0 0; 0 0 0; 0 0 E]
D=[e,0,E]

set_exchange!(sys,Jmat,Bond(1, 2, [0, 0, 0]))
S=spin_operators(sys,1)
set_onsite_coupling!(sys,S->2*e*S[1]^2+2*E*S[3]^2,1)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


q_points=[[0.5,0,0],[0.5,0.5,0],[1,0.0,0],[0,0,0],[0.5,0.5,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :full; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Spin wave dispersion", xticks)
ylims!(ax, 0.0, 60)
xlims!(ax, 1, size(disp, 1))
#colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
fig

γ = 1;
broadened_formula = intensity_formula(swt, :full; kernel=lorentzian(γ));
energies = collect(0:0.02:140);
is = intensities_broadened(swt, path, energies, broadened_formula);
is1=zeros(ComplexF64,length(path),length(energies));
for i  in 1:length(path)
    for j in 1:length(energies)
        is1[i,j] = is[i,j][1,1]+is[i,j][2,2]+is[i,j][3,3]##Sxx+Syy+Szz
    end
end 

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (Thz)", title="Convoluted spectra: ", xticks);
ylims!(ax, 0.0,60.0);
pl=heatmap!(ax, 1:size(is, 1), energies,real(is1),colorrange=(-0.01,0.05));
pl.colormap = Reverse(:redsblues);
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
fig