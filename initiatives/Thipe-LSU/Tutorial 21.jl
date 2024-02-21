
using Sunny,GLMakie,CairoMakie

a = 12.3563;
S=5/2;
latvecs = lattice_vectors(a, a, a, 90, 90, 90);
positions=[[0, 0, 0],[0.375, 0.0, 0.25]];
types=["A","B"];
cryst = Crystal(latvecs,positions,230;types);
#GLMakie.activate!();
#view_crystal(cryst)

print_symmetry_table(cryst,8)
sys=System(cryst, (1,1,1),[SpinInfo(1,S=1,g=1.0),SpinInfo(23,S=1,g=1.0)], :dipole, seed=0);


Jad = 59.9232/sqrt(S*(S+1))
Jdd = 20.22408/sqrt(S*(S+1))
Jaa = 5.74264/sqrt(S*(S+1))

set_exchange!(sys,Jad,Bond(2, 17, [0, 0, 0]))
set_exchange!(sys,Jdd,Bond(17, 21, [0, 0, 0]))
set_exchange!(sys,Jaa,Bond(2, 5, [0, 0, 0]))
set_exchange!(sys,Jaa,Bond(1, 5, [0, 0, 0]))
set_external_field!(sys,[0 0 0.01/(0.05788)])
sys_res=reshape_supercell(sys, [1/2 1/2 -1/2;-1/2 1/2 1/2;1/2 -1/2 1/2])

randomize_spins!(sys_res);
minimize_energy!(sys_res;maxiters=1000)
plot_spins(sys_res)

q_points=[[1.5,2.5,3.0],[1,2,3],[1,2,4]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt = SpinWaveTheory(sys_res);
formula = intensity_formula(swt, :full; kernel=delta_function_kernel);
disp, intensity = intensities_bands(swt, path, formula);



γ = 1;
broadened_formula = intensity_formula(swt, :full; kernel=lorentzian(γ));
energies = collect(0:0.02:140);
is = intensities_broadened(swt, path, energies, broadened_formula)

###calculating the difference Sxy-Sxy

is1=zeros(ComplexF64,length(path),length(energies));
for i  in 1:length(path)
    for j in 1:length(energies)
        is1[i,j] = is[i,j][1,2]-is[i,j][2,1]
    end
end 

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (Thz)", title="Convoluted spectra: ", xticks);
ylims!(ax, 0.0,28.0);
pl=heatmap!(ax, 1:size(is, 1), energies./4.136,imag(is1),colorrange=(-0.05,0.05));
pl.colormap = Reverse(:redsblues);
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]/4.136; color="black")
end
fig





#=
CairoMakie.activate!()

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="YIG low temperature spin wave spectrum", xticks)
ylims!(ax, 0.0, 30)
xlims!(ax, 1, size(disp, 1))
#colorrange = extrema(intensity)
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]/4.136; colorrange)
end
fig

=#








