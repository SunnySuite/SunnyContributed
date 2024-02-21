
using Sunny,GLMakie,CairoMakie


a = 4;
b = 4;
c = 3;
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[0,0,0]]
cryst = Crystal(latvecs,positions,1)
GLMakie.activate!()
view_crystal(cryst)
print_symmetry_table(cryst,8)
sys=System(cryst, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J_1=[-3 0 0;0 -4 0;0 0 -5]
set_exchange!(sys,J_1,Bond(1, 1, [1, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

q_points=[[0,0,0],[1,0,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt=SpinWaveTheory(sys)
formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="intensity",title="Spin-spin correlation")
ylims!(ax, 0, 0.8)
xlims!(ax, 1, size(disp, 1))

for i in axes(disp)[2]
    lines!(ax, 1:length(intensity[:,i]), intensity[:,i]; color="black")
end
fig


####anisotropic exchange on AFM chain
sys2=System(cryst, (2,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)
J_2=[3 0 0;0 4 0;0 0 4.1]

set_exchange!(sys2,J_2,Bond(1, 1, [1, 0, 0]))
randomize_spins!(sys2)
minimize_energy!(sys2)
plot_spins(sys2)

q_points=[[0,0,0],[1,0,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt=SpinWaveTheory(sys2)
formula = intensity_formula(swt, :full; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

γ = 0.1;
broadened_formula = intensity_formula(swt, :full; kernel=lorentzian(γ));
energies = collect(0:0.02:10);
is = intensities_broadened(swt, path, energies, broadened_formula)
###calculating Sxx,Syy,Szz
is1=zeros(ComplexF64,length(path),length(energies));
is2=zeros(ComplexF64,length(path),length(energies));
is3=zeros(ComplexF64,length(path),length(energies));
for i  in 1:length(path)
    for j in 1:length(energies)
        is1[i,j] = is[i,j][1,1]#Sxx
        is2[i,j] = is[i,j][2,2]#Syy
        is3[i,j] = is[i,j][3,3]#Szz
    end
end 

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="energy",title="Convoluted spectra: Re(Sxx)")
heatmap!(ax, 1:size(is, 1), energies,is1,colorrange=(0,1));
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
fig
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="energy",title="Convoluted spectra: Re(Syy)")
heatmap!(ax, 1:size(is, 1), energies,is2,colorrange=(0,1));
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
fig
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="energy",title="Convoluted spectra:Re(Szz)")
heatmap!(ax, 1:size(is, 1), energies,is3,colorrange=(0,1));
for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
fig



#####anisotropic exchange on the AFM chain rotated anisotropy
θ=45

rot_mat=[cos(θ*pi/180) -sin(θ*pi/180) 0;sin(θ*pi/180) cos(θ*pi/180) 0;0 0 1]
J_3=rot_mat*J_2*rot_mat'
sys3=System(cryst, (2,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)
set_exchange!(sys3,J_2,Bond(1, 1, [1, 0, 0]))
randomize_spins!(sys3)
minimize_energy!(sys3)
plot_spins(sys3)

q_points=[[0,0,0],[1,0,0]]
density = 200
path, xticks = reciprocal_space_path(cryst, q_points, density);
swt=SpinWaveTheory(sys3)
formula = intensity_formula(swt, :full; kernel=delta_function_kernel)
disp, intensity = intensities_bands(swt, path, formula);

γ = 0.1;
broadened_formula = intensity_formula(swt, :full; kernel=lorentzian(γ));
energies = collect(0:0.02:10);
is = intensities_broadened(swt, path, energies, broadened_formula)

is1=zeros(ComplexF64,length(path),length(energies));
is2=zeros(ComplexF64,length(path),length(energies));
###calculating Sxx+Sxy,Sxx-Sxy
for i  in 1:length(path)
    for j in 1:length(energies)
        is1[i,j] = is[i,j][1,1]+is[i,j][1,2]## Sxx+Sxy
        is2[i,j] = is[i,j][2,2]-is[i,j][1,2] ## Sxx-Sxy   
    end
end 

CairoMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="energy",title="Convoluted Spectra:Re(Sxx +Sxy)")
ylims!(ax, 0, 9)
xlims!(ax, 1, size(disp, 1))
pl=heatmap!(ax, 1:size(is, 1), energies,real(is1),colorrange=(-0.1,1));
#pl.colormap = Reverse(:redsblues)

for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
display(fig)
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="energy",title="Convoluted Spectra: Re(Sxx -Sxy)")
ylims!(ax, 0, 9)
xlims!(ax, 1, size(disp, 1))
pl=heatmap!(ax, 1:size(is, 1), energies,real(is2),colorrange=(-0.1,1));
#pl.colormap = Reverse(:redsblues)

for i in axes(disp)[2]
    lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black")
end
display(fig)











