using Sunny,GLMakie,CairoMakie


a = 10.0307
latvecs = lattice_vectors(a, a, a, 90, 90, 90)
positions=[[1/2, 1/2, 1/2]]
types=["A"]
cryst = Crystal(latvecs,positions,227;types,setting="2")
g_tensor=(-0.84)*[1 1 1;1 1 1;1 1 1]+4.32*[1 0 0; 0 1 0; 0 0 1]
GLMakie.activate!()
view_crystal(cryst)
print_symmetry_table(cryst,8)
sys=System(cryst, (1,1,1),[SpinInfo(11,S=1/2,g=g_tensor)], :dipole, seed=0)


J1 = -0.09; 
J2 = -0.22; 
J3 = -0.29; 
J4 = 0.01;

J_mat=[J2 J4 J4; 
       -J4 J1 J3; 
       -J4 J3 J1]

set_exchange!(sys,J_mat,Bond(1, 6, [0, 0, 0]))
Field=[5,2]

gaussian(x,η) = 1/(η/2.35482*sqrt(2π))*exp(-1/2*(x)^2/(η/2.35482)^2)
gaussian(η) = x -> gaussian(x,η)

CairoMakie.activate!()
fig1 = Figure(resolution=(1700, 750))
for i in 1:length(Field)
    set_external_field!(sys,Field[i]*[1 -1 0]/sqrt(2))
    randomize_spins!(sys)
    minimize_energy!(sys)
    q_points=[[[-0.5,-0.5,-0.5],[2,2,2]],[[1,1,-2],[1,1,1.5]],[[2,2,-2],[2,2,1.5]],[[-0.5,-0.5,0],[2.5,2.5,0]],[[0,0,1],[2.3,2.3,1]]]
    for j in 1:length(q_points)
        density = 200
        path, xticks = reciprocal_space_path(cryst, q_points[j], density);
        swt = SpinWaveTheory(sys)
        formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
        disp, intensity = intensities_bands(swt, path, formula);
        
        γ = 0.09 
        broadened_formula = intensity_formula(swt, :perp; kernel=gaussian(γ))
        energies = collect(0:0.02:12) 
        is = intensities_broadened(swt, path, energies, broadened_formula);
        ax = Axis(fig1[i,j]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", xticks)
        ylims!(ax, 0.0, 2.0)
        pl=heatmap!(ax, 1:size(is, 1), energies, is,color="black",colorrange=(0.0,40))
        pl.colormap = Reverse(:Spectral)
    end
    
end
display(fig1)
