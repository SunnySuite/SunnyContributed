using Sunny,GLMakie, CairoMakie, LinearAlgebra
using Revise


a=10.02
b=5.86
c=4.68

latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions=[[1/4, 1/4, 0]]
Ni = Crystal(latvecs,positions,62,setting="");
GLMakie.activate!()
view_crystal(Ni,6)
print_symmetry_table(Ni,8)
sys=System(Ni, (1,2,2),[SpinInfo(1,S=3/2,g=2)], :dipole, seed=0)

Jbc =  1.036;
Jb  =  0.6701;
Jc  = -0.0469;
Jac = -0.1121;
Jab =  0.2977;
Da  =  0.1969;
Db  =  0.9097;
Dc  =  0;

set_exchange!(sys,Jbc,Bond(2, 3, [0, 0, 0]))
set_exchange!(sys,Jc,Bond(1, 1, [0, 0, -1]))
set_exchange!(sys,Jb,Bond(1, 1, [0, 1, 0]))
set_exchange!(sys,Jab,Bond(1, 2, [0, 0, 0]))
set_exchange!(sys,Jab,Bond(3, 4, [0, 0, 0]))
set_exchange!(sys,Jac,Bond(3, 1, [0, 0, 0]))
set_exchange!(sys,Jac,Bond(4, 2, [0, 0, 0]))
S=spin_operators(sys,1)
set_onsite_coupling!(sys,Da*S[1]^2+Db*S[2]^2,1)
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)



q_points=[[[0,1,0],[2,1,0]],[[0,0,0],[0,2,0]],[[0,1,0],[0,1,2]]]
for i in 1:length(q_points)
    density = 200
    path, xticks = reciprocal_space_path(Ni, q_points[i], density);
    swt=SpinWaveTheory(sys)
    formula = intensity_formula(swt, :perp; kernel=delta_function_kernel)
    disp, intensity = intensities_bands(swt, path, formula);

    CairoMakie.activate!()
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", title="Convoluted spectra",xticks)
    ylims!(ax, 0.0, 12)
    xlims!(ax, 1, size(disp, 1))
    fig
    γ = 0.02 # width in meV
    broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(γ))

    energies = collect(0:0.02:12)  # 0 < ω < 10 (meV).
    is = intensities_broadened(swt, path, energies, broadened_formula);
    fig
    pl=heatmap!(ax, 1:size(is, 1), energies, is,colorrange=(0,1))
    pl.colormap = Reverse(:Spectral)
    
    colorrange = extrema(intensity)
    for i in axes(disp)[2]
        lines!(ax, 1:length(disp[:,i]), disp[:,i]; color="black", colorrange)
    end
    display(fig)

   
end







