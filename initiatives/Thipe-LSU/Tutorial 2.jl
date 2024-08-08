using Sunny, GLMakie, CairoMakie    

a=8.0
b=8.0
c=3.0
units=Units(:meV)
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


swt=SpinWaveTheory(sys; measure=ssf_perp(sys))
energies = range(0, 2.5, 400)

q_points = [[0,0,0],[0,0,1]]
density = 400
path = q_space_path(cryst, q_points, density);
res=intensities_bands(swt, path)

CairoMakie.activate!()
Sunny.BandIntensities{Float64}
plot_intensities(res; units)

int=log.(res.data[1,:])
fig = Figure()
ax = Axis(fig[1,1]; xlabel="Path(Å⁻¹)", ylabel="Intensity (meV)", title="Intensity of the spin-spin correlation function:")
ylims!(ax, -3, 10)
xlims!(ax, 0, 1)
reciprocalpath=LinRange(0,1,density)
lines!(ax, reciprocalpath, int;color="black")
fig
