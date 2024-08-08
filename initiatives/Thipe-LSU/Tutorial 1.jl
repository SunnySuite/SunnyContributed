using Sunny, GLMakie, CairoMakie    

a=3.0
b=8.0
c=8.0

units=Units(:meV)
latvecs = lattice_vectors(a, b, c, 90, 90, 90) #defining the lattice
positions=[[0, 0, 0]]
types=["Cu1"]
cryst = Crystal(latvecs, positions; types)
#GLMakie.activate!()
#view_crystal(cryst,10)
#print_symmetry_table(cryst,8.0)
sys=System(cryst, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=-1

set_exchange!(sys,J1,Bond(1, 1, [1, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

swt=SpinWaveTheory(sys; measure=ssf_perp(sys))
q_points = [[0,0,0], [1,0,0]]
density = 200
path = q_space_path(cryst, q_points, density);
res = intensities_bands(swt, path)

CairoMakie.activate!()
Sunny.BandIntensities{Float64}
plot_intensities(res; units)

radii = range(0, 2.5, 400) # (1/Å)
energies = range(0, 5, 400)
kernel = gaussian(fwhm=0.1)
res = powder_average(cryst, radii, 1600) do q_points
    intensities(swt, q_points; energies, kernel)
end
plot_intensities(res; units, colorrange=(0,5))



