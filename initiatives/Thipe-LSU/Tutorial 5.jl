
using Sunny, GLMakie, CairoMakie    

a=6.0
b=6.0
c=5.0

units=Units(:meV)
latvecs = lattice_vectors(a, b, c, 90, 90, 120) 
positions=[[1/2, 0, 0]]
types=["Cu1"]
Cu = Crystal(latvecs,positions,147;types)
GLMakie.activate!()
view_crystal(Cu,5)
sys=System(Cu, (1,1,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)


J1=-1.0

set_exchange!(sys,J1,Bond(2, 3, [0, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


swt=SpinWaveTheory(sys;measure=ssf_perp(sys))
q_points = [[-1/2,0,0], [0,0,0], [1/2,1/2,0]]
density = 400
path = q_space_path(Cu, q_points, density);
res = intensities_bands(swt, path)


CairoMakie.activate!()
Sunny.BandIntensities{Float64}
plot_intensities(res; units)

radii = range(0, 2.5, 400) # (1/Å)
energies = range(0, 6.5, 400)
kernel = gaussian(fwhm=0.1)
res = powder_average(Cu, radii, 1600) do q_points
    intensities(swt, q_points; energies, kernel)
end
plot_intensities(res; units, colorrange=(0,2.5))