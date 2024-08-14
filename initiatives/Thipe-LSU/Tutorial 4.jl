using Sunny, GLMakie, CairoMakie    

a=3.0
b=3.0
c=6.0

units=Units(:meV)
latvecs = lattice_vectors(a, b, c, 90, 90, 90) 
positions=[[0, 0, 0]]
types=["Cu1"]
Cu = Crystal(latvecs, positions; types)
cryst=subcrystal(Cu,"Cu1")
GLMakie.activate!()
view_crystal(cryst)

sys=System(cryst, (2,2,1),[SpinInfo(1,S=1,g=2)], :dipole, seed=0)

J1=1.0
J2=-0.1
set_exchange!(sys,J1,Bond(1, 1, [1, 0, 0]))
set_exchange!(sys,J2,Bond(1, 1, [1, 1, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)


swt=SpinWaveTheory(sys;measure=ssf_perp(sys))
q_points = [[0,0,0], [1/2,0,0], [1/2,1/2,0], [0,0,0]]
density = 400
path = q_space_path(cryst, q_points, density);
res = intensities_bands(swt, path)
CairoMakie.activate!()
Sunny.BandIntensities{Float64}
plot_intensities(res; units)


