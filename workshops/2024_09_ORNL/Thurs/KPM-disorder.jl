using Sunny, GLMakie, LinearAlgebra, Random
"""
This example calculates that neutron scattering spectrum of a S=1/2 triangular lattice in an applied magnetic
field along the z-axis. This is a simple example inspired by the more complicated case of YbMgGaO4 which is 
discussed in Paddison et al, Nature Physics, 13, 117–122 (2017) and Zhu et al, PRL 119, 157201 (2017). The 
presence of non-magnetic disorder of the the Mg/Ga occupancy leads to a distribution of exchange constants and g-factors. Here we 
take a normal distribution of J1 with a standard deviation of 1/3. The g-factors are also normally distributed with a standard deviation of 1/6. 
"""
# Set up minimal system and exchange interactions
latvecs = lattice_vectors(1, 1, 20.00, 90, 90, 120);
pos = [[0,0,0]]
cryst = Crystal(latvecs, pos)
infos = [ 1 => Moment(s=0.5, g=1)]
dims = (3,3,1)
sys = System(cryst, infos, :dipole; dims, seed=0)
J₁ = diagm([1,1,1])
set_exchange!(sys,J₁,Bond(1,1,[1,0,0]))
randomize_spins!(sys)
# minimize energy
minimize_energy!(sys)

# plot spins
plot_spins(sys;color = [s[3] for s ∈ sys.dipoles])
# reduce the size of the system according to the magnetic unit cell
print_wrapped_intensities(sys)
suggest_magnetic_supercell([[1/3,1/3,0],[-1/3,-1/3,0]])
sys_min = reshape_supercell(sys, [1 0 0; -1 3 0; 0 0 1])
randomize_spins!(sys_min)
minimize_energy!(sys_min)
# Set up (Q,E) grid
Γ = [0,0,0]
K = [1/3,1/3,0]
M = [1/2,0,0]
nqs = 150
path = q_space_path(cryst, [Γ, K, M, Γ], nqs)
Emax = 3.
nEs = 150
σ = 0.1
energies = range(0, Emax, nEs)
# create a standard SpinWaveTheory object
swt1 = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))
spec1 = intensities(swt1, path; energies, kernel=lorentzian(fwhm=2σ))
# Plot intensities
plot_intensities(spec1)

# The spectrum shows sharp modes associated with coherent excitations about the 120° ordered state.

"""
Now we introduce nonmagnetic site disorder. This will manifest in two ways, which we explore in turn. 
Firstly, we will introduce a distribution of exchange constants. Secondly, we will introduce a distribution 
of g-factors. In order to do this, we must create a large, inhomogeneous system and then assign exchange 
constants and g-factors to each site. 
"""

# Convert to inhomogeneous system of size (lenx,leny,1)
lenx = leny = 40
dims = (lenx,leny,1)
sys_big = resize_supercell(sys,dims)
sys_inhom = to_inhomogeneous(sys_big)
rng = MersenneTwister(234)

for (site1, site2, offset) in symmetry_equivalent_bonds(sys_inhom, Bond(1,1,[1,0,0]))
    set_exchange_at!(sys_inhom, 1.0+randn(rng)/3, site1, site2; offset)
end

# Finally re-minimize the energy.
randomize_spins!(sys_inhom)

minimize_energy!(sys_inhom,maxiters=5_000)

# Set up (Q,E) grid
Γ = [0,0,0]
K = [1/3,1/3,0]
M = [1/2,0,0]
nqs = 150
path = q_space_path(cryst, [Γ, K, M, Γ], nqs)
Emax = 4
nEs = 150
energies = range(0, Emax, nEs)

# Calculate the intensities
σ = 0.025*Emax
swt2 = SpinWaveTheoryKPM(sys_inhom; measure=ssf_perp(sys_inhom), resolution = 0.1)
spec2 = intensities(swt2, path; energies, kT=0.1, kernel=lorentzian(fwhm=2σ))
# Plot intensities
plot_intensities(spec2)

"""
A broad continuum of excitations is observed, which is a signature of the distribution of exchange constants.
We will now apply a magnetic field long z to polarize the spins. 
"""

set_field!(sys_inhom, 7.5*[0,0,1.0]) # apply a large field along the z-axis
randomize_spins!(sys_inhom)

minimize_energy!(sys_inhom;)

Emax = 8.
nEs = 150
σ = 0.2
energies = range(0, Emax, nEs)
swt3 = SpinWaveTheoryKPM(sys_inhom; measure=ssf_perp(sys_inhom), resolution = 0.1)
spec3 = intensities(swt3, path; energies, kT=0.1, kernel=lorentzian(fwhm=2σ))
# Plot intensities
plot_intensities(spec3)

"""
The spectrum shows a sharp mode which disperse from the zone center but which broadens into a 
continuum at the zone boundary. We will now introduce a distribution of g-factors.
"""

rng = MersenneTwister(1643);
for site in eachsite(sys_inhom)
    sys_inhom.gs[site] = Diagonal([1, 1, 1+randn(rng)/6])
end

randomize_spins!(sys_inhom)

minimize_energy!(sys_inhom,maxiters=5_000 )

swt4 = SpinWaveTheoryKPM(sys_inhom; measure=ssf_perp(sys_inhom), resolution = 0.1)
spec4 = intensities(swt4, path; energies, kT=0.1, kernel=lorentzian(fwhm=2σ))
# Plot intensities
plot_intensities(spec4)

"""
Finally, we will compare with the non-disordered case. 
"""

set_field!(sys_min, 7.5*[0,0,1.0])
randomize_spins!(sys_min)
minimize_energy!(sys_min)
# create a standard SpinWaveTheory object

swt5 = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))
spec5 = intensities(swt5, path; energies, kT=0.1, kernel=lorentzian(fwhm=2σ))
# Plot intensities
plot_intensities(spec5)

```
A single coherent mode is visible.
```