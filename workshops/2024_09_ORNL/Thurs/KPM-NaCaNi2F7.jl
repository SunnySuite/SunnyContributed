using Sunny, GLMakie, LinearAlgebra, Random
"""
This example calculates that neutron scattering spectrum of the S=1 pyrochlore antiferromagnet NaCaNi2F7, as  
reported in Zhang et al, PRL 122, 167203 (2019) and Plumb et al, Nature Physics 15, 54-59 (2019). The spectrum  
can be understood as originating from harmonic fluctuations about a disordered ground state. It is therefore
possible to reproduce the data by performing LSWT calculations about a disordered low-energy spin 
configuration. It is necessary to begin with a large enough system to capture the disordered nature of the
system.  
"""

# Set up crystal
latvecs = lattice_vectors(10.3, 10.3, 10.3, 90, 90, 90);
pos = [[0,0,0]]
cryst = Crystal(latvecs, pos,227;setting="2")
infos = [1 => Moment(s=1.0, g=2)] 
dims = (5,5,5)
sys = System(cryst, infos, :dipole; dims, seed=0)
# Set up exchange interactions
J₁ = J₂ = 3.2
J₃ = 0.019
J₄ = -0.070
Jnn = [J₂ J₄ J₄ ;
        -J₄ J₁ J₃;
        -J₄ J₃ J₁]

Jnnn = -0.025
set_exchange!(sys,Jnn,Bond(1,6,[0,0,0])) # pick the J01 bond from Zhang et al, PRL 122, 167203 (2019)
set_exchange!(sys,Jnnn,Bond(3,5,[0,0,0]))

# Set up a grid in (Q,E)
Q₁ = [2,2,-2]
Q₂ = [2,2,0]
Q₃ = [2,2,2]
Q₄ = [0,0,2]
Q₅ = [-2,-2,2]
nqs = 120
path = q_space_path(cryst, [Q₁,Q₂,Q₃,Q₄,Q₅], nqs)
Emax = 12
nEs = 150 
energies = range(0, Emax, nEs)

σ = 0.025*Emax # energy resolution

"""
For smaller unit cells it may be necessary to loop over realizations of minimum energy
configurations to capture the disordered nature of the system. However, for large unit cell
sizes, a simple realization is typically sufficient.
"""

randomize_spins!(sys)  
minimize_energy!(sys;maxiters=10_000)
swt = SpinWaveTheoryKPM(sys; measure=ssf_perp(sys), resolution = 0.2)
res = intensities(swt, path; energies, kernel=lorentzian(fwhm=2σ))

# plot the intensity
plot_intensities(res;colormap = :viridis)
