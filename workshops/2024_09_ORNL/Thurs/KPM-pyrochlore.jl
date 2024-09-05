using Sunny, LinearAlgebra, GLMakie, Random

"""
This example calculates that neutron scattering spectrum of the pyrochlore antiferromagnet.
In the Coulomb phase, the energy is mimimized for if all of the spins on each tetrahedron
sum to zero.
"""

latvecs    = lattice_vectors(5, 5, 5, 90, 90, 90)
positions  = [[0,0,0]]
spacegroup = 227 # Space Group Number
setting    = "2" # Space Group setting
cryst = Crystal(latvecs, positions, spacegroup; setting)

dims = (5, 5, 5)  # Supercell dimensions
spininfos = [1 => Moment(; s=1, g=2)]  # Specify spin information, note that all sites are symmetry equivalent
sys = System(cryst, spininfos, :dipole;dims); # Build system

view_crystal(cryst)

# Set exchange interaction
J₁ = 1.0
set_exchange!(sys, J₁, Bond(1, 2, [0,0,0]))

# Randomize spins
randomize_spins!(sys)
# minimize the energy
minimize_energy!(sys;maxiters = 5000) # specify a large number of maxiters to ensure that the the optimization converges.
plot_spins(sys;color = [s[3] for s in sys.dipoles])
nqs = 100
path = q_space_path(cryst, [[0,0,2],[2,2,2],[0,0,0]], nqs)
Emax = 6
nEs = 120 
energies = range(0, Emax, nEs)
σ = 0.2
swt = SpinWaveTheoryKPM(sys; measure=ssf_perp(sys), resolution = 0.1)
spec_QE = intensities(swt, path; energies, kT=0.1, kernel=lorentzian(fwhm=2σ))
plot_intensities(spec_QE)

"""
Now we calculate a constant energy slice at 0.25 meV. To do this, we build a grid of 
momenta.
"""

nqx = nqy = 30
grid = q_space_grid(cryst, [1, 1, 0], range(-3, 3, nqx), [0, 0, 1], range(-3, 3,nqy))
spec_QQ = intensities(swt, grid; energies=[0.25], kT=0.1, kernel=lorentzian(fwhm=2σ))
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "[H,H,0]", ylabel = "[0,0,L]") 
    heatmap!(ax,range(-5, 5, nqx),range(-5, 5, nqy),reshape(spec_QQ.data,nqx,nqy))
    fig
end

"""
Note that this is slower than the calculation for the QE slice. This is because the calculating 
the Chebyshev coefficients is very cheap. It should be noted that this still offers a considerable
speedup over standard LSWT.
"""