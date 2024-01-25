# The goal is to find spinwave spectrum using incommensurate magnetic structure.
# The crystal information is taken from: [[http://dx.doi.org/10.1103/PhysRevLett.101.247201].
# The exchange parameters are taken from: [[http://dx.doi.org/10.1103/PhysRevLett.106.207201]].

# load packages

using Sunny, CairoMakie, LinearAlgebra, Revise
using Optim , Optimization, OptimizationOptimJL
includet("DispersionAndIntensitiesDipoleSingleQ.jl")
includet("SWTSingleQcalculations.jl")

# Build a crystal
a = b = 8.539 # (Å)
c = 5.2414
latvecs = lattice_vectors(a, b, c, 90, 90, 120)
types = ["Fe", "Nb", "Ba", "Si", "O", "O", "O"]
positions = [[0.24964,0,0.5], [0,0,0], [0.56598,0,0], [2/3,1/3,0.5220], [2/3,1/3,0.2162], [0.5259,0.7024,0.3536], [0.7840,0.9002,0.7760]]
langasite = Crystal(latvecs, positions, 150; types)
cryst = subcrystal(langasite, "Fe")
latsize = (1,1,1)
S = 5/2
seed = 5
sys = System(cryst, latsize, [SpinInfo(1; S, g=2)], :dipole)

# Set exchange interaction parameters
J₁ = 0.85
J₂ = 0.24
J₃ = 0.053
J₄ = 0.017
J₅ = 0.24
set_exchange!(sys, J₁, Bond(3, 2, [1,1,0]))
set_exchange!(sys, J₄, Bond(1, 1, [0,0,1]))
set_exchange!(sys, J₂, Bond(1, 3, [0,0,0]))

# We define three chiral properties of the crystal and magnetic order:
# * epsilon_T: crystal chirality, in our case we select J3-J5 interactions according to the chirality (J5>J3). * epsilon_Delta: 
#Chirality of the triangular units. * epsilon_H: Sense of rotation of the spin helices along the c-axis (right handed rotation is positive).
# The three property are related: eT = eD*eH.

ϵD = -1
ϵH = +1
ϵT = ϵD * ϵH

if ϵT == -1
    set_exchange!(sys, J₃, Bond(2, 3, [-1,-1,1]))
    set_exchange!(sys, J₅, Bond(3, 2, [1,1,1]))
elseif ϵT == 1
    set_exchange!(sys, J₅, Bond(2, 3, [-1,-1,1]))
    set_exchange!(sys, J₃, Bond(3, 2, [1,1,1]))
else
    throw("Provide a valid chirality")
end

# Define axis of rotation
n = [0.,0.,1]

# Optimizing crystal structure

xmin = [-1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6] # Minimum value of x
xmax = [2π 2π 2π 1e-6 1e-6 1]  # Maximum value of x
x0 = [3.2 0.4 3.2 0. 0. 0.3]  # Initial value of x
k = optimagstr(gm_planar!,xmin,xmax,x0,n)

# Path in Reciprocal space.
points_rlu = [[0,1,-1],[0,1,-1+1],[0,1,-1+2],[0,1,-1+3]];
density = 100
path, xticks = reciprocal_space_path(cryst, points_rlu, density);

#Calculating spin wave spectrum 

swt = SpinWaveTheory(sys)
γ = 0.15 
energies = collect(0:0.01:6)
broadened_formula = intensity_formula_SingleQ(swt,k,n, :perp; kernel=lorentzian(γ),formfactors=nothing)
is = intensities_broadened_SingleQ(swt, path, energies, broadened_formula);

# Plotting the intensities.

begin
    CairoMakie.activate!()
    fig = Figure() 
    ax = Axis(fig[1,1];title="SingleQ", xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", xticks, xticklabelrotation=0)
    pl = heatmap!(ax,1:size(is, 1),energies,dropdims(sum(is[:,:,:,1:3],dims=[3,4]),dims=(3,4)))
    pl.colormap = Reverse(:Spectral)
    pl.colorrange = (0,10)
    fig
end