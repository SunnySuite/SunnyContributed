# The goal is to calculate linear spinwave spectrum for the Kagome antiferromagnet 
# with easy plane anisotropy using incommensurate spiral structure.

# Load packages

using Sunny, CairoMakie, LinearAlgebra, Revise
using Optim
includet("DispersionAndIntensitiesDipoleSingleQ .jl")
includet("SWTSingleQcalculations.jl")

# Build a crystal

a = 6
b = 6
c = 40
latvecs = lattice_vectors(a,b,c,90,90,120)
positions = [[0.5,0,0]]
types = ["A"]
cryst = Crystal(latvecs,positions,147;types)

# Build a System with antiferrogmanetic nearest neighbor exchange J = 1.

sys = System(cryst,(1,1,1),[SpinInfo(1,S=1,g=2)], :dipole,seed =0)
bond1 = Bond(2,3,[0,0,0])
J1 = 1.
set_exchange!(sys,[J1 0. 0.;0 J1 0;0 0 J1],bond1)

# Axis of rotation, n = [0.0,0.0,1], along c -direction. 
# Also, we are constructing single ion anisotropy which has U(1) symmetry. 
# To generate onsite coupling, you can use either use construct_uniaxial_anisotropy function or 
# use simply anonmynous function as shown below.
# We can also check whether the interaction of system is invariant under axis of rotation using check_rotational_symmetry function.

n = [0.,0,1]
set_onsite_coupling!(sys, S -> (n'*S)^2, 1)

Sunny.check_rotational_symmetry(sys; n, θ=0.01)

# Now we need to find the magnetic ground state. optimagstr function helps us to 
# find the ground state and the propagation vector for that ground state.
# It uses a constraint function (@gm_planar in this case) to reduce the number of paramteres that has to be optimised. 
# It works well if the number of free parameters are low. we will find that the right k-vector is [1/3,1/3,0].

randomize_spins!(sys)
#       Phi1   Phi2  Phi3  k_x  k_y   k_z
xmin = [-1e-6 -1e-6 -1e-6 -1e-6 -1e-6 -1e-6] # Minimum value of x
xmax = [2π 2π 2π 1 1 1e-6]  # Maximum value of x
x0 = [0.01 0.01 0.01 0.01 0.01 0.0]  # Initial value of x
k = optimagstr(gm_planar!,xmin,xmax,x0,n)

# Define a path in reciprocal space.

qs = [[-0.5,0,0.],[0.0,0.,0],[0.5,0.5,0]]
density = 250
path, xticks = reciprocal_space_path(cryst, qs, density)


# Calculate Intensities

swt = SpinWaveTheory(sys)
γ = 0.02 # width in meV
broadened_formula = intensity_formula_SingleQ(swt,k,n, :perp; kernel=lorentzian(γ),formfactors=nothing)
energies = collect(0:0.005:3)  # 0 < ω < 10 (meV).
is = intensities_broadened_SingleQ(swt, path, energies, broadened_formula);

# The calculated intensity is a tensor with dimension [path×energies×nmodes×branch]. 
# Here, branch corresponds to K,K+Q and K-Q modes of incommensurate spin structure.

# Plotting of intensities using CairoMakie.

begin
    CairoMakie.activate!()
    fig = Figure() 
    ax = Axis(fig[1,1];title="SingleQ", xlabel="Momentum (r.l.u.)", ylabel="Energy (meV)", xticks, xticklabelrotation=0)
    pl = heatmap!(ax,1:size(is, 1),energies,dropdims(sum(is[:,:,1:3,1:3],dims=[3,4]),dims=(3,4)))
    pl.colormap = Reverse(:Spectral)
    pl.colorrange = (0,3)
    fig
end


  

