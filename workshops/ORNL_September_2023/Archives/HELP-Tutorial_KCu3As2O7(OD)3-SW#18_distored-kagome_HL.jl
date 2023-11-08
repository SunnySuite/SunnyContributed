# ![](https://raw.githubusercontent.com/SunnySuite/Sunny.jl/main/assets/sunny_logo.jpg)
# _This is a [tutorial](https://github.com/SunnySuite/SunnyTutorials/tree/main/tutorials)
#  for the [Sunny](https://github.com/SunnySuite/Sunny.jl/) package,
#  which enables dynamical simulations of ordered and thermally disordered spins with dipole
#  and higher order moments._


## Welcome to a Sunny Tutorial on the Distorted Kagome Lattice System 
#  KCu<sub>3</sub>As<sub>2</sub>O<sub>7</sub>(OD)<sub>3</sub>
# **Script**: Distorted Kagome Lattice Calculation <br>
# **Inspired by**:KCu<sub>3</sub>As<sub>2</sub>O<sub>7</sub>(OD)<sub>3</sub> SpinW tutorial 
# (Gøran Nilsen and Sandor Toth https://spinw.org/tutorials/18tutorial). Parameters taken from 
# Nilsen et al. https://doi.org/10.1103/PhysRevB.89.140412 <br>
# **Authors**: Harry Lane <br>
# **Date**: September 11, 2023  (Sunny 0.5.4) <br>
# **Goal**: This script is to calculate the linear spin wave theory spectrum for 
# KCu<sub>3</sub>As<sub>2</sub>O<sub>7</sub>(OD)<sub>3</sub> and compare with the results from SpinW.  


# ---
# #### Loading Packages 
using Sunny, GLMakie, LinearAlgebra

# ---
# ### System Definition for  KCu<sub>3</sub>As<sub>2</sub>O<sub>7</sub>(OD)<sub>3</sub> 
# Set up a [`Crystal`](@ref) using the parameters from 
# <a href="https://doi.org/10.1103/PhysRevB.89.140412 ">Nilsen et al., Phys. Rev. B **89**, 140412(R)</a>.

a = 10.20 # (Å)
b = 5.94
c = 7.81
lat_vecs = lattice_vectors(a, b, c, 90, 117.7, 90)
basis_vecs = [[0,0,0],[1/4,1/4,0]]
spgr = 12
basis_types = ["Cu","Cu"]
crystal = Crystal(lat_vecs, basis_vecs, spgr; types=basis_types, setting="b1")

# The next step is to add interactions. The command [`print_symmetry_table`](@ref) shows 
# all symmetry-allowed interactions up to a cutoff distance.

print_symmetry_table(crystal,7.1)

# ---
# ### Fix the ground-state of the system

# The magnetic structure of KCu<sub>3</sub>As<sub>2</sub>O<sub>7</sub>(OD)<sub>3</sub> has the 
# incommensurate ordering wavevector $\mathbf{Q}\approx(0.77,0,0.11)$.
# We supply the experimentally determined wavevector and allow it to be rationalized. 
# Here we employ the helical mode to propagate the magnetic structure within the unit cell 
# using the fractional positions.

Slist=[[1,0,0]]
n=[0,0,1]
kvec=[0.77, 0, 0.115]
S=1/2
Spinfo= [SpinInfo(1; S, g=2), SpinInfo(3; S, g=2)]
sys = System(crystal,(1,1,1),Spinfo,:dipole)

set_spiral_order_on_sublattice!(sys, 1 ;q=kvec,axis=n,S0=Slist[1])
set_spiral_order_on_sublattice!(sys, 3 ;q=kvec,axis=n,S0=Slist[1])
ps=plot_spins(sys)
