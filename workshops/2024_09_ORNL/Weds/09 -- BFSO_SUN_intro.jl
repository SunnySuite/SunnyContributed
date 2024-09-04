# # 0. Introduction to the SU(_N_) formalism with Ba₂FeSi₂O₇
#
# Traditional classical and semiclassical approaches to spin systems start by
# placing an SU(2) coherent state on each site. An SU(2) coherent state may
# simply be thought of as a dipole, or as a state of a 2-level quantum system,
# that is, as a linear combination of combination of $\vert\uparrow\rangle$ and
# $\vert\downarrow\rangle$ states.

# The idea behind the SU(_N_) generalization of this approach is to instead put
# an SU(_N_) coherent state on each site. This is equivalent to having a state
# from an _N_-level system on each site, for example, some linear combination of
# $\vert\frac{N-1}{2}\rangle$, $\vert\frac{N-2}{2}\rangle$, $\ldots$, $\vert\frac{-(N-1)}{2}\rangle$.
# The expectation of any relevant operator (including the dipole operators) can
# always be evaluated by taking an expectation value in this coherent state, as
# we will demonstrate concretely below.

# The chief advantage of this approach is that enables a richer description of
# the local physics. In particular, non-magnetic states can be represented
# directly and the physics of single-ion anisotropies can be modeled more
# faithfully. A useful showcase for this formalism the the square lattice
# antiferromagnet Ba₂FeSi₂O₇. This is is a quasi-2D system with effective $S=2$
# and strong easy-plane anisotropy. BFSO (as we will refer to it) has been
# studied using the SU(_N_) formalism in a number of recent studies, in
# particular the following:

# - S.-H. Do et al., "Decay and renormalization of a longitudinal mode...," [Nature Communications **12** (2021)](https://doi.org/10.1038/s41467-021-25591-7).
# - M. Lee et al., "Field-induced spin level crossings...," [PRB **107** (2023)](https://doi.org/10.1103/PhysRevB.107.144427).
# - S.-H. Do et al., "Understanding temperature-dependent SU(3) spin dynamics...," [npj quantum materials **5** (2023)](https://doi.org/10.1038/s41535-022-00526-7).

# # 1. Anisotropies and large spins 
#
# Before specifying the complete Hamiltonian, we'll consider a cartoon picture
# of the single-ion physics. The Hamiltonian for a single $S=2$ spin with single
# ion anisotropy is simply $\mathcal{H}_{\mathrm{SI}} = D(\hat{S}^z)^2$, where
# $\hat{S}^z$ is in the $S=2$ representation. We can use Sunny to represent this
# as a matrix.

## Import relevant libraries 
using Sunny, GLMakie, LinearAlgebra, Statistics

S = spin_matrices(2)  # Returns a vector of Sx, Sy, Sz
Sx, Sy, Sz = S        # Julia's "unpacking" syntax
## EXERCISE: Write the single-ion anisotropy (with D=1) and call it H_SI
## EXERCISE: How would you add a Zeeman term?

H_SI = Sz^2  # + h*μB*g*Sz

# The result is a diagonal matrix. Ordering of the basis elements is simply
# $\vert 2\rangle$, $\vert 1\rangle$, $\vert 0\rangle$, $\vert -1\rangle$, and
# $\vert -2\rangle$. Clearly the ground state is the $\vert 0 \rangle$, which
# is a non-magnetic state as we can quickly verify. First we'll write a complex
# vector to represent the ground state:

Z = [0., 0, 1 + 0im, 0, 0]

# We can now use this to evaluate the expectation values of the dipole operators.
expectation(op, Z) = real(Z' * op * Z)
sx = expectation(Sx, Z) 
sy = expectation(Sy, Z) 
sz = expectation(Sz, Z) 

## EXERCISE: Examine the expectation value of a quadrupolar operator.

# This is obviously a non-magnetic state that cannot be represented as a
# classical dipole of fixed length $S$. The SU(_N_) formalism provides a way for
# modeling states like these and calculating their dynamics. To see this in
# action, we'll make a spin system with only this single-ion anisotropy and no
# other interactions. We'll start by constructing a primitive tetragonal
# lattice. 

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
positions = [[0, 0, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

# Next we'll create a spin system. This is just as in previous examples, only we
# will now set the mode to `:SUN`.

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=2)], :SUN; dims, seed=1)
set_onsite_coupling!(sys, Sz^2, 1) # Set the anisotropy term

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# Notice that the dipoles are very short -- but we expect them to be zero. This
# is because the tolerances of the `minimize_energy!` function were too loose.
# The tolerance can be made tighter by setting the keyword `g_tol` to a very
# small value, say, 1e-16.

randomize_spins!(sys)
minimize_energy!(sys; g_tol=1e-16)
plot_spins(sys)

# Notice now that no dipoles are visible. We can check this more carefully
# by examining the state of the spin system. Let's look at the first spin:

sys.dipoles[1,1,1,1]

# We see that this is close enough to zero for all intents and purposes. We can
# also check the coherent state itself to see if it is equal to `Z=(0, 0, 1, 0,
# 0)` up to overall phase. In SU(_N_) mode, this information is contained in the
# `System` field `coherents`:

sys.coherents[1,1,1,1]

## EXERCISE: Construct a primitive _cubic_ lattice and assign the same anisotropy.
## EXERCISE: See what happens when you perform the same procedure in `:dipole` mode.
## EXERCISE: See what happens when you change the sign of `D`. 

# The anisotropy of BFSO is more complicated than the above, but it has a
# predominantly easy-plane character -- though, from the exercise above, you may
# realize it would be more appropriate to call this a "hard-axis" anisotropy. If
# a Zeeman term is added to a hard-axis anisotropy, it induces a number of level
# crossings as the field is increased. When exchange interactions are added on
# top of this, we will find that the ground state evolves with field as a
# mixture of the $\left\vert 0\right\rangle$ and $\left\vert 1\right\rangle$
# states, into a mixture of the $\left\vert 0\right\rangle$ and $\left\vert
# 2\right\rangle$ states, until finally polarizing completely in the $\left\vert
# 2\right\rangle$ state.

# # 2. BFSO Hamiltonian specification
#
# We next turn to specification of the model Hamiltonian for BFSO. We'll start
# by defining a lattice for our magnetic Fe ions. This is a tetragonal lattice
# as above, but now well use real units. In particular, we'll have `a = b =
# 8.3194 Å` and `c= 5.336 Å`.

## EXERCISE: Specify this crystal
units = Units(:K, :angstrom)
a = 8.3194
c = 5.336
latvecs = lattice_vectors(a, a, c, 90, 90, 90)
positions = [[0, 0, 0]]
spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])
view_crystal(crystal)

# We use this `Crystal` to specify a `System`.

dims = (6, 6, 2)
sys = System(crystal, [1 => Moment(s=2, g=1.93)], :SUN; dims)

# The anisotropy is predominantly hard-axis, as we studied above, but includes a
# number of other terms that induce an in-plane XY-ordering as well.
#
# ```math
# \mathcal{H} = D\left(\hat{S}^z\right)^2 + A\left(\left(\hat{S}^x\right)^4 + \left(\hat{S}^y\right)\right) + C\left(\hat{S}^z\right)^4
# ````
#
# Here $A=1.16 K$, $C=-1.74 K$ and $D=28.65 K$. By default, Sunny using $meV$
# and $T$ for units. These values can be converted to $meV$ with the constant `units.K` 

## EXERCISE: Express the single ion anisotropy as a matrix and assign it to the `sys`.
A = 1.16
C = -1.74
D = 28.65

Sx, Sy, Sz = spin_matrices(2)
H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
set_onsite_coupling!(sys, H_SI, 1)

# The single-ion Hamiltonian was described as a polynomial in spin operators.
# Oftentimes one has access to a description of the crystal-field Hamiltoniann
# in terms of Stevens' operators. Sunny also provides a function for generating
# these matrices, very similar to `spin_matrices`. It returns a 2-dimensional
# array, where the first index corresponds to $k$ (irrep label) and the second
# to $q$ (row label):

O = stevens_matrices(2)
O[2,-2]
O[2,-1]
O[2,0]
O[2,1]
O[2,2]

# Note that the indexing for $q$ ranges from $-k$ to $k$. We will not use these
# further, but this is a useful resource.
#
# We next turn to defining the exchange interactions. We will define three
# antiferromagnetic Heisenberg couplings: nearest-neighbor in-plane,
# next-nearest neighbor in-plane, and nearest out-of-plane.

## EXERCISE: Use `view_crystal` to identify bonds representative of the exchange
## interactions just mentioned.
view_crystal(crystal)

bond1 = Bond(1, 2, [0, 0, 0])
bond2 = Bond(1, 1, [1, 0, 0])
bond3 = Bond(1, 1, [0, 0, 1])

J = 1.028 * units.K
J′ = 0.1J
set_exchange!(sys, J, bond1)
set_exchange!(sys, J′, bond2)
set_exchange!(sys, J′, bond3)

# We have now completely specified our Hamiltonian. Let's examine the zero-field
# ground state.

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# We clearly see an a staggered XY-ordering in the plane.

## EXERCISE: Examine the `dipoles` and `coherents` fields. 
## EXERCISE: using `set_external_field!` to see how the ground state develops with applied field.
## EXERCISE: Write a function that takes dimensions and returns a `System` for BFSO.

function BFSO(dims; mode=:SUN, seed=1)
    a = 8.3194
    c = 5.336
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    spacegroup = 113    # Want to use the space group for original lattice, of which the Fe ions form a subcrystal
    crystal = Crystal(latvecs, positions, spacegroup; types=["Fe"])

    sys = System(crystal, [1 => Moment(s=2, g=1.93)], mode; dims, seed)

    A = 1.16 * units.K
    C = -1.74 * units.K
    D = 28.65 * units.K

    Sx, Sy, Sz = spin_matrices(2)
    H_SI = D*(Sz)^2 + A*((Sx)^4 + (Sy)^4) + C*(Sz)^4
    set_onsite_coupling!(sys, H_SI, 1)

    bond1 = Bond(1, 2, [0, 0, 0])  
    bond2 = Bond(1, 1, [1, 0, 0]) 
    bond3 = Bond(1, 1, [0, 0, 1])

    J = 1.028 * units.K
    J′ = 0.1J
    set_exchange!(sys, J, bond1)
    set_exchange!(sys, J′, bond2)
    set_exchange!(sys, J′, bond3)

    return sys
end

# # 3. M vs. H
# Let's now proceed to generate a figure of $M$ vs $H$ for a range of field
# values. First we'll define a function to calculate the magnetization per site.

function magnetization(sys)
    nsites = prod(size(sys.dipoles))
    M_avg = sum(magnetic_moment(sys, site) for site in eachsite(sys)) / nsites
    return norm(M_avg)
end

magnetization(sys)

# We'll also define a function to calculate the relevant order parameter,
# which in this case is staggered magnetization in the plane. 

function order_parameter(sys)
    xy1 = [1/√2, 1/√2, 0]   # Unit vector in the (1, -1, 0) direction
    xy2 = [-1/√2, 1/√2, 0]  # Unit vector in the (1, 1, 0) direction
    M_xy1 = 0.0
    M_xy2 = 0.0
    for site in eachsite(sys)
        sublattice = (-1)^(site.I[4]) * (-1)^(site.I[3])  
        M_xy1 = sublattice * (magnetic_moment(sys, site) ⋅ xy1)
        M_xy2 = sublattice * (magnetic_moment(sys, site) ⋅ xy2)
    end
    return max(abs(M_xy1), abs(M_xy2))
end

order_parameter(sys)

# Then we'll simply generate a list of applied field values and iteratively
# apply those fields, reoptimizing the spin configuration each time.

Hs = range(0.0, 55.0, 50)
Ms = Float64[]
OPs = Float64[]
for H in Hs
    set_field!(sys, (0, 0, H*units.T))
    minimize_energy!(sys)
    push!(Ms, magnetization(sys))
    push!(OPs, order_parameter(sys))
end

fig = Figure(size=(1200,400))
scatter(fig[1,1], Hs, Ms; axis=(xlabel="H", ylabel="M"))
scatter(fig[1,2], Hs, OPs; axis=(xlabel="H", ylabel="Staggered XY Magnetization"))
fig

# These results agree very well with those reported in the Lee et al. paper
# listed above.