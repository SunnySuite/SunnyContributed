# # Quadratic Casimirs

using Sunny, GLMakie, LinearAlgebra, SparseArrays

# ## Background: The Killing Form and its associated Quadratic Casimir
# 
# A Lie algebra $L$ is a vector space with a bracket operation $[\cdot,\cdot] : L \times L \to L$ satisfying certain properties.
# Since Lie algebra elements are often square matrices, it will be convenient to flatten them into vectors to treat them using linear algebra:

matrix_to_flat(M) = M[:]
flat_to_matrix(v) = reshape(v,Int64(sqrt(length(v))),Int64(sqrt(length(v))))
nothing#hide

# The partially applied bracket $\mathrm{ad}_X = [X,\cdot] : L \to L$ (where $X \in L$) is called the *adjoint representation* of $L$.
# In the extremely common case that the bracket operation is the matrix commutator, we can easily write down a matrix representation of $\mathrm{ad}_X$ with respect to some basis:


function basis_to_transformation_matrix(basis)
  basis = matrix_to_flat.(basis)
  mat_size = length(basis[1])
  basis_elems = length(basis)

  ## A matrix mapping (a,b,c) -> a * basis[1] + b * basis[2] + c * basis[3]
  [basis[i][j] for j = 1:mat_size, i = 1:basis_elems]
end

function adX_matrep(X,basis)
  M = basis_to_transformation_matrix(basis)
  adX = zeros(ComplexF64,length(basis),length(basis))
  for j = 1:length(basis)
    adX[:,j] = M \ matrix_to_flat(X * basis[j] - basis[j] * X)
  end
  adX
end
nothing#hide

# Since $\mathrm{ad}_X$ is just a linear map, we can multiply it with itself and take a trace, $B(X,Y) = \mathrm{tr}(\mathrm{ad}_X \mathrm{ad}_Y)$, which defines the Killing form $B$:


function killing_form(basis)
  B = zeros(ComplexF64,length(basis),length(basis))
  ad = map(X -> adX_matrep(X,basis),basis)
  for i = 1:length(basis), j = 1:length(basis)
    B[i,j] = tr(ad[i] * ad[j])
  end
  round.(B;digits = 12) # Chop tiny numbers
end

#
# If $\{A,B,C\}$ is a basis for $L$, then $2A-4B$ is in $L$, but none of $AB$, $AB - BA$, or $C^2$ is in $L$, in the same way that $\hat x\hat y$ is not in $\mathbb{R}^3$.
# In order to make sense of these formal products of Lie algebra elements, one needs to consider the so-called Universal Enveloping Algebra (UEA) $U(L)$, which is the free algebra over $L$ (meaning arbitrary products of elements of $L$ are allowed), but quotiented by the relation $[X,Y] = XY-YX$ (meaning that the formal expression $AB - BA$ is considered equal to whatever $[A,B]$ evaluates to).
#
# Within any Lie algebra (including a UEA), the *center* is the subspace of the Lie algebra that commutes with everything: $X$ is in the center if and only if $[X,Y] = 0$ for all $Y$.
# The term "Casimir" sometimes refers to simply *any* element of the center of a Lie algebra, but we will be interested in defining a specific conventional (and basis-independent) Casimir element.
#
# Given a Lie algebra $L$, the *Killing form-derived quadratic casimir* is the following element of UEA of $L$: $\Omega \equiv \sum_{ij} [B^{-1}]^{ij} X^i X^j$ where $[B^{-1}]^{ij}$ is the inverse matrix of $B_{ij}$ in the $X^i$ basis.


function killing_casimir(basis; B = killing_form(basis))
  C = (permutedims(basis) * inv(B) * basis)[1]
  round.(C;digits = 12) # Chop tiny numbers
end

#
# # Example: SU(2)
#
# The $\mathfrak{su}(2)$ Lie algebra consists of real linear combinations of $S^x$, $S^y$ and $S^z$ (defined as half the pauli matrices), with the commutation relations $[S^x, S^y] = iS^z$ (pauli matrices would have $2i$ instead of $i$) and so on.


su2 = spin_matrices(1/2)
@assert su2[1] * su2[2] - su2[2] * su2[1] ≈ im * su2[3] # Check commutation relation

# The center of $\mathfrak{su}(2)$ is empty (only includes zero).
# The Killing form is diagonal, $B(S^i,S^j) = 2\delta^{ij}$.

killing_form(su2)

#
# The UEA $U(\mathfrak{su}(2))$ consists of arbitrary spin polynomials $[S^x]^2$, $[S^y]^3S^x - 4 S^z$, etc, with only the commutation relations imposed (and *no* further "coincidental" representation-dependent relations like $[S^x]^2 = I$ imposed).
# The Killing form-derived quadratic casimir of $\mathfrak{su}(2)$ is $\frac{1}{2}S^2 \equiv \frac{1}{2}([S^x]^2 + [S^y]^2 + [S^z]^2)$:


killing_casimir(su2)

#
# We will now demonstrate that this is basis independent.
# Consider first the Pauli matrices $\sigma^i = 2 S^i$:

pauli = 2 .* su2
killing_form(pauli)

# The killing form is different; but the casimir is the same:

@assert killing_casimir(su2) ≈ killing_casimir(pauli)

# Next, consider the ladder basis $(S^z, S^+, S^-)$, where the killing form is even off-diagonal:

ladder = [su2[3], su2[1] + im * su2[2], su2[1] - im * su2[2]]
Sz, Splus, Sminus = ladder

## Verify ladder operator properties
@assert Sz * Splus - Splus * Sz ≈ +Splus
@assert Sz * Sminus - Sminus * Sz ≈ -Sminus

killing_form(ladder)

# Yet the casimir is the same:

killing_casimir(ladder)

# In all representations of $\mathfrak{su}(2)$, i.e. for any spin $S$, the killing form-derived quadratic casimir is $S(S+1)/2$:

println()
println("su(2) representations and their casimir values:")
for S = (1/2):(1/2):5
  spinS = spin_matrices(S)
  C = killing_casimir(spinS)
  @assert C ≈ (S*(S+1)/2) * I(Int64(2S+1))
  casimir = Float64(C[1,1])
  println("S = $(Sunny.number_to_math_string(S)), S(S+1)/2 = $casimir")
end

# ## Stevens Operator Bases
#
# The $N^2 - 1$ non-identity stevens operators form a basis for the defining representation of $\mathfrak{su}(N)$. We can collect them into a basis:

function stevens_basis(S; Smax = S)
  O = stevens_matrices(S)
  basis = Vector{Any}(undef,0)
  for n = 1:Int64(2Smax)
    for k = -n:n
      push!(basis,O[n,k])
    end
  end
  basis
end
 
# Meanwhile, the 0,0 stevens operator is the identity matrix. Since the quadratic casimir of $\mathfrak{su}(2)$ is always a scalar multiple of the identity matrix, `print_stevens_expansion(killing_casimir(spin_matrices(S)))` is a convenient way to print the quadratic casimir constant (which is the unique eigenvalue of the quadratic casimir):

println()
print("For S = 5, Ω = ")
print_stevens_expansion(killing_casimir(spin_matrices(5)))

# Using the Stevens operators, we can compute Killing forms and derive quadratic casimir invariants from them.
# Note that, in the $\mathfrak{su}(N)$ case, the quadratic means quadratic in the $\mathfrak{su}(N)$ generators, *not* in the spin operators.

sparse(killing_form(stevens_basis(1)))

# Since the Killing form is always diagonal in the Steven's operator basis, we just need the numbers on the diagonal:

diag(killing_form(stevens_basis(3/2)))

# Using S = Inf to have Sunny produce spin polynomials, we can compute the killing form-derived quadratic casimirs in spin polynomial form.
# First, we do this for S = 1/2:

## Compute the Killing form numerically, using numerical stevens matrices
B_su2 = killing_form(stevens_basis(1/2)) # = diag(2,2,2)

## Compute the casimir operator symbolically
Ω = killing_casimir(stevens_basis(Inf;Smax = 1/2); B = real.(B_su2))
println()
println("su(2): Ω = $(round(Ω,digits=5))")

# The output expression, $\Omega = \frac{1}{2}\sum_i [S^i]^2$, takes place the UEA.
# Recall that for our usual spin-1/2 representation, we have $[S^i]^2 = \frac{1}{4}$ for $i=x,y,z$, so $\Omega = \frac{1}{2}(3 \times \frac{1}{4}) = 3/8$, which agrees with the $S(S+1)/2 = 0.375$ value from earlier.
#
# Plugging in the spin-1/2 representation for the $S^i$ makes this quantum mechanical.
# We could instead evaluate this spin polynomial by inserting the classical x,y,z components of the spin dipole to find the classical version: $\Omega = \frac{1}{2}\lvert S\rvert^2 = 0.125 \neq 0.375$ (assuming $\lvert S \rvert = \frac{1}{2}$ for spin-1/2, i.e. $\kappa = 1$).
# Sunny can do part of this computation for us, by writing $\Omega$ as a polynomial in $\lvert S\rvert^2$, via printing the stevens expansion, which is just the prefactor to the identity matrix O[0,0]:

print("  stevens: ")
print_stevens_expansion(Ω)

# The situation is similar but more complicated for spin-1:

B_su3 = killing_form(stevens_basis(1)) # = diag(12,12,12,12,3,36,3,12)
Ω = killing_casimir(stevens_basis(Inf;Smax = 1); B = real.(B_su3))
println()
println("su(3): Ω = (1/36) * [$(round(Ω * 36,digits = 5))]")
print("  stevens: ")
print_stevens_expansion(Ω)

# However, the situation becomes more clear numerically, where all these complicated expressions turn out proportional to the identity matrix, similar to the representations of $\mathfrak{su}(2)$:

println()
println("su(N) defining representations and their casimir values")
sun_cas = []
for S = (1/2):(1/2):3
  suN = stevens_basis(S)
  C = killing_casimir(suN)
  @assert allequal(diag(C))
  casimir = Float64(C[1,1])
  push!(sun_cas,casimir)
  println("S = $(Sunny.number_to_math_string(S)), Casimir diagonal = $casimir")
end

# The casimir value for $\mathfrak{su}(N)$ can be predicted as $\frac{1}{2}\frac{N^2-1}{N^2}$, and it approaches $\frac{1}{2}$ as $N\to\infty$.
#
# We can make a more complete table using the basis provided by D.D., original given in *Nemoto (2000), "Generalized coherent states for SU(n) systems."*, although we quickly run into scaling issues for larger $N$:

dd_file = "../../Dahlbom-ORNL\\generators-and-invariants\\sun_generators.jl"
if !isfile(dd_file)
  println("Couldn't find David's file :(")
else
  include(dd_file)
  println()
  println("su(N) casimir values")
  for S = (1/2):(1/2):4
    suN = sun_generators(Int64(2S+1))
    C = killing_casimir(suN)
    @assert allequal(diag(C))
    casimir = Float64(C[1,1])
    println("S = $(Sunny.number_to_math_string(S)), Casimir diagonal = $casimir")
  end
end

# ## Killing form is rotation-invariant

# The group of physics rotations, $\mathrm{SO}(3)$, acts on $\mathfrak{su}(N)$ by rotating the stevens operators amongst themselves.
# In particular, the group action is block diagonal in the stevens operators, only mixing between `O[q,k]` with fixed `q`.
# The following shows that, although the Killing form `B_su3` restricted to the `q=2` "quadrupole" part is not proportional to the identity matrix,

multipolar_killing_form = B_su3[4:8,4:8]
sparse(round.(multipolar_killing_form;digits = 8))

# it is still preserved under physical rotations:

## A random orthogonal matrix (a physical rotation)
R = Sunny.Mat3(Matrix(qr(randn(3,3)).Q))

## The matrix implementing the rotation of stevens operators
V = Sunny.operator_for_stevens_rotation(2,R)

V * multipolar_killing_form * transpose(V) ≈ multipolar_killing_form

# This can be checked more rigorously at the Lie algebra level by showing that the killing form is preserved under all infinitesimal rotations.
# In this case, the infinitesimal rotations are generated by the spin-2 representation of $\mathfrak{su}(2)$, since quadrupoles have dimension $5 = 2(2) + 1$, and $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ is the Lie algebra of physical rotations.
# The basis of stevens operators is related to the basis the spin matrices by the `Sunny.stevens_α` and `Sunny.stevens_αinv` matrices, so we use those to map the spin-2 Lie algebra `Sunny.spin_matrices(2)` to the Lie algebra generating rotations of stevens operators:

spin_generator_to_stevens_generator(S) = Sunny.stevens_α[2] * conj(S) * Sunny.stevens_αinv[2]
steven_gens = map(spin_generator_to_stevens_generator,Sunny.spin_matrices(2))
nothing#hide

# For example, the $S^x$ gets mapped to the following 5x5 generator of multipolar rotations:

sparse(steven_gens[1])

# Now, we can differentiate the rotation `V * B * transpose(V)` by setting $V = I + \epsilon A$ and taking a derivative with respect to $\epsilon$ at $\epsilon = 0$.
# The result is `A * B + B * transpose(A)` plus $O(\epsilon^2)$.
# Thus, we can show that `B = multipolar_killing_form` is invariant under any rotation:

for A in steven_gens
  ## Computes the derivative of the killing form matrix elements
  ## with respect to the angle of rotation
  deriv = A * multipolar_killing_form .+ multipolar_killing_form * transpose(A)

  ## If all angular derivatives vanish, it's invariant to rotations
  println(norm(deriv) < 1e-12)
end

