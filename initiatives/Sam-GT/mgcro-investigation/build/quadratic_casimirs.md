# Quadratic Casimirs

````julia
using Sunny, GLMakie, LinearAlgebra, SparseArrays
````

## Background: The Killing Form and its associated Quadratic Casimir

A Lie algebra $L$ is a vector space with a bracket operation $[\cdot,\cdot] : L \times L \to L$ satisfying certain properties.
Since Lie algebra elements are often square matrices, it will be convenient to flatten them into vectors to treat them using linear algebra:

````julia
matrix_to_flat(M) = M[:]
flat_to_matrix(v) = reshape(v,Int64(sqrt(length(v))),Int64(sqrt(length(v))))
````

The partially applied bracket $\mathrm{ad}_X = [X,\cdot] : L \to L$ (where $X \in L$) is called the *adjoint representation* of $L$.
In the extremely common case that the bracket operation is the matrix commutator, we can easily write down a matrix representation of $\mathrm{ad}_X$ with respect to some basis:

````julia
function basis_to_transformation_matrix(basis)
  basis = matrix_to_flat.(basis)
  mat_size = length(basis[1])
  basis_elems = length(basis)

  # A matrix mapping (a,b,c) -> a * basis[1] + b * basis[2] + c * basis[3]
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
````

Since $\mathrm{ad}_X$ is just a linear map, we can multiply it with itself and take a trace, $B(X,Y) = \mathrm{tr}(\mathrm{ad}_X \mathrm{ad}_Y)$, which defines the Killing form $B$:

````julia
function killing_form(basis)
  B = zeros(ComplexF64,length(basis),length(basis))
  ad = map(X -> adX_matrep(X,basis),basis)
  for i = 1:length(basis), j = 1:length(basis)
    B[i,j] = tr(ad[i] * ad[j])
  end
  round.(B;digits = 12) # Chop tiny numbers
end
````

````
killing_form (generic function with 1 method)
````

If $\{A,B,C\}$ is a basis for $L$, then $2A-4B$ is in $L$, but none of $AB$, $AB - BA$, or $C^2$ is in $L$, in the same way that $\hat x\hat y$ is not in $\mathbb{R}^3$.
In order to make sense of these formal products of Lie algebra elements, one needs to consider the so-called Universal Enveloping Algebra (UEA) $U(L)$, which is the free algebra over $L$ (meaning arbitrary products of elements of $L$ are allowed), but quotiented by the relation $[X,Y] = XY-YX$ (meaning that the formal expression $AB - BA$ is considered equal to whatever $[A,B]$ evaluates to).

Within any Lie algebra (including a UEA), the *center* is the subspace of the Lie algebra that commutes with everything: $X$ is in the center if and only if $[X,Y] = 0$ for all $Y$.
The term "Casimir" sometimes refers to simply *any* element of the center of a Lie algebra, but we will be interested in defining a specific conventional (and basis-independent) Casimir element.

Given a Lie algebra $L$, the *Killing form-derived quadratic casimir* is the following element of UEA of $L$: $\Omega \equiv \sum_{ij} [B^{-1}]^{ij} X^i X^j$ where $[B^{-1}]^{ij}$ is the inverse matrix of $B_{ij}$ in the $X^i$ basis.

````julia
function killing_casimir(basis; B = killing_form(basis))
  C = (permutedims(basis) * inv(B) * basis)[1]
  round.(C;digits = 12) # Chop tiny numbers
end
````

````
killing_casimir (generic function with 1 method)
````

# Example: SU(2)

The $\mathfrak{su}(2)$ Lie algebra consists of real linear combinations of $S^x$, $S^y$ and $S^z$ (defined as half the pauli matrices), with the commutation relations $[S^x, S^y] = iS^z$ (pauli matrices would have $2i$ instead of $i$) and so on.

````julia
su2 = spin_matrices(1/2)
@assert su2[1] * su2[2] - su2[2] * su2[1] ≈ im * su2[3] # Check commutation relation
````

The center of $\mathfrak{su}(2)$ is empty (only includes zero).
The Killing form is diagonal, $B(S^i,S^j) = 2\delta^{ij}$.

````julia
killing_form(su2)
````

````
3×3 Matrix{ComplexF64}:
  2.0-0.0im  -0.0-0.0im   0.0-0.0im
 -0.0-0.0im   2.0-0.0im  -0.0-0.0im
  0.0-0.0im  -0.0-0.0im   2.0-0.0im
````

The UEA $U(\mathfrak{su}(2))$ consists of arbitrary spin polynomials $[S^x]^2$, $[S^y]^3S^x - 4 S^z$, etc, with only the commutation relations imposed (and *no* further "coincidental" representation-dependent relations like $[S^x]^2 = I$ imposed).
The Killing form-derived quadratic casimir of $\mathfrak{su}(2)$ is $\frac{1}{2}S^2 \equiv \frac{1}{2}([S^x]^2 + [S^y]^2 + [S^z]^2)$:

````julia
killing_casimir(su2)
````

````
2×2 Matrix{ComplexF64}:
 0.375+0.0im    0.0+0.0im
   0.0+0.0im  0.375+0.0im
````

We will now demonstrate that this is basis independent.
Consider first the Pauli matrices $\sigma^i = 2 S^i$:

````julia
pauli = 2 .* su2
killing_form(pauli)
````

````
3×3 Matrix{ComplexF64}:
  8.0-0.0im  -0.0-0.0im   0.0-0.0im
 -0.0-0.0im   8.0-0.0im  -0.0-0.0im
  0.0-0.0im  -0.0-0.0im   8.0-0.0im
````

The killing form is different; but the casimir is the same:

````julia
@assert killing_casimir(su2) ≈ killing_casimir(pauli)
````

Next, consider the ladder basis $(S^z, S^+, S^-)$, where the killing form is even off-diagonal:

````julia
ladder = [su2[3], su2[1] + im * su2[2], su2[1] - im * su2[2]]
Sz, Splus, Sminus = ladder

# Verify ladder operator properties
@assert Sz * Splus - Splus * Sz ≈ +Splus
@assert Sz * Sminus - Sminus * Sz ≈ -Sminus

killing_form(ladder)
````

````
3×3 Matrix{ComplexF64}:
 2.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  4.0+0.0im
 0.0+0.0im  4.0+0.0im  0.0+0.0im
````

Yet the casimir is the same:

````julia
killing_casimir(ladder)
````

````
2×2 Matrix{ComplexF64}:
 0.375+0.0im    0.0+0.0im
   0.0+0.0im  0.375+0.0im
````

In all representations of $\mathfrak{su}(2)$, i.e. for any spin $S$, the killing form-derived quadratic casimir is $S(S+1)/2$:

````julia
println()
println("su(2) representations and their casimir values:")
for S = (1/2):(1/2):5
  spinS = spin_matrices(S)
  C = killing_casimir(spinS)
  @assert C ≈ (S*(S+1)/2) * I(Int64(2S+1))
  casimir = Float64(C[1,1])
  println("S = $(Sunny.number_to_math_string(S)), S(S+1)/2 = $casimir")
end
````

````

su(2) representations and their casimir values:
S = 1/2, S(S+1)/2 = 0.375
S = 1, S(S+1)/2 = 1.0
S = 3/2, S(S+1)/2 = 1.875
S = 2, S(S+1)/2 = 3.0
S = 5/2, S(S+1)/2 = 4.375
S = 3, S(S+1)/2 = 6.0
S = 7/2, S(S+1)/2 = 7.875
S = 4, S(S+1)/2 = 10.0
S = 9/2, S(S+1)/2 = 12.375
S = 5, S(S+1)/2 = 15.0

````

## Stevens Operator Bases

The $N^2 - 1$ non-identity stevens operators form a basis for the defining representation of $\mathfrak{su}(N)$. We can collect them into a basis:

````julia
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
````

````
stevens_basis (generic function with 1 method)
````

Meanwhile, the 0,0 stevens operator is the identity matrix. Since the quadratic casimir of $\mathfrak{su}(2)$ is always a scalar multiple of the identity matrix, `print_stevens_expansion(killing_casimir(spin_matrices(S)))` is a convenient way to print the quadratic casimir constant (which is the unique eigenvalue of the quadratic casimir):

````julia
println()
print("For S = 5, Ω = ")
print_stevens_expansion(killing_casimir(spin_matrices(5)))
````

````

For S = 5, Ω = 15

````

Using the Stevens operators, we can compute Killing forms and derive quadratic casimir invariants from them.
Note that, in the $\mathfrak{su}(N)$ case, the quadratic means quadratic in the $\mathfrak{su}(N)$ generators, *not* in the spin operators.

````julia
sparse(killing_form(stevens_basis(1)))
````

````
8×8 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
 12.0+0.0im       ⋅           ⋅           ⋅          ⋅           ⋅          ⋅           ⋅    
      ⋅      12.0-0.0im       ⋅           ⋅          ⋅           ⋅          ⋅           ⋅    
      ⋅           ⋅      12.0-0.0im       ⋅          ⋅           ⋅          ⋅           ⋅    
      ⋅           ⋅           ⋅      12.0-0.0im      ⋅           ⋅          ⋅           ⋅    
      ⋅           ⋅           ⋅           ⋅      3.0+0.0im       ⋅          ⋅           ⋅    
      ⋅           ⋅           ⋅           ⋅          ⋅      36.0+0.0im      ⋅           ⋅    
      ⋅           ⋅           ⋅           ⋅          ⋅           ⋅      3.0-0.0im       ⋅    
      ⋅           ⋅           ⋅           ⋅          ⋅           ⋅          ⋅      12.0-0.0im
````

Since the Killing form is always diagonal in the Steven's operator basis, we just need the numbers on the diagonal:

````julia
diag(killing_form(stevens_basis(3/2)))
````

````
15-element Vector{ComplexF64}:
  40.0 + 0.0im
  40.0 + 0.0im
  40.0 + 0.0im
  96.0 + 0.0im
  24.0 - 0.0im
 288.0 + 0.0im
  24.0 + 0.0im
  96.0 + 0.0im
 144.0 + 0.0im
  24.0 - 0.0im
 240.0 + 0.0im
 360.0 + 0.0im
 240.0 + 0.0im
  24.0 - 0.0im
 144.0 + 0.0im
````

Using S = Inf to have Sunny produce spin polynomials, we can compute the killing form-derived quadratic casimirs in spin polynomial form.
First, we do this for S = 1/2:

````julia
# Compute the Killing form numerically, using numerical stevens matrices
B_su2 = killing_form(stevens_basis(1/2)) # = diag(2,2,2)

# Compute the casimir operator symbolically
Ω = killing_casimir(stevens_basis(Inf;Smax = 1/2); B = real.(B_su2))
println()
println("su(2): Ω = $(round(Ω,digits=5))")
````

````

su(2): Ω = 0.5*𝒮ˣ^2 + 0.5*𝒮ʸ^2 + 0.5*𝒮ᶻ^2

````

The output expression, $\Omega = \frac{1}{2}\sum_i [S^i]^2$, takes place the UEA.
Recall that for our usual spin-1/2 representation, we have $[S^i]^2 = \frac{1}{4}$ for $i=x,y,z$, so $\Omega = \frac{1}{2}(3 \times \frac{1}{4}) = 3/8$, which agrees with the $S(S+1)/2 = 0.375$ value from earlier.

Plugging in the spin-1/2 representation for the $S^i$ makes this quantum mechanical.
We could instead evaluate this spin polynomial by inserting the classical x,y,z components of the spin dipole to find the classical version: $\Omega = \frac{1}{2}\lvert S\rvert^2 = 0.125 \neq 0.375$ (assuming $\lvert S \rvert = \frac{1}{2}$ for spin-1/2, i.e. $\kappa = 1$).
Sunny can do part of this computation for us, by writing $\Omega$ as a polynomial in $\lvert S\rvert^2$, via printing the stevens expansion, which is just the prefactor to the identity matrix O[0,0]:

````julia
print("  stevens: ")
print_stevens_expansion(Ω)
````

````
  stevens: (1/2)𝒮²

````

The situation is similar but more complicated for spin-1:

````julia
B_su3 = killing_form(stevens_basis(1)) # = diag(12,12,12,12,3,36,3,12)
Ω = killing_casimir(stevens_basis(Inf;Smax = 1); B = real.(B_su3))
println()
println("su(3): Ω = (1/36) * [$(round(Ω * 36,digits = 5))]")
print("  stevens: ")
print_stevens_expansion(Ω)
````

````

su(3): Ω = (1/36) * [3.0*𝒮ˣ^2 + 3.0*𝒮ʸ^2 + 3.0*𝒮ᶻ^2 + 4.0*𝒮ˣ^4 + 8.0*𝒮ʸ^2*𝒮ˣ^2 + 4.0*𝒮ʸ^4 + 8.0*𝒮ᶻ^2*𝒮ˣ^2 + 8.0*𝒮ᶻ^2*𝒮ʸ^2 + 4.0*𝒮ᶻ^4]
  stevens: (1/12)𝒮² + (1/9)𝒮⁴

````

However, the situation becomes more clear numerically, where all these complicated expressions turn out proportional to the identity matrix, similar to the representations of $\mathfrak{su}(2)$:

````julia
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
````

````

su(N) defining representations and their casimir values
S = 1/2, Casimir diagonal = 0.375
S = 1, Casimir diagonal = 0.444444444444
S = 3/2, Casimir diagonal = 0.46875
S = 2, Casimir diagonal = 0.48
S = 5/2, Casimir diagonal = 0.486111111111
S = 3, Casimir diagonal = 0.489795918367

````

The casimir value for $\mathfrak{su}(N)$ can be predicted as $\frac{1}{2}\frac{N^2-1}{N^2}$, and it approaches $\frac{1}{2}$ as $N\to\infty$.

We can make a more complete table using the basis provided by D.D., original given in *Nemoto (2000), "Generalized coherent states for SU(n) systems."*, although we quickly run into scaling issues for larger $N$:

````julia
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
````

````
Couldn't find David's file :(

````

## Killing form is rotation-invariant

The group of physics rotations, $\mathrm{SO}(3)$, acts on $\mathfrak{su}(N)$ by rotating the stevens operators amongst themselves.
In particular, the group action is block diagonal in the stevens operators, only mixing between `O[q,k]` with fixed `q`.
The following shows that, although the Killing form `B_su3` restricted to the `q=2` "quadrupole" part is not proportional to the identity matrix,

````julia
multipolar_killing_form = B_su3[4:8,4:8]
sparse(round.(multipolar_killing_form;digits = 8))
````

````
5×5 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 5 stored entries:
 12.0-0.0im      ⋅           ⋅          ⋅           ⋅    
      ⋅      3.0+0.0im       ⋅          ⋅           ⋅    
      ⋅          ⋅      36.0+0.0im      ⋅           ⋅    
      ⋅          ⋅           ⋅      3.0-0.0im       ⋅    
      ⋅          ⋅           ⋅          ⋅      12.0-0.0im
````

it is still preserved under physical rotations:

````julia
# A random orthogonal matrix (a physical rotation)
R = Sunny.Mat3(Matrix(qr(randn(3,3)).Q))

# The matrix implementing the rotation of stevens operators
V = Sunny.operator_for_stevens_rotation(2,R)

V * multipolar_killing_form * transpose(V) ≈ multipolar_killing_form
````

````
true
````

This can be checked more rigorously at the Lie algebra level by showing that the killing form is preserved under all infinitesimal rotations.
In this case, the infinitesimal rotations are generated by the spin-2 representation of $\mathfrak{su}(2)$, since quadrupoles have dimension $5 = 2(2) + 1$, and $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ is the Lie algebra of physical rotations.
The basis of stevens operators is related to the basis the spin matrices by the `Sunny.stevens_α` and `Sunny.stevens_αinv` matrices, so we use those to map the spin-2 Lie algebra `Sunny.spin_matrices(2)` to the Lie algebra generating rotations of stevens operators:

````julia
spin_generator_to_stevens_generator(S) = Sunny.stevens_α[2] * conj(S) * Sunny.stevens_αinv[2]
steven_gens = map(spin_generator_to_stevens_generator,Sunny.spin_matrices(2))
````

For example, the $S^x$ gets mapped to the following 5x5 generator of multipolar rotations:

````julia
sparse(steven_gens[1])
````

````
5×5 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 6 stored entries:
     ⋅          ⋅          ⋅      0.0-2.0im      ⋅    
     ⋅          ⋅          ⋅          ⋅      0.0-0.5im
     ⋅          ⋅          ⋅      0.0-6.0im      ⋅    
 0.0+0.5im      ⋅      0.0+0.5im      ⋅          ⋅    
     ⋅      0.0+2.0im      ⋅          ⋅          ⋅    
````

Now, we can differentiate the rotation `V * B * transpose(V)` by setting $V = I + \epsilon A$ and taking a derivative with respect to $\epsilon$ at $\epsilon = 0$.
The result is `A * B + B * transpose(A)` plus $O(\epsilon^2)$.
Thus, we can show that `B = multipolar_killing_form` is invariant under any rotation:

````julia
for A in steven_gens
  # Computes the derivative of the killing form matrix elements
  # with respect to the angle of rotation
  deriv = A * multipolar_killing_form .+ multipolar_killing_form * transpose(A)

  # If all angular derivatives vanish, it's invariant to rotations
  println(norm(deriv) < 1e-12)
end
````

````
true
true
true

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

