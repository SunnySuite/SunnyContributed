using LinearAlgebra

### Getting around

exp(im * pi)

exp(im * π) # Enter \pi<TAB>

exp(im * π) ≈ -1 # Enter \approx<TAB>

rationalize(π; tol=0.0001) # keyword argument
Float64(π)
333/106

isapprox(333/106, π; atol=1e-4)
isapprox(333/106, π; atol=1e-5)

### Getting help

# From terminal: ? rationalize
# Or hover from VSCode extension
# Or ctrl-click from VSCode extension!

### Broadcasting notation

x = [1, 2]
y = [3, 4]

x + y
# x * y
x .* y

[sqrt(x) for x in [1, 2, 3, 4]]
sqrt.([1, 2, 3, 4])

### Array aliasing

a = [1, 2]
b = a
b[1] = 11
a

### Linear algebra

A = randn(3, 3)
x = randn(3)
b = A * x

norm(x - inv(A) * b)
x ≈ inv(A) * b
x ≈ A \ b

x' ≈ b' * inv(A)' ≈ b' / A'

v1, v2, v3 = eachcol(A)
dot(v1, cross(v2, v3)) ≈ det(A)
v1 ⋅ (v2 × v3) ≈ det(A)

exp.(A) ≈ [exp(a) for a in A]
exp.(A) ≈ exp(A)

norm(I - exp(A))
norm((I + A + A^2/2 + A^3/6 + A^4/24) - exp(A))
sum(A^n / factorial(n) for n in 0:15) ≈ exp(A)

### Plotting with Makie

using GLMakie, SpecialFunctions

xs = range(0, 20, 100)
ax = lines(xs, besselj.(1/2, xs); label="ν = 1/2")

fig = Figure()
ax = Axis(fig[1, 1])
for ν in (1, 2, 3)
    lines!(ax, xs, besselj.(ν, xs); label="ν = $ν")
end
axislegend(ax)

A = randn(1000, 1000)
eigvals(A)
hist(eigvals(hermitianpart(A)))

sinc(r) = iszero(r) ? 1 : sin(r) / r
xs = range(-5, 5, 100)
ys = range(-5, 5, 100)
zs = [5 * sinc(4 * sqrt(x^2 + y^2)) for x in xs, y in ys]
surface(xs, ys, zs; colormap = :Spectral)

cs = range(-10, 10, 100)
cube = [(x^2 + y^2 + z^2) for x in cs, y in cs, z in cs]
contour(cube, alpha=0.5)

dots = [sinc(x) * sinc(y) * sinc(z) for x in cs, y in cs, z in cs]
volume(dots, colorrange=(0, 0.2))

# More ideas: https://beautiful.makie.org/dev/


### Matrix representation of spin operators

using Sunny
@assert pkgversion(Sunny) >= v"0.7.1"

S = spin_matrices(1)
S.z
S.x
S.x * S.y - S.y * S.x ≈ im * S.z

n = normalize(randn(3))
θ = π/3
R = exp(im * θ * n' * S)

R' ≈ inv(R)
R ≈ I + im * (n' * S) * sin(θ) + (n' * S)^2 * (cos(θ) - 1) # Rodrigues


### Performance tuning

x = 2

function f(y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time f(3.0, 50_000_000)

@code_warntype f(3.0, 20_000_000)


function g(x, y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time g(2, 3.0, 20_000_000)

@code_warntype g(2, 3.0, 100)
@code_native g(2, 3.0, 100)


### Inspecting code


###



### Spin dynamics on Heisenberg ferromagnet

# Square lattice

latvecs = lattice_vectors(1, 1, 10, 90, 90, 90)
positions = [[0,0,0]]
cryst = Crystal(latvecs, positions)
view_crystal(cryst; ndims=2, ghost_radius=2)

# Ferromagnetic couplings

sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole; seed=1, dims=(10, 10, 1))
J = -1.0
set_exchange!(sys, J, Bond(1, 1, (1, 0, 0)))
randomize_spins!(sys)
fig = plot_spins(sys; colorfn=i->sys.dipoles[i].z, colorrange=(-1, 1), ndims=2)

# Dynamics of local magnetic moments
dt = 0.05/abs(J)
integrator = Langevin(dt; damping=0.05, kT=0)

# View animation in real time
for _ in 1:500
    for _ in 1:5
        step!(sys, integrator)
    end
    notify(fig)
    sleep(1/60)
end


### Frustrated Kagome

latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
positions = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]
cryst = Crystal(latvecs, positions)
view_crystal(cryst; ndims=2, ghost_radius=3)

sys = System(cryst, (10,10,1), [SpinInfo(1, S=1, g=2)], :dipole; seed=1)
J1 = -1.0
J2 = +0.5
set_exchange!(sys, J1, Bond(1, 2, (0, 0, 0)))
set_exchange!(sys, J2, Bond(1, 2, (0, 1, 0)))
set_onsite_coupling!(sys, S -> 0.1*S[3]^2, 1)

randomize_spins!(sys)
minimize_energy!(sys; maxiters=1000)
energy_per_site(sys)
plot_spins(sys; color=[s[3] for s in sys.dipoles], dims=2)


### Debugger

view_crystal(cryst)
print_bond(cryst, Bond(1, 3, [0, 0, 0]))

@enter print_bond(cryst, Bond(1, 3, [0, 0, 0]))

# Hot code replacement with Revise
