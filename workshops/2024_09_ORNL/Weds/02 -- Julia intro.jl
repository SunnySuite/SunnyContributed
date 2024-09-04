using LinearAlgebra

### Unicode

exp(im * pi)

exp(im * π) # Enter \pi<TAB>

exp(im * π) ≈ -1 # Enter \approx<TAB>

rationalize(π; tol=0.0001) # keyword argument
Float64(π)
333/106

333/106 ≈ π
isapprox(333/106, π; atol=1e-4)

### Getting help

# From terminal: ? rationalize
# Or hover from VSCode extension
# Or ctrl-click from VSCode extension!

### Array aliasing

a = [1, 2]
b = a
b[1] = 11
a

### Broadcasting notation

a = [1.0, 2.0]
b = [3.0, 4.0]

a + b
# a * b
a .* b

[sqrt(x) for x in [1, 2, 3, 4]]
sqrt.([1, 2, 3, 4])

for i in eachindex(a)
    a[i] = sin(b[i]) * b[i]
end

a .= sin.(b) .* b
@. a = sin(b) * b  # Macro


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


### Plotting with Makie

using Sunny
using GLMakie
using SpecialFunctions

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
@assert pkgversion(Sunny) >= v"0.7"

S = spin_matrices(1)
S.z
S.x
S.x * S.y - S.y * S.x ≈ im * S.z

n = normalize(randn(3))
θ = π/3
R = exp(im * θ * n' * S)
R' ≈ inv(R)
R ≈ I + im * (n' * S) * sin(θ) + (n' * S)^2 * (cos(θ) - 1) # Rodrigues

S = spin_matrices(2)
print_stevens_expansion(S.x^4 + S.y^4 + S.z^4)

O = stevens_matrices(2)
S.x^4 + S.y^4 + S.z^4 ≈ (1/20)*O[4,0] + (1/4)*O[4,4] + 102/5 * I


### Performance tuning

x = 2

function f(y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time f(3.0, 50_000_000)

@code_warntype f(3.0, 50_000_000)


function g(x, y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@time g(2, 3.0, 20_000_000)

@code_warntype g(2, 3.0, 100)
@code_native g(2, 3.0, 100)

# Lots more at https://docs.julialang.org/en/v1/manual/performance-tips/


### Visual profiler

@profview f(3.0, 10_000_000)
@profview g(2, 3.0, 10_000_000)


### Inspecting code

using SpecialFunctions

@which erfinv(0.2)

@edit erfinv(0.2)

@enter erfinv(0.2)


### Writing your own scripts, hot code reloading

include("auxiliary.jl")

A = randn(3, 3)

norm(custom_exponential(A) - exp(A))

using Revise

includet("auxiliary.jl")

norm(custom_exponential(A) - exp(A))

