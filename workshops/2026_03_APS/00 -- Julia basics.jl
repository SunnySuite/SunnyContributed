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


### Anonymous function syntax

(x -> x^2)(5)

map(x -> x^2, [1, 2, 3])

map([1, 2, 3]) do x
    x^2
end


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
