using Sunny, GLMakie


### Spin dynamics on Heisenberg ferromagnet

# Square lattice
latvecs = lattice_vectors(1, 1, 10, 90, 90, 90)
positions = [[0,0,0]]
cryst = Crystal(latvecs, positions)
view_crystal(cryst; dims=2, ghost_radius=2)

# Ferromagnetic couplings
sys = System(cryst, (10,10,1), [SpinInfo(1, S=1, g=2)], :dipole; seed=1)
J = -1.0
set_exchange!(sys, J, Bond(1, 1, (1, 0, 0)))

randomize_spins!(sys)
fig = plot_spins(sys; colorfn=i->sys.dipoles[i][3], colorrange=(-1, 1), dims=2)

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
view_crystal(cryst; dims=2, ghost_radius=3)

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


### Math notation

A = randn(10, 10)
x = randn(10)

b = A * x

@assert inv(A) * b ≈ x
@assert A \ b ≈ x


using LinearAlgebra

A = randn(3, 3)
v1, v2, v3 = eachcol(A)
@assert isapprox(dot(v1, cross(v2, v3)), det(A))
@assert v1 ⋅ (v2 × v3) ≈ det(A)

@assert [exp(a) for a in A] ≈ exp.(A)
norm(exp.(A) - exp(A))

norm(I - exp(A))
norm((I + A + A^2/2 + A^3/6 + A^4/24) - exp(A))


### Performance tuning

x = 2

function f(y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

f(3.0, 10)
f(3.0, 100)

using Chairmarks
@b f(3.0, 100)

@code_warntype f(3.0, 100)


function g(x, y, n)
    for i in 1:n
        y = 0.4 * (x + y)
    end
    return y
end

@b g(2, 3.0, 100)

@code_warntype g(2, 3.0, 100)
@code_native g(2, 3.0, 100)


### Debugger

view_crystal(cryst)
print_bond(cryst, Bond(1, 3, [0, 0, 0]))

@enter print_bond(cryst, Bond(1, 3, [0, 0, 0]))

# Hot code replacement with Revise

# Interactively run tests
