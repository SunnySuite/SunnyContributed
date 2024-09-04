using Sunny, GLMakie
@assert pkgversion(Sunny) >= v"0.7"

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
