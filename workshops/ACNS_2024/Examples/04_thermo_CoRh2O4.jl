
using Sunny, GLMakie


# Diamond crystal

a = 8.5031 # (Å)
latvecs = lattice_vectors(a, a, a, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]], 227, setting="1")
view_crystal(cryst)

# Antiferromagnetic interactions for CoRh₂O₄

S = 3/2
J = 0.63 # (meV)
sys = System(cryst, (8,8,8), [SpinInfo(1; S, g=2)], :dipole; seed=0)
set_exchange!(sys, J, Bond(1, 3, [0,0,0]))

# Find ground state

randomize_spins!(sys)
minimize_energy!(sys)
@assert energy_per_site(sys) ≈ -2J*S^2
plot_spins(sys; color=[s'*s0 for s in sys.dipoles])

# Construct Langevin integrator with appropriate timestep

kT_max = meV_per_K * 40
langevin = Langevin(; damping=0.2, kT=kT_max)
suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.025

# Define range of kT values

kTs = 0:0.1:kT_max

# Measure the energy per site at each kT

nsteps_per_kT = 200
Es = Float64[]
for kT in kTs
    langevin.kT = kT
    for _ in 1:nsteps_per_kT
        step!(sys, langevin)
    end

    push!(Es, energy_per_site(sys))
end

# Plot energy

Ts = kTs / meV_per_K
scatter(Ts, Es; axis=(; xlabel="Temperature (K)", ylabel="⟨E⟩ (meV)"))

# Plot heat capacity using finite differences

Ts_reduced = (Ts[2:end] + Ts[1:end-1]) / 2
dT = step(kTs)
Cs = (Es[2:end] - Es[1:end-1]) / dT
scatter(Ts_reduced, Cs; axis=(; xlabel="Temperature (K)", ylabel="Cᵥ (meV)"))


### Use averaging to collect better statistics

# Initialize to ground state again

randomize_spins!(sys)
minimize_energy!(sys)

# Now collect energy averages for each kT

nsteps_per_kT = 2000
Es = Float64[]
@time for kT in kTs
    E_accum = 0
    langevin.kT = kT
    for _ in 1:nsteps_per_kT
        step!(sys, langevin)
        E_accum += energy_per_site(sys)
    end

    push!(Es, E_accum / nsteps_per_kT)
end

# The resulting heat capacities are more smooth

Ts_reduced = (Ts[2:end] + Ts[1:end-1]) / 2
dT = step(kTs)
Cs = (Es[2:end] - Es[1:end-1]) / dT
lines(Ts_reduced, Cs; axis=(; xlabel="Temperature (K)", ylabel="Cᵥ (meV)"))
