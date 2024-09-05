using Sunny, LinearAlgebra, GLMakie

################################################################################
# Picking a step size
################################################################################

# Make a simple system
units = Units(:meV, :angstrom);
a = 8.5031 # (Å)
latvecs = lattice_vectors(a, a, a, 90, 90, 90)
positions = [[0, 0, 0]]
cryst = Crystal(latvecs, positions, 227; types=["Co"], setting="1")
view_crystal(cryst)

sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)

J = +0.63 # (meV)
set_exchange!(sys, J, Bond(1, 3, [0, 0, 0]))

plot_spins(sys)

# Themalize a ground state
damping = 0.05
kT = 0.1
integrator = Langevin(; damping, kT)
randomize_spins!(sys)
minimize_energy!(sys)
suggest_timestep(sys, integrator; tol=1e-2)
dt = 0.01
integrator.dt = dt

dur = 10.0
nsteps = round(Int, dur/dt)
for i in 1:nsteps
    step!(sys, integrator)
end


# Look at a T=0 trajectory
suggest_timestep(sys, integrator; tol=1e-2)
integrator.kT = 0
integrator.dt = 0.02
dur = 10.0
nsteps = round(Int, dur/dt)
xs = zeros(nsteps)
for i in 1:nsteps
    step!(sys, integrator)
    xs[i] = sys.dipoles[1,1,1,1][1]
end
lines((0:nsteps-1) .* dt, xs)


## EXERCISE: Make dt much smaller and much larger and see what happens
## EXERCISE: Try the above with the ImplicitMidpoint integrator


################################################################################
# Example of step size affecting statistics 
################################################################################
function su3_mean_energy(kT, D)
    a = D/kT
    return D * (2 - (2 + 2a + a^2)*exp(-a)) / (a * (1 - (1+a)*exp(-a)))
end 

seed = 1
latvecs = lattice_vectors(1, 1, 1, 90, 90, 90)
positions = [[0,0,0]]
cryst = Crystal(latvecs, positions, 1)
L=20
D=1.0
sys = System(cryst, [1 => Moment(s=1, g=2)], :SUN; dims=(L, 1, 1), seed)
set_onsite_coupling!(sys, S -> D*S[3]^2, 1)
randomize_spins!(sys)

function thermalize!(sys, integrator, dur)
    numsteps = round(Int, dur/integrator.dt)
    for _ in 1:numsteps
        step!(sys, integrator)
    end
end

function calc_mean_energy(sys, integrator, dur)
    numsteps = round(Int, dur/integrator.dt)
    Es = zeros(numsteps)
    for i in 1:numsteps
        step!(sys, integrator)
        Es[i] = energy_per_site(sys)
    end
    sum(Es)/length(Es) 
end


damping = 1.0
dt = 0.07
kTs = range(0.1, 1, 10) 
thermalize_dur = 10.0
collect_dur = 100.0

heun = Langevin(dt; damping, kT=0)

suggest_timestep(sys, heun; tol=1e-2)

E_refs = Float64[]
Es = Float64[]
for kT in kTs
    heun.kT = kT
    thermalize!(sys, heun, thermalize_dur)
    E = calc_mean_energy(sys, heun, collect_dur)
    E_ref = su3_mean_energy(kT, D)
    # @test isapprox(E, E_ref; rtol=0.1)
    push!(Es, E)
    push!(E_refs, E_ref)
end

l = lines(kTs, E_refs; axis=(xlabel="kT", ylabel="E"), label="Analytical")
scatter!(kTs, Es; label="Calculated")
axislegend(l.axis; position=:rb)


################################################################################
# Decorrelation times
################################################################################

