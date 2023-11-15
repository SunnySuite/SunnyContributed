using Sunny, GLMakie, LinearAlgebra, Statistics

include("sun_practical_util.jl")


# # Hands-on examples of working with SU(N) in Sunny 

# ## What happens when you set the mode to `:SUN`? 
# Sunny has two modes: `:dipole` and `:SUN` . The package is designed so that a
# user may make use of the same functions in both modes without much thought:
# plotting, dynamics, spin wave calculations, and so on. We'll look at a few
# examples that demonstrate where the differences lie.
#
# To build intuition, consider a AFM system of S=1 spins on a square lattice
# with a simple crystal field term:
# ```math
# \mathcal{H}=\sum_{i} (S_i^z)^2
# ```
# In the traditional classical limit, this anisotropy would be hard-axis/easy
# plane. We will see that this is not quite the right interpretation in the
# generalized classical theory. We'll compare the two approaches.

## Create dipole and SU(N) system for the Hamiltonian above
latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])
sys_dip = System(cryst, (6,6,1), [SpinInfo(1, g=1, S=1)], :dipole)
sys_sun = System(cryst, (6,6,1), [SpinInfo(1, g=1, S=1)], :SUN)

## Set crystal field 
Ss_dip = spin_operators(sys_dip, 1)
Ss_sun = spin_operators(sys_sun, 1)
set_onsite_coupling!(sys_dip, Ss_dip[3]^2, 1)
set_onsite_coupling!(sys_sun, Ss_sun[3]^2, 1)

## Randomize the initial condition and search for the ground state.
randomize_spins!(sys_dip)
randomize_spins!(sys_sun)

minimize_energy!(sys_dip)
minimize_energy!(sys_sun)

fig = Figure()
ax1 = LScene(fig[1,1])
ax2 = LScene(fig[1,2])
plot_spins!(ax1, sys_dip)
plot_spins!(ax2, sys_sun)
fig

# In dipole mode, the anisotropy does indeed result in an easy-plane behavior,
# and we find our spins all lying in the ab-plane. The result of the SU(N) is
# qualitatively different, as the magnitude of the dipoles is drastically
# reduced. In the SU(N) theory (as for a real S=1 spin), the magnitude of the
# dipole is _not_ conserved. Note that this reduction in ordered moment has
# nothing to do with entanglement between sites. Rather, it is simply the result
# of the fact that we are modeling each local Hilbert space as a three-level
# system (not as a simple 2-level system, i.e. dipole). In particular, we are
# able to capture the paramagnetic state that is the true ground state of the
# operator ``(S^z)^2``. 

# While the dipole moment is not invariant, there is an invariant of the total
# "generalized" spin living on each site. To investigate this, we point out a
# few details about Sunny's internals. For a `:dipole`-mode system, information
# about the state of the spins is kept in 

sys_dip.dipoles

# In this case we have a 6x6x1 array, each element of which is a 3-vector containing
# information about the dipoles. Note that the magnitude of any one of these is 1.

norm(sys_dip.dipoles[1,1,1,1])

# The SU(N) system also has a `dipoles` field.

sys_sun.dipoles

# And the norm of these dipoles is practically zero.

norm(sys_sun.dipoles[1,1,1,1])

# The SU(N) system has an additional field, however, called coherents.

sys_sun.coherents

# Here we have have a complex 3-vector on each site, just as we would have
# for a local three-dimensional Hilbert space. Formally, each site hosts
# a coherent state of SU(3), and we can verify that that the expectation
# value of the three spin operators with respect to this coherent state
# matches what is in the `dipoles` field.

## Extract corresponding dipole and coherent state
dipole = sys_sun.dipoles[1,1,1,1] |> Vector
Z = sys_sun.coherents[1,1,1,1]

## Calculate the expectation of the spin operators with respect to the coherent state
Ss = spin_matrices(N=3) # Spin matrices in S=1 representation
expectation(op, Z) = real(Z' * op * Z)
calculated_dipole = map(op -> expectation(op, Z), Ss) 

calculated_dipole ≈ dipole

# While there is essentially no weight in the dipole sector, we hasten to add
# that our simple model of course represents a vegetable rather than a spin
# liquid. The relevant order parameter here is quadrupolar rather than dipolar,
# and symmetry breaking has occurred in the quadrupole sector. To verify this,
# define a set of quadrupole operators and evaluate their expectation value.

Qs = [
        -(Ss[1]*Ss[3] + Ss[3]*Ss[1]),    # -(Sx*Sz + Sz*Sx)
        -(Ss[2]*Ss[3] + Ss[3]*Ss[2]),    # -(SySz + Sz*Sy)
        Ss[1]*Ss[1] - Ss[2]*Ss[2],       # Sx*Sx - Sy*Sy
        Ss[1]*Ss[2] + Ss[2]*Ss[1],       # Sx*Sy + Sy*Sx
        (√3 .* Ss[3]*Ss[3]) - I*(2/√3)   # Sz*Sz (with trace removed)
]

quad_sector = map(op -> expectation(op, Z), Qs)

# Observe that all the weight is in the final quadrupolar operator.
#
# The key observations to take from this are these:
# 1) In SU(N) mode, Sunny stores the state of each spin as an SU(N) coherent
#    state, which is simply a set of N complex amplitudes.
# 2) This "generalized" spin can be associated with the expectation values of an
#    enlarged set of operators, not just spin operators. In particular, it is no
#    longer the case that the dipole moment of our classical spin is conserved.
# 3) The spin therefore cannot be associated with a point on a sphere. It has
#    more degrees of freedom (in general, N-1 degrees of freedom).
# 4) Basic intuitions about the ground states and excitations of classical spin
#    systems need to be reexamined when S > 1/2. For example, one can model a
#    paramagnet "classically," as we have done above.


# ## Static properties of generalized spins: BFSO
#
# Let us now look at a material where these properties are of real interest.
# Consider BFSO, an anti-ferromagnetic, highly-anisotropic, easy-plane quantum
# magnet on an square lattice. We provide a function, located in
# `sun_practical_util.jl`, to make the spin system.

dims = (6,6,6)
sys, cryst = BFSO(dims)

randomize_spins!(sys)
minimize_energy!(sys; subiters=20, maxiters=1000)
plot_spins(sys)

# We now wish to calculate the magnetization vs. field. We'll do this by
# successively updating the external field, minimizing the energy, and
# calculating the magnetization, and saving the results.

function magnetization(sys, g=1.93)
    M_avg = g*sum(sys.dipoles) / prod(size(sys.dipoles)) 
    return norm(M_avg)
end

function avg_dipole(sys)
    sum(norm.(sys.dipoles)) / prod(size(sys.dipoles))
end

Bs = range(0.0, 55.0, 50)
Ms = Float64[]
dips = Float64[]
for B in Bs
    set_external_field!(sys, [0,0,B])
    minimize_energy!(sys)

    M = magnetization(sys)
    push!(Ms, M)

    dip = avg_dipole(sys)
    push!(dips, dip)
end

fig = Figure(; resolution = (1200,500))
ax = Axis(fig[1,1]; ylabel = "M (μ_B/Fe²⁺)", xlabel = "μ₀B (T)")
scatter!(ax, Bs, Ms)
ax = Axis(fig[1,2]; ylabel = "Mean |S|", xlabel = "μ₀B (T)")
scatter!(ax, Bs, dips)
fig

# Let's also examine what the real space spin configuration looks like at
# a number of field values.

dims = (2,2,2)
sys, cryst = BFSO(dims)
B = 20.0

randomize_spins!(sys)
set_external_field!(sys, [0,0,B])
minimize_energy!(sys; maxiters=1000)

plot_spins(sys)



# ## The statistical mechanics of generalized spins

# Because there are more degrees of freedom associated with a generalized spin,
# there is also more entropy. In particular, to evaluate the partition function,
# one no longer integrates over a sphere, but instead over a higher-dimensional
# manifold (CP^(N-1)). For this reason, many familiar formulas for classical
# spins derived in the large-S formalism do not apply when using the SU(N)
# formalism.
#
# We'll give an illustration of such a difference by again examining our
# single-site "hard-axis" Hamiltonian, for which analytical results may
# be derived.

## Make a crystal with no space-group symmetries
latvecs = lattice_vectors(1, 1, 1, 90, 90, 90)
positions = [[0,0,0]]
cryst = Crystal(latvecs, positions, 1)

## System and integrator parameters
D = -1.0
L = 20   # number of (non-interacting) sites
λ = 1.0
Δt = 0.01
seed = 1

## Make the system
sys = System(cryst, (L,1,1), [SpinInfo(1, S=1, g=2)], :SUN; seed)
S = spin_operators(sys, 1)
set_onsite_coupling!(sys, D*S[3]^2, 1)

## Let's first estimate how long it takes to thermalize the system.
randomize_spins!(sys)
dur = 40.0
nsteps = round(Int, dur/Δt)
kT = 0.1
integrator = Langevin(Δt; λ, kT)

Es = zeros(nsteps)
for i in 1:nsteps
    step!(sys, integrator)
    Es[i] = energy(sys)
end

ts = collect(0:(nsteps-1)) .* Δt
lines(ts, Es; axis=(xlabel="Time (J⁻¹)", ylabel="Energy (J)"))

# It looks like the system is thermalized after about 5 J⁻¹. Since this is a
# relatively low temperature, this duration should be sufficient for higher
# temperatures as well.

therm_dur = 5.0
collect_dur = 5.0

# Let's now write functions to peform thermalization
# and to calcualte the mean energy.


function thermalize!(sys, langevin, dur)
    Δt = langevin.Δt
    numsteps = round(Int, dur/Δt)
    for _ in 1:numsteps
        step!(sys, langevin)
    end
end

function calc_mean_energy(sys, langevin, dur)
    L = size(sys.dipoles)[1]
    numsteps = round(Int, dur/langevin.Δt)
    Es = zeros(numsteps)
    for i in 1:numsteps
        step!(sys, langevin)
        Es[i] = energy(sys) / L
    end
    sum(Es)/length(Es) 
end


# Finally, let's calculate the mean energy at a range of temperatures.

lo, hi = 0.0, 2.0
kTs_test = lo:0.2:hi 
kTs_ref = lo:0.01:hi

randomize_spins!(sys)

μs = zeros(length(kTs_test))
for (i, kT) in enumerate(kTs)
    integrator.kT = kT
    thermalize!(sys, integrator, therm_dur)
    μs[i] = calc_mean_energy(sys, integrator, collect_dur)
end

function su3_mean_energy(kT, D)
    a = D/kT
    return D * (2 - (2 + 2a + a^2)*exp(-a)) / (a * (1 - (1+a)*exp(-a)))
end 

E_ref = su3_mean_energy.(kTs_ref, D)

scatter(kTs_test, μs; axis=(xlabel="Temperature (J)", ylabel="<E>"))
lines!(kTs_ref, E_ref)



# ## How to get large-S 

# Whenever you make a spin system in `:dipole` mode, you will be using the RCS
# theory. This section will instead be focused on how you can "undo" the RCS
# theory so that you can get a sense of what it does for you. We will consider a
# simple

latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])

view_crystal(cryst, 2.0)

units = Sunny.Units.theory
sys_sun     = System(cryst, (10, 10, 1), [SpinInfo(1, S=1, g=1)], :SUN; units)
sys_rcs     = System(cryst, (10, 10, 1), [SpinInfo(1, S=1, g=1)], :dipole; units)
sys_large_S = System(cryst, (10, 10, 1), [SpinInfo(1, S=1, g=1)], :dipole; units)

## Model parameter
J = 1.0
h = 0.1 
D = 0.05

## Set exchange interactions
set_exchange!(sys_sun, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_rcs, J, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_large_S, J, Bond(1, 1, [1, 0, 0]))

## Single-ion anisotropy
Ss = spin_operators(sys_sun, 1)
set_onsite_coupling!(sys_sun, D*Ss[3]^2, 1)

Ss = spin_operators(sys_rcs, 1)
set_onsite_coupling!(sys_rcs, D*Ss[3]^2, 1)

set_onsite_coupling!(sys_large_S, D*large_S_spin_operators[3]^2, 1)

## External field
set_external_field!(sys_large_S, h*[0,0,3])
set_external_field!(sys_rcs, h*[0,0,3])
set_external_field!(sys_sun, h*[0,0,3])


names = ["Large S (SpinW)", "SU(N)", "RCS"]
syss = [sys_large_S, sys_sun, sys_rcs]

fig = Figure(resolution=(1200,400))
for (n, (sys, name)) in enumerate(zip(syss, names))
    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=1000)
    sys_min = reshape_supercell(sys, [1 -1 0; 1 1 0; 0 0 1])
    swt = SpinWaveTheory(sys_min)

    path, xticks = reciprocal_space_path(cryst, [[0,0,0], [1/2, 1/2,0], [1,1,0]], 50)
    formula = intensity_formula(swt, :trace; kernel=delta_function_kernel)
    disp, intensity = intensities_bands(swt, path, formula)

    ax = Axis(fig[1,n]; xlabel="𝐪", ylabel="Energy (meV)", xticks, xticklabelrotation=π/6, title=name)
    ylims!(ax, 0.0, 10.0)
    xlims!(ax, 1, size(disp, 1))
    colorrange = extrema(intensity)
    for i in axes(disp)[2]
        lines!(ax, 1:length(disp[:,i]), disp[:,i]; color=intensity[:,i], colorrange)
    end
end
fig