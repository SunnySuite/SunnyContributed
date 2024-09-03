include("spidy-sunny.jl")

ωL = 1 # Larmor frequency (reference time scale)
Δt = 0.1 / ωL # time step for the dynamics evaluation
tend = 150 / ωL # final time of the dynamics
N = round(Int, tend/Δt) # number of total steps
tspan = (0, N*Δt) # tuple of initial and final time
saveat = (0:1:N)*Δt # vector of times at which the solution is saved
α = 10 * ωL # Lorentzian coupling amplitude
ω0 = 7 * ωL # Lorentzian resonant frequency
Γ = 5 * ωL # Lorentzian width
Jsd = LorentzianSD(α, ω0, Γ) # Lorentzian spectral density
Cw = IsoCoupling(1) # isotropic coupling tensor
# the resulting coupling tensor is equivalent to the following
# Cw = AnisoCoupling([1 0 0
#                     0 1 0
#                     0 0 1]);
T = 0.8 * ωL # temperature at which the dynamics takes place (where ħ=1, kB=1)
noise = ClassicalNoise(T) # noise profile for the stochastic field
s0 = [1.0; 0.0; 0.0] # initial conditions of the spin vector for the dynamics
ntraj = 10 # number of trajectories (stochastic realizations)

### running the dynamics ###
sols = zeros(ntraj, 3, length(saveat)) # solution matrix
for i in 1:ntraj # iterations through the number of trajectories
    # we use the Lorentzian spectral density Jsd to generate the stochastic
    # field. This ensures the field obeys the FDR as noted in the main text
    local bfields = [bfield(N, Δt, Jsd, noise),
                     bfield(N, Δt, Jsd, noise), # vector of independent
                     bfield(N, Δt, Jsd, noise)] # stochastic fields
    # diffeqsolver (below) solves the system for the single trajectory
    local sol = diffeqsolver(s0, tspan, Jsd, bfields, Cw; saveat=saveat)
    sols[i,:,:] = sol[:,:] # store the trajectory into the matrix of solutions
end
