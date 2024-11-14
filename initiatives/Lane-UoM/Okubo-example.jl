using Sunny, GLMakie

a, c = 1.0, 10.0
latvecs = lattice_vectors(a, a, c, 90, 90, 120)
positions = [[0, 0, 0]]
cryst = Crystal(latvecs, positions)
   
J₁ = -1
J₂ = 0.0
J₃ = -3J₁
qmod = 2*acos(0.25*(1+sqrt(1-2*(J₁/J₃))))
x=2π/(qmod)
x_rat=rationalize(x;tol=0.001)
dims = (denominator(x_rat),denominator(x_rat),1)
spinfos = [1 => Moment(s=1,g=1)]
sys = System(cryst, spinfos, :dipole;dims, seed=0)
set_exchange!(sys, J₁, Bond(1, 1, [1,0,0]))
set_exchange!(sys, J₂, Bond(1, 1, [1,2,0]))
set_exchange!(sys, J₃, Bond(1, 1, [2,0,0]))

# Parameters from Okubo (close to lattice size 26)
h = -2.5*J₃
kT_target = 0.3*J₃
set_field!(sys, [0, 0, h])
function anneal!(sys, sampler, kTs,nsweeps)
    Es = zeros(length(kTs))        # Buffer for saving energy as we proceed
    for (i, kT) in enumerate(kTs)
        sampler.kT = kT
        for j ∈ 1:nsweeps
            step!(sys, sampler)
        end                
        Es[i] = energy(sys)   # Query the energy
    end
    return Es    # Return the energy values collected during annealing
end;
Δt = 0.02
λ = 0.1

randomize_spins!(sys)
kT_upper =10*kT_target  # some large temperature to start with (in meV)
langevin = Langevin(Δt; kT=kT_upper, λ) # initialize the Langevin integrator
kTs = [kT_target + (kT_upper-kT_target) * 0.9^k  for k in 0:100] # define a temperature schedule
Es = anneal!(sys, langevin, kTs,2_000 ) # run the anneal
langevin.kT = kT_target

for _ in 1:2_000
    step!(sys, langevin)
end

minimize_energy!(sys;maxiters=10_000) 
# you may need to do the simulated anneal more carefully to find the true minimum.

########################################
# Plot scalar spin chirality 
function spin_chirality(s₁, s₂, s₃)
    s₁ ⋅ (s₂ × s₃)
end
include(joinpath(pkgdir(Sunny), "examples", "extra", "Plotting","plotting2d.jl"))

begin 
    fig = Figure()
    ax = Axis(fig[1,1];)
    crange = (-0.5,0.5)
    fig=plot_triangular_plaquettes(spin_chirality, [sys.dipoles];resolution = (1600, 800),fontsize = 48, colorrange = crange  )
    Colorbar(fig[1,2];label = "Spin chirality", colormap = :RdBu , colorrange = crange);
    fig
end


plot_spins(sys;color = [s[3] for s ∈ sys.dipoles] )
energy_per_site(sys) 


measure = ssf_perp(sys; )
swt = SpinWaveTheoryKPM(sys; measure,tol=0.01)


# Γ—K—M-Γ
Γ = [0,0,0]
K = [1/3,1/3,0]
M = [1/2,0,0]

q_points = [Γ,K,M,Γ]
path = q_space_path(cryst, q_points, 150)
Emax = 17.5
σin = 0.025*Emax
kernel = lorentzian(fwhm=σin)

@time begin
    energies = range(0,Emax,150) 
    res = intensities(swt, path;energies,kernel)
    plot_intensities(res)
end

