using Sunny, GLMakie, LinearAlgebra, Random

latvecs = lattice_vectors(1, 1, 20.00, 90, 90, 120);
pos = [[0,0,0]]
cryst = Crystal(latvecs, pos)
rng = MersenneTwister(1643);
lenx=30
leny=30
ranperp=randn(rng,Float64 , lenx*leny)/6
ranpar=randn(rng,Float64 , lenx*leny)/6
gpar=1
gperp = 1
infos=[1 => Moment(s=1/2,g =1 ) ]
dims = (lenx,leny,1)
sys = System(cryst, infos, :dipole;dims, seed=0)
J₁=diagm([1,1,1])
set_exchange!(sys,J₁,Bond(1,1,[-1,0,0]))


sys_inhom=to_inhomogeneous(sys)
for i ∈ 1:lenx*leny
    sys_inhom.gs[i]= [1.0 0.0 0.0;
                        0.0 1.0 0.0;
                        0.0 0.0 1.0+ranperp[i]];
end
hsat = 9*1*0.5
set_field!(sys_inhom, 2.5hsat*[0,0,1.0])
kT_target =0.0
Δt = 0.02
λ = 0.1
langevin = Langevin(Δt; kT=kT_target,damping= λ)

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

kT_upper = 3.0
kTs = [kT_target + (kT_upper-kT_target) * 0.9^k  for k in 0:100] 
randomize_spins!(sys_inhom) 
Es = anneal!(sys_inhom, langevin, kTs,2_000 ) 
for _ in 1:10_000
    step!(sys_inhom, langevin)
end
for i ∈ 1:100
    minimize_energy!(sys_inhom;maxiters=10_000)
end

plot_spins(sys_inhom; color = [S[3] for S ∈ sys_inhom.dipoles])
print_wrapped_intensities(sys_inhom)

measure = ssf_perp(sys_inhom; )
swt = SpinWaveTheoryKPM(sys_inhom; measure,tol=0.01)

# Γ—K—M-Γ
Γ = [0,0,0]
K = [1/3,1/3,0]
M = [1/2,0,0]

q_points = [Γ,K,M,Γ]

path = q_space_path(cryst, q_points, 200)
Emax = 17.5
σin = 0.025*Emax
kernel = lorentzian(fwhm=σin)


@time begin
    energies = range(0,Emax,150) 
    res = intensities(swt, path;energies,kernel)
    plot_intensities(res)
end

