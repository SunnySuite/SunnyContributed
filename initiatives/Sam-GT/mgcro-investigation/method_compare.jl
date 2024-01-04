using Sunny # The main package
using GLMakie # Plotting package
using LinearAlgebra, StaticArrays, Statistics, ProgressMeter

function mk_fresh_mgcro()
  cif = "MgCr2O4_160953_2009.cif"
  xtal_mgcro = Crystal(cif; symprec=0.001)
  xtal_mgcro = subcrystal(xtal_mgcro,"Cr1")

  dims = (3, 3, 1)  # Supercell dimensions 
  spininfos = [SpinInfo(1, S=3/2, g=2)]  # Specify spin information, note that all sites are symmetry equivalent 
  sys_mgcro = System(xtal_mgcro, dims, spininfos, :dipole); # Same on MgCr2O4 crystal

  #print_symmetry_table(xtal_mgcro, 6.0) 

  #view_crystal(xtal_mgcro, 5.9)
   
  J1      = 3.27/(3/2)*5/2  # value of J1 in meV from Bai's PRL paper
  J_mgcro = [1.00,0.0815,0.1050,0.0085]*J1; # PRL model

  ## === Assign exchange interactions to MgCr2O4 system ===
  set_exchange!(sys_mgcro, J_mgcro[1], Bond(1, 2, [0,0,0]))  # J1
  set_exchange!(sys_mgcro, J_mgcro[2], Bond(1, 7, [0,0,0]))  # J2
  set_exchange!(sys_mgcro, J_mgcro[3], Bond(1, 3, [1,0,0]))  # J3a -- Careful here!  
  set_exchange!(sys_mgcro, J_mgcro[4], Bond(1, 3, [0,0,0])); # J3b -- And here!

  randomize_spins!(sys_mgcro);
  sys_mgcro
end

function compute_instant_correlations(sys; nsample, λ, kT, Δt)
  isc = instant_correlations(sys)
  langevin = Langevin(Δt; λ, kT);
  sample_from_equilibrium(sys,langevin;nsample) do
    add_sample!(isc, sys)
  end
  isc
end

function compute_dynamical_correlations(sys; nsample, Δtlangevin, Δtmidpoint, nω, ωmax, kT, λ)
  dsc = dynamical_correlations(sys; Δt = Δtmidpoint, nω, ωmax)
  langevin = Langevin(Δtlangevin; λ, kT);
  sample_from_equilibrium(sys,langevin;nsample) do
    add_sample!(dsc, sys; alg = :window)
  end
  dsc
end

include("../susceptibility/online_correlations.jl")
function compute_online_correlations(sys; Δt, nt, nt_record, kT, λ, measperiod, thermal_steps = 2000)
  langevin = Langevin(Δt; kT, λ);

  println("Thermalizing")
  for _ in 1:thermal_steps
      step!(sys, langevin)
  end

  oc = mk_oc(sys; measperiod,nt, integrator = langevin, observables = nothing, correlations = nothing)
  prog = Progress(nt_record;desc = "Walking: ")
  for l = 1:nt_record
    walk_online!(oc)
    next!(prog)
  end
  finish!(prog)
  oc
end


function sample_from_equilibrium(f::Function,sys,langevin; thermal_steps = 2000, decorr_steps = 500,nsample = 30)
  # We can now thermalize our systems by running the integrator.
  println("Thermalizing")
  for _ in 1:thermal_steps
      step!(sys, langevin)
  end

  prog = Progress(nsample;desc = "Sampling: ")
  for i in 1:nsample
      for _ in 1:decorr_steps
          step!(sys, langevin)
      end
      f()
      next!(prog)
  end
  finish!(prog)
end

kT_20K = 1.8   # Desired temperature in meV

## Dynamical correlation methods
sys = mk_fresh_mgcro()
dsc = compute_dynamical_correlations(sys
  ;nsample = 30
  ,Δtlangevin = 0.05
  ,Δtmidpoint = 0.05
  ,nω = 30
  ,ωmax = 30.
  ,kT = kT_20K
  ,λ = 0.1)

params_dyn = unit_resolution_binning_parameters(dsc;negative_energies = true)
params_dyn.binstart[1:2] .-= 4
params_dyn.binend[1:2] .+= 3
#params_dyn.binstart[1:2] .-= 2
#params_dyn.binend[1:2] .+= 1
params_dyn.binstart[3] -= 1
params_dyn.binend[3] += 1

### Submethods: temperature correction
formula_corrected = intensity_formula(dsc, :trace, kT = kT_20K)
is, counts = intensities_binned(dsc,params_dyn,formula_corrected)
is ./= counts
is[counts .== 0.] .= 0
is_corrected = sum(is,dims=4)

formula_uncorrected = intensity_formula(dsc, :trace)
is, counts = intensities_binned(dsc,params_dyn,formula_uncorrected)
is ./= counts
is[counts .== 0.] .= 0
is_uncorrected = sum(is,dims=4)

## Instant correlation method
sys = mk_fresh_mgcro()
isc = compute_instant_correlations(sys; nsample = 90, λ = 0.1, kT = kT_20K, Δt = 0.05)

params = copy(params_dyn)
integrate_axes!(params,axes = 4)

formula_instant = intensity_formula(isc, :trace)
is, counts = intensities_binned(isc,params,formula_instant)
is_instant = sum(is ./ counts,dims = 4)

sys = mk_fresh_mgcro()
oc = compute_online_correlations(sys
  ;nt = 89
  ,nt_record = 600
  ,Δt = 0.05
  ,measperiod = 2
  ,kT = kT_20K
  ,λ = 0.1)

sc_online = online_to_sampled(oc)

params_online = unit_resolution_binning_parameters(sc_online;negative_energies = true)
params_online.binstart[1:2] .-= 4
params_online.binend[1:2] .+= 3
#params_online.binstart[1:2] .-= 2
#params_online.binend[1:2] .+= 1
params_online.binstart[3] -= 1
params_online.binend[3] += 1

formula_corrected = intensity_formula(sc_online, :trace, kT = 1.8)
is, counts = intensities_binned(sc_online,params_online,formula_corrected)
is ./= counts
is[counts .== 0.] .= 0
is_online_corrected = sum(is,dims=4)

formula_uncorrected = intensity_formula(sc_online, :trace)
is, counts = intensities_binned(sc_online,params_online,formula_uncorrected)
is ./= counts
is[counts .== 0.] .= 0
is_online_uncorrected = sum(is,dims=4)

sys = mk_fresh_mgcro()
is_scga = scga_bincenters(params,sys,1/kT_20K)


println("Sum rule for is_instant:")
nbzs = map(x -> x[end] - x[1],Sunny.axes_binedges(params))[1:3]
A = (sys.gs[1][1] * sys.κs[1])^2 * length(Sunny.eachsite(sys)) * prod(nbzs)
println("  A = (g * κ)² × num_sites × num_BZs = $(A)")
println("  sum(is_instant)/A = $(sum(is_instant) / A)")
println("  sum(is_uncorrected)/A = $(sum(is_uncorrected) / A)")
println("  sum(is_corrected)/A = $(sum(is_corrected) / A)")
println("  sum(is_online_uncorrected)/A = $(sum(is_online_uncorrected) / A)")
println("  sum(is_online_corrected)/A = $(sum(is_online_corrected) / A)")
println("  sum(is_scga)/A = $(sum(is_scga) / A)")

f = Figure()

dats = [is_online_uncorrected, is_online_corrected, is_instant, is_uncorrected, is_corrected, is_scga]
names = ["Online","Online, Corrected","Instant","Sampled","Sampled, Corrected","SCGA"]

bcs = axes_bincenters(params)

for i in eachindex(dats)
  ax = Axis(f[fldmod1(i,3)...],title = names[i])
  heatmap!(ax,bcs[1],bcs[2],dats[i][:,:,2,1],colorrange = (0,500))
end

f
