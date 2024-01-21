using Sunny, GLMakie, LinearAlgebra, StaticArrays, Statistics

cif = "MgCr2O4_160953_2009.cif"
xtal_mgcro = Crystal(cif; symprec=0.001)
xtal_mgcro = subcrystal(xtal_mgcro,"Cr1")

dims = (3, 3, 3)  # Supercell dimensions 
spininfos = [SpinInfo(1, S=3/2, g=2)]  # Specify spin information, note that all sites are symmetry equivalent 
sys_mgcro = System(xtal_mgcro, dims, spininfos, :dipole); # Same on MgCr2O4 crystal
randomize_spins!(sys_mgcro)

J1      = 3.27/(3/2)*5/2  # value of J1 in meV from Bai's PRL paper
J_mgcro = [1.00,0.0815,0.1050,0.0085]*J1; # PRL model
set_exchange!(sys_mgcro, J_mgcro[1], Bond(1, 2, [0,0,0]))  # J1
set_exchange!(sys_mgcro, J_mgcro[2], Bond(1, 7, [0,0,0]))  # J2
set_exchange!(sys_mgcro, J_mgcro[3], Bond(1, 3, [1,0,0]))  # J3a -- Careful here!  
set_exchange!(sys_mgcro, J_mgcro[4], Bond(1, 3, [0,0,0])); # J3b -- And here!

kT_20K = 1.8   # Desired temperature in meV

dsc = compute_dynamical_correlations(sys_mgcro
  ;nsample = 30
  ,Δtlangevin = 0.05
  ,Δtmidpoint = 0.05
  ,nω = 30
  ,ωmax = 30.
  ,kT = kT_20K
  ,λ = 0.1, alg = :window)

params_lores = unit_resolution_binning_parameters(dsc)
params_lores.binstart[1:3] .-= 2
params_lores.binend[1:3] .+= 1

#formula_sum_rule = intensity_formula(dsc, :trace, kT = Inf)
formula = intensity_formula(dsc, :perp, kT = kT_20K)
Sqw, counts = intensities_binned(dsc,params_lores,formula)
Sqw ./= counts
Sqw[counts .== 0.] .= 0
Sq = sum(Sqw,dims = 4) # Integrate out ω (correct because intensities_binned is correct)

#params_hires = copy(params_lores)

# High resolution in X and Y:
#params_hires.binwidth[1:2] ./= 1

# Restrict to just the Z bin at zero:
#params_hires.binstart[3] = -params_hires.binwidth[3]/2
#params_hires.binend[3] = 0

include("../realspace/classical.jl") # Import periodic extension
bsc = rep_sc(sys_mgcro,dsc,(4,4,1))
params_hires = unit_resolution_binning_parameters(bsc)
params_hires.binstart[1:3] .-= 2
params_hires.binend[1:3] .+= 1
formula_periodic = intensity_formula(bsc,:trace,kT = kT_20K)
Sq_periodic_extension = intensities_binned(bsc,params_hires,formula_periodic)

nothing
