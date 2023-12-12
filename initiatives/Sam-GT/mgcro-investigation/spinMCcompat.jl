using Sunny
using SpinMC
using JuliaSCGA
using LinearAlgebra, StaticArrays

SMC = JuliaSCGA

function to_spinmc_crystal(cryst)
  a1 = Tuple(cryst.latvecs[:,1])
  a2 = Tuple(cryst.latvecs[:,2])
  a3 = Tuple(cryst.latvecs[:,3])
  uc = SMC.UnitCell(a1,a2,a3)
  for j = 1:length(cryst.positions)
    SMC.addBasisSite!(uc, Tuple(cryst.positions[j]))
  end
  uc
end

function to_spinmc_system(sys)
  uc = to_spinmc_crystal(sys.crystal)
  for j = 1:length(sys.interactions_union) # Loop over basis sites
    this_site_pair_couplings = sys.interactions_union[j].pair
    for k = 1:length(this_site_pair_couplings) # Loop over couplings
      c = this_site_pair_couplings[k]
      if c.isculled
        continue # SpinMC doesn't want to know about the culled bonds
      end
      if !iszero(c.biquad)
        error("Unsupported biquadratic interaction $c")
      end
      if !isempty(c.general.data)
        error("Unsupported general interaction $c")
      end
      if !iszero(c.scalar)
        error("Unsupported scalar interaction $c")
      end
      SMC.addInteraction!(uc,c.bond.i,c.bond.j,Matrix(c.bilin * I(3)),Tuple(c.bond.n))
    end
  end
  #SMC.Lattice(uc,sys.latsize)
  uc
end

function run_sim(lat)
  # Define simulation parameters
  thermalizationSweeps = 50000 # Number of sweeps to thermalize the system. 
  measurementSweeps = 50000 # Number of sweeps after thermalization to perform measurements.
  beta = 10.0 #inverse temperature

  # Create and run simulation
  m = MonteCarlo(lat, beta, thermalizationSweeps, measurementSweeps)
  run!(m, outfile="simulation.h5") # Run simulation and write result file "simulation.h5".
end

function scga_bincenters(params,sys,beta)
  uc = to_spinmc_system(sys)
  #uc = lat.unitcell
  dist = getDist(uc)

  bin_centers = axes_bincenters(params)

  # coords = covectors * (q,ω)
  coords_to_q = inv(params.covectors[1:3,1:3])

  is = zeros(Float64,params.numbins[1:3]...)

  cis = CartesianIndices(params.numbins.data[1:3])
  q_crys = Matrix{Float64}(undef,length(cis),3)
  # Loop over qs
  for (i,ci) in enumerate(cis)
      x_center = bin_centers[1][ci[1]]
      y_center = bin_centers[2][ci[2]]
      z_center = bin_centers[3][ci[3]]

      q = coords_to_q * [x_center;y_center;z_center]
      q_crys[i,:] .= q
  end
  Jq_calc = getFourier_aniso(uc,dist,q_crys)
  λ = solveLambda_aniso(uc,beta)
  correl = getCorr_aniso(uc,q_crys,Jq_calc,beta,λ)
  for (i,ci) in enumerate(cis)
    is[ci] = correl[i]
  end
  is
end
