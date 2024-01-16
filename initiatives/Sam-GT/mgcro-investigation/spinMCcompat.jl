using Sunny
using SpinMC # ] add https://github.com/fbuessen/SpinMC.jl
using JuliaSCGA # ] add https://github.com/moon-dust/JuliaSCGA.jl
using LinearAlgebra, StaticArrays, Statistics

SMC = JuliaSCGA

function to_spinmc_crystal(cryst)
  a1 = Tuple(cryst.latvecs[:,1])
  a2 = Tuple(cryst.latvecs[:,2])
  a3 = Tuple(cryst.latvecs[:,3])
  uc = SMC.UnitCell(a1,a2,a3)
  for j = 1:length(cryst.positions)
    @assert j == SMC.addBasisSite!(uc, Tuple(cryst.latvecs * cryst.positions[j]))
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
      spinS = sys.κs[1]
      SMC.addInteraction!(uc,c.bond.i,c.bond.j,Matrix(spinS * spinS * c.bilin * I(3)),Tuple(c.bond.n))
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
  run!(m)#, outfile="simulation.h5") # Run simulation and write result file "simulation.h5".
  m
end

function get_sim_sf(m,L)
  lattice = m.lattice
  # Fourier transform correlations to compute structure factor. 
  N = 256
  correlation = mean(m.observables.correlation) # The correlation is measured with respect to spins on the lattice basis sites, i.e. the (i,j)-th entry of the matrix is the correlation dot(S_i,S_j), where i runs over all lattice sites and j runs over all basis sites. 
  kx = collect(range(-2pi,2pi,length=N))
  ky = collect(range(-2pi,2pi,length=N))
  structurefactor = zeros(N,N)
  for i in 1:N
      for j in 1:N
          z = 0.0
          # Compute Fourier transformation at momentum (kx, ky). The real-space position of the i-th spin is obtained via getSitePosition(lattice,i). 
          for b in 1:length(lattice.unitcell.basis)
              for k in 1:length(lattice)
                  z += cos(dot((kx[i],ky[j],L),getSitePosition(lattice,k).-getSitePosition(lattice,b))) * correlation[k,b]
              end
          end
          structurefactor[j,i] = z / (length(lattice) * length(lattice.unitcell.basis))
      end
  end
  structurefactor
end

function scga_bincenters(params,sys,beta)
  uc = to_spinmc_system(sys)
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

      # absolute wave vector of bin center
      k = sys.crystal.recipvecs * coords_to_q * [x_center;y_center;z_center]

      # This satisfies q_lab = transpose(recipvecs) * q_crys
      q_crys[i,:] .= inv(transpose(sys.crystal.recipvecs)) * k

      # Later, in the SCGA code, this is used as:
      #
      #   r1 = uc.basis[bond[1]]
      #   r2 = uc.basis[bond[2]] + ∑ᵢ bond[3][i] .* uc.primitive[i]
      #   dist = r1 - r2
      #   dot(q_lab,dist)
      #
      # which means that uc.basis[1] must be in physical units
  end
  Jq_calc = getFourier_iso(uc,dist,q_crys)
  λ = solveLambda_iso(uc,beta)
  correl = getCorr_iso(uc,Jq_calc,beta,λ)
  for (i,ci) in enumerate(cis)
    is[ci] = correl[i]
  end

  # SCGA reports the intensity, so we need to "integrate" it over the bin
  # by multiplying by the binwidth
  is .*= prod(params.binwidth[1:3])
  
  # SCGA uses unit-length dimensionless spins, so we need to restore the units
  spinS = sys.κs[1]
  is .*= spinS * spinS

  # SCGA computes the correlation per site (for some reason??) so we need
  # to convert back to the actual mean correlation here:
  #is .*= length(Sunny.eachsite(sys))

  is
end
