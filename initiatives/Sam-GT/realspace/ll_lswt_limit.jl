using Sunny, LinearAlgebra, StaticArrays

cryst = Crystal(I(3),[[0,0,0],[0.5,0.5,0.5]],1)
sys = System(cryst,(1,1,2),[SpinInfo(1,S=1,g=2),SpinInfo(2,S=1,g=2)],:dipole)

B_z_ext = 2.3
set_external_field!(sys,[0,0,B_z_ext])
#set_exchange!(sys,1.,Bond(1,1,[0,0,1]))
#set_exchange!(sys,1.,Bond(2,2,[0,0,1]))
#set_onsite_coupling!(sys,S -> -0.1 * S[3]^2,1)
#set_onsite_coupling!(sys,S -> -0.1 * S[3]^2,2)

randomize_spins!(sys)
minimize_energy!(sys)
println("Dipoles:")
display(sys.dipoles)

latsize_final = (2,3,4)
#function finite_spin_wave_theory(sys,latsize_final)

  # Enforce periodic system
  cellsize_mag = Sunny.cell_shape(sys) * diagm(Sunny.Vec3(sys.latsize))
  sys_unit = Sunny.reshape_supercell_aux(sys, (1,1,1), cellsize_mag)
  sys_periodic = repeat_periodically(sys_unit,latsize_final)

  swt = SpinWaveTheory(sys_unit)


  isc = instant_correlations(sys_periodic)
  add_sample!(isc,sys_periodic)

sys_periodic_orig = Sunny.clone_system(sys_periodic)
function mk_dt()
  s = Sunny.clone_system(sys_periodic_orig)
  dsc = dynamical_correlations(s;Δt = 0.05,nω = 100,ωmax = 1.0)
  formula_classical_transverse = intensity_formula(dsc,[(:Sx,:Sx),(:Sy,:Sy)]) do k,ω,corr
    @assert imag(corr[1]) < 1e-8
    real(corr[1] + corr[2])
  end
  params = unit_resolution_binning_parameters(dsc;negative_energies=true)
  params.binend[1:2] .+= 1
  params.binend[3] += 3
  function(;kT_use=0.1)
    dsc.data .= 0
    dsc.nsamples[1] = 0
    s.dipoles .= sys_periodic_orig.dipoles
    langevin = Langevin(0.05,λ=0.1,kT=kT_use)
    for j = 1:20
      for i = 1:1000
        step!(s,langevin)
      end
      add_sample!(dsc,s)
    end

    formula_classical_transverse_kT = intensity_formula(dsc,[(:Sx,:Sx),(:Sy,:Sy)],kT = kT_use) do k,ω,corr
      @assert imag(corr[1]) < 1e-8
      real(corr[1] + corr[2])
    end
    is_Sqw_transverse, counts = intensities_binned(dsc,params,formula_classical_transverse)
    is_Sqw_transverse_kT, counts = intensities_binned(dsc,params,formula_classical_transverse_kT)
    sum(is_Sqw_transverse ./ counts)/A, sum(is_Sqw_transverse_kT ./ counts)/A
  end
end
dynamic_transverse = mk_dt()


  kT_use = 0.1
  langevin = Langevin(0.05,λ=0.1,kT=kT_use)
  dsc = dynamical_correlations(sys_periodic;Δt = 0.05,nω = 100,ωmax = 1.0)
  for j = 1:20
    for i = 1:1000
      step!(sys_periodic,langevin)
    end
    add_sample!(dsc,sys_periodic)
  end

  #formula_classical = intensity_formula(isc,:trace)
  formula_classical_static = intensity_formula(isc,:trace)
  formula_classical_trace = intensity_formula(dsc,:trace)
  formula_classical_transverse = intensity_formula(dsc,[(:Sx,:Sx),(:Sy,:Sy)],kT = kT_use) do k,ω,corr
    @assert imag(corr[1]) < 1e-8
    real(corr[1] + corr[2])
  end

  params = unit_resolution_binning_parameters(dsc;negative_energies=true)
  params.binend[1:2] .+= 1
  params.binend[3] += 3
  params_instant = copy(params)
  params_instant = unit_resolution_binning_parameters(isc)
  params_instant.binend[1:2] .+= 1
  params_instant.binend[3] += 3
  bcs = Sunny.axes_bincenters(params)

  is_Sq, counts = intensities_binned(isc,params_instant,formula_classical_static)
  is_Sqw_transverse, counts = intensities_binned(dsc,params,formula_classical_transverse)
  is_Sqw_trace, counts = intensities_binned(dsc,params,formula_classical_trace)
  spin_S = sys.κs[1]
  nbzs = map(x -> x[end] - x[1],Sunny.axes_binedges(params))[1:3]
  A = (sys.gs[1][1] * spin_S)^2 * prod(nbzs)
  println("Classical sum rule: A = $A = (gS)^2 * num BZ")
  println("            sum(is_Sq) / A = $(sum(is_Sq)/A)")
  println("sum(is_Sqw_transverse) / A = $(sum(is_Sqw_transverse)/A)")
  println("     sum(is_Sqw_trace) / A = $(sum(is_Sqw_trace)/A)")

  formula = intensity_formula(swt,:full;kernel = delta_function_kernel)

  # coords = covectors * (q,ω)
  coords_to_q = inv(params.covectors[1:3,1:3])

  is = Array{Sunny.BandStructure,3}(undef,params.numbins[1:3]...)

  # Loop over qs
  for ci in CartesianIndices(params.numbins.data[1:3])
    x_center = bcs[1][ci[1]]
    y_center = bcs[2][ci[2]]
    z_center = bcs[3][ci[3]]

    q = SVector{3}(coords_to_q * [x_center;y_center;z_center])
    is[ci] = formula.calc_intensity(swt,q)
  end

  # Spin wave intensities (after summing over bands) are in S²/BZ. To smear this over a bin,
  # we need [bin size in 1/BZ] * [new value in S²] = [original value in S²/BZ].
  # The bin size as a fraction of the BZ is:
  bin_size = prod(params.binwidth[1:3]) / det(params.covectors[1:3,1:3])

  g_factor = 2

  sw_is_transverse = g_factor^2 * bin_size * map(x -> sum(map(y -> y[1,1] + y[2,2],x.intensity)),is)
  sw_is_trace = g_factor^2 * bin_size * map(x -> sum(map(y -> y[1,1] + y[2,2] + y[3,3],x.intensity)),is)
  println("Now for spin wave:")
  println(" sum(sw_is_transverse) / A = $(sum(sw_is_transverse)/A)")
  println("      sum(sw_is_trace) / A = $(sum(sw_is_trace)/A)")
#end

#finite_spin_wave_theory(sys,(2,3,4))
