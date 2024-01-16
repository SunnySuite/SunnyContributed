include("method_compare.jl")

function mk_sys()
  cryst = Crystal(I(3),[[0.,0,0]],1)
  sys = System(cryst, (6,6,1), [SpinInfo(1;S=1/2,g=1)], :SUN, units = Units.theory)
  set_external_field!(sys,[0,0,1.2])
  #set_onsite_coupling!(sys, S -> -1.0 * S[1]^2,1)
  J = -1.0
  set_exchange!(sys,J,Bond(1,1,[0,1,0]))
  set_exchange!(sys,J,Bond(1,1,[1,0,0]))
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 3000)

  sys
end

#=
function longitudinal_part_lswt(swt,params_qgrid)
  # Number of magnetic atoms
  Nm = length(swt.sys.dipoles)
  N = swt.sys.Ns[1]
  nmodes = Sunny.nbands(swt)
  bin_centers = axes_bincenters(params_qgrid)

  # coords = covectors * (q,ω)
  coords_to_q = inv(params_qgrid.covectors[1:3,1:3])

  #is = zeros(Float64,params_qgrid.numbins...)
  H = zeros(ComplexF64, 2*nmodes, 2*nmodes)
  V = zeros(ComplexF64, 2*nmodes, 2*nmodes)
  Avec_pref = zeros(ComplexF64, Nm)
  #intensity = zeros(return_type, nmodes)

  # Loop over qs
  for ci in CartesianIndices(params_qgrid.numbins.data[1:3])
    x_center = bin_centers[1][ci[1]]
    y_center = bin_centers[2][ci[2]]
    z_center = bin_centers[3][ci[3]]

    q = SVector{3}(coords_to_q * [x_center;y_center;z_center])
    ωvals = bin_centers[4]

    #intensity_as_function_of_ω = formula.calc_intensity(swt,q)
    #is[ci,:] .= intensity_as_function_of_ω(ωvals)

    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)
    q_absolute = swt.sys.crystal.recipvecs * q_reshaped

    @assert sys.mode == :SUN
    swt_hamiltonian_SUN!(H, swt, q_reshaped)
    #swt_hamiltonian_dipole!(H, swt, q_reshaped)

    disp = try
        bogoliubov!(V, H)
    catch e
        error("Instability at wavevector q = $q")
    end

    for i = 1:Nm
        @assert Nm == natoms(sys.crystal)
        phase = exp(-2π*im * dot(q_reshaped, sys.crystal.positions[i]))
        Avec_pref[i] = sqrt_Ns * phase

        # TODO: move form factor into `f`, then delete this rescaling
        Avec_pref[i] *= compute_form_factor(ff_atoms[i], q_absolute⋅q_absolute)
    end

    Nobs = num_observables(swt.observables)
    longitudinal_part = zeros(ComplexF64,Nm,Nobs)
    for band = 1:nmodes
      v = reshape(view(V, :, band), N-1, Nm, 2)
      Avec = zeros(ComplexF64, Nobs)
      (; observable_operators) = data
      @views O = observable_operators[:, :, μ, i]
      for obs = 1:Nobs, atom = 1:Nm
        for i = eachindex(is_full)
          Sab = is_full[i]
          longitudinal_part[atom,obs] = 
        end
      end
    end
  end
end
=#

kT_use = 0.1

sys = mk_sys()
dsc = compute_dynamical_correlations(sys
    ;nsample = 30
    ,Δtlangevin = 0.005
    ,Δtmidpoint = 0.005
    ,nω = 200
    ,ωmax = 8.
    ,kT = kT_use
    ,λ = 0.1)

params = unit_resolution_binning_parameters(dsc;negative_energies = true)
params.binstart[1:2] .-= 1
params.binend[1:2] .+= 1
#params.binstart[3] -= 1
#params.binend[3] += 1

#params.binwidth[3] /= 8
#params_sw.binwidth[4] /= 8
#params.binend[3] = 1.5
#params.binend[4] = -params.binstart[4]

ωs = axes_bincenters(params)[4]
betaOmegaCorrection = reshape(ωs ./ kT_use,(1,1,1,length(ωs)))

sig = 1.8

formula_corrected = intensity_formula(dsc, [9,5,1], kT = kT_use) do k,w,Sii
  real(Sii[2] + Sii[1])
end
is_full_corrected, counts = intensities_binned(dsc,params,formula_corrected)
#is_full_corrected ./= counts
#is_full_corrected[counts .== 0.] .= 0
# is comes in units including ×fs, and we integrate over exactly [0,fs]; this is the same as taking the mean.
# The remaining quantity has units of "S^2 per BZ"
is_corrected = sum(is_full_corrected,dims=4) / size(is_full_corrected,4)

formula_uncorrected = intensity_formula(dsc, [9,5,1]) do k,w,Sii
  real(Sii[2] + Sii[1])
end
is_full_uncorrected, counts = intensities_binned(dsc,params,formula_uncorrected)
#is_full_uncorrected ./= counts
#is_full_uncorrected[counts .== 0.] .= 0
is_uncorrected = sum(is_full_uncorrected,dims=4) / size(is_full_uncorrected,4)

formula_uncorrected_tr = intensity_formula(dsc, :trace)
is_full_uncorrected_tr, counts = intensities_binned(dsc,params,formula_uncorrected_tr)
#is_full_uncorrected_tr ./= counts
#is_full_uncorrected_tr[counts .== 0.] .= 0
is_uncorrected_tr = sum(is_full_uncorrected_tr,dims=4) / size(is_full_uncorrected_tr,4)

formula_corrected_tr = intensity_formula(dsc, :trace, kT = kT_use)
is_full_corrected_tr, counts = intensities_binned(dsc,params,formula_corrected_tr)
#is_full_corrected_tr ./= counts
#is_full_corrected_tr[counts .== 0.] .= 0
is_corrected_tr = sum(is_full_corrected_tr,dims=4) / size(is_full_corrected_tr,4)

zz_only = is_uncorrected_tr .- is_uncorrected
zz_only_corr = is_corrected_tr .- is_corrected

#=
formula_projection = intensity_formula(dsc,[9,5,1]) do k,w,Sii
  real(Sii[2] + Sii[1])
end
is, counts = intensities_binned(dsc,params,formula_projection)
is ./= counts
is[counts .== 0.] .= 0
is_projected = sum(is,dims=4) / size(is,4)
=#

params_sw = copy(params)
#params_sw.binwidth[1:2] ./= 4
#params_sw.binwidth[3] /= 8
#params_sw.binwidth[4] /= 8
#params_sw.binend[3] = 1.5
#params_sw.binend[4] = -params_sw.binstart[4]

#sys_re = reshape_supercell(sys, [1 -1 0; 1 1 0; 0 0 1])
sys_re = Sunny.clone_system(sys)
minimize_energy!(sys_re)
swt = SpinWaveTheory(sys_re)
formula_lswt = intensity_formula(swt,[9,5,1],kernel = delta_function_kernel) do k,w,Sii
  real(Sii[2] + Sii[1])
end

is_full_lswt, counts = Sunny.intensities_bin_multisample(swt, params_sw,[[0.5,0.5,0.5]],[], formula_lswt)
#is_full_lswt ./= counts
#is_full_lswt[counts .== 0.] .= 0
# LSWT intensities have units "per BZ". Additionally, for broadened LSWT, when the result is sampled
# with spacing Δω in energy, we need to multiply by Δω when we sum along the energy axis. This way,
# the lorentzians will sum to one.
is_lswt = sum(is_full_lswt,dims=4) * prod(params_sw.binwidth[1:3]) #* 2π
nbzs_sw = map(x -> x[end] - x[1],Sunny.axes_binedges(params_sw))[1:3]

#Aswt = (1/2) * (1 + 1/2) * prod(nbzs_sw)
Aswt = (1/2) * (1/2) * prod(nbzs_sw)

# Sum rule:
nbzs = map(x -> x[end] - x[1],Sunny.axes_binedges(params))[1:3]
A = (sys.gs[1][1] * sys.κs[1])^2 * length(Sunny.eachcell(sys)) * prod(nbzs)

f = Figure()

pars = [params,params,params_sw]
dats = [is_uncorrected[:,:,1,1], is_corrected[:,:,1,1],(params.binwidth[3] / params_sw.binwidth[3]) * is_lswt[:,:,1,1]]
names = ["Sampled","Sampled, Corrected","LSWT"]

for i in eachindex(dats)
  ax = Axis(f[fldmod1(i,3)...],title = names[i])
  bcs = axes_bincenters(pars[i])
  heatmap!(ax,bcs[1],bcs[2],dats[i],colorrange = (0,5))
end

f

