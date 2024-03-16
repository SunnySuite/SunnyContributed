using Sunny, LinearAlgebra, StaticArrays, GLMakie

# Do spin wave theory on a "finite system" by:
#
# - Collapsing to the crystallographic unit cell
#   (periodizing the ground state with period the crystallographic cell; equiv. sampling in BZ with Δq = an entire BZ)
#
# - Performing Spin Wave analysis on the collapsed system
#   (Spin Wave assumes a continuum in reciprocal space; equiv. an infinitely extended system in real space)
#
# - Sampling the resulting S(q,ω) from spin wave, with Δq = 1/latsize
#   (Sampling in k-space; equiv. periodizing the infinite system to be of size latsize in real space)
#
# - Re-rendering the ground state in real space
function finite_spin_wave(sys)
  # Assess if all unit cells have the same data
  all_repeats = allequal([sys.dipoles[i,:] for i = Sunny.eachcell(sys)]) && allequal([sys.coherents[i,:] for i = Sunny.eachcell(sys)])
  if !all_repeats
    @warn "Not all unit cells in the system have the same spin data! Collapsing to first unit cell anyway"
  end

  sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
  swt = SpinWaveTheory(sys_collapsed)

  # Find polyatomic required number of unit cells:
  na = Sunny.natoms(sys.crystal)
  ΔRs = [map(x -> iszero(x) ? Inf : x,abs.(sys.crystal.positions[i] - sys.crystal.positions[j])) for i = 1:na, j = 1:na]
  nbzs = round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))

  comm_ixs = CartesianIndices(ntuple(i -> sys.latsize[i] .* nbzs[i],3))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  formula = intensity_formula(swt,:full;kernel = delta_function_kernel)

  bs = intensities_band_structure(swt,ks_comm,formula)

  sys_periodic = repeat_periodically(sys_collapsed,sys.latsize)

  isc = instant_correlations(sys_periodic)
  add_sample!(isc,sys_periodic)

  params = unit_resolution_binning_parameters(isc)
  params.binend[1:3] .+= nbzs .- 1

  is_static = intensities_binned(isc,params,intensity_formula(isc,:full))[1]

  enhanced_nbands = Sunny.nbands(swt) + 1
  is_type = typeof(bs[1]).parameters[2]
  enhanced_bs = Array{Sunny.BandStructure{enhanced_nbands,is_type}}(undef,size(bs)...)
  for i = comm_ixs
    x = bs[i]
    enhanced_dispersion = SVector{enhanced_nbands,Float64}([0., x.dispersion...])
    enhanced_intensity = SVector{enhanced_nbands,is_type}([is_static[i], (x.intensity ./ prod(sys.latsize))...])
    enhanced_bs[i] = Sunny.BandStructure{enhanced_nbands,is_type}(enhanced_dispersion,enhanced_intensity)
  end

  is_swt = map(x -> sum(x.intensity),bs) / prod(sys.latsize)

  is_all = is_static + is_swt

  enhanced_bs, is_all, sys_periodic
end

function intensities_band_structure(swt::SpinWaveTheory, ks, formula::Sunny.SpinWaveIntensityFormula)
    if !isnothing(formula.kernel)
        # This is only triggered if the user has explicitly specified a formula with e.g. kT
        # corrections applied, but has not disabled the broadening kernel.
        error("intensities_bands: Can't compute band intensities if a broadening kernel is applied.\nTry intensity_formula(...; kernel = delta_function_kernel)")
    end

    map(k -> formula.calc_intensity(swt,Sunny.Vec3(k)),ks)
end

function plot_bands_1d(band_structures)
  dispersion = hcat(map(x -> x.dispersion, band_structures)...)
  intensity = hcat(map(x -> x.intensity, band_structures)...)

  f = Figure()
  ax = Axis(f[1,1]; xlabel = "Momentum", ylabel = "Energy (meV)", xticklabelsvisible = false)
  ylims!(ax, min(-0.1,minimum(dispersion)), 1.1 * maximum(dispersion))
  nq = size(dispersion,2)
  xlims!(ax, 1, nq)
  colorrange = extrema(intensity)

  # Loop over bands
  for i in axes(dispersion)[1]
    lines!(ax, 1:nq, Vector(dispersion[i,:]); color=Vector(intensity[i,:]), colorrange,linewidth = 5.0)
    scatter!(ax, 1:nq, Vector(dispersion[i,:]); color=Vector(intensity[i,:]), colorrange,markersize = 15)
  end
  Colorbar(f[1,2];colorrange)
  f
end

Base.map(f::Function,bs::Sunny.BandStructure{N,T}) where {N,T} = Sunny.BandStructure(bs.dispersion,map(f,bs.intensity))
