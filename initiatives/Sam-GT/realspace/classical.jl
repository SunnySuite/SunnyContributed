using Sunny
using FFTW

struct DirectSpaceIntensityFormula{T} <: Sunny.IntensityFormula
    kT :: Float64
    formfactors :: Union{Nothing, Vector{FormFactor}}
    string_formula :: String
    calc_intensity :: Function
end

function Base.show(io::IO, formula::DirectSpaceIntensityFormula{T}) where T
    print(io,"DirectSpaceIntensityFormula{$T}")
end

function Base.show(io::IO, ::MIME"text/plain", formula::DirectSpaceIntensityFormula{T}) where T
    printstyled(io, "Direct Space Scattering Intensity Formula\n";bold=true, color=:underline)

    formula_lines = split(formula.string_formula,'\n')

    intensity_equals = "  Intensity(q)[ix_ω] = ∑_Δx exp(iq⋅Δx) "
    println(io,"At any wavevector q, use the discretely many S(Δx)[ix_ω]:")
    println(io)
    println(io,intensity_equals,formula_lines[1])
    for i = 2:length(formula_lines)
        precursor = repeat(' ', textwidth(intensity_equals))
        println(io,precursor,formula_lines[i])
    end
    println(io)

    if isnothing(formula.formfactors)
        printstyled(io, "No form factors specified\n";color=:yellow)
    else
        #printstyled(io, "Form factors included in S ✓\n";color=:green)
        printstyled(io, "Form factors not yet implemented!\n";color=:red)
    end
    if formula.kT == Inf
        printstyled(io, "No temperature correction";color=:yellow)
        print(io, " (kT = ∞)\n")
    else
        printstyled(io, "Temperature corrected (kT = $(formula.kT)) ✓\n";color = :green)
    end
    if T != Float64
        println(io,"Intensity :: $(T)")
    end
end

function intensity_formula_periodic_extension(f::Function, sc::SampledCorrelations, corr_ix::AbstractVector{Int64}; 
    kT = Inf, 
    formfactors = nothing, 
    return_type = Float64, 
    string_formula = "f(Q,ω,S{α,β}[ix_ω])",
    lolim = -Inf, hilim = Inf
    )

  # If temperature given, ensure it's greater than 0.0
  if kT != Inf
      if iszero(kT)
          error("`kT` must be greater than zero.")
      end
      # Only apply c2q factor if have dynamical correlations
      if isnan(sc.Δω)
          error("`kT`-dependent corrections not available when using correlation data generated with `instant_correlations`. Do not set `kT` keyword.")
      end
  end

  ωs_sc = available_energies(sc;negative_energies = true)

  ff_atoms = Sunny.propagate_form_factors_to_atoms(formfactors, sc.crystal)

  real_corr = real.(ifft(sc.data,(4,5,6)))

  # Displacements associated with realspace correlations
  ls = size(sc.data)[4:6]
  dri = []
  for i = 1:3
    push!(dri,fftfreq(ls[i],ls[i]))
  end

  R = Array{Sunny.Vec3,3}(undef,ls...)
  for cell in CartesianIndices(ls)
    R[cell] = sc.crystal.latvecs * [dri[1][cell[1]];dri[2][cell[2]];dri[3][cell[3]]]
  end
  atom_pos = map(p -> sc.crystal.latvecs * p, sc.crystal.positions)
  na = Sunny.natoms(sc.crystal)

  calc_intensity = function(fBack::Function, ix_ω::Int64, q_absolute::Sunny.Vec3)
    val = zeros(ComplexF64,length(corr_ix))
    # Correlations from every offset Δx
    for cell in CartesianIndices(ls), a = 1:na, b = 1:na

      # The detailed Δx includes the lattice displacement and the sublattice displacement
      this_R = R[cell] .+ (atom_pos[a] .- atom_pos[b])

      # Range-resolved correlations
      if !(norm(this_R) > lolim && norm(this_R) < hilim)
        continue
      end

      # TODO: form factors!
      for i = eachindex(val)
        correlation = real_corr[corr_ix[i],a,b,cell,ix_ω]
        val[i] = val[i] + fBack(correlation,this_R) / na # Mean over a
      end
    end

    # This is NaN if sc is instant_correlations
    ω = (typeof(ωs_sc) == Float64 && isnan(ωs_sc)) ? NaN : ωs_sc[ix_ω] 

    return f(q_absolute, ω, val) * Sunny.classical_to_quantum(ω,kT)
  end
  DirectSpaceIntensityFormula{return_type}(kT, formfactors, string_formula, calc_intensity)
end

function intensity_formula_periodic_extension(sc::SampledCorrelations, elem::Tuple{Symbol,Symbol}; kwargs...)
    string_formula = "S(q){$(elem[1]),$(elem[2])}[ix_ω]"
    intensity_formula_periodic_extension(sc,Element(sc, elem); string_formula, kwargs...)
end

function intensity_formula_periodic_extension(sc::SampledCorrelations, mode::Symbol; kwargs...)
    contractor, string_formula = Sunny.contractor_from_mode(sc, mode)
    intensity_formula_periodic_extension(sc, contractor; string_formula, kwargs...)
end

function intensity_formula_periodic_extension(sc::SampledCorrelations, contractor::Sunny.Contraction{T}; kwargs...) where T
    intensity_formula_periodic_extension(sc,Sunny.required_correlations(contractor); return_type = T,kwargs...) do k, ω, correlations
        intensity = Sunny.contract(correlations, k, contractor)
    end
end



function Sunny.intensities_binned(sc::SampledCorrelations, params::BinningParameters,formula::DirectSpaceIntensityFormula;fast_mode = false)
  if any(params.covectors[1:3,4] .!= 0.) || any(params.covectors[4,1:3] .!= 0.)
    error("Complicated binning parameters not supported")
  end
  # coords = covectors * (q,ω)
  # Also the columns of this are the e^i triple duals
  coords_to_q = inv(params.covectors[1:3,1:3])
  coords_to_k = sc.crystal.recipvecs * coords_to_q

  nbin = count_bins(params.binstart,params.binend,params.binwidth)
  is = zeros(Float64,nbin...)

  # Loop over qs
  for bin_number in CartesianIndices(ntuple(i -> nbin[i],3)), ix_ω = 1:nbin[4]

    # This line causes the dipole factor to be applied at the bin center only
    k0 = params.binstart[1:3] .+ params.binwidth[1:3] .* (bin_number.I .- 0.5)

    bin_val = formula.calc_intensity(ix_ω,Sunny.Vec3(k0)) do val, this_R
      for j = 1:3
        sj = params.binstart[j]
        wj = params.binwidth[j]
        left_edge = sj + wj * (bin_number[j]-1)
        ejdx = view(coords_to_k,:,j) ⋅ this_R
        if iszero(ejdx)
          val = val * wj # No variation; can just multiply by bin width!
          continue
        else
          val = val * exp(im * left_edge * ejdx)
          if fast_mode
            val = val * wj
          else
            val = val * (exp(im * wj * ejdx) - 1)
            val = val / (im * ejdx)
          end
        end
      end
      val
    end
    # Missing det(covectors) and 1/binwidth factors here?
    is[bin_number,ix_ω] = bin_val # / prod(params.binwidth[1:3])
  end
  is
end

function Sunny.intensities_interpolated(sc::SampledCorrelations, qs, formula::DirectSpaceIntensityFormula)
  ks = map(q -> sc.crystal.recipvecs * q,qs)
  ωs = available_energies(sc)

  return_type = typeof(formula).parameters[1]
  is = zeros(return_type,size(ks)...,length(ωs))

  # Loop over qs
  for i in eachindex(ks), ix_ω in eachindex(ωs)
    bin_val = formula.calc_intensity(ix_ω,ks[i]) do val, this_R
      val * exp(im * (ks[i] ⋅ this_R))
    end
    # Missing det(covectors) and 1/binwidth factors here?
    is[i,ix_ω] = real(bin_val)
  end
  is
end

function binned_components(sys, sc, params::BinningParameters; lolim = -Inf, hilim = Inf)
    dat = sc.data
    real_corr = real.(ifft(dat,(4,5,6)))
    Sq = sum(real_corr,dims = 7) / size(real_corr,7)

    # Displacements associated with realspace correlations
    dri = []
    for i = 1:3
      push!(dri,fftfreq(sys.latsize[i],sys.latsize[i]))
    end

    R = Array{Sunny.Vec3,3}(undef,sys.latsize...)
    for cell in Sunny.eachcell(sys)
      R[cell] = sys.crystal.latvecs * [dri[1][cell[1]];dri[2][cell[2]];dri[3][cell[3]]]
    end
    atom_pos = map(p -> sys.crystal.latvecs * p, sys.crystal.positions)
    na = Sunny.natoms(sys.crystal)

    if any(params.covectors[1:3,4] .!= 0.) || any(params.covectors[4,1:3] .!= 0.)
      error("Complicated binning parameters not supported")
    end

    # coords = covectors * (q,ω)
    # Also the columns of this are the e^i triple duals
    coords_to_q = inv(params.covectors[1:3,1:3])
    coords_to_k = sc.crystal.recipvecs * coords_to_q

    #nbin = params.numbins[1:3]
    nbin = count_bins(params.binstart,params.binend,params.binwidth)[1:3]
    is = zeros(Float64,nbin...)

    # Loop over qs
    for ci in CartesianIndices(ntuple(i -> nbin[i],3))
      bin_val = 0. + 0im
      for cell in Sunny.eachcell(sys), a = 1:na, b = 1:na

        this_R = R[cell] .+ (atom_pos[a] .- atom_pos[b])
        if !(norm(this_R) > lolim && norm(this_R) < hilim)
          continue
        end

        val = Sq[1,a,b,cell]

        for j = 1:3
          sj = params.binstart[j]
          wj = params.binwidth[j]
          left_edge = sj + wj * (ci[j]-1)
          ejdx = view(coords_to_k,:,j) ⋅ this_R
        #val = val + exp(2π * im * this_R ⋅ q) * Sq[1,i,j,cell]
          if iszero(ejdx)
            val = val * wj # No variation; can just multiply by bin width!
            continue
          else
            exppref = 0. + 1im # * 2π
            val = val * exp(exppref * left_edge * ejdx)
            val = val * (exp(exppref * wj * ejdx) - 1)
            val = val / (exppref * ejdx)
          end
        end
        bin_val = bin_val + val / (prod(sys.latsize) * na * na)
      end

      # Missing det(covectors) and 1/binwidth factors here?
      is[ci] = real(bin_val)

        #q = SVector{3}(coords_to_q * [x_center;y_center;z_center])
        #ωvals = bin_centers[4]

        #intensity_as_function_of_ω = formula.calc_intensity(swt,q)
        #is[ci,:] .= intensity_as_function_of_ω(ωvals)
    end
    is
end


# Show several S(q,w) plots, resolved by interaction range
function show_ranges(sys,sc)
  f = Figure()
  params = unit_resolution_binning_parameters(sc)
  params.binwidth[1:2] ./= 4

  params.binwidth[3] = 0.2
  params.binstart[3] = -0.1
  params.binend[3] = 0.

  rangs = 5:5:45
  for i = 1:9
    ax = Axis(f[fldmod1(i,3)...], title = "Range = $(rangs[i])")
    is = binned_components(sys,sc,params,hilim = rangs[i])
    heatmap!(ax,bcs[1],bcs[2],is[:,:,1])
  end
  f
end

