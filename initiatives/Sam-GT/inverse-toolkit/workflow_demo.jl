using Sunny, GLMakie, HDF5

# The `.nxs` file exported from Mantid looks something like this:

fn = "../inverse-toolkit/data/LaSrCrO4_prepared.nxs"
h5open(fn,"r")

# `Sunny.jl` provides a function, `load_nxs`, which loads that data into
# native Sunny data structures which include both the binning metadata:

if !(:params ∈ names(Main))#hide
params, data = load_nxs(fn)
end#hide
display(params)

# and the data itself:

"typeof(data) = $(typeof(data))\nsize(data) = $(size(data))"#hide

# The `signal` is loaded by default, but the `errors_squared` (as computed by Mantid) and other fields
# are also accessible:
#params_err, data_err = load_nxs(fn;field = "errors_squared")
#nothing#hide

function intensities_binned_psf(sc::SampledCorrelations, params::BinningParameters, formula::Sunny.ClassicalIntensityFormula;
    integrated_kernel = nothing, point_spread_function = nothing
)
    (; binwidth, binstart, binend, covectors, numbins) = params
    return_type = typeof(formula).parameters[1]
    output_intensities = zeros(return_type,numbins...)
    output_counts = zeros(Float64,numbins...)
    ωvals = Sunny.available_energies_including_zero(sc;negative_energies=true)
    q0vals = available_wave_vectors(sc;bzsize = (1,1,1))

    # Find an axis-aligned bounding box containing the histogram.
    # The AABB needs to be in r.l.u for the (possibly reshaped) crystal
    # because that's where we index over the scattering modes.
    lower_aabb_q, upper_aabb_q = Sunny.binning_parameters_aabb(params)

    # Round the axis-aligned bounding box *outwards* to lattice sites
    # SQTODO: are these bounds optimal?
    Ls = sc.latsize
    lower_aabb_cell = floor.(Int64,lower_aabb_q .* Ls .+ 1) 
    upper_aabb_cell = ceil.(Int64,upper_aabb_q .* Ls .+ 1)

    k = MVector{3,Float64}(undef)
    v = MVector{4,Float64}(undef)
    q = view(v,1:3)
    coords = MVector{4,Float64}(undef)
    xyztBin = MVector{4,Int64}(undef)
    xyzBin = view(xyztBin,1:3)

    # Pre-compute discrete broadening kernel from continuous one provided
    if !isnothing(integrated_kernel)

        # Upgrade to 2-argument kernel if needed
        integrated_kernel_edep = try
            integrated_kernel(0.,0.)
            integrated_kernel
        catch MethodError
            (ω,Δω) -> integrated_kernel(Δω)
        end

        fraction_in_bin = Vector{Vector{Float64}}(undef,length(ωvals))
        for (iω,ω) in enumerate(ωvals)
            fraction_in_bin[iω] = Vector{Float64}(undef,numbins[4])
            for iωother = 1:numbins[4]
                # Start and end points of the target bin
                a = binstart[4] + (iωother - 1) * binwidth[4]
                b = binstart[4] + iωother * binwidth[4]

                # P(ω picked up in bin [a,b]) = ∫ₐᵇ Kernel(ω' - ω) dω'
                fraction_in_bin[iω][iωother] = integrated_kernel_edep(ω,b - ω) - integrated_kernel_edep(ω,a - ω)
            end
        end
    end

    # Pre-compute discrete point spread function from continuous one provided
    if !isnothing(point_spread_function)

        # Upgrade to 2-argument kernel if needed
        psf_qdep = if point_spread_function isa Function
          point_spread_function
        else
          q -> point_spread_function # Constant
        end

        @assert iszero(params.covectors[1:3,4]) && iszero(params.covectors[4,1:3])
        hist_axes_psf = Array{SMatrix{3,3,Float64}}(undef,size(q0vals)...)
        largest_bin_span = [0.,0,0]
        for ix_q in CartesianIndices(q0vals)
          hist_axes_psf[ix_q] = params.covectors[1:3,1:3] * psf_qdep(q0vals[ix_q])
          for j = 1:3
            largest_bin_span[j] = max(largest_bin_span[j],maximum(abs.(hist_axes_psf[ix_q][j,:]))/params.binwidth[j])
          end
        end

        sigma_tol = 4.0
        psf_kern_span = ceil.(Int64,sigma_tol * largest_bin_span)
        psf_kern_size = 1 .+ 2psf_kern_span
        psf_kern_span_plus_one_ix = CartesianIndex(ntuple(i -> 1 + psf_kern_span[i],3))

        psf_kerns = Array{Float64}(undef,psf_kern_size...,size(q0vals)...)
        for ix_q_source in CartesianIndices(q0vals)
            center_pt = params.covectors[1:3,1:3] * q0vals[ix_q_source]
            center_bin = 1 .+ floor.(Int64,(center_pt .- params.binstart[1:3]) ./ params.binwidth[1:3])
            psf_matrix = inv(hist_axes_psf[ix_q_source])
            #psf_matrix = (hist_axes_psf[ix_q_source])
            center_bin_ix = CartesianIndex(ntuple(i -> center_bin[i],3))
            psf_kern_size_ix = CartesianIndex(ntuple(i -> psf_kern_size[i],3))
            for kern_arr_ix = CartesianIndices(ntuple(i -> psf_kern_size[i],3))
                
                diff_ix = kern_arr_ix - psf_kern_span_plus_one_ix
                
                # Start and end points of the target bin
                bin_ix = center_bin_ix + diff_ix
                l = params.binstart[1:3] .+ collect(bin_ix.I .- 1) .* params.binwidth[1:3]
                h = params.binstart[1:3] .+ collect(bin_ix.I) .* params.binwidth[1:3]

                # Relative to actual center point
                lr = l .- center_pt
                hr = h .- center_pt

                # SQ TODO: exact integrated gaussian!
                center_relative = (lr + hr) / 2
                factor = exp(-(center_relative ⋅ (psf_matrix * psf_matrix * center_relative)) / 2)
                psf_kerns[kern_arr_ix,ix_q_source] = factor
            end
        end
    else
        psf_kern_span = [0,0,0]
        psf_kern_size = 1 .+ 2psf_kern_span
        psf_kern_span_plus_one_ix = CartesianIndex(ntuple(i -> 1 + psf_kern_span[i],3))
        psf_kerns = ones(Float64,1,1,1,size(q0vals)...)
    end

    # Loop over every scattering vector in the bounding box
    for cell in CartesianIndices(Tuple(((:).(lower_aabb_cell,upper_aabb_cell))))
        # Which is the analog of this scattering mode in the first BZ?
        base_cell = CartesianIndex(mod1.(cell.I,Ls)...)
        q .= ((cell.I .- 1) ./ Ls) # q is in R.L.U.
        k .= sc.crystal.recipvecs * q
        for (iω,ω) in enumerate(ωvals)
            if isnothing(integrated_kernel) # `Delta-function energy' logic
                # Figure out which bin this goes in
                v[4] = ω
                mul!(coords,covectors,v)
                xyztBin .= 1 .+ floor.(Int64,(coords .- binstart) ./ binwidth)

                # Check this bin is within the 4D histogram bounds
                if all(xyztBin .<= numbins) && all(xyztBin .>= 1)
                    intensity = formula.calc_intensity(sc,SVector{3,Float64}(k),base_cell,iω)

                    ci = CartesianIndex(xyztBin.data)
                    center_bin_ix = CartesianIndex(xyztBin[1],xyztBin[2],xyztBin[3])
                    for kern_arr_ix = CartesianIndices(ntuple(i -> psf_kern_size[i],3))
                      diff_ix = kern_arr_ix - psf_kern_span_plus_one_ix
                      ci_other = center_bin_ix + diff_ix
                      if all(ci_other.I .<= view(numbins,1:3)) && all(ci_other.I .>= 1)
                        output_intensities[ci_other,xyztBin[4]] += psf_kerns[kern_arr_ix,base_cell] * intensity
                        output_counts[ci_other,xyztBin[4]] += psf_kerns[kern_arr_ix,base_cell]
                      end
                    end
                    #output_intensities[ci] += intensity
                    #output_counts[ci] += 1
                end
            else # `Energy broadening into bins' logic
                # For now, only support broadening for `simple' energy axes
                if covectors[4,:] == [0,0,0,1] && norm(covectors[1:3,:] * [0,0,0,1]) == 0

                    # Check this bin is within the *spatial* 3D histogram bounds
                    # If we are energy-broadening, then scattering vectors outside the histogram
                    # in the energy direction need to be considered
                    mul!(view(coords,1:3),view(covectors,1:3,1:3), view(v,1:3))
                    xyzBin .= 1 .+ floor.(Int64,(view(coords,1:3) .- view(binstart,1:3)) ./ view(binwidth,1:3))
                    if all(xyzBin .<= view(numbins,1:3)) &&  all(xyzBin .>= 1)

                        # Calculate source scattering vector intensity only once
                        intensity = formula.calc_intensity(sc,SVector{3,Float64}(k),base_cell,iω)

                        # Broaden from the source scattering vector (k,ω) to
                        # each target bin ci_other
                        center_bin_ix = CartesianIndex(xyzBin[1],xyzBin[2],xyzBin[3])
                        for kern_arr_ix = CartesianIndices(ntuple(i -> psf_kern_size[i],3))
                          diff_ix = kern_arr_ix - psf_kern_span_plus_one_ix
                          ci_other = center_bin_ix + diff_ix
                          if all(ci_other.I .<= view(numbins,1:3)) && all(ci_other.I .>= 1)
                            view(output_intensities,ci_other,:) .+= psf_kerns[kern_arr_ix,base_cell] * fraction_in_bin[iω] .* Ref(intensity)
                            view(output_counts,ci_other,:) .+= psf_kerns[kern_arr_ix,base_cell] * fraction_in_bin[iω]
                          end
                        end
                    end
                else
                    error("Energy broadening not yet implemented for histograms with complicated energy axes")
                end
            end
        end
    end

    # `output_intensities` is in units of S²/BZ/fs, and includes the sum
    # of `output_counts`-many individual scattering intensities.
    # To give the value integrated over the bin, we need to multiply by
    # the binwidth, Δω×ΠᵢΔqᵢ. But Δq/BZ = 1/N, where N is the number of
    # bins covering one BZ, which is itself equal to the latsize.
    #
    # To find the number of bins covering one BZ, we first compute the volume of the BZ
    # in histogram label space: it is det(covectors[1:3,1:3]). Next, we compute the volume
    # of one bin in label space: it is prod(binwidth[1:3]).
    # Then, N = det(covectors[1:3,1:3])/prod(binwidth[1:3]).
    #
    # For the time axis, Δω/fs = 1/N, where N is the number of frequency bins
    # 
    # So the division by N here makes it so the result has units of
    # raw S² (to be summed over M-many BZs to recover M-times the sum rule)
    N_bins_in_BZ = abs(det(covectors[1:3,1:3])) / prod(binwidth[1:3])
    return output_intensities ./ N_bins_in_BZ ./ length(ωvals), output_counts
end


