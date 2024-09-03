using Sunny, GLMakie, HDF5, StaticArrays

# N.B. 9/2/2024: This is an unfinished attempt at integrating an arbitrary
# Point Spread Function (in momentum) into the calculation of histogrammed intensities.
# This is because Sunny lacks most usual kinds of "Q-resolution" despite this being
# highly important for correct comparison with experimental data!
#
# However: please note that the binning effect captures the Q-resolution effect
# whenever the point spread function is much sharper than a histogram bin.
# This means that you can always make an accurate comparison without this PSF code
# by simply making your bins coarser than the PSF!

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
    integrated_kernel = nothing, point_spread_function = nothing, approach = :centers
)
    (; binwidth, binstart, binend, covectors, numbins) = params
    return_type = typeof(formula).parameters[1]
    output_intensities = zeros(return_type,numbins...)
    output_intensities = MArray{Tuple{numbins[1],numbins[2],numbins[3],numbins[4]},return_type}(undef)
    output_intensities .= 0
    output_counts = MArray{Tuple{numbins[1],numbins[2],numbins[3],numbins[4]},Float64}(undef)
    output_counts .= 0
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

    all_q_scattering_source = CartesianIndices(Tuple(((:).(lower_aabb_cell,upper_aabb_cell))))

    # Pre-compute discrete point spread function from continuous one provided
    psf_kern_size = if !isnothing(point_spread_function)

        # Upgrade to 2-argument kernel if needed
        is_constant_kernel = !(point_spread_function isa Function)
        
        # Find how many bins over a signal can be spread
        # along each histogram axis.
        bin_span = if is_constant_kernel
          # Find the size for the single constant kernel
          unique_bin_span = zeros(Float64,4)

          # Eigenvectors * √λ of point_spread_function give the 1σ spread distance.
          # Here, we measure those displacements in terms of the histogram
          # coordinates; the rows of hist_axes_psf label histogram axes,
          # and the columns label point spread directions.
          F = eigen(point_spread_function)
          hist_axes_psf = params.covectors * F.vectors * sqrt.(diagm(F.values))
          #hist_axes_psf = params.covectors * point_spread_function * inv(params.covectors)
          for j = 1:4
            # Find the maximum (along all spread directions) of the 1σ distance
            # measured in bins
            unique_bin_span[j] = maximum(abs.(hist_axes_psf[j,:]))/params.binwidth[j]
          end
          unique_bin_span
        else
          # Need to evaluate psf at all scattering (q,w) sources to determine
          # optimal size for kernel
          hist_axes_psf = Array{SMatrix{4,4,Float64}}(undef,size(all_q_scattering_source)...,length(ωvals))
          lower_aabb_cell_ix = CartesianIndex(lower_aabb_cell)
          largest_bin_span = zeros(Float64,4)
          for ix_q in all_q_scattering_source, ix_w in CartesianIndices(ωvals)
            q = ((ix_q.I .- 1) ./ Ls) # q is in R.L.U.
            iq = ix_q + (CartesianIndex(1,1,1) - lower_aabb_cell_ix)
            hist_axes_psf[iq,ix_w] = params.covectors * point_spread_function(q,ωvals[ix_w])
            for j = 1:4
              largest_bin_span[j] = max(largest_bin_span[j],maximum(abs.(hist_axes_psf[iq,ix_w][j,:]))/params.binwidth[j])
            end
          end
          largest_bin_span
        end

        sigma_tol = 4.0
        psf_kern_span1 = convert(Vector{Int64},ceil.(Int64,sigma_tol * bin_span))
        1 .+ 2 .* psf_kern_span1
    else
        psf_kern_span = Int64[0,0,0,0]
        1 .+ 2 .* psf_kern_span
    end

    println(psf_kern_size)
    psf_matrix = params.covectors * pinv(point_spread_function) * inv(params.covectors)
    display(psf_matrix)

    is, counts = intensities_binned_aux(output_counts,output_intensities,SMatrix{4,4}(psf_matrix),all_q_scattering_source,SVector{4,Int64}(psf_kern_size),isnothing(point_spread_function),Ls,sc,sc.crystal.recipvecs,binstart,binwidth,covectors,ωvals;approach)
    Array(is), Array(counts)
end

function intensities_binned_aux(output_counts::MArray,output_intensities::MArray,psf_matrix::SMatrix,all_q_scattering_source,psf_kern_size::SVector,skip_qres,Ls,sc,recipvecs,binstart,binwidth,covectors,ωvals; approach = :centers)
    psf_kern_span_plus_one_ix = CartesianIndex(ntuple(i -> 1 + (psf_kern_size[i] - 1)÷2,4))
    kern_ixs = CartesianIndices(ntuple(i -> psf_kern_size[i],4))
    maxbin = ntuple(i -> size(output_counts,i),4)

    carloMatrix = pinv(psf_matrix)
    Fcarlo = eigen(carloMatrix)

    k = MVector{3,Float64}(undef)
    v = MVector{4,Float64}(undef)
    q = view(v,1:3)
    coords = MVector{4,Float64}(undef)
    xyztBin = MVector{4,Int64}(undef)
    xyzBin = view(xyztBin,1:3)

    # Loop over every scattering vector in the bounding box
    for cell in all_q_scattering_source
        # Which is the analog of this scattering mode in the first BZ?
        base_cell = CartesianIndex(mod1.(cell.I,Ls)...)
        q .= ((cell.I .- 1) ./ Ls) # q is in R.L.U.
        k .= recipvecs * q
        for (iω,ω) in enumerate(ωvals)
          # Figure out which bin this goes in
          v[4] = ω
          mul!(coords,covectors,v)
          xyztBin .= 1 .+ floor.(Int64,(coords .- binstart) ./ binwidth)
          intensity = formula.calc_intensity(sc,SVector{3,Float64}(k),base_cell,iω)

          fractional_coords = ((coords .- binstart) ./ binwidth) .- xyztBin
          center_bin_ix = CartesianIndex(xyztBin.data)
          if skip_qres
            if all(xyztBin .<= numbins) && all(xyztBin .>= 1) # Check bounds
              output_intensities[center_bin_ix] += intensity
              output_counts[center_bin_ix] += 1
            end
          else
            if approach == :centers
              for kern_arr_ix = kern_ixs
                diff_ix = kern_arr_ix - psf_kern_span_plus_one_ix
                ci_other = center_bin_ix + diff_ix
                b = ci_other.I
                # Check this bin is within the 4D histogram bounds
                if b[1] > maxbin[1] || b[2] > maxbin[2] || b[3] > maxbin[3] || b[4] > maxbin[4] || b[1] < 1 || b[2] < 1 || b[3] < 1 || b[4] < 1
                #if !(all(ci_other.I .<= maxbin) && all(ci_other.I .>= (1,1,1,1)))
                  continue
                end
          
                # Start and end points of the target bin, relative to origin of center bin
                l = collect(diff_ix.I .- 1) .* binwidth
                h = collect(diff_ix.I) .* binwidth

                # Relative to actual scattering point source
                lr = l .- fractional_coords .* binwidth
                hr = h .- fractional_coords .* binwidth

                # SQ TODO: exact integrated gaussian!
                center_relative = (lr + hr) / 2
                #println()
                #println(center_relative)
                #display(psf_matrix)
                factor = exp(-(dot(center_relative,psf_matrix,center_relative)) / 2)
          #println(factor)
                output_intensities[ci_other] += factor * intensity
                output_counts[ci_other] += factor
              end
            elseif approach == :montecarlo
              ncarlo = 10000
              for j = 1:ncarlo
                dq = carloMatrix * randn(4)/2
                diff = floor.(Int64,dq ./ binwidth)
                ci_other = center_bin_ix + CartesianIndex(ntuple(i -> diff[i],4))
                b = ci_other.I
                if b[1] > maxbin[1] || b[2] > maxbin[2] || b[3] > maxbin[3] || b[4] > maxbin[4] || b[1] < 1 || b[2] < 1 || b[3] < 1 || b[4] < 1
                  continue
                end
                output_intensities[ci_other] += intensity/ncarlo
                output_counts[ci_other] += 1/ncarlo
              end
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
    output_intensities ./= N_bins_in_BZ * length(ωvals)
    return output_intensities, output_counts
end


