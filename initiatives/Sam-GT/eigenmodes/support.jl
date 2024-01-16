### Support functions

function plot_band_intensities(dispersion, intensity)
    f = Makie.Figure()
    ax = Makie.Axis(f[1,1]; xlabel = "Momentum", ylabel = "Energy (meV)", xticklabelsvisible = false)
    plot_band_intensities!(ax,dispersion,intensity)
    f
end

function plot_band_intensities!(ax, dispersion, intensity)
    Makie.ylims!(ax, min(0.0,minimum(dispersion)), maximum(dispersion))
    Makie.xlims!(ax, 1, size(dispersion, 1))
    colorrange = extrema(intensity)
    for i in axes(dispersion)[2]
        Makie.lines!(ax, 1:length(dispersion[:,i]), dispersion[:,i]; color=intensity[:,i], colorrange)
    end
    nothing
end

function plot_spin_data(sys::System; resolution=(768, 512), show_axis=false, kwargs...)
    fig = Makie.Figure(; resolution)
    ax = Makie.LScene(fig[1, 1]; show_axis)
    plot_spin_data!(ax, sys; kwargs...)
    return fig
end

function plot_spin_data!(ax, sys::System; arrowscale=1.0, stemcolor=:lightgray, color=:red, show_cell=true,
                     orthographic=false, ghost_radius=0, rescale=1.0, dims=3, spin_data = Makie.Observable(sys.dipoles))
    if dims == 2
        sys.latsize[3] == 1 || error("System not two-dimensional in (a₁, a₂)")
    elseif dims == 1
        sys.latsize[[2,3]] == [1,1] || error("System not one-dimensional in (a₁)")
    end

    supervecs = sys.crystal.latvecs * diagm(Vec3(sys.latsize))

    ### Plot spins ###

    # Show bounding box of magnetic supercell in gray (this needs to come first
    # to set a scale for the scene in case there is only one atom).
    supervecs = sys.crystal.latvecs * diagm(Vec3(sys.latsize))
    Makie.linesegments!(ax, Sunny.Plotting.cell_wireframe(supervecs, dims); color=:gray, linewidth=rescale*1.5)

    # Bounding box of original crystal unit cell in teal
    if show_cell
        Makie.linesegments!(ax, Sunny.Plotting.cell_wireframe(Sunny.orig_crystal(sys).latvecs, dims); color=:teal, linewidth=rescale*1.5)
    end

    # Infer characteristic length scale between sites
    ℓ0 = Sunny.Plotting.characteristic_length_between_atoms(Sunny.orig_crystal(sys))

    # Quantum spin-S, averaged over all sites. Will be used to normalize
    # dipoles.
    S0 = (sum(sys.Ns)/length(sys.Ns) - 1) / 2

    # Parameters defining arrow shape
    a0 = arrowscale * ℓ0
    arrowsize = 0.4a0
    linewidth = 0.12a0
    lengthscale = 0.6a0
    markersize = 0.8linewidth
    arrow_fractional_shift = 0.6
   
    # Make sure colors are indexable by site
    color0 = fill_colors(color, size(sys.dipoles))

    # Find all sites within max_dist of the system center
    rs = [supervecs \ global_position(sys, site) for site in eachsite(sys)]
    if dims == 3
        r0 = [0.5, 0.5, 0.5]
    elseif dims == 2
        r0 = [0.5, 0.5, 0]
    end

    # Require separate drawing calls with `transparency=true` for ghost sites
    for isghost in (false, true)
        if isghost
            alpha = 0.08
            images = Sunny.Plotting.all_ghost_images_within_distance(supervecs, rs, Sunny.Plotting.cell_center(dims); max_dist=ghost_radius)
        else
            alpha = 1.0
            images = [[zero(Sunny.Vec3)] for _ in rs]
        end
        pts = Makie.Point3f0[]
        vecs = Makie.Observable(Makie.Vec3f0[])
        arrowcolor = Tuple{eltype(color0), Float64}[]
        for site in eachindex(images)
            vec = (lengthscale / S0) * sys.dipoles[site]
            # Loop over all periodic images of site within radius
            for n in images[site]
                pt = supervecs * (rs[site] + n)
                push!(pts, Makie.Point3f0(pt))
                push!(vecs[], Makie.Vec3f0(vec))
                push!(arrowcolor, (color0[site], alpha))
            end
        end

        Makie.on(spin_data, update = true) do dipoles
            ix = 1
            for site in eachindex(images)
                vec = (lengthscale / S0) * dipoles[site]
                for n in images[site]
                    iszero(n) == isghost && continue
                    vecs[][ix] = Makie.Vec3f0(vec)
                    ix += 1
                end
            end
            notify(vecs)
        end

        shifted_pts = map(vs -> pts - arrow_fractional_shift * vs, vecs)

        linecolor = @something (stemcolor, alpha) arrowcolor
        Makie.arrows!(ax, shifted_pts, vecs; arrowsize, linewidth, linecolor, arrowcolor, transparency=isghost)

        # Small sphere inside arrow to mark atom position
        Makie.meshscatter!(ax, pts; markersize, color=linecolor, transparency=isghost)
    end

    if show_cell
        # Labels for lattice vectors. This needs to come last for
        # `overdraw=true` to work.
        pos = [(3/4)*Makie.Point3f0(p) for p in eachcol(Sunny.orig_crystal(sys).latvecs)[1:dims]]
        text = [Makie.rich("a", Makie.subscript(repr(i))) for i in 1:dims]
        Makie.text!(ax, pos; text, color=:black, fontsize=rescale*20, font=:bold, glowwidth=4.0,
                    glowcolor=(:white, 0.6), align=(:center, :center), overdraw=true)
    end

    Sunny.Plotting.orient_camera!(ax, supervecs; ghost_radius, ℓ0, orthographic, dims)

    return ax
end


function fill_colors(c::AbstractArray, sz)
    size(c) == sz || error("Colors array must have size $sz.")
    if eltype(c) <: Number
        c = Sunny.Plotting.numbers_to_colors(c)
    end
    return c
end
fill_colors(c, sz) = fill(c, sz)


