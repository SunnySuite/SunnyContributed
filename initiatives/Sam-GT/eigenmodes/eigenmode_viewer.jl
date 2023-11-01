# This script is an experiment which visualizes individual spin wave oscillations
# in real space. Run one of the example_* functions at the bottom to see it in action!
# 
# Works on Sunny main as of this writing! (Oct 27)

#
# The correctness of the visualization is not yet verified--see comments within--and also
# currently only really shows the dipole sector.

using Sunny, GLMakie, Observables, LinearAlgebra

# Steal these internal functions required for the eigenmode computation
import Sunny: bogoliubov!, to_reshaped_rlu, swt_hamiltonian_SUN!, natoms


function get_eigenmodes(swt,q; verbose = false)
    (; sys, data, observables) = swt

    Nm = natoms(sys.crystal) # Number of atoms in magnetic unit cell
    N = sys.Ns[1] # The N in SU(N), giving the total number of boson flavors
    Nf = N - 1 # Number of uncondensed boson flavors
    nmodes = Nf * Nm

    Hmat = zeros(ComplexF64, 2*nmodes, 2*nmodes)
    Vmat = zeros(ComplexF64, 2*nmodes, 2*nmodes)
    disp = zeros(Float64, nmodes)


    q_reshaped = to_reshaped_rlu(swt.sys, Sunny.Vec3(q))

    if sys.mode == :SUN
        swt_hamiltonian_SUN!(Hmat, swt, q_reshaped)
    elseif sys.mode == :dipole
        error("Dipole mode not supported")
    end
    #println("Original Hamiltonian:")
    H0 = copy(Hmat)
    #display(H0)
    #display(eigen(H0))

    disp = bogoliubov!(Vmat, Hmat)

    # Now, Vmat contains the information about the eigenmodes as columns.
    # The way this works is that the first [nmodes] columns describe the
    # [nmodes] many boson deletion operators, as linear combinations of both
    # the original boson deletion operators (top half of column) and the original boson
    # creation operators (bottom half of column).
    #
    # [[The second half of the columns contain the boson creation operators in a similar
    # format, but in reverse order, e.g. it goes [b1,b2,b†2,b†1]. But the creation operators
    # are not needed because they can be inferred from the deletion operators. The fact
    # that they can be inferred is equivalent to the dagger operation being preserved
    # by the bogoliubov transform V]]
    #
    # The operators within each half-column come in blocks.
    # The length of the block corresponds to the boson flavor index, which runs
    # from 2 to N (where N is as in SU(N)). The #1 is the 'ground state' boson which
    # was condensed away.
    #
    # There is one such (2:N) block for each atom in the magnetic unit cell


    bases = swt.data.local_unitary

    eigen_mode_displacements = zeros(ComplexF64,N,Nm,nmodes)

    # Only loops over the boson deletion operators
    for eigen_mode = 1:nmodes
      # Describes the operator by its coefficients in the linear
      # combination ∑ᵢ λᵢaᵢ where aᵢ runs over both the deletion and
      # creation operators for the *original* non-bogoliubov bosons
      bogoliubov_deletion_operator = Vmat[:,eigen_mode]

      # Now we construct the (real-space, eventually) displacement
      # associated with each eigenmode
      for atom = 1:Nm
        # Skip the first (atom-1) blocks
        offset = (atom - 1) * Nf

        # Get the combination of original bosons relevant for this atom
        deletion_operators_on_this_atom = bogoliubov_deletion_operator[offset .+ (1:Nf)]
        creation_operators_on_this_atom = bogoliubov_deletion_operator[nmodes .+ offset .+ (1:Nf)]

        # Get the local quantization basis at this atom.
        # This is actually a basis for the *tangent space* of the space
        # of coherent states, rooted at the ground state coherent state.
        # The basis is orthogonal, and there is one boson flavor associated
        # with each basis vector.
        basis = bases[:,:,atom]

        # The first basis vector in the basis is along the 'ground state'/longitudinal mode
        # and it got condensed away. There will be no eigenmode displacements in that
        # direction.
        uncondensed_basis = basis[:,2:N]

        # Loop over only the uncondensed bosons
        for original_uncondensed_boson = 2:N
          # The bosons all got shifted one to the left when we chopped of the
          # one which was condensed away.
          boson_ix = original_uncondensed_boson - 1

          this_displacement = uncondensed_basis[:,boson_ix]
          this_amplitude = deletion_operators_on_this_atom[boson_ix]
          eigen_mode_displacements[:,atom,eigen_mode] .+= this_displacement .* this_amplitude

          # This says that the basis vectors corresponding to the creation operators
          # are the imaginary unit (1im) times the basis vectors corresponding to the deletion
          # operators. Is this true? Who knows...
          this_conj_displacement = 1im .* uncondensed_basis[:,boson_ix]
          this_conj_amplitude = creation_operators_on_this_atom[boson_ix]
          eigen_mode_displacements[:,atom,eigen_mode] .+= this_conj_displacement .* this_conj_amplitude

        end
      end
    end



    if verbose
      println("V matrix:")
      display(Vmat)
      println("Diagonalized V'HV:")
      display(Vmat' * H0 * Vmat)
      println("Bases")
      display(bases)
      println("Eigenmodes (columns are atoms)")
      for m = 1:nmodes
        println()
        println("Mode #$m with energy $(disp[m])")
        display(eigen_mode_displacements[:,:,m])
        #=
        for i = 1:Nm
          println("Atom#$i:")
          display(eigen_mode_displacements[:,i,m])
          println("Δ[dipole] = ")
          orig_dipole = expected_spin(sys.coherents[i])
          eps = 1e-5
          new_dipole = expected_spin(SVector{2}(sys.coherents[i] .+ eps .* eigen_mode_displacements[:,i,m]))
          #display(orig_dipole)
          #display(new_dipole)
          display((new_dipole .- orig_dipole) ./ eps)
        end
        =#
      end
      println("Eigenenergies")
      display(disp)
      println("Nm = $Nm, N = $N, Nf = $Nf, nmodes = $nmodes")
      display(sys.coherents)
    end
    return H0, Vmat, bases, eigen_mode_displacements, disp
end

function plot_eigenmode(displacements, swt::SpinWaveTheory; kwargs...)
    fig = Figure()
    ax = LScene(fig[1, 1]; show_axis = false)
    plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; kwargs...)
    fig
end

function plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; t = nothing, kwargs...)
  plot_spin_data!(ax,swt.sys;color=:grey,arrowscale = 0.9,kwargs...)

  tweaked = Observable(zeros(Vec3f,size(swt.sys.dipoles)))
  coherents_scratch = copy(swt.sys.coherents)

  t = isnothing(t) ? Observable(0.) : t

  on(displacements,update = true) do disps
    notify(t)
  end
  
  on(t) do time
    disps = displacements[]
    # Requires single-site swt system
    for i = 1:size(coherents_scratch,4)
      coherents_scratch[i] = swt.sys.coherents[i] .+ 0.5 .* exp(im * time) .* disps[:,i]
      tweaked[][i] = Sunny.expected_spin(coherents_scratch[i])
    end
    notify(tweaked)
  end

  # TODO: ghost spins are currently inaccurate, since they should pick up a phase factor
  plot_spin_data!(ax,swt.sys;color = :blue,spin_data = tweaked,kwargs...)
end

if !(:eigenmode_viewer_screen ∈ names(Main))
  global eigenmode_viewer_screen = nothing
end
function interact_eigenmodes(swt::SpinWaveTheory, qs, formula; kwargs...)
    # The background band structure plot
  fig = Figure()
  ax = Axis(fig[1,1], title = "Click a mode! (Spacebar to animate)", xticklabelsvisible = false, xrectzoom = false, yrectzoom = false)
  dispersion, intensity = intensities_bands(swt, qs, formula)
  plot_band_intensities!(ax, dispersion, intensity)

  # The marker showing the user-selected oscillation
  marker_points = Observable(Point2f[Point2f(NaN,NaN)])
  marker_colors = Observable(Float64[NaN])
  sc_marker = scatter!(ax, marker_points, color = marker_colors, strokewidth = 1.)

  # The eigenmode viewer window
  global eigenmode_viewer_screen
  if isnothing(eigenmode_viewer_screen) || eigenmode_viewer_screen.window_open[] == false
    eigenmode_viewer_screen = GLMakie.Screen()
  end
  fig_mode = Figure()
  ax_mode = LScene(fig_mode[1,1]; show_axis = false)
  t = Observable(0.) # Animation time
  rendered_displacements = Observable(zeros(ComplexF64,swt.sys.Ns[1],natoms(swt.sys.crystal)))
  plot_eigenmode!(ax_mode, rendered_displacements, swt; t)

  # Update the eigenviewer based on the user-selected position
  function do_update(;mp = nothing)
    # mouse position = nothing on click, but we already know mp when dragging
    mp = isnothing(mp) ? events(ax).mouseposition[] : mp

    # Convert from screen-space to data-space
    bbox = ax.layoutobservables.computedbbox[]
    c = (mp .- bbox.origin) ./ bbox.widths
    if 0 < c[1] < 1 && 0 < c[2] < 1
      data_bbox = ax.finallimits[]
      data_space_c = data_bbox.origin .+ c .* data_bbox.widths
      q,ωclick = data_space_c

      # Interpolate between the provided q points
      q_int = floor(Int64,q)
      if q_int + 1 > length(qs) || q_int < 1 # if outside q range of plot
        return
      end
      τ = q - q_int
      q_interp = (1-τ) .* qs[q_int] .+ τ .* qs[q_int+1]

      # Perform the eigenmode analysis
      _H, _V, _bases, displacements, disp = get_eigenmodes(swt,q_interp)

      # Snap to the nearest band (vertically only)
      _, ix = findmin(abs.(disp .- ωclick))

      # Move the marker
      marker_points[][1] = Point2f(q,disp[ix])
      marker_colors[][1] = 1. # TODO: intensity coloring?
      notify(marker_points)

      # Update eigenmode viewer with the newly selected mode
      rendered_displacements[] .= displacements[:,:,ix]
      notify(rendered_displacements)
      ax.title[] = "q = [$(join(Sunny.number_to_simple_string.(q_interp,digits=3),","))], ω = $(Sunny.number_to_simple_string(disp[ix],digits=3))"
    end
  end

  # Update on left click
  on(events(fig).mousebutton, priority = 2) do event
    if event.button == Mouse.left
      if event.action == Mouse.press
        do_update()
        return Consume(true)
      end
    end
  end

  # Update on left click-drag
  on(async_latest(events(fig).mouseposition), priority = 2) do mp
    if events(fig).mousebutton[].button == Mouse.left
      if events(fig).mousebutton[].action == Mouse.press
        do_update(;mp)
        return Consume(true)
      end
    end
  end

  # Lock so if we're already animating
  # we don't do it again over top of itself
  lck = ReentrantLock()

  # Animate on spacebar
  on(events(fig_mode).keyboardbutton) do event
    if event.key == Keyboard.space
      if event.action == Keyboard.press
          @async begin
            if trylock(lck)
              for t0 in range(0,2π,length = 80)
                omega = marker_points[][1][2]
                t[] = omega .* t0
                sleep(1/30)
              end
              unlock(lck)
            end
          end
        return Consume(true)
      end
    end
  end
  # Forward spacebar to other window if needed
  connect!(events(fig_mode).keyboardbutton,events(fig).keyboardbutton)
  
  # Clear eigenmode screen and display both windows
  empty!(eigenmode_viewer_screen)
  display(eigenmode_viewer_screen,fig_mode)
  display(eigenmode_viewer_screen)
  fig
end

function example_eigenmodes()
  sys = System(Sunny.cubic_crystal(), (1,2,1), [SpinInfo(1;S=1/2,g=1)], :SUN, units = Units.theory)
  set_external_field!(sys,[0,0,0.5]) # Field along Z
  set_exchange!(sys,-1.,Bond(1,1,[0,1,0])) # Strong Ferromagnetic J
  randomize_spins!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)

  swt = SpinWaveTheory(sys)
  qs = [[0,k,0] for k = range(0,1,length=20)]
  get_eigenmodes(swt,qs[3]; verbose = true)
  formula = intensity_formula(swt,:perp, kernel = delta_function_kernel)
  interact_eigenmodes(swt, qs, formula)
end

function example_fei2()
  a = b = 4.05012  # Lattice constants for triangular lattice
  c = 6.75214      # Spacing in the z-direction
  latvecs = lattice_vectors(a, b, c, 90, 90, 120) # A 3x3 matrix of lattice vectors that
                                                  # define the conventional unit cell
  positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]  # Positions of atoms in fractions
                                                             # of lattice vectors
  types = ["Fe", "I", "I"]
  FeI2 = Crystal(latvecs, positions; types)
  cryst = subcrystal(FeI2, "Fe")
  sys = System(cryst, (4,4,4), [SpinInfo(1, S=1, g=2)], :SUN, seed=2)
  J1pm   = -0.236
  J1pmpm = -0.161
  J1zpm  = -0.261
  J2pm   = 0.026
  J3pm   = 0.166
  J′0pm  = 0.037
  J′1pm  = 0.013
  J′2apm = 0.068

  J1zz   = -0.236
  J2zz   = 0.113
  J3zz   = 0.211
  J′0zz  = -0.036
  J′1zz  = 0.051
  J′2azz = 0.073

  J1xx = J1pm + J1pmpm
  J1yy = J1pm - J1pmpm
  J1yz = J1zpm

  set_exchange!(sys, [J1xx   0.0    0.0;
                      0.0    J1yy   J1yz;
                      0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
  set_exchange!(sys, [J2pm   0.0    0.0;
                      0.0    J2pm   0.0;
                      0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
  set_exchange!(sys, [J3pm   0.0    0.0;
                      0.0    J3pm   0.0;
                      0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
  set_exchange!(sys, [J′0pm  0.0    0.0;
                      0.0    J′0pm  0.0;
                      0.0    0.0    J′0zz], Bond(1,1,[0,0,1]))
  set_exchange!(sys, [J′1pm  0.0    0.0;
                      0.0    J′1pm  0.0;
                      0.0    0.0    J′1zz], Bond(1,1,[1,0,1]))
  set_exchange!(sys, [J′2apm 0.0    0.0;
                      0.0    J′2apm 0.0;
                      0.0    0.0    J′2azz], Bond(1,1,[1,2,1]))

  D = 2.165
  S = spin_operators(sys, 1)
  set_onsite_coupling!(sys, -D*S[3]^2, 1)

  randomize_spins!(sys)
  minimize_energy!(sys);

  sys_min = reshape_supercell(sys, [1 0 0; 0 1 -2; 0 1 2])
  randomize_spins!(sys_min)
  minimize_energy!(sys_min)

  swt = SpinWaveTheory(sys_min)

  q_points = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]];
  density = 50
  path, xticks = reciprocal_space_path(cryst, q_points, density);
  formula = intensity_formula(swt,:perp, kernel = delta_function_kernel)
  interact_eigenmodes(swt, path, formula)
end

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
    images = Sunny.Plotting.all_images_within_distance(supervecs, rs, [r0]; max_dist=ghost_radius, include_zeros=true)

    # Require separate drawing calls with `transparency=true` for ghost sites
    for (isghost, alpha) in ((false, 1.0), (true, 0.08))
        pts = Makie.Point3f0[]
        vecs = Makie.Observable(Makie.Vec3f0[])
        arrowcolor = Tuple{eltype(color0), Float64}[]
        for site in eachindex(images)
            vec = (lengthscale / S0) * sys.dipoles[site]
            # Loop over all periodic images of site within radius
            for n in images[site]
                # If drawing ghosts, require !iszero(n), and vice versa
                iszero(n) == isghost && continue
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

    Sunny.Plotting.orient_camera!(ax, supervecs; ghost_radius, orthographic, dims)

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

