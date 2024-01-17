# This script visualizes individual spin wave oscillations in real space.
#
# To use the GUI viewer, call `interact_eigenmodes(swt, qs, formula)` using the same
# arguments as `intensities_interpolated(::SpinWaveTheory, ...)`.
#
# To run the eigenmode analysis at a single wavevector, call `get_eigenmodes(swt,q,verbose = true)`.
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
    # [nmodes] many boson deletion operators at +k, as linear combinations of both
    # the original boson deletion operators at +k (top half of column) and the original boson
    # creation operators at -k (bottom half of column).
    #
    # [[The second half of the columns contain the boson creation operators at -k in a similar
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

    # Sunny changed from (2:N) to 1:(N-1) at some point
    ix_uncondensed = 1:(N-1)

    bases = swt.data.local_unitaries

    eigen_mode_displacements = zeros(ComplexF64,N,Nm,2nmodes)

    # Only loops over the boson deletion operators
    for eigen_mode = 1:(2nmodes) # NEW: and creation

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
        uncondensed_basis = basis[:,ix_uncondensed]

        # Loop over only the uncondensed bosons
        for (boson_ix,original_uncondensed_boson) = enumerate(ix_uncondensed)
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
  
  on(t;update = true) do time
    disps = displacements[]
    # Requires single-site swt system
    for i = 1:size(coherents_scratch,4)
      coherents_scratch[i] = swt.sys.coherents[i] .+ 0.5 .* exp(im * time) .* disps[:,i]
      tweaked[][i] = Sunny.expected_spin(coherents_scratch[i])
    end
    notify(tweaked)
  end

  # TODO: ghost spins are currently inaccurate, since they should pick up a phase factor
  plot_spin_data!(ax,swt.sys;color = :blue,spin_data = tweaked,show_cell=false,kwargs...)
end

if !(:eigenmode_viewer_screen ∈ names(Main))
  global eigenmode_viewer_screen = nothing
end
function interact_eigenmodes(swt::SpinWaveTheory, qs, formula)
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
              for t0 in range(0,100 * 2π,length = 8000)
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

include("support.jl")


