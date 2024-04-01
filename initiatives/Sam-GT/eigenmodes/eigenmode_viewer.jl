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
import Sunny: bogoliubov!, to_reshaped_rlu, swt_hamiltonian_SUN!, swt_hamiltonian_dipole!, natoms

function get_eigenmodes(swt,q; verbose = false)
    (; sys, data, observables) = swt


    # The spin wave system is described by classical variables (α1,...,αN,α†1,..,α†N).
    # The N pairs of variables are organized by sublattice:
    Nm = natoms(sys.crystal) # Number of sublattice = number of atoms in magnetic unit cell
    # and each sublattice may have several flavors of excitation; there is one flavor of
    # boson for each transverse mode. In :dipole mode, there is only one flavor per sublattice.
    # For SU(N), there are N total components per sublattice, but one is longitudinal, so (N-1) flavors.
    Nf = sys.mode == :dipole ? 1 : (sys.Ns[1] - 1)
    # 
    # The variables are ordered as:
    #
    #   First atom: α1,...,αNf
    #   Second atom: α(Nf+1),...,α(2Nf)
    #   ...
    #   Nmth atom: α((Nm - 1)*Nf+1),...,α(Nm * Nf)
    #   
    #   [followed by † version of the above]
    #
    # The number of eigenmodes (number of columns of Vmat) is one for each classical variable.
    num_dagger_vs_non_dagger = 2
    num_variables = Nf * Nm * num_dagger_vs_non_dagger
    num_eigenmodes = num_variables

    # The Hamiltonian is a bilinear form between the classical variables:
    Hmat = zeros(ComplexF64, num_variables, num_variables)

    # The `Vmat' bogoliubov matrix has one column for each eigenmode, and
    # each column contains initial configurations for every variable:
    Vmat = zeros(ComplexF64, num_variables, num_eigenmodes)

    # Ask Sunny to fill in the Hamiltonian matrix
    q_reshaped = to_reshaped_rlu(swt.sys, Sunny.Vec3(q))
    if sys.mode == :SUN
        swt_hamiltonian_SUN!(Hmat, swt, q_reshaped)
    elseif sys.mode == :dipole
        swt_hamiltonian_dipole!(Hmat, swt, q_reshaped)
    end

    # Keep a copy of the original Hamiltonian, since it
    # gets overwritten by `bogoliubov!'
    H0 = copy(Hmat)

    # Ask Sunny to perform the bogoliubov diagonalization to give us the
    # modeshapes in Vmat. The eigenmodes are ordered by decreasing ω
    #
    #   WARNING: The `disp' returned by Sunny only includes some of the eigenvalues.
    #
    disp = bogoliubov!(Vmat, Hmat)

    # Break down each eigenmode in the bogoliubov matrix
    # by variable: flavor, site, and dagger-vs-non-dagger
    modeshapes_α = reshape(Vmat,Nf,Nm,num_dagger_vs_non_dagger,num_eigenmodes)

    # Next, we want to make a change of coordinates from (α,α†)
    # to q = (α + α†)/2, p = (α - α†)/(2i), which are the more
    # usual coordinates for each oscillator. This matrix
    # will multiply the dagger-vs-non-dagger dimension of the modeshape:
    boson_coords_to_qp = [1 1; (1/im) (-1/im)]/2

    # The (q,p) coordinates are a specific choice of coordinates for the tangent
    # space to the ground state. To visualize the eigenmode however, we want to
    # embed the tangent space into the ambient space. The embedding is given by:
    #
    #   (q,p) ↦ [ground state] + ε R*(q,p,0)
    #
    # where ε is a small number, R is the rotation matrix relevant for the given site
    # and [ground state] is the point in state space that the LSWT is linearized around.
    R = if sys.mode == :SUN
      # Only need the transverse columns
      [swt.data.local_unitaries[:,1:Nf,i] for i = 1:Nm]
    elseif sys.mode == :dipole
      # Only need the first two columns (for q and p) since
      # the longitudinal displacement is always zero
      map(x -> x[:,1:2],swt.data.local_rotations)
    end

    # Historical note:
    # [[The second half of the columns contain the boson creation operators at -k in a similar
    # format, but in reverse order, e.g. it goes [b1,b2,b†2,b†1]. But the creation operators
    # are not needed because they can be inferred from the deletion operators. The fact
    # that they can be inferred is equivalent to the dagger operation being preserved
    # by the bogoliubov transform V]]
    

    dim_global = sys.mode == :dipole ? 3 : sys.Ns[1]
    num_sin_cos = 2
    sin_cos_displacements = zeros(ComplexF64,dim_global,Nm,num_sin_cos,num_eigenmodes)

    # The eigenmodes described by Vmat were generated using the ansatz:
    #
    #   (α(t),α†(t)) = (α0,α†0) exp(iωt)
    #
    # which is a *complex* solution. To get classical trajectories we need to
    # add or subtract this from a solution with the opposite complex phase, exp(i(-ω)t).
    # In a centrosymmetric system, the eigenvalues `eigvals(Itilde,H0)` come in ±pairs,
    # so we could use those to construct real-valued classical solutions---but in general
    # systems this eigenvalue spectrum is *not* symmetric so we can't do this.
    #
    # Instead, we need to rely on a special property of Linear Spin Wave Hamiltonians, namely
    # that the eigenvalue spectrum for +k is negative of the spectrum of -k, with corresponding
    # eigenvectors mapping as x ↦ conj(swap(x)), where swap(...) swaps † operators with non-† operators.
    #
    # Using this, we create classical trajectories that are associated with the entire
    # ±k system rather than just +k or just -k:
    #
    #   (αk(t),αk†(t)) = (αk0,αk†0) exp(iωt)
    #   (α-k(t),α-k†(t)) = (α-k0,α-k†0) exp(-iωt)
    #
    #   (αj(t),αj†(t)) =   exp(-ik Rj) c1 (αk0,αk†0) exp(iωt)
    #                    + exp(+ik Rj) c2 (α-k0,α-k†0) exp(-iωt)
    #
    for apply_swap = [false,true]
      modeshapes_α_swap = copy(modeshapes_α)
      if apply_swap
        for (i,j) = [(1,2),(2,1)]
          modeshapes_α_swap[:,:,i,:] .= conj.(modeshapes_α[:,:,j,:])
        end
      end

      # Matrix multiply (α,α†) → (q,p)
      num_qp = 2
      modeshapes_qp = zeros(ComplexF64,Nf,Nm,num_qp,num_eigenmodes)
      for i = 1:num_qp, j = 1:num_dagger_vs_non_dagger
        modeshapes_qp[:,:,i,:] .+= boson_coords_to_qp[i,j] * modeshapes_α_swap[:,:,j,:]
      end

      # Rotate local spin wave theory frame → lab frame
      modeshapes_global = zeros(ComplexF64,dim_global,Nm,num_eigenmodes)
      for i = 1:dim_global, j = 1:num_qp, atom = 1:Nm
        if sys.mode == :dipole
          # In dipole mode, there is only one boson flavor and the
          # rotation/embedding acts on the (q,p)
          modeshapes_global[i,atom,:] .+= R[atom][i,j] * modeshapes_qp[1,atom,j,:]
        elseif sys.mode == :SUN
          # In SU(N) mode, q is the real part and p is the imaginary part,
          # and the rotation acts on the flavor index
          modeshapes_global[i,atom,:] .+= R[atom][i,j] * (modeshapes_qp[j,atom,1,:] + im * modeshapes_qp[j,atom,2,:])
        end
      end
      
      # Convert unswapped [exp(ix)] and swapped [exp(-ix)] complex solutions
      # to sin-like and cos-like real solutions by linear combination:
      #
      # cos(x) = [exp(ix) + exp(-ix)]/2      [[c1 = 1/2,  c2 = 1/2  ]]
      # sin(x) = [exp(ix) - exp(-ix)]/(2i)   [[c1 = 1/2i, c2 = -1/2i]]
      if apply_swap
        # exp(-ix) contribution
        sin_cos_displacements[:,:,1,:] .+= modeshapes_global / 2 # cos
        sin_cos_displacements[:,:,2,:] .+= -modeshapes_global / (2im) # sin
      else
        # exp(+ix) contribution
        sin_cos_displacements[:,:,1,:] .+= modeshapes_global / 2 # cos
        sin_cos_displacements[:,:,2,:] .+= modeshapes_global / (2im) # sin
      end
    end

    It = diagm([repeat([1],Nf * Nm); repeat([-1],Nf * Nm)])
    disp_full = 2 ./ eigvals(It,H0; sortby = x -> -1/x)

    @assert norm(imag(sin_cos_displacements)) < 1e-8
    sin_cos_displacements = real(round.(sin_cos_displacements,digits = 8))
    if verbose
      println("V matrix:")
      display(Vmat)
      println("Diagonalized V'HV:")
      display(Vmat' * H0 * Vmat)
      println("Eigenmodes (columns are sites)")
      for m = 1:num_eigenmodes
        println()
        println("Mode #$m with energy $(disp_full[m])")
        println("cos → ")
        display(sin_cos_displacements[:,:,1,m])
        println("sin →")
        display(sin_cos_displacements[:,:,2,m])
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
      display(disp_full)
      println("Nm = $Nm, Nf = $Nf, dim_global = $dim_global, num_eigenmodes = $num_eigenmodes")
      display(sys.coherents)
    end
    return H0, Vmat, round.(sin_cos_displacements,digits = 12), disp_full
end

function plot_eigenmode(displacements, swt::SpinWaveTheory; kwargs...)
    fig = Figure()
    ax = LScene(fig[1, 1]; show_axis = false)
    plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; kwargs...)
    fig
end

function plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; t = nothing, k, kwargs...)
  super_size = (5,1,2)
  sys_large = resize_supercell(swt.sys,super_size)
  
  plot_spin_data!(ax,sys_large;color=:grey,arrowscale = 0.9,kwargs...)

  tweaked = Observable(zeros(Vec3f,size(sys_large.dipoles)))
  coherents_scratch = copy(sys_large.coherents)
  dipole_scratch = copy(sys_large.dipoles)

  t = isnothing(t) ? Observable(0.) : t

  on(displacements,update = true) do disps
    notify(t)
  end

  on(t;update = true) do time
    disps = displacements[]
    for i = eachsite(sys_large)
      #spatial_phase = -2π * ((collect(i.I[1:3]) .+ sys_large.crystal.positions[i.I[4]]) ⋅ k[])
      spatial_phase = -2π * ((collect(i.I[1:3])) ⋅ k[])
      atom = i.I[4]
      phase = spatial_phase + time
      if sys_large.mode == :SUN
        coherents_scratch[i] = sys_large.coherents[i] .+ cos(phase) * disps[:,atom,1] .+ sin(phase) * disps[:,atom,2]
        tweaked[][i] = Sunny.expected_spin(coherents_scratch[i])
      elseif sys_large.mode == :dipole
        dipole_scratch[i] = sys_large.dipoles[i] .+ cos(phase) * disps[:,atom,1] .+ sin(phase) * disps[:,atom,2]
        tweaked[][i] = dipole_scratch[i]
      end
    end
    notify(tweaked)
  end

  # TODO: ghost spins are currently inaccurate, since they should pick up a phase factor
  plot_spin_data!(ax,sys_large;color = :blue,spin_data = tweaked,show_cell=false,kwargs...)
end

if !(:eigenmode_viewer_screen ∈ names(Main))
  global eigenmode_viewer_screen = nothing
end
function interact_eigenmodes(swt::SpinWaveTheory, qs, formula;time_scale = 1.0)
  # The background band structure plot
  fig = Figure()
  ax = Axis(fig[1,1], title = "Click a mode! (Spacebar to animate)", xticklabelsvisible = false, xrectzoom = false, yrectzoom = false)
  dispersion, intensity = intensities_bands(swt, qs, formula)
  dispersionmq, intensitymq = intensities_bands(swt, -qs, formula)
  plot_band_intensities!(ax, dispersion, intensity)
  plot_band_intensities!(ax, -dispersionmq, intensitymq, colormap = :spring)

  ylims!(ax, 1.1 * minimum([dispersion;-dispersionmq]), 1.1 * maximum([dispersion;-dispersionmq]))

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
  k = Observable([0. + 0im,0,0]) # SWT Wavevector

  dim_global = swt.sys.mode == :dipole ? 3 : sys.Ns[1]
  rendered_displacements = Observable(zeros(Float64,dim_global,natoms(swt.sys.crystal),2))
  plot_eigenmode!(ax_mode, rendered_displacements, swt; t, k)

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
      _H, _V, sin_cos_displacements, disp = get_eigenmodes(swt,q_interp)

      k[] .= q_interp
      notify(k)

      # Snap to the nearest band (vertically only)
      _, ix = findmin(abs.(disp .- ωclick))

      # Move the marker
      marker_points[][1] = Point2f(q,disp[ix])
      marker_colors[][1] = 1. # TODO: intensity coloring?
      notify(marker_points)

      # Update eigenmode viewer with the newly selected mode
      rendered_displacements[] .= real(sin_cos_displacements[:,:,:,ix])
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
              for t0 in range(0,time_scale * 100 * 2π,length = 8000)
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


