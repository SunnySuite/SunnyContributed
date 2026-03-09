# Visualize individual spin wave oscillations in real space.
# Author: Sam Quinn
#
# To use the GUI viewer, call `interact_eigenmodes(swt, qs)`.
#
# To run the eigenmode analysis at a single wavevector, call `get_eigenmodes(swt,q,verbose = true)`.
# 
# The correctness of the visualization is not yet verified--see comments within--and also
# currently only really shows the dipole sector.

using Sunny, GLMakie, LinearAlgebra

function get_eigenmodes(swt, q; verbose = false)
    (; sys, data) = swt

    ground_states, rotations = if sys.mode == :SUN
      sys.coherents, data.local_unitaries
    elseif sys.mode == :dipole
      error("NYI: dipole mode")
      # The story is exactly the same for dipole mode, they are
      # just called differently!

      #for i = 1:natoms(sys.crystal)
        #sys.dipoles
      #map(x -> ComplexF64.(Vector(x)),sys.dipoles), map(x -> ComplexF64.(Matrix(x)),data.local_rotations)
    end

    energies, T = excitations(swt,q)

    # The spin wave system is described by classical variables (α1,...,αN,α†1,..,α†N).
    # The N pairs of variables are organized by sublattice:
    Nm = Sunny.natoms(sys.crystal) # Number of sublattice = number of atoms in magnetic unit cell
    @assert size(ground_states) == (1,1,1,Nm) "SpinWaveTheory has more than one unit cell!"
    # and each sublattice may have several flavors of excitation; there is one flavor of
    # boson for each transverse mode. In :dipole mode, there is only one flavor per sublattice.
    # For SU(N), there are N total components per sublattice, but one is longitudinal, so (N-1) flavors.
    Nf = sys.mode == :dipole ? 1 : (sys.Ns[1] - 1)
    # 
    # The variables are ordered as:
    #
    #   First sublattice: α1,...,αNf
    #   Second sublattice: α(Nf+1),...,α(2Nf)
    #   ...
    #   Nmth sublattice: α((Nm - 1)*Nf+1),...,α(Nm * Nf)
    #   
    #   [followed by † version of the above]
    #
    # The number of eigenmodes (number of columns of T) is one for each classical variable.
    num_dagger_vs_non_dagger = 2
    num_variables = Nf * Nm * num_dagger_vs_non_dagger
    num_eigenmodes = num_variables

    # The `T' bogoliubov matrix has one column for each eigenmode, and
    # each column contains initial configurations for every variable:
    @assert size(T,1) == num_variables
    @assert size(T,2) == num_eigenmodes
    
    # Break down each eigenmode in the bogoliubov matrix
    # by: flavor, sublattice, and dagger-vs-non-dagger
    modeshapes = Array{ComplexF64,3}[]
    for i = 1:num_eigenmodes
      push!(modeshapes,reshape(T[:,i],Nf,Nm,num_dagger_vs_non_dagger))
    end


    # The (q,p) coordinates are a specific choice of coordinates for the tangent
    # space to the ground state. To visualize the eigenmode however, we want to
    # embed the tangent space into the ambient space. The embedding is given by:
    #
    #   (q,p) ↦ [ground state] + ε R*(q,p,0)
    #
    # where ε is a small number, R is the rotation matrix (part of the swt.data)
    # that embeds the (q,p,0) vector into the tangent space (considered as a subspace of
    # the ambient space), and [ground state] is the point in state space that we are
    # linearizing around, in coordinates in the ambient space.
    Rs = begin
      Rs = Matrix{ComplexF64}[]
      for i = 1:Nm
        U = rotations[i]

        # Assert that, if we invert the embedding, viz.
        U_inv = U'
        #   originally: tangent coordinates → ambient
        #   inverse: ambient space → tangent coordinates
        # then the ground state should map to (0,…,0,1),
        # proving that the first Nf columns are transverse and
        # the final column is longitudinal.
        ground_state = ground_states[i]
        target = zeros(ComplexF64,Nf+1); target[end] = 1.0
        @assert isapprox(U_inv * ground_state, target;atol = 1e-12) "The SpinWaveTheory rotation failed to map the ground state to (0,...,0,1)"

        # Since the last column is longitudinal, and we compute
        # only transverse fluctuations, we can drop that column
        # entirely!
        push!(Rs,U[:,1:end-1])
      end
      Rs
    end

    # Historical note:
    # [[The second half of the columns contain the boson creation operators at -k in a similar
    # format, but in reverse order, e.g. it goes [b1,b2,b†2,b†1]. But the creation operators
    # are not needed because they can be inferred from the deletion operators. The fact
    # that they can be inferred is equivalent to the dagger operation being preserved
    # by the bogoliubov transform V]]
    

    #dim_global = sys.mode == :dipole ? 3 : sys.Ns[1]
    #num_sin_cos = 2
    #sin_cos_displacements = zeros(ComplexF64,dim_global,Nm,num_sin_cos,num_eigenmodes)

    # The eigenmodes described by T were generated using the ansatz:
    #
    #   (α(R,t),α†(R,t)) = (α0,α†0) exp(-ikR) exp(iωt)
    #
    # which is a *complex* solution. To get classical trajectories we need to
    # add or subtract this from a solution with the opposite complex phase.
    # In a centrosymmetric system, the eigenvalues at any particular wavevector come in ±pairs,
    # so we could use those to construct real-valued classical solutions---but in general
    # systems this eigenvalue spectrum is *not* symmetric so we can't do this.
    #
    # Instead, we need to rely on a special property of Linear Spin Wave Hamiltonians, namely
    # that the eigenvalue spectrum for +k is negative of the spectrum of -k, with corresponding
    # eigenvectors mapping as x ↦ conj(swap(x)), where swap(...) swaps † operators with non-† operators.
    #
    # In detail, it means that (β(R,t),β†(R,t)) = (β0,β†0) exp(+ikR) exp(-iωt),
    # where (β0,β†0) = conj((α†0,α0)), is another valid eigenmode.

    # Step 1: Perform the x → conj(swap(x)) operation on each mode individually
    swapped_ix = [2,1]
    conjswap_modeshapes = map(x -> conj.(x[:,:,swapped_ix]),modeshapes)
    # If the frequencies for the modeshapes were originally ordered like:
    #   exp(-ikR) exp(iωt) where ω = +a,+b,…,+m,-n,…,-z
    # then the conjswap_modeshapes frequencies are:
    #   exp(+ikR) exp(iωt) where ω = -a,-b,…,-m,+n,…,+z
    #
    # ! However, ! this fact is complicated to observe directly by the fact that
    # the dynamical matrix for exp(-ikR)-type modes is a different matrix from
    # the one that is applicable for exp(+ikR)-type waves. Here, we calculate
    # both and assert the appropriate relations as an active check that
    # everything is working how we think it is:

    # Internal hackery to get access to the dynamical matrix
    dyn_mat = 0*copy(T)
    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)
    Sunny.dynamical_matrix!(dyn_mat, swt, q_reshaped)
    dyn_mat_neg = 0*copy(T)
    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, -q)
    Sunny.dynamical_matrix!(dyn_mat_neg, swt, q_reshaped)
    Itilde = diagm([ones(size(dyn_mat,1)÷2); -ones(size(dyn_mat,1)÷2)])

    for j = 1:length(modeshapes)
      v = modeshapes[j][:]
      w = conjswap_modeshapes[j][:]

      #=
      println()
      println("Modeshape $j:")
      display(round.(v,digits=12))
      println("has rayleigh quotient $(v' * dyn_mat * v) wrt D_k")
      println("and its conjugate modeshape $j:")
      display(round.(w,digits=12))
      println("has rayleigh quotient $(w' * dyn_mat_neg * w) wrt D_-k")
      println()
      =#

      # This is that weird extra minus sign from Itilde
      #sgn = j > length(modeshapes)÷2 ? -1 : 1

      #display(energies)
      #display(energies[j] * v)
      #display(Itilde * dyn_mat * v)
      @assert isapprox(energies[j] * v,Itilde * dyn_mat * v;atol = 1e-8) "Unexpected eigenvalue for an eigenvector. Do you have degeneracies in the spectrum? Consider breaking any degeneracies"
      #display(energies[j] * w)
      #display(-Itilde * dyn_mat_neg * w)
      @assert isapprox(energies[j] * w,-Itilde * dyn_mat_neg * w;atol = 1e-8) "Unexpected eigenvalue for a conjugate eigenvector. Do you have degeneracies in the spectrum? Consider breaking any degeneracies"
    end

    # Using this, we create classical trajectories that are associated with the entire
    # ±k system rather than just +k or just -k, even though we only actually have
    # the eigendata at +k:
    #
    #   (αj(t),αj†(t)) =   exp(-ik Rj) c1 (α0,α†0) exp(iωt)
    #                    + exp(+ik Rj) c2 (β0,β†0) exp(-iωt)
    #
    # where different choices of c1,c2 give differently phased trajectories.
    q_cos_disp = Matrix{Float64}[]
    q_sin_disp = Matrix{Float64}[]
    p_cos_disp = Matrix{Float64}[]
    p_sin_disp = Matrix{Float64}[]
    # TODO: remove degeneracy before doing this or it goes wrong!
    for j = 1:num_eigenmodes
      # We want to make a change of classical coordinates from (α,α†)
      # to q = (α + α†)/2, p = (α - α†)/(2i), which are the more
      # usual coordinates for each oscillator. They have the advantage
      # of not introducing any additional complex factors to the calculation.
      q_shape = (modeshapes[j][:,:,1] .+ modeshapes[j][:,:,2]) ./ 2
      p_shape = (modeshapes[j][:,:,1] .- modeshapes[j][:,:,2]) ./ (2im)
      q_shape_conjswap = (conjswap_modeshapes[j][:,:,1] .+ conjswap_modeshapes[j][:,:,2]) ./ 2
      p_shape_conjswap = (conjswap_modeshapes[j][:,:,1] .- conjswap_modeshapes[j][:,:,2]) ./ (2im)

      # First term evolves like exp(-ikR)exp(iωt), second like exp(+ikR)exp(-iωt)
      q_cos = (q_shape .+ q_shape_conjswap) ./ 2 # 2 cos(phi) = exp(phi) + exp(-phi)
      # This will be real because: (i) the R,t dependence was just shown to be real
      # and (ii) the coefficients in front of that dependence are now real numbers
      # because they are the actual q and p displacements. For SU(N) mode, even though
      # the coherent state Z is complex, we describe its oscillations by breaking
      # it up into Z = q + ip real-valued tangent coordinates.
      @assert norm(imag(q_cos)) < 1e-8
      push!(q_cos_disp, real(q_cos))

      q_sin = (q_shape .- q_shape_conjswap) ./ (2im) # 2i sin(phi) = exp(phi) - exp(-phi)
      @assert norm(imag(q_sin)) < 1e-8
      push!(q_sin_disp, real(q_sin))

      p_cos = (p_shape .+ p_shape_conjswap) ./ 2 # 2 cos(phi) = exp(phi) + exp(-phi)
      @assert norm(imag(p_cos)) < 1e-8
      push!(p_cos_disp, real(p_cos))

      p_sin = (p_shape .- p_shape_conjswap) ./ (2im) # 2i sin(phi) = exp(phi) - exp(-phi)
      @assert norm(imag(p_sin)) < 1e-8
      push!(p_sin_disp, real(p_sin))

    end
    #=
    display(round.(q_cos_disp[1];digits=8))
    display(round.(q_sin_disp[1];digits=8))
    display(round.(p_cos_disp[1];digits=8))
    display(round.(p_sin_disp[1];digits=8))
    println()
    =#

    # Assemble the q and p tangent coordinate displacements into
    # actual spin-space (ambient space) displacements using the 
    # rotation data.
    Z_cos_disp = Matrix{ComplexF64}[]
    Z_sin_disp = Matrix{ComplexF64}[]
    for mode = 1:num_eigenmodes
      # Z = q + ip within the tangent space...
      δz_tangent = q_cos_disp[mode] .+ im * p_cos_disp[mode]
      # and then embed in ambient space:
      dim_ambient = Nf + 1
      δz_ambient = [sum([Rs[site][i,k] * δz_tangent[k,site] for k = 1:Nf])
                    for i = 1:dim_ambient, site = 1:Nm]
      push!(Z_cos_disp, δz_ambient)

      δz_tangent = q_sin_disp[mode] .+ im * p_sin_disp[mode]
      δz_ambient = [sum([Rs[site][i,k] * δz_tangent[k,site] for k = 1:Nf])
                    for i = 1:dim_ambient, site = 1:Nm]
      push!(Z_sin_disp, δz_ambient)
    end
    return energies, Z_cos_disp, Z_sin_disp
end

function graph_eigenmode(ω,I,Q,k,swt::SpinWaveTheory;temporal = true,amp = 0.3)
  color_cycle = [:blue,:orange,:green,:pink]

  Na = Sunny.natoms(swt.sys.crystal)
  Nsun = swt.sys.Ns[1]

  fig = Figure()
  ax_base = Axis(fig[1,1];title = "unit cell 0",xlabel = "Time")
  ax_a = Axis(fig[1,2];title = "unit cell 0+a",xlabel = "Time")
  ax_b = Axis(fig[2,1];title = "unit cell 0+b",xlabel = "Time")
  ax_c = Axis(fig[2,2];title = "unit cell 0+c",xlabel = "Time")

  for a = [ax_base,ax_a,ax_b,ax_c]
    if temporal
      a.xlabel[] = "Time"
    else
      a.xlabel[] = "Re Z"
      a.ylabel[] = "Im Z"
    end
  end

  overlap = 0.1
  period = 2π/ω
  ts = range(-overlap*period,(1+overlap)*period,length = 350)

  for j = 1:Na, i = 1:Nsun
    gs = swt.sys.coherents[1,1,1,j][i]

    for (v,ax) = [([0,0,0],ax_base),([1,0,0],ax_a),([0,1,0],ax_b),([0,0,1],ax_c)]
      # spatial_phase = -2π * (Sunny.position(swt.sys,(v[1],v[2],v[3],j)) ⋅ k)
      spatial_phase = -2π * (Sunny.global_position(swt.sys)[v[1],v[2],v[3],j] ⋅ k)
      phase = spatial_phase .+ ω * ts
      Zt = gs .+ amp * (cos.(phase) .* I[i,j] + sin.(phase) .* Q[i,j])
      if temporal
        lines!(ax,ts,real(Zt),color = color_cycle[i],linestyle = :solid)
        lines!(ax,ts,imag(Zt),color = color_cycle[i],linestyle = :dash)
        if Na > 1
          text!(ax,ts[1] * 1.05,real(Zt)[1],color = color_cycle[i],text = "$j",align = (:right,:center))
          text!(ax,ts[1] * 1.05,imag(Zt)[1],color = color_cycle[i],text = "$j",align = (:right,:center))
        end
      else
        Z0 = gs .+ amp * (cos.(0) .* I[i,j] + sin.(0) .* Q[i,j])
        scatter!(ax,real(Z0),imag(Z0),color = color_cycle[i])
        Z90 = gs .+ amp * (cos.(π/2) .* I[i,j] + sin.(π/2) .* Q[i,j])
        scatter!(ax,real(Z90),imag(Z90),color = color_cycle[i],marker = 'x')
        lines!(ax,real(Zt),imag(Zt),color = color_cycle[i],linestyle = :solid)
        if Na > 1
          text!(ax,real(Zt[1]),imag(Zt[1]),color = color_cycle[i],text = "$j",align = (:center,:top))
        end
      end
    end
  end
  display(fig)
end

function plot_eigenmode(displacements, swt::SpinWaveTheory; super_size, kwargs...)
    fig = Figure()
    ax = LScene(fig[1, 1]; show_axis = false)
    plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; super_size, kwargs...)
    fig
end

function plot_eigenmode!(ax, displacements, swt::SpinWaveTheory; amp=1.0, t=nothing, k, super_size, kwargs...)
  sys_large = resize_supercell(swt.sys, super_size)
  
  notif_sunny = Observable(nothing)
  plot_spins!(ax,sys_large;notifier = notif_sunny, kwargs...)

  tweaked = Observable(zeros(Vec3f,size(sys_large.dipoles)))
  coherents_scratch = copy(sys_large.coherents)
  dipole_scratch = copy(sys_large.dipoles)

  t = isnothing(t) ? Observable(0.) : t

  on(displacements,update = true) do disps
    notify(t)
  end

  on(t;update = true) do time
    Z_cos, Z_sin = displacements[]
    for i = eachsite(sys_large)
      spatial_phase = -2π * (Sunny.position_at(sys_large,i) ⋅ k[])
      atom = i.I[4]
      phase = spatial_phase + time
      if sys_large.mode == :SUN
        coherents_scratch[i] = sys_large.coherents[i] .+ amp * (cos(phase) * collect(Z_cos[:,atom]) .+ sin(phase) * collect(Z_sin[:,atom]))
        sys_large.dipoles[i] = Sunny.expected_spin(coherents_scratch[i])
      elseif sys_large.mode == :dipole
        error("NYI")
        #sys_large.dipoles[i] = dipole_scratch[i]
      end
    end
    notify(notif_sunny)
  end

  #plot_spin_data!(ax,sys_large;color = :blue,spin_data = tweaked,show_cell=false,kwargs...)
  nothing
end

if !(:eigenmode_viewer_screen ∈ names(Main))
  global eigenmode_viewer_screen = nothing
end
function interact_eigenmodes(swt::SpinWaveTheory, qs_spec::Sunny.QPath; super_size, time_scale=1.0, kwargs...)
  # The background band structure plot
  fig = Figure()
  ax = Axis(fig[1,1], title = "Click a mode! (Spacebar to animate)", xticklabelsvisible = false, xrectzoom = false, yrectzoom = false)
  res = intensities_bands(swt,qs_spec)
  dispersion, intensity = res.disp', res.data'

  qs_specm = Sunny.QPath(-qs_spec.qs,qs_spec.xticks)
  resm = intensities_bands(swt, qs_specm)
  dispersionm, intensitym = resm.disp', resm.data'

  qs = qs_spec.qs
  plot_band_intensities!(ax, dispersion, intensity)
  plot_band_intensities!(ax, -dispersionm, intensitym, colormap = :spring)

  ylims!(ax, 1.1 * minimum([dispersion;-dispersionm]), 1.1 * maximum([dispersion;-dispersionm]))

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

  dim_global = swt.sys.mode == :dipole ? 3 : swt.sys.Ns[1]
  ωs, zc, zs = get_eigenmodes(swt,[0,0,0])
  rendered_displacements = Observable((zc[1],zs[1]))
  plot_eigenmode!(ax_mode, rendered_displacements, swt; super_size, t, k, kwargs...)

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
      energies,zc,zs = get_eigenmodes(swt,q_interp)

      k[] .= q_interp
      notify(k)

      # Snap to the nearest band (vertically only)
      _, ix = findmin(abs.(energies .- ωclick))

      # Move the marker
      marker_points[][1] = Point2f(q,energies[ix])
      marker_colors[][1] = 1. # TODO: intensity coloring?
      notify(marker_points)

      # Update eigenmode viewer with the newly selected mode
      rendered_displacements[] = (zc[ix],zs[ix])
      notify(rendered_displacements)
      ax.title[] = "q = [$(join(Sunny.number_to_simple_string.(q_interp,digits=3),","))], ω = $(Sunny.number_to_simple_string(energies[ix],digits=3))"
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
  on(events(fig).mouseposition) do mp
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
                if fig.scene.isclosed
                  break
                end
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

function plot_band_intensities(dispersion, intensity; kwargs...)
    f = Makie.Figure()
    ax = Makie.Axis(f[1,1]; xlabel = "Momentum", ylabel = "Energy (meV)", xticklabelsvisible = false)
    plot_band_intensities!(ax, dispersion, intensity; kwargs...)
    f
end

function plot_band_intensities!(ax, dispersion, intensity; rev_x=false, kwargs...)
    Makie.ylims!(ax, min(0.0,minimum(dispersion)), maximum(dispersion))
    Makie.xlims!(ax, 1, size(dispersion, 1))
    colorrange = extrema(intensity)
    for i in axes(dispersion)[2]
      Makie.lines!(ax, rev_x ? reverse(1:length(dispersion[:,i])) : 1:length(dispersion[:,i]), dispersion[:,i]; color=intensity[:,i], colorrange,kwargs...)
    end
    nothing
end
