
using Observables, Statistics
using DataStructures
include("susceptibility.jl")

function example_viewer()
  cryst = Crystal(I(3),[[0.,0,0]],1)
  sys = System(cryst, (12,12,1), [SpinInfo(1;S=1,g=1)], :dipole, units = Units.theory)
  #set_external_field!(sys,[0,0,0.5])
  set_onsite_coupling!(sys, S -> -1.0 * S[1]^2,1)
  J = 3.0
  set_exchange!(sys,J,Bond(1,1,[0,1,0]))
  set_exchange!(sys,J,Bond(1,1,[1,0,0]))
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 3000)

  swt = SpinWaveTheory(sys)
  #formula = intensity_formula(swt, [(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
    #S[1]
  #end
  #energies = range(-15,15,length = 200)

  #Bzs = range(0,20,length = 300)
  #dat = zeros(ComplexF64,length(Bzs),length(energies))
  #for (i,Bz) in enumerate(Bzs)
    #set_external_field!(sys,[0,0,Bz])
    #randomize_spins!(sys)
    #minimize_energy!(sys)
    #minimize_energy!(sys)
    #minimize_energy!(sys)
    #swt = SpinWaveTheory(sys)
    #formula = intensity_formula(swt, [(:Sx,:Sx),(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
      ##S[1]
    #end
    #dat[i,:] = intensities_spectral_function(swt, [[0.,0,0]], energies, formula, decay = 0.2, susceptibility = true)
  #end
  #four_panel_plot(Bzs,energies,dat,"χxx(B)")
  sys
end

function field_viewer(Bzs,energies,data,name)
  f = Figure()
  ax_r = Axis(f[1,2],title = "Real part of $name")
  ax_i = Axis(f[1,3],title = "Imag part of $name")
  ax_abs = Axis(f[2,2],title = "|$name|")
  ax_arg = Axis(f[2,3],title = "arg $name")

  hm = heatmap!(ax_r,ks,energies,real.(data),colormap=:redgreensplit)
  cr = hm.calculated_colors[].colorrange[]
  sym_range = maximum(abs.(cr))
  hm.colorrange[] = [-sym_range,sym_range]
  Colorbar(f[1,1],hm)

  hm = heatmap!(ax_i,ks,energies,imag.(data))
  Colorbar(f[1,4],hm)

  hm = heatmap!(ax_abs,ks,energies,abs.(data))
  Colorbar(f[2,1],hm)

  csp = ComplexPortraits.cs_p()
  coloring = z -> csp(0.,z)
  hm = image!(ax_arg,ks,energies,coloring.(data))
  #Colorbar(f[2,4],hm)
  f
end


if !(:view_screen ∈ names(Main))
  global view_screen = nothing
end
function sim_AC_applied(sys)
  global view_screen
  if isnothing(view_screen) || view_screen.window_open[] == false
    view_screen = GLMakie.Screen()
  end

  f = Figure();

  f_main = f[1,1]
  f_sliders = f[2,1]

  loc_sys_view = f_main[1,1]
  loc_graphs = f_main[1,2]
  loc_controls = f_main[2,1]
  loc_corr = f_main[2,2]

  # View of system
  ax_sys = LScene(loc_sys_view; show_axis = false)
  current_dipoles = Observable(zeros(Vec3f,size(sys.dipoles)))
  plot_spin_data!(ax_sys,sys;color = :blue,spin_data = current_dipoles)
  gs_spin_config = Observable(copy(sys.dipoles))
  plot_spin_data!(ax_sys,sys;color = :black,spin_data = gs_spin_config)

  # Applied field graph
  ax_applied = Axis(loc_graphs[1,1],xlabel = "Time", ylabel = "Applied Field ⋅ AC direction")
  dt = 0.01
  n_memory = 1500
  time_axis = dt * (1:n_memory) .- (dt * n_memory)
  bs = Observable(CircularBuffer{Float64}(n_memory))
  fill!(bs[],0)
  applied_field_magnitude = Observable(1.0)
  omega0 = Observable(2.0)
  lines!(ax_applied,time_axis,bs)

  # Measured magnetization graph
  ax_mag = Axis(loc_graphs[2,1],xlabel = "Time", ylabel = "Magnetization")
  ms = Observable(CircularBuffer{Float64}(n_memory))
  fill!(ms[],0)
  lines!(ax_mag,time_axis,ms)
  ms_predicted = Observable(CircularBuffer{Float64}(n_memory))
  fill!(ms_predicted[],NaN)
  lines!(ax_mag,time_axis,ms_predicted)

  sg = SliderGrid(f_sliders[1,1],
    (label = "Amp.", range = -3:0.01:1, startvalue = -3, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))"),
    (label = "kT", range = -4:0.01:2, startvalue = -2, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))"),
    (label = "λ", range = -4:0.01:2, startvalue = -1, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))")
   )

  on(sg.sliders[1].value;update = true) do logB
    applied_field_magnitude[] = 10^logB
  end

  kT = Observable(0.)
  on(sg.sliders[2].value;update = true) do logkT
    kT[] = 10^logkT
  end

  λ = Observable(0.1)
  on(sg.sliders[3].value;update = true) do logλ
    λ[] = 10^logλ
  end

  tracer_points = Observable(Point2f[Point2f(NaN,NaN)])

  display(f)
  empty!(view_screen)
  display(view_screen,f)
  display(view_screen)

  swt = SpinWaveTheory(sys)
  onames = Dict([(v,k) for (k,v) in swt.observables.observable_ixs])
  options = [("($(string(onames[k.I[1]])),$(string(onames[k.I[2]])))",k) for (k,v) in swt.observables.correlations]
  corr_selector = Menu(loc_controls[1,1]; options)
  selected_correlation = Observable(CartesianIndex(1,1))

  # Formula updates based on selected correlation
  spectral_formula = map(corr_selector.selection) do ci
    a,b = ci.I
    intensity_formula(swt, [(onames[a],onames[b])], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
      S[1]
    end
  end

  function name_to_unit_vector(s::Symbol)
    if s == :Sx
      [1,0,0]
    elseif s == :Sy
      [0,1,0]
    elseif s == :Sz
      [0,0,1]
    else
      println("Invalid observable $s")
      [0,0,0]
    end
  end

  # TODO: check convention
  ac_unit_vector = map(ci -> name_to_unit_vector(onames[ci.I[2]]),corr_selector.selection)
  mag_unit_vector = map(ci -> name_to_unit_vector(onames[ci.I[1]]),corr_selector.selection)

  # Heatmap window
  fig = Figure(); ax_click = Axis(fig[1,1],xlabel = "Static Bx Applied", ylabel = "ω applied magnetic field")
  f_spectrum = fig[2,1]
  static_B_fields = 0.1 * (0:120)
  applied_B_freqs = range(-25,25,length = 200)
  chi_background = Observable(zeros(Float64,length(static_B_fields),length(applied_B_freqs)))
  hm = heatmap!(ax_click,static_B_fields,applied_B_freqs,chi_background)

  # Draw heatmap based on selected correlation
  δ = 1.5
  sys_hm = Sunny.clone_system(sys)
  on(corr_selector.selection; update = true, priority = -1) do ci
    formula = spectral_formula[]
    for (i,Bx) in enumerate(static_B_fields)
      set_external_field!(sys_hm,[Bx,0,0]) # Parameter
      randomize_spins!(sys_hm)
      minimize_energy!(sys_hm;maxiters = 3000)
      swt_hm = SpinWaveTheory(sys_hm)
      try
        chi_background[][i,:] .= abs.(intensities_spectral_function(swt_hm, [[0.,0,0]], applied_B_freqs, formula, decay = δ, susceptibility = true)[1,:])
      catch e
        println("Issue computing spectrum, NaN'ing part of the heatmap: ")
        println(e)
        chi_background[][i,:] .= NaN
      end
    end
    notify(chi_background)
  end

  rand_button = Button(loc_controls[2,1], label = "Randomize spins")
  on(rand_button.clicks) do event
    randomize_spins!(sys)
  end

  Bxstatic = Observable(0.0)
  refresh_button = Button(loc_controls[3,1], label = "Refresh ground state")
  on(refresh_button.clicks) do event
    mouse_hook((Bxstatic[],omega0[]))
  end

  # Update ylims on applied field display
  on(applied_field_magnitude,update = true) do b0
    Bstat = Bxstatic[]
    ac_vec = ac_unit_vector[]
    ylims!(ax_applied,(-b0 + [Bstat,0,0] ⋅ ac_vec,b0 + [Bstat,0,0] ⋅ ac_vec))
    #ylims!(ax_mag,(-2b0,2b0))
  end

  on(Bxstatic) do bxs
    notify(applied_field_magnitude)
  end

  on(corr_selector.selection) do event
    notify(applied_field_magnitude)
  end

  on(mag_unit_vector) do mag_hat
  end

  int = Langevin(dt,λ = λ[],kT = kT[])
  nt = 160#320
  measperiod = 8
  oc = mk_oc(sys; measperiod,nt, integrator = int, observables = nothing, correlations = nothing)

  selected_correlation_index = map(corr_selector.selection) do ci
    a,b = ci.I
    Sunny.lookup_correlations(oc.observables,[(onames[a],onames[b])])
  end

  # Marker interaction
  sys_gs = Sunny.clone_system(sys)
  chi_current = Observable(NaN + 0im) # Records the LSWT-predicted χ
  mouse_hook = hook_mouse_marker(fig,ax_click) do B, ωclick
    tracer_points[][1] = Point2f(ωclick,tracer_points[][1][2])
    notify(tracer_points)
    omega0[] = ωclick
    Bxstatic[] = B

    # Re-find ground state at this exact field
    set_external_field!(sys_gs,[B,0,0])
    sys_gs.dipoles .= oc.sys.dipoles # Use the current laboratory spin configuration
    minimize_energy!(sys_gs; maxiters = 3000)

    gs_spin_config[] .= sys_gs.dipoles
    notify(gs_spin_config)

    this_swt = SpinWaveTheory(sys_gs)
    formula = spectral_formula[]
    this_chi = intensities_spectral_function(this_swt, [[0.,0,0]], [applied_B_freqs;ωclick], formula, decay = δ, susceptibility = true)[:]
    notify(mag_unit_vector)
    chi_current[] = this_chi[end]

    #chiOmega[] .= this_chi[1:end-1]
    #notify(chiOmega)
    #yy = maximum(abs.(chiOmega[])) * 3.5
    #ylims!(ax_energy,(-yy,yy))

    ax_click.title[] = "B = $(Sunny.number_to_simple_string(B,digits=3)), ω = $(Sunny.number_to_simple_string(ωclick,digits=3)), |χ| = $(Sunny.number_to_simple_string(abs(chi_current[]),digits=3)), arg χ = $(Sunny.number_to_simple_string(angle(chi_current[]),digits=3))"
  end

  on(corr_selector.selection;update = true) do event
    mouse_hook((Bxstatic[],omega0[]))
  end

  display(fig)




  # Online correlations graph
  ax_corr = Axis(loc_corr[1,1])
  corr_time = dt * oc.measperiod * fftshift(fftfreq(nt,nt))
  corr_vals = Observable(zeros(Float64,nt))
  corr_vals_neighbor = Observable(zeros(Float64,nt))
  lines!(ax_corr,corr_time,corr_vals)
  lines!(ax_corr,corr_time,corr_vals_neighbor,color = :black)

  # Precompute interpolation constants (TODO: move to online correlations code)
  points = [[1/2,   0, 0],  # List of wave vectors that define a path
            [0,   1/2, 0],
            [0,   1/2, 0],
            [1/2,   1/2, 0],
            [0,   0, 0],
            [1/2, 0 ,0]]
  density = 8
  path, xticks = reciprocal_space_path(oc.sys.crystal, points, density);
  formfactors = [FormFactor("Fe2"; g_lande=3/2)]
  new_formula = intensity_formula(oc, :trace; kT = Inf, formfactors)
  ixqs = Vector{CartesianIndex{3}}(undef,length(path))
  ks = Vector{Sunny.Vec3}(undef,length(path))
  hmdat = Observable(ones(Float64,length(path),nt))
  ax = Axis(f_spectrum[1,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
  dw = 2π / (oc.integrator.Δt * oc.measperiod * nt)
  hm = heatmap!(ax,1:length(path),dw * fftshift(fftfreq(nt,nt)),map(x -> log10.(abs.(x)),hmdat))
  Colorbar(f_spectrum[1,2],hm)
  for (j,q) in enumerate(path)
    m = round.(Int, oc.sys.latsize .* q)
    ixqs[j] = map(i -> mod(m[i], oc.sys.latsize[i])+1, (1, 2, 3)) |> CartesianIndex{3}
    ks[j] = Sunny.Vec3(oc.sys.crystal.recipvecs * q)
  end

  # Simulation loop
  i = 0
  @async begin
    while true
      i = i + 1
      t = i * dt
      if length(f.scene.current_screens) < 1
      println(length(f.scene.current_screens))
        break
      end

      # Apply field
      Bamp = applied_field_magnitude[] * cos(omega0[] * t)
      ac_vec = ac_unit_vector[]
      app_field = Bxstatic[] * [1,0,0] + Bamp * ac_vec
      set_external_field!(oc.sys,app_field) # Update system
      push!(bs[],app_field ⋅ ac_vec) # Update graph

      # Grab latest integrator parameters
      oc.integrator.λ = λ[]
      oc.integrator.kT = kT[]

      # Integrate forward
      step!(oc.sys,oc.integrator)
      # Walk the correlator forward once every oc.measperiod steps
      if mod(i,oc.measperiod) == 0
        walk_online_no_step!(oc)
      end

      # Actual magnetization (per site)
      mag_hat = mag_unit_vector[]
      push!(ms[],magnetization_along_axis(mag_hat,oc.sys.dipoles))

      # LSWT prediction of magnetization (per site)
      ground_state_mag = magnetization_along_axis(mag_hat, sys_gs.dipoles)
      lswt_prediction = applied_field_magnitude[] * abs(chi_current[]) * cos(omega0[] * t + angle(chi_current[]) - (pi/2)) + ground_state_mag
      push!(ms_predicted[],lswt_prediction)

      if mod(i,2000) == 0
        # Correlation spectrum
        fft!(oc.data,1)
        for (j,q) in enumerate(path)
          for t = 1:nt
            hmdat[][j,t] = new_formula.calc_intensity(oc, ks[j], ixqs[j], t)
          end
        end
        hmdat[] .= fftshift(hmdat[],2)
        ifft!(oc.data,1)
        notify(hmdat)
      end

      if mod(i,20) == 0
        notify(ms)
        notify(ms_predicted)
        current_dipoles[] .= oc.sys.dipoles
        notify(current_dipoles)
        notify(bs)

        # Real space correlations
        real_dat = real.(ifft(oc.data,(2,3,4)))

        sel_ix = selected_correlation_index[]
        corr_vals[] .= fftshift(real_dat[:,1,1,1,1,1,sel_ix])
        corr_vals_neighbor[] .= fftshift(real_dat[:,2,1,1,1,1,sel_ix])
        el,eh = extrema(corr_vals[])
        ylims!(ax_corr,(el-0.0001,eh+0.0001))
        notify(corr_vals)
        notify(corr_vals_neighbor)

        rms_field = sqrt(sum((bs[] .- mean(bs[])) .^ 2) / length(bs[]))
        rms_mag = sqrt(sum((ms[] .- mean(ms[])) .^ 2) / length(ms[]))

        ml,mh = extrema(ms[])
        ylims!(ax_mag,(ml-rms_mag/2,mh+rms_mag/2))
        ax_mag.title[] = "RMS mag = $(rms_mag)"
        ax_applied.title[] = "RMS field = $(rms_field)"
        tracer_points[][1] = Point2f(tracer_points[][1][1],rms_mag / rms_field)
        notify(tracer_points)
        sleep(0.01)
      end
    end
    println("Done!")
  end


end

function hook_mouse_marker(f,fig,ax; marker = true)
  if marker
    marker_points = Observable(Point2f[Point2f(NaN,NaN)])
    scatter!(ax, marker_points, strokewidth = 1.)
  end

  function mouse_hook(data_space_c)
    if marker
      # Move the marker
      marker_points[][1] = Point2f(data_space_c...)
      notify(marker_points)
    end

    try
      f(data_space_c...)
    catch e
      println("hook_mouse_marker: Error during callback!")
      println(e)
    end
  end

  function do_update(;mp = nothing)
    # mouse position = nothing on click, but we already know mp when dragging
    mp = isnothing(mp) ? events(ax).mouseposition[] : mp

    # Convert from screen-space to data-space
    bbox = ax.layoutobservables.computedbbox[]
    c = (mp .- bbox.origin) ./ bbox.widths
    if 0 < c[1] < 1 && 0 < c[2] < 1
      data_bbox = ax.finallimits[]
      data_space_c = data_bbox.origin .+ c .* data_bbox.widths
      mouse_hook(data_space_c)
    end
  end

  # Update on left click
  on(events(fig).mousebutton, priority = 2; update = true) do event
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
  mouse_hook
end

function magnetization_along_axis(mag_hat,dipoles)
  sum(map(x -> x ⋅ mag_hat,dipoles)) / length(dipoles)
end

include("online_correlations.jl")
include("../eigenmodes/support.jl")
using Printf
include("../susceptibility/support.jl")
