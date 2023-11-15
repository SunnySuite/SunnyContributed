
using Observables, Statistics
include("susceptibility.jl")

function example_viewer()
  cryst = Crystal(I(3),[[0.,0,0]],1)
  sys = System(cryst, (30,30,1), [SpinInfo(1;S=1,g=1)], :dipole, units = Units.theory)
  #set_external_field!(sys,[0,0,0.5])
  #set_onsite_coupling!(sys, S -> -15 * S[3]^2,1)
  set_exchange!(sys,-1.,Bond(1,1,[0,1,0]))
  set_exchange!(sys,-1.,Bond(1,1,[1,0,0]))
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 3000)

  swt = SpinWaveTheory(sys)
  #formula = intensity_formula(swt, [(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
    #S[1]
  #end
  energies = range(-15,15,length = 200)

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

  sys_hm = Sunny.clone_system(sys)
  sys_gs = Sunny.clone_system(sys)

  global view_screen
  if isnothing(view_screen) || view_screen.window_open[] == false
    view_screen = GLMakie.Screen()
  end


  B_fields = 0.1 * (0:120)
  energies = range(-25,25,length = 200)

    f = Figure();

  ax_sys = LScene(f[1, 1]; show_axis = false)
  tweaked = Observable(zeros(Vec3f,size(sys.dipoles)))
  plot_spin_data!(ax_sys,sys;color = :blue,spin_data = tweaked)
  gs_spin_config = Observable(copy(sys.dipoles))
  plot_spin_data!(ax_sys,sys;color = :black,spin_data = gs_spin_config)


  ax_applied = Axis(f[1, 2])
  dt = 0.001
  timeline = round(Int64,5/dt)
  bs = Observable(zeros(Float64,timeline))
  Bxstatic = Observable(0.0)
  B0 = Observable(1.0)
  omega0 = Observable(2.0)
  lines!(ax_applied,dt * (1:timeline),bs)

  ax_mag = Axis(f[2, 2])
  ms = Observable(zeros(Float64,timeline))
  lines!(ax_mag,dt * (1:timeline),ms)
  chi_current = Observable(NaN + 0im)
  ms_predicted = Observable(NaN * zeros(Float64,timeline))
  lines!(ax_mag,dt * (1:timeline),ms_predicted)

  sl_B = Slider(f[1,3], range = -3:0.01:1, horizontal = false, startvalue = -1)
  on(sl_B.value;update = true) do logB
    B0[] = 10^logB
  end


  ax_energy = Axis(f[2, 1])
  es = Observable(energy(sys) * ones(Float64,timeline))
  chiOmega = Observable(NaN * collect(energies) .+ 0im)
  powerSpectrum = Observable(NaN * ms[])
  #lines!(ax_energy,dt * (1:timeline),es)

  tracer_points = Observable(Point2f[Point2f(NaN,NaN)])
  scatter!(ax_energy, tracer_points, strokewidth = 1.4, color = RGBAf(0,0,0,0))
  lines!(ax_energy,energies,map(x -> abs.(x),chiOmega), color = :black)
  lines!(ax_energy,energies,map(x -> imag.(x),chiOmega), color = :red, linestyle = :dash)
  lines!(ax_energy,energies,map(x -> real.(x),chiOmega), color = :green, linestyle = :dash)
  lines!(ax_energy,2π .* fftfreq(timeline,1/dt),powerSpectrum, color = :blue)
  xlims!(ax_energy,extrema(energies))

  display(f)
  empty!(view_screen)
  display(view_screen,f)
  display(view_screen)

  #spectrum_formula = intensity_formula(swt, [(:Sx,:Sx),(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
    #S[1]
  #end
  #function get_spectrum(sys)
    #minimize_energy!(sys)
    #swt = SpinWaveTheory(sys)
    #intensities_spectral_function(swt, [[0.,0,0]], energies, spectrum_formula, decay = δ, susceptibility = true)
  #end

  swt = SpinWaveTheory(sys)
  onames = Dict([(v,k) for (k,v) in swt.observables.observable_ixs])
  options = [("($(string(onames[k.I[1]])),$(string(onames[k.I[2]])))",k) for (k,v) in swt.observables.correlations]
  corr_selector = Menu(f[3,1]; options)
  selected_correlation = Observable(CartesianIndex(1,1))

  rand_button = Button(f[4,1], label = "Randomize spins")
  on(rand_button.clicks) do event
    randomize_spins!(sys)
  end

  refresh_button = Button(f[5,1], label = "Refresh ground state")
  on(refresh_button.clicks) do event
    mouse_hook((Bxstatic[],omega0[]))
  end


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
  fig = Figure(); ax_click = Axis(fig[1,1])
  chi_background = Observable(zeros(Float64,length(B_fields),length(energies)))
  hm = heatmap!(ax_click,B_fields,energies,chi_background)

  # Draw heatmap based on selected correlation
  δ = 1.5
  on(corr_selector.selection; update = true, priority = -1) do ci
    formula = spectral_formula[]
    for (i,Bx) in enumerate(B_fields)
      set_external_field!(sys_hm,[Bx,0,0]) # Parameter
      randomize_spins!(sys_hm)
      minimize_energy!(sys_hm;maxiters = 3000)
      swt_hm = SpinWaveTheory(sys_hm)
      try
        chi_background[][i,:] .= abs.(intensities_spectral_function(swt_hm, [[0.,0,0]], energies, formula, decay = δ, susceptibility = true)[1,:])
      catch e
        println("Issue computing spectrum, NaN'ing part of the heatmap: ")
        println(e)
        chi_background[][i,:] .= NaN
      end
    end
    notify(chi_background)
  end

  # Update ylims on applied field display
  on(B0,update = true) do b0
    Bstat = Bxstatic[]
    ac_vec = ac_unit_vector[]
    ylims!(ax_applied,(-b0 + [Bstat,0,0] ⋅ ac_vec,b0 + [Bstat,0,0] ⋅ ac_vec))
    #ylims!(ax_mag,(-2b0,2b0))
  end

  on(Bxstatic) do bxs
    notify(B0)
  end

  on(corr_selector.selection) do event
    notify(B0)
  end

  on(mag_unit_vector) do mag_hat
  end

  # Marker interaction
  mouse_hook = hook_mouse_marker(fig,ax_click) do B, ωclick
    tracer_points[][1] = Point2f(ωclick,tracer_points[][1][2])
    notify(tracer_points)
    omega0[] = ωclick
    Bxstatic[] = B

    # Re-find ground state at this exact field
    set_external_field!(sys_gs,[B,0,0])
    sys_gs.dipoles .= sys.dipoles # Use the current laboratory spin configuration
    minimize_energy!(sys_gs; maxiters = 3000)

    gs_spin_config[] .= sys_gs.dipoles
    notify(gs_spin_config)

    this_swt = SpinWaveTheory(sys_gs)
    formula = spectral_formula[]
    this_chi = intensities_spectral_function(this_swt, [[0.,0,0]], [energies;ωclick], formula, decay = δ, susceptibility = true)[:]
    notify(mag_unit_vector)
    chi_current[] = this_chi[end]

    chiOmega[] .= this_chi[1:end-1]
    notify(chiOmega)
    yy = maximum(abs.(chiOmega[])) * 3.5
    ylims!(ax_energy,(-yy,yy))

    ax_click.title[] = "B = $(Sunny.number_to_simple_string(B,digits=3)), ω = $(Sunny.number_to_simple_string(ωclick,digits=3)), |χ| = $(Sunny.number_to_simple_string(abs(this_chi[2]),digits=3)), arg χ = $(Sunny.number_to_simple_string(angle(this_chi[2]),digits=3))"
  end

  on(corr_selector.selection;update = true) do event
    mouse_hook((Bxstatic[],omega0[]))
  end

  display(fig)

  # Render loop
  kT = Observable(0.)
  sl_kT = Slider(f[2,3], range = -4:0.01:2, horizontal = false, startvalue = -3)
  on(sl_kT.value;update = true) do logkT
    kT[] = 10^logkT
  end
  int = Langevin(dt,λ = 0.1,kT = kT[])
  #int = ImplicitMidpoint(dt)
  i = 0
  @async begin
    while true
  #begin
    #for i = 1:10
      i = i + 1
      if length(f.scene.current_screens) < 1
      println(length(f.scene.current_screens))
        break
      end
      t = i * dt
      Bamp = B0[] * cos(omega0[] * t)
      ac_vec = ac_unit_vector[]
      app_field = Bxstatic[] * [1,0,0] + Bamp * ac_vec
      set_external_field!(sys,app_field)
      int.kT = kT[] # Grab latest temperature
      step!(sys,int)
      tweaked[] .= sys.dipoles

      bs[][1:(timeline - 1)] .= bs[][2:timeline]
      bs[][timeline] = app_field ⋅ ac_vec


      mag_hat = mag_unit_vector[]
      ground_state_mag = sum(map(x -> x ⋅ mag_hat,sys_gs.dipoles)) / 2

      ms_predicted[][1:(timeline - 1)] .= ms_predicted[][2:timeline]
      ms_predicted[][timeline] = B0[] * abs(chi_current[]) * cos(omega0[] * t + angle(chi_current[]) - (pi/2)) + ground_state_mag

      ms[][1:(timeline - 1)] .= ms[][2:timeline]
      ms[][timeline] = sum(map(x -> x⋅mag_hat,sys.dipoles)) / 2 # magnetization per atom

      es[][1:(timeline - 1)] .= es[][2:timeline]
      es[][timeline] = energy(sys)

      if mod(i,20) == 0
        #el,eh = extrema(es[])
        #ylims!(ax_energy,(el-0.1,eh+0.1))
        ml,mh = extrema(ms[])
        notify(es)
        notify(ms)
        notify(ms_predicted)
        notify(tweaked)
        notify(bs)

        rms_field = sqrt(sum((bs[] .- mean(bs[])) .^ 2) / length(bs[]))
        rms_mag = sqrt(sum((ms[] .- mean(ms[])) .^ 2) / length(ms[]))

        mags = ms[]
        mags = mags .- mean(mags)
        mags = mags .* cos.(((1:timeline) .- (timeline/2)) .* π ./ timeline) .^ 2
        powerSpectrum[] .= real.(fft(mags) .* conj.(fft(mags))) ./ timeline
        notify(powerSpectrum)

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

include("../eigenmodes/support.jl")
using Printf
include("../susceptibility/support.jl")
