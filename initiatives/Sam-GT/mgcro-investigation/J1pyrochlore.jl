using Sunny, GLMakie, Observables

#include("../susceptibility/support.jl")
include("../eigenmodes/support.jl")
if !(:view_screen ∈ names(Main))
  global view_screen = nothing
end
function sim_pyrochlore()
  global view_screen
  if isnothing(view_screen) || view_screen.window_open[] == false
    view_screen = GLMakie.Screen()
  end

  pyro = Sunny.pyrochlore_crystal()


  model_mode = Observable(:dipole)
  model_spin = Observable(3/2)
  model_latsize = Observable((1,1,1))
  model_J1 = Observable(1.)
  model_B_field = Observable(0.)

  current_sys = Observable{System}(System(pyro, (1,1,1), [SpinInfo(1, S=3/2, g=2)], :dipole))

  f_view = Figure();
  ax_view = LScene(f_view[1,1],show_axis=false)
  spin_data_obs = Observable(current_sys[].dipoles)
  volume_data_obs = Observable(zeros(Float64,1,1,1))

  f = Figure()
  f_main = f[1,1]
  f_Sq = f[2,1]
  loc_buttons = f_main[1,1]
  loc_graphs = f_main[1,2]
  loc_controls = f_main[2,1]
  loc_corr = f_main[2,2]

  volume!(LScene(f_Sq,show_axis=false),volume_data_obs,height = 300)

  energy_readout = Label(loc_buttons[2,1], text = "Energy = ?",tellwidth=false)

  function update_energy_readout()
    energy_readout.text[] = "Energy = $(Sunny.number_to_simple_string(energy(current_sys[]),digits = 4)) meV"
  end

  function recreate_sys()
    println("Recreated System")
    mode = model_mode[]
    spin = model_spin[]
    latsize = model_latsize[]
    old_sys = current_sys[]
    new_sys = System(pyro, latsize, [SpinInfo(1, S=spin, g=2)], mode)
    set_exchange!(new_sys, model_J1[], Bond(1, 2, [0,0,0]))  # J1
    set_external_field!(new_sys,[0,0,model_B_field[]])

    # Copy over spin state
    dd_copy = mode == :dipole || old_sys.mode == :dipole || length(old_sys.coherents[1]) != length(new_sys.coherents[1])
    if all(new_sys.latsize .<= old_sys.latsize)
      for i = CartesianIndices(tuple(new_sys.latsize...,Sunny.natoms(old_sys.crystal)))
        if dd_copy
          set_dipole!(new_sys,old_sys.dipoles[i],i)
        else
          set_coherent!(new_sys,old_sys.coherents[i],i)
        end
      end
    else
      # Repeat unit cell periodically
      for i = Sunny.eachcell(new_sys)
        for a = 1:Sunny.natoms(old_sys.crystal)
          if dd_copy
            set_dipole!(new_sys,old_sys.dipoles[1,1,1,a],CartesianIndex(i.I...,a))
          else
            set_coherent!(new_sys,old_sys.coherents[1,1,1,a],CartesianIndex(i.I...,a))
          end
        end
      end
    end
    current_sys[] = new_sys

    empty!(ax_view)
    spin_data_obs = Observable(new_sys.dipoles)
    isc = instant_correlations(new_sys)
    formula = intensity_formula(isc,:trace)
    params = unit_resolution_binning_parameters(isc)
    params.binstart[1:3] .-= 2
    params.binend[1:3] .+= 1
    on(spin_data_obs;update = true) do spin_data
      update_energy_readout()
      isc.data .= 0
      isc.nsamples[1] = 0
      #new_sys.dipoles .= spin_data
      add_sample!(isc,new_sys)
      is, counts = intensities_binned(isc,params,formula)
      volume_data_obs[] = is[:,:,:,1] ./ counts[:,:,:,1]
      notify(volume_data_obs)
    end

    Sunny.Plotting.plot_spins!(ax_view,new_sys,notifier = spin_data_obs)
    #plot_spin_data!(ax_view,new_sys,spin_data = spin_data_obs)
    display(new_sys)
  end
  
  f_sliders = f[2,1]

  options = [("Renormalized Dipoles",:dipole), ("SU(N) Coherent States",:SUN)]
  mode_select = Menu(loc_controls[1,1]; options)
  on(mode_select.selection;update = true) do mode
    model_mode[] = mode
  end

  options = [("S = 1/2",1/2), ("S = 1",1), ("S = 3/2",3/2), ("S = 13/2",13/2)]
  spin_select = Menu(loc_controls[2,1]; options)
  on(spin_select.selection;update = true) do spin
    model_spin[] = spin
  end
 
  latsize_sg = SliderGrid(loc_controls[1,2],
    (label = "X extent", range = 1:6, startvalue = 1, format = x -> "$x unit cells"),
    (label = "Y extent", range = 1:6, startvalue = 1, format = x -> "$x unit cells"),
    (label = "Z extent", range = 1:6, startvalue = 1, format = x -> "$x unit cells"),
    (label = "J₁", range = range(0,2,length = 1001), startvalue = 1, format = x -> "$(Sunny.number_to_simple_string(x,digits=3)) meV × S²"),
    (label = "Bᶻ", range = range(0,15,length = 1001), startvalue = 0, format = x -> "$(Sunny.number_to_simple_string(x,digits=3)) Tesla"),
   )

  on(latsize_sg.sliders[1].value) do xExt
    ls = model_latsize[]
    ls[1] == xExt && return
    model_latsize[] = (xExt,ls[2],ls[3])
  end

  on(latsize_sg.sliders[2].value) do yExt
    ls = model_latsize[]
    ls[2] == yExt && return
    model_latsize[] = (ls[1],yExt,ls[3])
  end

  on(latsize_sg.sliders[3].value) do zExt
    ls = model_latsize[]
    ls[3] == zExt && return
    model_latsize[] = (ls[1],ls[2],zExt)
  end

  on(latsize_sg.sliders[4].value) do J1
    model_J1[] == J1 && return
    set_exchange!(current_sys[], J1, Bond(1, 2, [0,0,0])) # overwrite
    model_J1[] = J1
  end

  on(latsize_sg.sliders[5].value) do B_field
    model_B_field[] == B_field && return
    set_external_field!(current_sys[],[0,0,B_field])
    model_B_field[] = B_field
  end




  on(x -> recreate_sys(),model_spin)
  on(x -> recreate_sys(),model_mode)
  on(x -> recreate_sys(),model_latsize)
  on(x -> update_energy_readout(),model_J1)
  on(x -> update_energy_readout(),model_B_field)

  recreate_sys()

  rand_button = Button(loc_buttons[1,1], label = "Randomize spins",tellwidth=false)
  on(rand_button.clicks) do event
    sys = current_sys[]
    randomize_spins!(sys)
    notify(spin_data_obs)
  end


  step_button = Button(loc_buttons[1,2], label = "Relax spins",tellwidth=false)
  on(step_button.clicks) do event
    sys = current_sys[]
    minimize_energy!(sys,maxiters = 1)
    notify(spin_data_obs)
  end

  refresh_button = Button(loc_buttons[1,3], label = "Refresh",tellwidth=false)
  on(refresh_button.clicks) do event
    notify(spin_data_obs)
  end


  display(f) # On main screen

  empty!(view_screen)
  display(view_screen,f_view)
  display(view_screen)

  current_sys
end

if !(:heat_screen ∈ names(Main))
  global heat_screen = nothing
end
function heat_capacity_widget(cur_sys)
  global heat_screen
  if isnothing(heat_screen) || heat_screen.window_open[] == false
    heat_screen = GLMakie.Screen()
  end

  f = Figure()
  f_control = f[1,1]
  f_graph = f[2,1]
  f_aux = f[3,1]
  ax_aux = Axis(f_aux)

  reading_temperatures = Observable(Float64[]) # meV
  reading_energy = Observable(Float64[])

  ax = Axis(f_graph,xlabel = "T [Kelvin]",ylabel = "Energy per Site [meV]")

  selected_temperature = Observable(0.) # meV
  vlines!(ax,map(x -> [x / meV_per_K],selected_temperature))
  scatter!(ax,map(x -> x ./ meV_per_K,reading_temperatures),reading_energy)
  Tmax = Observable(15. * meV_per_K) # meV
  on(Tmax;update = true) do tmax
    xlims!(ax,(0.,tmax / meV_per_K))
  end

  sg = SliderGrid(f_control[2,1],
    (label = "T max", range = 10 .^ range(-2,3,length = 1001), startvalue = 15, format = x -> "$(Sunny.number_to_simple_string(x,digits=3)) Kelvin"),
   )

  on(async_latest(sg.sliders[1].value,1)) do tmax
    tmax * meV_per_K == Tmax[] && return
    Tmax[] = tmax * meV_per_K
  end

  function make_reading(sys0,kT0)
    langevin = Langevin(0.05;λ = 0.1, kT = kT0)
    e_val = 0.

    # Burn in
    for i = 1:3000
      step!(sys0,langevin)
    end

    # Read energy
    for i = 1:30000
      step!(sys0,langevin)
      e_val = e_val + (energy_per_site(sys0) - e_val) / i
    end
    #display("made reading: $kT0, $e_val")

    push!(reading_temperatures[],kT0)
    push!(reading_energy[],e_val)
    ylims!(ax,extrema(reading_energy[],init = (-1e-8,1e-8)))
    notify(reading_temperatures)
  end

  reading_button = Button(f_control[1,1], label = "Take Reading",tellwidth=false)
  on(async_latest(reading_button.clicks, 8)) do event
    sys0 = Sunny.clone_system(cur_sys[])
    kT0 = selected_temperature[]
    make_reading(sys0,kT0)
  end

  reset_data_button = Button(f_control[3,1], label = "Reset Data",tellwidth=false)
  on(reset_data_button.clicks) do event
    empty!(reading_temperatures[])
    empty!(reading_energy[])
    notify(reading_temperatures)
    nothing
  end

  reset_data_button = Button(f_control[3,2], label = "Heat Capacity",tellwidth=false)
  on(reset_data_button.clicks) do event
    Ts = reading_temperatures[]
    Es = reading_energy[]
    ix = sortperm(Ts)

    dT = diff(Ts[ix])
    dE = diff(Es[ix])
    empty!(ax_aux)
    plot!(ax_aux,(Ts[ix][1:end-1] .+ Ts[ix][2:end]) ./ 2 ./ meV_per_K,dE ./ dT)
    xlims!(ax_aux,(0.,Tmax[] / meV_per_K))
    nothing
  end

  lck = ReentrantLock()
  sweep_level = 1
  sweep_button = Button(f_control[1,2], label = Observable("Make Sweep"),tellwidth=false)
  on(sweep_button.clicks) do event
    lock(lck)
    try
      hi_T = Tmax[]
      lo_T = 0
      kts = lo_T .+ (hi_T - lo_T) .* range(1,step=2,length=2 ^ (sweep_level-1)) ./ (2 ^ sweep_level)
      sys0 = Sunny.clone_system(cur_sys[])
      for i = eachindex(kts)
        make_reading(sys0,kts[i])
        sleep(1/30)
      end
      sweep_level = sweep_level + 1
      sweep_button.label[] = "Make Sweep ($(2^(sweep_level - 1)))"
      notify(sweep_button.label)
    finally
      unlock(lck)
    end
  end

  reset_sweep_button = Button(f_control[2,2], label = "Reset Sweep Level",tellwidth=false)
  on(reset_sweep_button.clicks) do event
    sweep_level = 1
    sweep_button.label[] = "Make Sweep ($(2^(sweep_level - 1)))"
    notify(sweep_button.label)
  end


  mouse_hook = hook_mouse_marker(f_graph,ax) do T, energy_click
    selected_temperature[] = T * meV_per_K
  end

  empty!(heat_screen)
  display(heat_screen,f)
  display(heat_screen)
  nothing
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
      return true
    end
    return false
  end

  # Update on left click
  on(events(fig).mousebutton, priority = 2; update = true) do event
    if event.button == Mouse.left
      if event.action == Mouse.press
        return Consume(do_update())
      end
    end
  end

  # Update on left click-drag
  on(async_latest(events(fig).mouseposition), priority = 2) do mp
    if events(fig).mousebutton[].button == Mouse.left
      if events(fig).mousebutton[].action == Mouse.press
        return Consume(do_update(;mp))
      end
    end
  end
  mouse_hook
end


