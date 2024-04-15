
using Sunny, LinearAlgebra
using ProgressMeter
using GLMakie
using Observables, Statistics, FFTW
using DataStructures
using Random

function spin_in_field_system(;ls = (1,1,1))
  cryst = Crystal(I(3),[[0.,0,0]],1)
  sys = System(cryst, ls, [SpinInfo(1;S=1,g=1)], :dipole)
  set_external_field!(sys,[0,0,-1])
  randomize_spins!(sys)

  sys
end

function copper_sulfate(;inter_chain_coupling = false,chain_length = 4, n_chains = (4,4))
  crystal= Crystal("CuSO4_main.cif", symprec=0.001)
  subc= subcrystal(crystal, "Cu1","Cu2")

  sys = System(subc,(chain_length,n_chains...), [SpinInfo(1;S=1/2, g=2.51),SpinInfo(2;S=1/2, g=2.35)], :dipole)

  # row (g) of Table 4.1 Mourigal Thesis
  # N.B.: Interactions will be renormalized!! (due to :dipole mode Sunny)
  J = 0.252 # Intrachain coupling (a axis)
  J12 = -0.025 # Intracell coupling (Cu1-Cu2)
  Jinter = 0.005 # Interchain (c axis); no interchain (b axis) because that bond is very long

  set_exchange!(sys,J,Bond(1, 1, [1, 0, 0]))
  set_exchange!(sys,J12,Bond(1, 2, [0, 0, 0]))
  if inter_chain_coupling
    set_exchange!(sys,Jinter,Bond(1, 1, [0, 0, 1]))
  end
  set_external_field!(sys,[0,0.2,0])

  # ESR units
  meV_per_GHz = 1/241.8
  meV_per_kHz = 1e-6 * meV_per_GHz

  # ESR parameters
  ωc = 3.5 * meV_per_GHz
  kappa_e = 556 * meV_per_kHz
  kappa_i = 381 * meV_per_kHz
  gamma = 0.001 # This is the line-broadening for the calculation of χ

  # Calculating |g(ωc)|²:
  # should be roughly 88 MHz
  coupling_G_sq = 88000 * meV_per_kHz
  # formula might be: abs2(im * filling_factor * sqrt(sys.units.μ0))
  #   (with filling_factor = 0.4)
  # This works for plotting purposes:
  #coupling_G_sq = 0.00005 * 20

  # The DC field applied along y axis
  #nB = 375 # Resolution in DC Field
  #B_min = 0 # Oe
  #B_max = 3000 # Oe
  #tesla_per_Oe = 1e-4
  #B_fields = tesla_per_Oe * range(B_min,B_max,length = nB) # Tesla
  #println("ΔB = $(Sunny.number_to_simple_string((B_max - B_min)/nB,digits = 8)) Oe")
  sys
end

function bake_cuso4(;kwargs...)
  sys = copper_sulfate(;n_chains = (1,1))
  meV_per_GHz = 1/241.8
  bake_chi(sys;kT = 1e-3,dt = 0.5,n_energies = 250, max_energy = meV_per_GHz * 60,kwargs...)
end

#struct ChiModel
#end

function bake_chi(sys;kT = 0.01,damping = 0.1,dt = 0.05,n_energies = 600, max_energy = 1.0,n_sample = 200)
  s_dot_obs(i) = Sunny.NonLocalObservableOperator(sys -> map(x -> x[i],sys.dipoles .× Sunny.energy_grad_dipoles(sys)))

  dsc = dynamical_correlations(sys;Δt = dt, nω = n_energies, ωmax = max_energy, observables = [:Sx => [1. 0 0], :Sy => [0. 1 0], :Sz => [0. 0 1], :Sxdot => s_dot_obs(1), :Sydot => s_dot_obs(2), :Szdot => s_dot_obs(3)])
  display(dsc)

  langevin = Langevin(dt;λ = damping, kT)
  #println("x = βω₀ = $(beta * omega0)")

  viewed_thetas = Float64[]

  prog = Progress(n_sample,"Baking")
  for k = 1:n_sample
    for j = 1:1000
      step!(sys,langevin)
    end
    theta = acos(sys.dipoles[1][3])
    #push!(viewed_thetas,theta)

    Sunny.new_sample!(dsc,sys,() -> nothing,ImplicitMidpoint(dt;λ = damping,kT))
    Sunny.accum_sample!(dsc)
    next!(prog)
  end
  finish!(prog)

  N = size(dsc.data,7)
  params = unit_resolution_binning_parameters(dsc;negative_energies = true)
  is_full = intensities_binned(dsc,params,intensity_formula(dsc,:full))[1][1,1,1,:]

  # Normalize to unitary FFT
  is_full *= sqrt(N)

  chi = (1/kT) * sys.units.μB * sys.gs[1][1] * map(x -> x[1:3,4:6],is_full)

  chi_model = Array{Vector{ComplexF64}}(undef,3,3)
  for i = 1:3, j = 1:3
    ff = ifftshift(map(x -> x[i,j],chi))
    ifft!(ff)
    ff[fftfreq(length(ff)) .< 0] .= 0
    fft!(ff)
    chi_model[i,j] = ff
  end

  chi_model, dsc.Δt, dsc.measperiod
end

if !(:view_screen ∈ names(Main))
  global view_screen = nothing
end
function sim_AC_applied(sys,chi_model,dt_sim,measperiod;static_field = [0,0,-1])
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
  plot_spin_data!(ax_sys,sys;color = :orange,spin_data = current_dipoles)

  # Input graph
  ax_input = Axis(loc_graphs[1,1],xlabel = "Time", ylabel = "λ")
  n_memory = (length(chi_model[1])÷2) + 1
  zbuf = zeros(ComplexF64,length(chi_model[1]))
  time_axis = (dt_sim * measperiod) * (1:n_memory) .- ((dt_sim * measperiod) * n_memory)
  λs = Observable(CircularBuffer{Float64}(n_memory))
  fill!(λs[],0)
  lines!(ax_input,time_axis,λs; color = :lightblue)

  output_chi = Observable(CircularBuffer{Float64}(n_memory))
  fill!(output_chi[],NaN)
  lines!(ax_input,time_axis,output_chi,color = :red)


  # Output graph
  ax_output = Axis(loc_graphs[2,1],xlabel = "Time", ylabel = "Observable")

  output_reference = Observable(CircularBuffer{Float64}(n_memory))
  fill!(output_reference[],0)
  lines!(ax_output,time_axis,output_reference,color = :orange)

  output_response = Observable(CircularBuffer{Float64}(n_memory))
  fill!(output_response[],0)
  lines!(ax_output,time_axis,output_response,color = :blue)

  output_predicted = Observable(CircularBuffer{Float64}(n_memory))
  fill!(output_predicted[],NaN)
  lines!(ax_output,time_axis,output_predicted,color = :black, linestyle = :dash)

  ax_response = Axis(loc_graphs[3,1],xlabel = "Time", ylabel = "δB")


  fluct_actual = Observable(CircularBuffer{Float64}(n_memory))
  fill!(fluct_actual[],0)
  lines!(ax_response,time_axis,fluct_actual,color = :blue)

  fluct_predict = Observable(CircularBuffer{Float64}(n_memory))
  fill!(fluct_predict[],0)
  lines!(ax_response,time_axis,fluct_predict,color = :black, linestyle = :dash)

  lines!(ax_response,time_axis,0 .* fluct_actual[],color = :orange)

  sg = SliderGrid(f_sliders[1,1],
    (label = "F", range = 0:0.01:10, startvalue = 3, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 4))"),
    (label = "kT", range = -4:0.01:2, startvalue = -2, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))"),
    (label = "damping", range = -4:0.01:2, startvalue = -1, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))")
   )

  control_fudge = Observable(0.)
  on(sg.sliders[1].value;update = true) do w
    control_fudge[] = 10^w
  end

  kT = Observable(0.)
  on(sg.sliders[2].value;update = true) do logkT
    kT[] = 10^logkT
  end

  damping = Observable(0.1)
  on(sg.sliders[3].value;update = true) do logλ
    damping[] = 10^logλ
  end

  display(f)
  empty!(view_screen)
  display(view_screen,f)
  display(view_screen)

  onames = [:x,:y,:z]
  options = [("(Spin $(string(onames[k[1]])), Field $(string(onames[k[2]])))",k) for k in [(1,1),(2,2),(3,3),(1,2),(2,3),(3,1),(2,1),(3,2),(1,3)]]
  corr_selector = Menu(loc_controls[1,1]; options)
  selected_correlation = Observable(CartesianIndex(1,1))

  function name_to_unit_vector(s::Symbol)
    if s == :x
      [1,0,0]
    elseif s == :y
      [0,1,0]
    elseif s == :z
      [0,0,1]
    else
      println("Invalid observable $s")
      [0,0,0]
    end
  end

  # TODO: check convention
  perturbation_field_vector = map(ci -> name_to_unit_vector(onames[ci[2]]),corr_selector.selection)
  mag_unit_vector = map(ci -> name_to_unit_vector(onames[ci[1]]),corr_selector.selection)

  # Extra window:
  #
  # Joystick to apply perturbation
  # 
  fig = Figure(); ax_click = Axis(fig[1,1])
  xlims!(ax_click,-3,3)
  ylims!(ax_click,-3,3)
  δ = 1.5

  sgj = SliderGrid(fig[2,1],
    (label = "ω", range = 0:1e-4:0.1, startvalue = 0.1, format = x -> "$(Sunny.number_to_simple_string(x; digits = 4))"),
    (label = "s", range = 0:1e-4:4, startvalue = 1.0, format = x -> "$(Sunny.number_to_simple_string(1 - 10^(-x); digits = 4))")
   )

  control_frequency = Observable(0.1)
  on(sgj.sliders[1].value;update = true) do w
    control_frequency[] = w
  end

  control_falloff = Observable(0.9)
  on(sgj.sliders[2].value;update = true) do s
    control_falloff[] = 1 - 10^(-s)
  end

  ax_spectrum = Axis(fig[3,1])
  spectrum_axis = 2π * fftfreq(n_memory, 1 / (dt_sim * measperiod))
  spectrum_ix = sortperm(spectrum_axis)

  vlines!(map(x -> -x,control_frequency),color = :lightblue)
  vlines!(control_frequency,color = :lightblue)

  chi_spectrum_re = Observable(Vector{Float64}(undef,n_memory))
  chi_spectrum_re[] .= 0
  chi_spectrum_im = Observable(Vector{Float64}(undef,n_memory))
  chi_spectrum_im[] .= 0
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],chi_spectrum_re,color = :red)
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],chi_spectrum_im,color = :red,linestyle = :dash)

  lambda_spectrum_re = Observable(Vector{Float64}(undef,n_memory))
  lambda_spectrum_re[] .= 0
  lambda_spectrum_im = Observable(Vector{Float64}(undef,n_memory))
  lambda_spectrum_im[] .= 0
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],lambda_spectrum_re,color = :lightblue)
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],lambda_spectrum_im,color = :lightblue,linestyle = :dash)

  response_spectrum_re = Observable(Vector{Float64}(undef,n_memory))
  response_spectrum_re[] .= 0
  response_spectrum_im = Observable(Vector{Float64}(undef,n_memory))
  response_spectrum_im[] .= 0
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],response_spectrum_re,color = :blue)
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],response_spectrum_im,color = :blue,linestyle = :dash)

  reference_spectrum_re = Observable(Vector{Float64}(undef,n_memory))
  reference_spectrum_re[] .= 0
  reference_spectrum_im = Observable(Vector{Float64}(undef,n_memory))
  reference_spectrum_im[] .= 0
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],reference_spectrum_re,color = :orange)
  lines!(ax_spectrum,spectrum_axis[spectrum_ix],reference_spectrum_im,color = :orange,linestyle = :dash)

  on(corr_selector.selection; update = true, priority = -1) do ci
    # Update labels on joystick
  end

  rand_button = Button(loc_controls[2,1], label = "Randomize spins")
  on(rand_button.clicks) do event
    randomize_spins!(sys)
  end

  
  int = ImplicitMidpoint(dt_sim,λ = damping[],kT = kT[])

  control_abs = Observable(1.0)
  control_phase = Observable(0.0)
  tracer_points = Observable(Point2f[Point2f(NaN,NaN)])
  scatter!(ax_click,tracer_points)

  on(control_abs,update = true) do b0
    l,h = extrema(λs[])
    amp = max(1e-8,abs(b0))
    ylims!(ax_input,min(l,-amp),max(h,amp))

    ctrl = exp(im * control_phase[]) * b0
    tracer_points[][1] = Point2f(real(ctrl),imag(ctrl))
    notify(tracer_points)
  end

  # Marker interaction
  dragging = Observable(true)
  mouse_hook = hook_mouse_marker(fig,ax_click,marker = false) do xclick, yclick

    control_phase[] = angle(xclick + im * yclick)
    control_abs[] = abs(xclick + im * yclick)
    dragging[] = true

    #ax_click.title[] = "B = $(Sunny.number_to_simple_string(B,digits=3)), ω = $(Sunny.number_to_simple_string(ωclick,digits=3)), |χ| = $(Sunny.number_to_simple_string(abs(chi_current[]),digits=3)), arg χ = $(Sunny.number_to_simple_string(angle(chi_current[]),digits=3))"
  end


  display(fig)


  sys_response = Sunny.clone_system(sys)
  Random.seed!(sys_response.rng,15)

  current_dipoles_responding = Observable(zeros(Vec3f,size(sys_response.dipoles)))
  plot_spin_data!(ax_sys,sys_response;color = :blue,spin_data = current_dipoles_responding)

  sync_button = Button(loc_controls[3,1], label = "Synchronize")
  on(sync_button.clicks) do event
    sys_response.dipoles .= sys.dipoles
    current_dipoles_responding[] .= sys_response.dipoles
    notify(current_dipoles_responding)
    empty!(λs[])
    fill!(λs[],0)
    notify(λs)
    control_abs[] = 0.0 * control_abs[]
    notify(control_abs)
  end


  # Simulation loop
  i = 0
  #@async begin
  begin
    while true
      i = i + 1
      t = i * dt_sim

      if length(f.scene.current_screens) < 1
        break
      end

      # Apply field
      λ = control_abs[] * cos(control_phase[] + control_frequency[] * t)
      total_field = static_field .+ λ * perturbation_field_vector[]

      set_external_field!(sys,static_field) # Update system
      set_external_field!(sys_response,total_field)

      # Grab latest integrator parameters
      int.λ = damping[]
      int.kT = kT[]

      # Integrate forward
      step!(sys,int)
      step!(sys_response,int)

      if mod(i,measperiod) == 0
        push!(λs[],λ) # Update input graph

        lambda_spectrum = fft(λs[])
        lambda_spectrum_re[] .= real(lambda_spectrum[spectrum_ix])
        lambda_spectrum_im[] .= imag(lambda_spectrum[spectrum_ix])

        # Actual magnetization (per site)
        mag_hat = mag_unit_vector[]
        out_ref = magnetization_along_axis(mag_hat,sys.dipoles)
        out_resp = magnetization_along_axis(mag_hat,sys_response.dipoles)
        push!(output_reference[],out_ref)
        push!(output_response[],out_resp)

        reference_spectrum = fft(output_reference[])
        reference_spectrum_re[] .= real(reference_spectrum[spectrum_ix])
        reference_spectrum_im[] .= imag(reference_spectrum[spectrum_ix])
        
        response_spectrum = fft(output_response[])
        response_spectrum_re[] .= real(response_spectrum[spectrum_ix])
        response_spectrum_im[] .= imag(response_spectrum[spectrum_ix])

        # Linear Response prediction of magnetization (per site)
        zbuf .= 0
        zbuf[1:n_memory] .= λs[]
        fft!(zbuf)
        # chi_model has 1/sqrt(N) unitary normalization (from bake_chi + intensities_binned)
        # which is correct here
        zbuf .*= control_fudge[] * chi_model[corr_selector.selection[]...]
        ifft!(zbuf)
        output_chi[] .= control_fudge[] * real(ifft(chi_model[corr_selector.selection[]...])[1:n_memory])

        chi_spectrum = fft(output_chi[])
        chi_spectrum_re[] .= real(chi_spectrum[spectrum_ix])
        chi_spectrum_im[] .= imag(chi_spectrum[spectrum_ix])

        predicted = real(zbuf[1:n_memory])
        output_predicted[] .= predicted .+ output_reference[]
        #push!(output_predicted[],real(zbuf[1:n_memory]))
        #push!(output_predicted[],real(zbuf[1:n_memory]))
        
        push!(fluct_actual[],out_resp .- out_ref)
        fluct_predict[] .= predicted

        is_dragging = dragging[]
        if is_dragging
          control_abs[] = control_falloff[] * control_abs[]
          notify(control_abs)
        else
          dragging[] = false
        end
        notify(output_reference)
        notify(output_response)
        notify(output_predicted)
        notify(output_chi)

        notify(chi_spectrum_re)
        notify(chi_spectrum_im)
        notify(lambda_spectrum_re)
        notify(lambda_spectrum_im)
        notify(response_spectrum_re)
        notify(response_spectrum_im)
        notify(reference_spectrum_re)
        notify(reference_spectrum_im)

        current_dipoles[] .= sys.dipoles
        current_dipoles_responding[] .= sys_response.dipoles
        notify(current_dipoles)
        notify(current_dipoles_responding)
        notify(λs)
        notify(fluct_predict)
        notify(fluct_actual)
        sleep(1/70)
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

function magnetization_along_axis(mag_hat,dipoles)
  sum(map(x -> x ⋅ mag_hat,dipoles)) / length(dipoles)
end

include("online_correlations.jl")
include("../eigenmodes/support.jl")
using Printf
include("../susceptibility/support.jl")
