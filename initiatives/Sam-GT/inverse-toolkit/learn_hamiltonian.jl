using MKL
using Sunny, LinearAlgebra, GLMakie, Optim, Random, LineSearches, Observables

# This is a worked example of how to define a loss function and fit to it using the other
# example code in this directory. You probably can't run this unless you have my copy of
# the .nxs referenced below, sorry :(. But you can base your own fit code on this anyway!

function load_data()
  # Load experiment data (so we have histogram_parameters)
  data_folder = "C:\\Users\\Sam\\Dropbox (GaTech)\\Sam-Research\\Projects\\LaSrCrO4\\Data\\sliced"
  fn = joinpath(data_folder,"normData_LaSrCrO4_120meV_5K_no_symmetrize_skew.nxs")
  #fn = joinpath(data_folder,"normData_LaSrCrO4_120meV_5K_symmetrize_skew.nxs")
  histogram_parameters, data = load_nxs(fn)
end

function chunk_params(params_orig,data_orig;chunking = (1,1,1,1), nan = true)
  params = copy(params_orig)
  data = zeros(Float64,size(data_orig).÷chunking)
  data_nonan = if nan
    data_orig
  else
    data_nonan = copy(data_orig)
    data_nonan[isnan.(data_orig)] .= 0.
    data_nonan
  end
  for i = 1:4
    params.binwidth[i] *= chunking[i]
    for ci = CartesianIndices(size(data))
      i1 = (1:chunking[1]) .+ (ci[1] - 1) * chunking[1]
      i2 = (1:chunking[2]) .+ (ci[2] - 1) * chunking[2]
      i3 = (1:chunking[3]) .+ (ci[3] - 1) * chunking[3]
      i4 = (1:chunking[4]) .+ (ci[4] - 1) * chunking[4]
      data[ci] = sum(data_nonan[i1,i2,i3,i4])
    end
  end
  params, data
end

function approximate_bintegrate(ax,desire_start,desire_end,params,data;restrict = false,nan=true)
  x0 = params.binstart[ax]
  dx = params.binwidth[ax]
  x1 = (desire_start - x0)/dx
  x2 = (desire_end - x0)/dx

  x1 = max(0,x1)
  x2 = min(params.numbins[ax],x2)

  # Edge case
  x2 = max(1,x2)
  x1 = min(params.numbins[ax]-1,x1)

  x1 = round(Int64,x1)
  x2 = round(Int64,x2)
  bes = Sunny.axes_binedges(params)[ax]
  ix = (x1+1):x2
  data_nonan = if nan
    data
  else
    data_nonan = copy(data)
    data_nonan[isnan.(data)] .= 0.
    data_nonan
  end
  new_data = if restrict
    if ax == 1
      data_nonan[ix,:,:,:]
    elseif ax == 2
      data_nonan[:,ix,:,:]
    elseif ax == 3
      data_nonan[:,:,ix,:]
    elseif ax == 4
      data_nonan[:,:,:,ix]
    end
  else
    if ax == 1
      sum(data_nonan[ix,:,:,:],dims=1)
    elseif ax == 2
      sum(data_nonan[:,ix,:,:],dims=2)
    elseif ax == 3
      sum(data_nonan[:,:,ix,:],dims=3)
    elseif ax == 4
      sum(data_nonan[:,:,:,ix],dims=4)
    end
  end
  new_params = copy(params)
  new_params.binstart[ax] = bes[x1+1]
  new_params.binend[ax] = bes[x2+1]
  new_params.binwidth[ax] = (restrict ? 1 : (0.1 + x2 - x1)) * params.binwidth[ax]
  new_params, new_data
end

figure_chunking = (1,5,1,142)

function show_figure2b(params,data;elastic = false, chunking = (1,1,1,1))
  params, data = approximate_bintegrate(2,-0.1,0.12,params,data)
  params, data = approximate_bintegrate(4,elastic ? -Inf : 15,Inf,params,data,nan=false)
  params, data = chunk_params(params,data;chunking)
  bcs = axes_bincenters(params)
  #display(data[:,1,:,1])
  #display(params)
  f = Figure(); ax = Axis(f[1,1],xlabel = "[H,H,0]",ylabel="[0,0,L]")
  hm = heatmap!(ax,bcs[1],bcs[3],log10.(abs.(data[:,1,:,1])),colormap = :jet1)
  Colorbar(f[1,2],hm)
  #heatmap(bcs[1],bcs[3],(data[:,1,:,1]),colormap = :jet1)
  f
end

function show_afm_curve(params,data;elastic = false)
  params, data = approximate_bintegrate(2,-0.1,0.12,params,data)
  params, data = approximate_bintegrate(4,elastic ? -Inf : 8,Inf,params,data,restrict = true)
  params, data = approximate_bintegrate(3,-8,8,params,data)
  #params, data = chunk_params(params,data;chunking=(2,1,1,2))
  display(params)
  bcs = axes_bincenters(params)
  f = Figure(); ax = Axis(f[1,1],xlabel = "[H,H,0]",ylabel="meV")
  #hm = heatmap!(ax,bcs[1],bcs[4],log10.(abs.(data[:,1,1,:])),colormap = :jet1)
  hm = heatmap!(ax,bcs[1],bcs[4],data[:,1,1,:],colormap = :jet1)
  Colorbar(f[1,2],hm)
  f
end

function record_L_int_range_sweep(params,data)
  function for_range(dL)
    params1, data1 = approximate_bintegrate(2,-0.1,0.12,params,data)
    elastic = false
    params1, data1 = approximate_bintegrate(4,elastic ? -Inf : 8,Inf,params1,data1,restrict = true)
    #params1, data1 = approximate_bintegrate(3,-dL,dL,params1,data1)
    params1, data1 = approximate_bintegrate(3,dL,dL+0.4,params1,data1)
    params1, data1
  end
  p, d = for_range(8)
  display(p)
  bcs = axes_bincenters(p)
  dat = Observable(d[:,1,1,:])
  f = Figure()
  ax = Axis(f[1,1],xlabel = "[H,H,0]",ylabel = "meV")
  dat_nonan = copy(dat[])
  dat_nonan[isnan.(dat_nonan)] .= 0
  hm = heatmap!(ax,bcs[1],bcs[4],map(x -> log10.(abs.(x)),dat),colormap = :jet1,colorrange = (-3,-2))
  #hm = heatmap!(ax,bcs[1],bcs[4],dat,colormap = :jet1,colorrange = extrema(dat_nonan[:]))
  Colorbar(f[1,2],hm)
  #record(f,"test.mp4") do io
  display(f)
    @async begin
      for dL = range(0.01,8,step = 0.1)
      p, d = for_range(dL)
      dat[] .= d[:,1,1,:]
      notify(dat)
      #recordframe!(io)
      sleep(0.1)
    end
  end
  #end
end

function show_line!(ax,params,data)
  params, data = approximate_bintegrate(2,-0.1,0.12,params,data)
  elastic = false
  params, data = approximate_bintegrate(4,elastic ? -Inf : 15,Inf,params,data,nan=false)
  params, data = approximate_bintegrate(3,1.0,1.4,params,data)
  bcs = axes_bincenters(params)
  lines!(ax,bcs[1],data[:,1,1,1])
  nothing
end

function show_lineshape!(ax,params,data)
  #params, data = approximate_bintegrate(1,1.2,1.6,params,data)
  params, data = approximate_bintegrate(1,0.7,1.1,params,data)
  params, data = approximate_bintegrate(2,-0.1,0.12,params,data)
  #params, data = approximate_bintegrate(3,3.9,4.0,params,data)
  params, data = approximate_bintegrate(3,0.9,1.0,params,data)
  elastic = false
  params, data = approximate_bintegrate(4,elastic ? -Inf : 15,Inf,params,data,restrict = true)
  display(params)
  bcs = axes_bincenters(params)
  lines!(ax,bcs[4],data[1,1,1,:])
  nothing
end

function show_disp(params,data)
  params, data = approximate_bintegrate(1,0.49,0.51,params,data)
  #params, data = approximate_bintegrate(1,0.7,1.3,params,data)
  params, data = approximate_bintegrate(2,-0.1,0.12,params,data)
  elastic = false
  params, data = approximate_bintegrate(4,elastic ? -Inf : 15,Inf,params,data,restrict = true)
  bcs = axes_bincenters(params)
  f = Figure(); ax = Axis(f[1,1],xlabel = "[0,0,L]",ylabel="meV")
  #hm = heatmap!(ax,bcs[3],bcs[4],log10.(abs.(data[1,1,:,:])),colormap = :jet1)
  hm = heatmap!(ax,bcs[3],bcs[4],data[1,1,:,:],colormap = :jet1)
  Colorbar(f[1,2],hm)
  f
end


function reduce_hist(params,data)
  params, data = approximate_bintegrate(1,0,1.0,params,data;restrict=true)
  params, data = approximate_bintegrate(2,-0.1,0.1,params,data)
  params, data = approximate_bintegrate(3,0,5.,params,data;restrict=true)
  params, data = chunk_params(params,data;chunking = (2,1,1,1))
  params, data
end

function setup_sys()
  # Crystallography & Chemistry
  cryst = Crystal("example_cif.cif"; symprec=1e-4)
  sys_chemical = System(subcrystal(cryst,"Cr"), (1,1,1), [SpinInfo(1,S=3/2,g=2)], :SUN)
  sys = reshape_supercell(sys_chemical, [1 1 0; 1 -1 0; 0 0 1]) # Neel state

  sys
end


function get_Z(is)
  #1#sum(is[7:14,7:14,:,4]) # Qx, Qy, and E range of magnetic bragg peak
  rparamsint, is = approximate_bintegrate(1,0.45,0.55,rparams,is)
  rparamsint, is = approximate_bintegrate(2,-0.05,0.05,rparamsint,is)
  rparamsint, is = approximate_bintegrate(3,-0.1,0.1,rparamsint,is)
  rparamsint, is = approximate_bintegrate(4,2,16,rparamsint,is)
  sum(is)
end

# Multi-sampling magic numbers
msaa1 = [[0.5, 0.5, 0.5]]
msaa4 = [[0.625, 0.625, 0.125]
        ,[0.875, 0.125, 0.375]
        ,[0.375, 0.375, 0.875]
        ,[0.125, 0.875, 0.625]]
energy_multisample = [(n + 0.5)/5 for n = 1:5]


function forward_problem(histogram_parameters;J1 = 10.6,J2 = 0.16,A = 0.08,D = 0.01,K = 0.1, kwargs...)
  forward_problem(histogram_parameters,J1,J2,A,D,K; kwargs...)
end

function forward_problem(histogram_parameters,J1,J2,A,D,K)#; sys = setup_sys())
  cryst = Crystal("example_cif.cif"; symprec=1e-4)
  sys = System(subcrystal(cryst,"Cr"), (1,1,1), [SpinInfo(1,S=3/2,g=2)], :dipole)
  # J1
  set_exchange!(sys,J1,Bond(1,1,[1,0,0]))

  # J2
  set_exchange!(sys,J2,Bond(1,1,[1,1,0]))
  
  # A
  Sz = spin_matrices(spin_label(sys,1))[3]
  set_onsite_coupling!(sys,A*Sz^2,1)

  # D and K
  #cryst = sys.origin.crystal
  nHat = normalize(cryst.latvecs * (cryst.positions[2] .- cryst.positions[1]))
  exchange_matrix = I(3) - 3(nHat * nHat')
  set_pair_coupling!(sys,(Si,Sj) -> Si'*(D .* exchange_matrix)*Sj + (Si'*K*Sj)^2,Bond(1,2,[0,0,0]))

  sys = reshape_supercell(sys, [1 1 0; 1 -1 0; 0 0 1]) # Neel state

  # Standard calculation:
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters=1000,subiters=80)
  if minimize_energy!(sys;maxiters=5) < 0
    display(sys.dipoles)
    return NaN .* rdata
  end

  # Modify to:
  # :perp, but with out-of-plane allowed
  # x occ. and (1-x) occ. of a,b ordered-by-disorder
  # (discrete group average)

  swt = SpinWaveTheory(sys)
  formula = intensity_formula(swt,:perp;
    # TODO: instrument-adapted broadening
    kernel = lorentzian(2.0)
    ,formfactors = [FormFactor("Cr3")]
    )

  #=
  swt = SpinWaveTheory(sys;correlations = [(:Sx,:Sx),(:Sy,:Sy),(:Sz,:Sz)])
  formula = intensity_formula(swt,[1,2,3];
    # TODO: instrument-adapted broadening
    kernel = lorentzian(2.0)
    ,mode_fast = true
    ,formfactors = [FormFactor("Cr3")]
    ) do k,ω,corr
    #(Sxx + Syy)/2 + Szz
    (corr[1] + corr[2])/2 + corr[3]
  end
  =#

  #=
  swt = SpinWaveTheory(sys)
  formula = intensity_formula(swt,:perp;
    # TODO: instrument-adapted broadening
    kernel = lorentzian(2.)
    ,mode_fast = true
    ,formfactors = [FormFactor("Cr3")]
    )
  =#
  
  intensity, counts = try
    Sunny.intensities_bin_multisample(swt
                                                       ,histogram_parameters
                                                       ,msaa4
                                                       ,energy_multisample
                                                       ,formula)
  catch e
    println("LSWT error!")
    println(e)
    return NaN .* rdata
  end
  # Sunny LSWT is computing a density; even if the bin size goes to zero the value
  # returned from LSWT is still finite. We need to multiply by the bin size!
  intensity .*= prod(histogram_parameters.binwidth)

  return intensity ./ counts
end

function squared_errors(experiment_data,simulation_data)
  Z_experiment = 1#get_Z(experiment_data)
  normalized_exp_data = experiment_data ./ Z_experiment

  Z_sunny = 1#get_Z(simulation_data)
  normalized_sim_data = simulation_data ./ Z_sunny

  weights = 1.

  # Compute squared error over every histogram bin
  squared_errors = (normalized_exp_data .- normalized_sim_data).^2
  squared_errors[isnan.(experiment_data)] .= 0 # Filter out missing experiment data
  squared_errors[:,:,:,1:17] .= 0 # Filter out elastic line
  squared_errors .* weights
end

function loss_function(experiment_data,simulation_data)
  sqr = squared_errors(experiment_data,simulation_data)
  sqrt(sum(sqr))
end

function get_loss_scaled(parameters)
  println("Trying ", parameters)
  J1,A,λ = parameters
  simulation_data = forward_problem(rparams;J1,A)
  return loss_function(rdata, λ * simulation_data)
end

function thermal_loss()
  cent, cov = thermal_basin(get_loss_scaled,[10.6,0.098,10.0],0.002;j_max = 50,noise_scale = [1,0.01,1.0])
  F = eigen(cov;sortby = λ -> -λ)
  n(x) = Sunny.number_to_simple_string(x,digits = 8)
  for i = 1:3
    println("x[$i] = $(n(cent[i])) ± $(n(sqrt(cov[i,i])))")
  end
  println()
  println("Loosest mode (σ = $(sqrt(F.values[1]))):")
  println(F.vectors[:,1])
  println("Strictest mode (σ = $(sqrt(F.values[3]))):")
  println(F.vectors[:,3])
  cent, cov
end

function get_loss(parameters)
  println("Trying ", parameters)
  J1,A = parameters
  #=
  if J1 > 11 || J1 < 9
    return Inf
  end
  if A > 1 || A < 0.1
    return Inf
  end
  =#
  simulation_data = forward_problem(rparams;J1,A)
  return loss_function(rdata, simulation_data)
end

# Parameter sweep to generate loss landscape

if !(:main_screen ∈ names(Main))
  global main_screen = nothing
  global side_screen = nothing
end
function doSweep(scale;J0 = 10.11, A0 = 0.069)
  global main_screen
  global side_screen
  if isnothing(main_screen) || main_screen.window_open[] == false
    main_screen = GLMakie.Screen()
  end
  if isnothing(side_screen) || side_screen.window_open[] == false
    side_screen = GLMakie.Screen()
  end

  fig = Figure()
  ax = Axis(fig[1,1])

  ax.xzoomlock[] = true
  ax.yzoomlock[] = true
  ax.xpanlock[] = true
  ax.ypanlock[] = true
  ax.xrectzoom[] = false
  ax.yrectzoom[] = false
  
  empty!(main_screen)
  display(main_screen,fig)

  nJ = 20
  nA = 20
  loss_landscape = Observable(NaN .* zeros(Float64,nJ,nA))
  dJ = 1
  dA = 0.03
  Js = range(J0 - dJ * scale/2, J0 + dJ * scale/2,length=nJ)
  As = range(A0 - dA * scale/2, A0 + dA * scale/2,length=nA)

  points = Observable(Point2f[])
  point_colors = Observable(Float64[])

  hm = heatmap!(ax,Js,As,loss_landscape)
  sc = scatter!(ax, points, color = point_colors, strokewidth = 1.)

  main_screen_render_lock = ReentrantLock()
  side_screen_render_lock = ReentrantLock()

  on(events(ax).mousebutton) do event
      println("CLICK")
      lock(main_screen_render_lock)
      notify(point_colors)
      notify(points)
      notify(sc.colorrange)
      notify(hm.colorrange)
      unlock(main_screen_render_lock)
      if event.button == Mouse.left
          if event.action == Mouse.press
              mp = events(ax).mouseposition[]
              bbox = ax.layoutobservables.computedbbox[]
              c = (mp .- bbox.origin) ./ bbox.widths
              if 0 < c[1] < 1 && 0 < c[2] < 1
                x_lim = Sunny.axes_binedges(unit_resolution_binning_parameters(Js)...)[1]
                y_lim = Sunny.axes_binedges(unit_resolution_binning_parameters(As)...)[1]
                J = x_lim[1] + c[1] * (x_lim[end] - x_lim[1])
                A = y_lim[1] + c[2] * (y_lim[end] - y_lim[1])
                println("Telling model: $J, $A")
                #@async model_evaluator[] = [J,A]

                @async begin
                  params = (J,A)
                  println("Got request $(hash(params))")

                  lock(main_screen_render_lock)
                  ix_red = length(point_colors[]) + 1
                  push!(points[], Point2f($J,$A))
                  push!(point_colors[], Inf)
                  notify(point_colors)
                  notify(points)
                  unlock(main_screen_render_lock)

                  loss = plot_model($J,$A;scr = side_screen)
                  println("Filled request $(hash(params)) at $ix_red")

                  lock(main_screen_render_lock)
                  point_colors[][Int64(ix_red)] = loss
                  point_cols = point_colors[]
                  hmrange = hm.calculated_colors[].colorrange[]

                  point_cols = point_cols[isfinite.(point_cols)]
                  cmax = max(maximum(hmrange),maximum(point_cols))
                  cmin = min(minimum(hmrange),minimum(point_cols))
                  sc.colorrange[] = [cmin,cmax]
                  hm.colorrange[] = [cmin,cmax]

                  notify(point_colors)
                  notify(points)
                  notify(sc.colorrange)
                  notify(hm.colorrange)
                  unlock(main_screen_render_lock)
                end
              end
          end
      end
  end

  @async begin
    all_tasks = shuffle(collect(Iterators.product(enumerate(Js),enumerate(As))))
    for tasks in Iterators.partition(all_tasks,Threads.nthreads())
      #Threads.@threads for ((ij,J),(id,A)) in tasks
      for ((ij,J),(id,A)) in tasks
        this_loss = get_loss([J,A])
        loss_landscape[][ij,id] = this_loss
      end
      lock(main_screen_render_lock)
      notify(loss_landscape)
      unlock(main_screen_render_lock)
    end

    fig, Js, As, loss_landscape[]
  end
end

function plot_model(J,A;scr = nothing)
  # Plot plausible model result
  #print("Trying J = $J, A = $A ...")
  is = forward_problem(J,A;sys = setup_sys())
  #println(" Done!")
  sqr = squared_errors(data,is)
  is ./= get_Z(is)
  is[isnan.(data)] .= NaN

  fig = Figure()
  heatmap!(Axis(fig[1,1],title = "J = $J"),sum(is[12:14,:,1,3:end],dims=1)[1,:,:])
  heatmap!(Axis(fig[1,2]),sum(data[12:14,:,1,3:end],dims=1)[1,:,:] ./ get_Z(data))
  heatmap!(Axis(fig[1,3],title = "$(sum(sqr[12:14,:,1,3:end]))"),sum(sqr[12:14,:,1,3:end],dims=1)[1,:,:])
  #display(sum(sqr[12:14,:,1,3:end],dims=1)[1,:,:])

  heatmap!(Axis(fig[2,1],title = "A = $A"),sum(is[:,:,1,3:5],dims=4)[:,:,1])
  heatmap!(Axis(fig[2,2]),sum(data[:,:,1,3:5],dims=4)[:,:,1] ./ get_Z(data))
  heatmap!(Axis(fig[2,3], title = "$(sum(sqr[:,:,1,3:5]))"),sum(sqr[:,:,1,3:5],dims=4)[:,:,1])
  if isnothing(scr)
    display(fig)
  else
    empty!(scr)
    display(scr,fig)
  end
  loss_function(data,is)
end

function do_optim()
  Optim.optimize(get_loss,[10.25,0.069], GradientDescent(;alphaguess = LineSearches.InitialStatic(;alpha=0.8),linesearch = LineSearches.BackTracking()))
end

function multigrid_histograms(params)
  nx,ny,nz,_ = params.numbins
  block_sizes = zeros(Int64,nx,ny,nz,3)
  data_fraction = zeros(Float64,nx,ny,nz)
  for ci = CartesianIndices((nx,ny,nz))
    dx,dy,dz = ci.I
    xb,xdrop = divrem(nx,dx)
    yb,ydrop = divrem(ny,dy)
    zb,zdrop = divrem(nz,dz)

    block_sizes[ci,1] = xb
    block_sizes[ci,2] = yb
    block_sizes[ci,3] = zb

    data_fraction[ci] = (nx - xdrop) * (ny - ydrop) * (nz - zdrop) / (nx*ny*nz)
  end

  f = Figure()
  ax = Axis(f[1,1])
  ax.xzoomlock[] = true
  ax.yzoomlock[] = true
  ax.xpanlock[] = true
  ax.ypanlock[] = true
  ax.xrectzoom[] = false
  ax.yrectzoom[] = false
  heatmap!(ax,data_fraction[:,:,1])
  on(events(f).mousebutton) do event
    if event.button == Mouse.left
        if event.action == Mouse.press || event.action == Mouse.release
            mp = events(f).mouseposition[]
            bbox = ax.layoutobservables.computedbbox[]
            c = (mp .- bbox.origin) ./ bbox.widths
            if 0 < c[1] < 1 && 0 < c[2] < 1
              box_ix = 1 .+ floor.(Int64,c .* [nx,ny])
              #println(data_fraction[box_ix[1],box_ix[2],1])
              #println(block_sizes[box_ix[1],box_ix[2],1,:])
              println("Bins of size $(box_ix) use $(data_fraction[box_ix[1],box_ix[2],1]) of data")
            end
            #push!(points[], mp)
            #notify(points)
        end
    end
  end
  display(f)
  block_sizes, data_fraction
end

if !(:rparams ∈ names(Main))
  global params
  global data
  global rparams
  global rdata
  params, data = load_data()
  rparams, rdata = reduce_hist(params,data)
end
nothing
