using Sunny, GLMakie, Observables, FFTW, Printf

function view_correlations(sc)
  real_sc_data = real.(ifft(sc.data,[4,5,6,7]));
  ls = size(sc.data)[4:6]

  f = Figure()
  title = Observable("Correlations")
  ax = Axis(f[1,1]; title)

  ax_f = Axis(f[2,1]; xlabel = "q = [H,1/2,0]", ylabel = "ω (meV)")
  params = unit_resolution_binning_parameters(sc)
  kT_use = Inf
  formula_all = intensity_formula(sc,:all_available;kT = kT_use)
  is_all, counts_all = intensities_binned(sc,params,formula_all)
  is_all = is_all ./ counts_all
  max_sz = log10(maximum(maximum.(map(x -> abs.(x),is_all))))
  min_sz = log10(minimum(minimum.(map(x -> abs.(x),is_all))))

  bcs = axes_bincenters(params)
  hm_data = Observable(zeros(Float64,params.numbins[1],params.numbins[4]))
  hm = heatmap!(ax_f,bcs[1],bcs[4],hm_data,colorrange = (-3,max_sz))
  Colorbar(f[2,2],hm)

  neighs = 4
  n_colors = [:black,:red,:blue,:yellow]
  n_step = Observable([1,0,0])
  corr_ix = Observable(1)

  time_2T = size(sc.data,7)
  fourier_mode = Observable(false)
  ts_data = Observable(zeros(Float64,time_2T))
  corr_data = Vector{Observable}(undef,neighs)

  for n = 1:neighs
    corr_data[n] = Observable(zeros(Float64,time_2T))
    plot!(ax,ts_data,corr_data[n],color = n_colors[n])
  end

  onames = Dict([(v,k) for (k,v) in sc.observables.observable_ixs])
  cnames = Dict([(v,"($(string(onames[k.I[1]])),$(string(onames[k.I[2]])))") for (k,v) in sc.observables.correlations])


  fourier_step_listener = function(st)
    c = corr_ix[]
    for n = 1:neighs
      if n > 1
        corr_data[n][] .= NaN
        notify(corr_data[n])
        continue
      end
      dat = sc.data[c,1,1,mod1(st[1] + 1,ls[1]),mod1(st[2] + 1,ls[2]),mod1(st[3] + 1,ls[3]),:]
      corr_data[n][] .= log10.(abs.(dat))
      notify(corr_data[n])
    end
    title[] = "Correlation spectrum $(cnames[c]) vs ω, q = $(st ./ ls)"
  end

  real_step_listener = function(st)
    c = corr_ix[]
    for n = 1:neighs
      dat = real_sc_data[c,1,1,mod1(1 + n * st[1],ls[1]),mod1(1 + n * st[2],ls[2]),mod1(1 + n * st[3],ls[3]),:]
      corr_data[n][] .= dat
      notify(corr_data[n])
    end
    # First neighbor data
    dat = real_sc_data[c,1,1,mod1(1 + st[1],ls[1]),mod1(1 + st[2],ls[2]),mod1(1 + st[3],ls[3]),:]
    positive_time_dat = dat[1:(time_2T÷2)]
    dt = sc.Δt
    power = sum(positive_time_dat .* positive_time_dat .* dt)
    tau = power ./ (positive_time_dat[1] * positive_time_dat[1])
    title[] = "Correlation $(cnames[c]) vs Time, Δ = $(st), τ = $(Sunny.number_to_simple_string(tau,digits=4)), power = $(Sunny.number_to_simple_string(power,digits=4))"
  end

  on(real_step_listener,n_step)
  on(fourier_mode;update = true) do is_fourier
    if is_fourier
      off(n_step,real_step_listener)
      ts_data[] .= fftfreq(time_2T,1/sc.Δt)
      notify(ts_data)
      xlims!(ax,minimum(ts_data[]),maximum(ts_data[]))
      ylims!(ax,-9,1 + maximum(log10.(abs.(sc.data))))
      on(fourier_step_listener,n_step)
    else
      off(n_step,fourier_step_listener)
      ts_data[] .= sc.Δt .* (1:time_2T)
      notify(ts_data)
      xlims!(ax,minimum(ts_data[]),maximum(ts_data[]))
      ylims!(ax,minimum(real_sc_data),maximum(real_sc_data))
      on(real_step_listener,n_step)
    end
    notify(n_step)
  end

  on(corr_ix;update = true) do ci
    formula = intensity_formula(sc,[ci];kT = kT_use) do k,ω,corrs
      abs(corrs[1])
    end
    is, counts = intensities_binned(sc,params,formula)
    ix_half = (size(is,2)÷2) + 1
    for k = 1:size(is,1)
      column = is[k,ix_half,1,:]
      ω = available_energies(sc)
      ### UPGRADED FORMULA
      #ωMean = sum(ω .* column) / sum(column)
      ωMean = ω[findmax(column)[2]]
      #display(ωMean)
      is[k,ix_half,1,:] .*= Sunny.classical_to_quantum(ωMean,0.01)
    end
    hm_data[] .= log10.(is[:,ix_half,1,:] ./ counts[:,ix_half,1,:])
    #hm_data[] .= (is[:,ix_half,1,:] ./ counts[:,ix_half,1,:])
    notify(hm_data)

    notify(n_step)
  end

  display(f)

  display(sc.observables)
  display(n_colors)
  obs_names = [string.(onames[i]) for i = 1:length(onames)]

  # [q]uit
  # [c]orrelation selection
  # [s]tep size, e.g. select t in <a(t) b(0)>
  # [f]ourier transform
  while true
    print("> ")
    #try
      r = readline()
    #catch e
      #println("Error:")
      #println(e)
    #end

    if r == ""
      continue
    end

    if r[1] == 'q'
      break
    end

    if lowercase(r[1]) == 'c'
      if length(r) < 2
        println("Available observables:")
        println(obs_names)
        continue
      end
      rest = r[2:end]

      ix_1, ix_2 = if length(rest) <= 4 # single-character mode
        rest = strip(rest)
        if length(rest) < 2
          println("Invalid correlations")
          continue
        end
        c1 = rest[1]
        c2 = rest[2]
        if count(contains.(obs_names,c1)) == 1 && count(contains.(obs_names,c2)) == 1
          ix_1 = findfirst(contains.(obs_names,c1))
          ix_2 = findfirst(contains.(obs_names,c2))
          ix_1,ix_2
        else
          println("Non-unique labels!")
          continue
        end
      else # comma mode

        parts = split(strip(rest),",")
        if length(parts) != 2
          println("Invalid correlations")
          continue
        end

        ix_1 = findfirst(obs_names .== parts[1])
        ix_2 = findfirst(obs_names .== parts[2])
        if isnothing(ix_1) || isnothing(ix_2)
          println("Invalid correlations")
          continue
        end
        ix_1,ix_2
      end
      ci = CartesianIndex(ix_1,ix_2)
      corr_ix[] = sc.observables.correlations[ci]
    end

    if r[1] == 's'
      if length(r) < 2
        println("Lattice size is $(ls[1]),$(ls[2]),$(ls[3])")
        continue
      end
      rest = r[2:end]
      parts = if contains(rest,',')
        split(strip(rest),",")
      else
        split(strip(rest),"")
      end

      if length(parts) != 3
        println("Invalid step")
        continue
      end

      step = try
        parse.(Int64,parts)
      catch e
        println(e)
        continue
      end
      if fourier_mode[]
        if any(step .>= ls) || any(step .< 0)
          println("Warning: outside first BZ")
          println("Q index $(step)")
          println("Lattice size is $(ls[1]),$(ls[2]),$(ls[3])")
        end
      else
        max_wrap = neighs .* step
        if any(max_wrap .> ls)
          println("Warning: wrapping system")
          println("Furthest neighbor: $(max_wrap)")
          println("Lattice size is $(ls[1]),$(ls[2]),$(ls[3])")
        end
      end
      n_step[] = step
    end

    if r[1] == 'f'
      fourier_mode[] = !fourier_mode[]
    end
  end
end

function oscillatory_sc()
  latvecs = lattice_vectors(1, 1, 1, 90, 90, 90)
  cryst = Crystal(latvecs, [[0,0,0]])

  units = Sunny.Units.theory
  seed = 101
  sys = System(cryst, (1, 1, 1), [SpinInfo(1, S=1, g=1)], :dipole; units, seed)

  ## Model parameter
  h = 1.0

  ## External field
  set_external_field!(sys, h*[0,0,1])

  ##
  randomize_spins!(sys)
  minimize_energy!(sys)
  out = minimize_energy!(sys)
  #println(out)

  Δt = 0.0005
  kT = 40.8
  λ = 0.1
  langevin = Langevin(Δt; kT, λ)

  for _ in 1:10_000
      step!(sys, langevin)
  end

  sc = dynamical_correlations(sys; Δt=2Δt, nω=400, ωmax=10.0)

  nsamples = 50
  for _ in 1:nsamples
      for _ in 1:1_000
          step!(sys, langevin)
      end
      println(sys.dipoles[1])
      @time add_sample!(sc, sys; alg = :no_window) # Never observe decorrelation due to ImplicitMidpoint
  end
  sc, sys
end

function example_sc()
  latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
  cryst = Crystal(latvecs, [[0,0,0]])

  units = Sunny.Units.theory
  seed = 101
  sys_rcs = System(cryst, (20, 20, 1), [SpinInfo(1, S=1, g=1)], :dipole; units, seed)

  ## Model parameter
  J = 1.0
  h = 0.5 
  D = 0.05

  ## Set exchange interactions
  set_exchange!(sys_rcs, J, Bond(1, 1, [1, 0, 0]))

  ## Single-ion anisotropy
  Ss = spin_operators(sys_rcs, 1)
  set_onsite_coupling!(sys_rcs, D*Ss[3]^2, 1)

  ## External field
  set_external_field!(sys_rcs, h*[0,0,3])

  ##
  randomize_spins!(sys_rcs)
  minimize_energy!(sys_rcs)
  out = minimize_energy!(sys_rcs)
  #println(out)

  Δt = 0.025
  kT = 0.2
  λ = 0.1
  langevin = Langevin(Δt; kT, λ)

  for _ in 1:10_000
      step!(sys_rcs, langevin)
  end

  scgs = dynamical_correlations(sys_rcs; Δt=2Δt, nω=100, ωmax=5.0)

  nsamples = 50
  for _ in 1:nsamples
      for _ in 1:1_000
          step!(sys_rcs, langevin)
      end
      @time add_sample!(scgs, sys_rcs; alg = :window)
  end
  scgs, sys_rcs
end

include("support.jl")
