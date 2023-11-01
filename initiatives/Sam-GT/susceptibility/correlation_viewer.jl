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
    title[] = "Correlation $(cnames[c]) vs Time, Δ = $(st)"
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

  while true
    print("> ")
    r = readline()

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

function example_sc()
  latvecs = lattice_vectors(1, 1, 1.2, 90, 90, 90)
  cryst = Crystal(latvecs, [[0,0,0]])

  units = Sunny.Units.theory
  seed = 101
  sys_rcs = System(cryst, (10, 10, 1), [SpinInfo(1, S=1, g=1)], :dipole; units, seed)

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
  @profview for _ in 1:nsamples
      for _ in 1:1_000
          step!(sys_rcs, langevin)
      end
      @time add_sample!(scgs, sys_rcs; alg = :window)
  end
  scgs, sys_rcs
end

function detail_sys(sys::System{N}) where N
    io = stdout
    modename = if sys.mode==:SUN
        "SU($N)"
    elseif sys.mode==:dipole
        "Dipole mode"
    elseif sys.mode==:large_S
        "Large-S classical limit"
    else
        error("Unreachable")
    end
    printstyled(io, "System [$modename]\n"; bold=true, color=:underline)
    println(io, "Lattice: $(sys.latsize)×$(Sunny.natoms(sys.crystal))")
    if !isnothing(sys.origin)
        #println(io, "Reshaped cell geometry $(cell_dimensions(sys))")
        is_enlarged = abs(det(sys.crystal.latvecs)) > abs(det(sys.origin.crystal.latvecs))
        println(io)
        println(io, "Unit cell has been $(is_enlarged ? "enlarged" : "reshaped") from original such that:")
        printstyled(io, "  [Original lattice vectors]";bold=true,color=:red)
        print(io," * $(cell_dimensions(sys)) = ")
        printstyled(io,"[Reshaped lattice vectors]\n";bold=true,color=:blue)
        println(io, "where")
        printstyled(io, "  [Original] ";bold=true,color=:red)
        show(io, sys.origin.crystal)
        printstyled(io, "\n  [Reshaped] ";bold=true,color=:blue)
        show(io, sys.crystal)
        println(io)
    else
        show(io, sys.crystal)
    end
    println(io)

    if Sunny.is_homogeneous(sys)
        ints = Sunny.interactions_homog(sys)
        if isempty(ints)
            println(io, "No interactions")
        else
            print(io, "Homogeneous interactions by atom:\n")
            for (i,int) in enumerate(ints)
                if sys.crystal.types[i] != ""
                    print(io,"  $i. '$(sys.crystal.types[i])' atom has ")
                else
                    print(io,"  Atom $i has ")
                end
                show(io,int)
                println(io)
            end
        end
    else
        print(io, "Inhomogeneous interactions (may differ at every site)")
    end

    if !isnothing(sys.ewald)
        println(io, "Long range dipole-dipole interactions enabled!")
    end

    if !iszero(sys.extfield)
        if allequal(sys.extfield)
            println(io, "Uniform magnetic field B = $(sys.extfield[1]) applied")
        else
            mean_field = sum(sys.extfield) ./ length(sys.extfield)
            rms_field = sqrt.(sum([(B .- mean_field) .^ 2 for B in sys.extfield]) ./ length(sys.extfield))

            mean_field_str = @sprintf "[%.4g %.4g %.4g]" mean_field[1] mean_field[2] mean_field[3]
            rms_field_str = @sprintf "[%.4g %.4g %.4g]" rms_field[1] rms_field[2] rms_field[3]

            println(io, "Spatially periodic magnetic field B = (mean $mean_field_str ± $rms_field_str RMS) applied")
        end
    end
end

function Base.show(io::IO, stvexp::Sunny.StevensExpansion)
    print(io,"StevensExpansion{c0=$(stvexp.c0),0,$(stvexp.c2),0,$(stvexp.c4),0,$(stvexp.c6)}")
end

function Base.show(io::IO, ::MIME"text/plain", stvexp::Sunny.StevensExpansion)
    print(io,show_stevens_expansion(stvexp))
end

function Base.iszero(s::Sunny.StevensExpansion)
  iszero(s.c0) && iszero(s.c2) && iszero(s.c4) && iszero(s.c6)
end

function Base.show(io::IO, ints::Sunny.Interactions)
    has_onsite = !iszero(ints.onsite)
    count_pair = length(ints.pair)
    if !has_onsite && count_pair == 0
        print(io,"[No Interactions]")
    else
        print(io,"Interactions($(has_onsite ? "Onsite Coupling, " : "")$(count_pair) Pair Couplings)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ints::Sunny.Interactions)
    if !iszero(ints.onsite)
      if ints.onsite isa Sunny.StevensExpansion
        println(io,"Onsite coupling stevens expansion: ",show_stevens_expansion(ints.onsite))
      else
        println(io,"Onsite coupling matrix: ",display(ints.onsite))
      end
    end

    if isempty(ints.pair)
        if iszero(ints.onsite.matrep)
           println(io,"No interactions")
        end
        return
    end
    println(io,"Pair couplings:")
    count_culled = 0
    for pair in ints.pair
        if pair.isculled
          count_culled += 1
          continue
        end
        print(io,"  ")
        show(io,pair)
        println(io)
    end
    if count_culled > 0
        println(io,"  + $(count_culled) culled couplings")
    end
end

function show_stevens_expansion(stvexp::Sunny.StevensExpansion)
    c = map(1:6) do k
        if k == 2
            stvexp.c2
        elseif k == 4
            stvexp.c4
        elseif k == 6
            stvexp.c6
        else
            zeros(Float64, 2k+1)
        end
    end

    terms = String[]
    for k in 1:6
        for (c_km, m) in zip(reverse(c[k]), -k:k)
            abs(c_km) < 1e-12 && continue
            push!(terms, *(Sunny.coefficient_to_math_string(c_km), "𝒪", Sunny.int_to_underscore_string.((k,m))...))
        end
    end

    # Linear shift c_00 is not included in StevensExpansion
    push!(terms, "trace")

    # Concatenate with plus signs
    str = join(terms, " + ")
    # Remove redundant plus signs and print
    str = replace(str, "+ -" => "- ")
    str
end

function Base.show(io::IO, pair::Sunny.PairCoupling)
    cull_string = pair.isculled ? "(CULLED)," : ""
    bilin_string = iszero(pair.bilin) ? "" : (pair.bilin isa Float64 ? ",J = $(pair.bilin)" : ",J = Exchange Matrix")
    biquad_string = iszero(pair.biquad) ? "" : ",biquad = $(pair.biquad)"
    print(io,"PairCoupling($(cull_string)$(repr(pair.bond))$(bilin_string)$(biquad_string))")
end

function Base.show(io::IO, ::MIME"text/plain", pair::Sunny.PairCoupling)
    cull_string = pair.isculled ? "(CULLED) " : ""
    printstyled(io,"Pair Coupling $(cull_string)on $(repr(pair.bond))\n";bold=true,underline=true)
    #printstyled(io, repr(b); bold=true, color=:underline)

    atol = 1e-12
    digits = 8
    max_denom = 20

    if !iszero(pair.scalar)
      println(io,"Scalar (diagonal pure Heisenberg) = $(Sunny.number_to_math_string(pair.scalar;digits,atol,max_denom))")
    end

    if !iszero(pair.bilin)
      if pair.bilin isa Sunny.Mat3
          strs = Sunny.number_to_math_string.(pair.bilin;digits,atol,max_denom)
          Sunny.print_allowed_coupling(io,strs; prefix="Bilinear exchange matrix: ")
      else
        println(io,"Heisenberg bilinear (pure diagonal) exchange J = $(Sunny.number_to_math_string(pair.bilin;digits,atol,max_denom))")
      end
    end

    if !iszero(pair.biquad)
      if pair.biquad isa Sunny.Mat5
          strs = Sunny.number_to_math_string.(pair.biquad;digits,atol,max_denom)
          Sunny.print_allowed_coupling(io,strs; prefix="Biquadratic exchange matrix: ")
      else
        println(io,"Heisenberg biquadratic exchange J = $(Sunny.number_to_math_string(pair.biquad;digits,atol,max_denom))")
      end
    end
end

function Base.show(io::IO, ::MIME"text/plain", ten::Sunny.TensorDecomposition)
  print(io,"General interaction with $(length(ten.data)) SVD terms")
  #for (a,b) in ten.data
  #end
end

