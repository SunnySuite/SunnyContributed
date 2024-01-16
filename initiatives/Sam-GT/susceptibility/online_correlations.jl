using Sunny, GLMakie, FFTW, LinearAlgebra

mutable struct OnlineCorrelations
  sys :: System
  measperiod :: Int
  data :: Array{ComplexF64, 7} # nt x latsize x natom x natom x ncorr
  samplebuf :: Array{ComplexF64, 6} # nt x latsize x natom x nobs
  backbuf :: Array{ComplexF64, 6}
  nsamples :: Int
  observables :: Sunny.ObservableInfo
  pfft
  corr_buf :: Array{ComplexF64, 4}
  integrator
end

function online_to_sampled(oc)
  nt = size(oc.data,1)
  dw = 2π / (oc.integrator.Δt * oc.measperiod * nt)
  ωmax = dw * nt / 2
  sc = dynamical_correlations(oc.sys; Δt=oc.integrator.Δt, nω=(nt+1)÷2, ωmax)
  println(size(sc.data))
  println(size(oc.data))
  sc.data .= permutedims(fft(oc.data,1),[7,5,6,2,3,4,1])
  sc
end

function mk_oc(sys;measperiod = 5,nt = 250, integrator = Langevin(0.001,λ=0.1,kT = 0.1), observables = nothing, correlations = nothing)
  N = typeof(sys).parameters[1]
  observables = Sunny.parse_observables(N;observables,correlations)
  na = Sunny.natoms(sys.crystal)
  data = zeros(ComplexF64,nt,sys.latsize...,na,na,Sunny.num_correlations(observables))
  samplebuf = zeros(ComplexF64,nt,sys.latsize...,na,Sunny.num_observables(observables))
  backbuf = copy(samplebuf)
  pfft = plan_fft(samplebuf,(2,3,4)) * (1 / sqrt(prod(sys.latsize)))
  corr_buf = zeros(ComplexF64,nt,sys.latsize...)
  OnlineCorrelations(sys,measperiod,data,samplebuf,backbuf,0,observables,pfft,corr_buf,integrator)
end


function walk_online!(oc)
  # Walk forward for several states
  for j = 1:oc.measperiod
    step!(oc.sys,oc.integrator)
  end
  walk_online_no_step!(oc)
end

function walk_online_no_step!(oc)
  # Put the new observable values at the end of the back buffer
  nt = size(oc.samplebuf,1)
  now_buf = reshape(copy(oc.backbuf[nt,:,:,:,:,:]),1,size(oc.backbuf)[2:6]...)

  # observable_values!
  N = typeof(oc.sys).parameters[1]
  if N == 0
    for site in eachsite(oc.sys), (i, op) in enumerate(oc.observables.observables)
      dipole = oc.sys.dipoles[site]
      apply_g = true
      if apply_g
        dipole = oc.sys.gs[site] * dipole
      end
      now_buf[1,site,i] = op * dipole
    end
  else
    Zs = oc.sys.coherents
    for (i, op) in enumerate(oc.observables.observables)
      matrix_operator = convert(Matrix{ComplexF64},op)
        for site in eachsite(oc.sys)
          now_buf[1,site,i] = dot(Zs[site], matrix_operator, Zs[site])
        end
    end
  end

  # Shift in new value
  new_value_index = (nt+1)÷2
  circshift!(oc.backbuf,oc.samplebuf,(-1,0,0,0,0,0))
  oc.backbuf[new_value_index,:,:,:,:,:] .= view(now_buf,1,:,:,:,:,:)

  # Swap buffers!
  oc.backbuf, oc.samplebuf = oc.samplebuf, oc.backbuf
  # samplebuf now holds a history of the latest samples

  # Go to spatial fourier space because we're correlating
  #oc.backbuf .= oc.samplebuf
  #fft!(oc.backbuf,(2,3,4))
  oc.backbuf .= oc.pfft * oc.samplebuf

  # Correlate this new observable value with the existing ones
  oc.nsamples += 1
  na = Sunny.natoms(oc.sys.crystal)
  corr_buf = oc.corr_buf
  for j = 1:na, i = 1:na, (ci, c) in oc.observables.correlations
    a,b = ci.I

    # The 'present' is at index 1 of the history
    now_b_value = oc.backbuf[1,:,:,:,j,b]
    #now_b_value = view(oc.backbuf,1,:,:,:,j,b)

    # Relative to the present, the history is aligned so that
    # future values exist at 'positive fftfreq' indices and vice versa
    then_a_values = view(oc.backbuf,:,:,:,:,i,a)

    # This performs the spatial correlation, in spatial fourier space
    #window_func = cos.(range(0,π,length = nt)).^2 ./ pls
    @inbounds for t = 1:nt, l = CartesianIndices(now_b_value)
      corr_buf[t,l] = then_a_values[t,l] * conj(now_b_value[l]) #* window_func[t]
    end


    # Average the corr_buf into oc.data
    databuf = view(oc.data,:,:,:,:,i,j,c)
    for k in eachindex(databuf)
      diff = corr_buf[k] - databuf[k]
      databuf[k] += diff / nt #oc.nsamples
    end
  end

  nothing
end

function show_corrs(ax,oc)
  heatmap!(ax,log10.(abs.(real.(ifft(oc.data,(2,3,4)))[:,1,1,1,1,1,:])))
end

function sqw(ax,oc)
  heatmap!(ax,log10.(abs.(fft(oc.data,1)[:,:,1,1,1,1,1])))
end

function Sunny.intensity_formula(f::Function, oc::OnlineCorrelations, corr_ix::AbstractVector{Int64}; 
    kT = Inf, 
    formfactors = nothing, 
    return_type = Float64, 
    string_formula = "f(Q,ω,S{α,β}[ix_q,ix_ω])"
)
    # If temperature given, ensure it's greater than 0.0
    if kT != Inf
        if iszero(kT)
            error("`kT` must be greater than zero.")
        end
    end

    #ωs_sc = available_energies(sc;negative_energies = true)
    ωs = fftfreq(size(oc.data,1),size(oc.data,1))

    ff_atoms = Sunny.propagate_form_factors_to_atoms(formfactors, oc.sys.crystal)
    na = Val(Sunny.natoms(oc.sys.crystal))
    nc = Val(length(corr_ix))

    dt = oc.integrator.Δt
    cryst = oc.sys.crystal

    # Intensity is calculated at the discrete (ix_q,ix_ω) modes available to the system.
    # Additionally, for momentum transfers outside of the first BZ, the norm `q_absolute` of the
    # momentum transfer may be different than the one inferred from `ix_q`, so it needs
    # to be provided independently of `ix_q`.
    calc_intensity = function(oc::OnlineCorrelations, q_absolute::Sunny.Vec3, ix_q::CartesianIndex{3}, ix_ω::Int64)
      correlations = Sunny.phase_averaged_elements(permutedims(view(oc.data, ix_ω, ix_q, :, :, corr_ix),[3,1,2]), q_absolute, cryst, ff_atoms, nc, na)

      ω = ωs[ix_ω] * 2π / (dt * oc.measperiod * size(oc.data,1))
      return f(q_absolute, ω, correlations) * Sunny.classical_to_quantum(ω,kT)
    end
    Sunny.ClassicalIntensityFormula{return_type}(kT, formfactors, string_formula, calc_intensity)
end


function Sunny.intensity_formula(sc::OnlineCorrelations, elem::Tuple{Symbol,Symbol}; kwargs...)
    string_formula = "S{$(elem[1]),$(elem[2])}[ix_q,ix_ω]"
    intensity_formula(sc,Element(sc, elem); string_formula, kwargs...)
end

function Sunny.intensity_formula(sc::OnlineCorrelations, mode::Symbol; kwargs...)
    contractor, string_formula = Sunny.contractor_from_mode(sc, mode)
    intensity_formula(sc, contractor; string_formula, kwargs...)
end

function Sunny.intensity_formula(sc::OnlineCorrelations, contractor::Sunny.Contraction{T}; kwargs...) where T
    intensity_formula(sc,Sunny.required_correlations(contractor); return_type = T,kwargs...) do k,ixω,correlations
        intensity = Sunny.contract(correlations, k, contractor)
    end
end

function streaming_fei2()
  a = b = 4.05012#hide
  c = 6.75214#hide
  latvecs = lattice_vectors(a, b, c, 90, 90, 120)#hide
  positions = [[0,0,0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]#hide
  types = ["Fe", "I", "I"]#hide
  FeI2 = Crystal(latvecs, positions; types)#hide
  cryst = subcrystal(FeI2, "Fe")#hide
  sys = System(cryst, (4,4,4), [SpinInfo(1,S=1,g=2)], :SUN, seed=2)#hide
  J1pm   = -0.236#hide
  J1pmpm = -0.161#hide
  J1zpm  = -0.261#hide
  J2pm   = 0.026#hide
  J3pm   = 0.166#hide
  J′0pm  = 0.037#hide
  J′1pm  = 0.013#hide
  J′2apm = 0.068#hide
  J1zz   = -0.236#hide
  J2zz   = 0.113#hide
  J3zz   = 0.211#hide
  J′0zz  = -0.036#hide
  J′1zz  = 0.051#hide
  J′2azz = 0.073#hide
  J1xx = J1pm + J1pmpm#hide
  J1yy = J1pm - J1pmpm#hide
  J1yz = J1zpm#hide
  set_exchange!(sys, [J1xx 0.0 0.0; 0.0 J1yy J1yz; 0.0 J1yz J1zz], Bond(1,1,[1,0,0]))#hide
  set_exchange!(sys, [J2pm 0.0 0.0; 0.0 J2pm 0.0; 0.0 0.0 J2zz], Bond(1,1,[1,2,0]))#hide
  set_exchange!(sys, [J3pm 0.0 0.0; 0.0 J3pm 0.0; 0.0 0.0 J3zz], Bond(1,1,[2,0,0]))#hide
  set_exchange!(sys, [J′0pm 0.0 0.0; 0.0 J′0pm 0.0; 0.0 0.0 J′0zz], Bond(1,1,[0,0,1]))#hide
  set_exchange!(sys, [J′1pm 0.0 0.0; 0.0 J′1pm 0.0; 0.0 0.0 J′1zz], Bond(1,1,[1,0,1]))#hide
  set_exchange!(sys, [J′2apm 0.0 0.0; 0.0 J′2apm 0.0; 0.0 0.0 J′2azz], Bond(1,1,[1,2,1]))#hide
  D = 2.165#hide
  S = spin_operators(sys, 1)#hide
  set_onsite_coupling!(sys, -D*S[3]^2, 1)#hide

  Δt = 0.05/D
  kT = 0.2  
  λ = 0.1  
  langevin = Langevin(Δt; kT, λ);

  sys_large = resize_supercell(sys, (16,16,4)) # 16x16x4 copies of the original unit cell
  plot_spins(sys_large; color=[s[3] for s in sys_large.dipoles])

  kT = 3.5 * meV_per_K     # 3.5K ≈ 0.30 meV
  langevin.kT = kT
  println("Thermalize")
  @time for _ in 1:10_000
    step!(sys_large, langevin)
  end
  thermal_langevin = copy(langevin)

  #=
  seed = 101
  crystal = Crystal(I(3), [[0,0,0]], 1)
  units = Sunny.Units.theory
  sys_large = System(crystal, (20,20,1), [SpinInfo(1; S=1, g=2)], :dipole; units, seed)
  J = 1.0
  h = 0.4
  D = 0.1
  set_exchange!(sys_large, J, Bond(1, 1, [1, 0,0]))
  set_exchange!(sys_large, J, Bond(1, 1, [0, 1,0]))
  set_external_field!(sys_large, [h, 0, 0])
  S = spin_matrices(1)
  set_onsite_coupling!(sys, D*S[3]^2, 1)
  =#

  nt = 120
  langevin.Δt = 0.05/D
  langevin.kT = 0.5
  langevin.λ = 0.2
  oc = mk_oc(sys_large; measperiod = 12,nt, integrator = langevin, observables = nothing, correlations = nothing)
  dw = 2π / (oc.integrator.Δt * oc.measperiod * nt)
  ωmax = dw * nt / 2
  sc = dynamical_correlations(sys_large; Δt=langevin.Δt, nω=(nt-1)÷2, ωmax)
  add_sample!(sc,sys_large;alg = :no_window)
  for _ in 1:1000
    step!(sys_large, thermal_langevin)
  end
  add_sample!(sc,sys_large;alg = :no_window)
  for _ in 1:1000
    step!(sys_large, thermal_langevin)
  end
  add_sample!(sc,sys_large;alg = :no_window)
  display(sc)

  points = [[0,   0, 0],  # List of wave vectors that define a path
            [1,   0, 0],
            [0,   1, 0],
            [1/2, 0, 0],
            [0,   1, 0],
            [0,   0, 0]]
  density = 8
  path, xticks = reciprocal_space_path(cryst, points, density);

  formfactors = [FormFactor("Fe2"; g_lande=3/2)]
  new_formula = intensity_formula(oc, :trace; kT = Inf, formfactors)
  new_formula_sc = intensity_formula(sc, :trace; kT = Inf, formfactors)

  ixqs = Vector{CartesianIndex{3}}(undef,length(path))
  ks = Vector{Sunny.Vec3}(undef,length(path))

  hmdat = Observable(zeros(Float64,length(path),nt))

  f = Figure();

  ax = Axis(f[1,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
  dw = 2π / (oc.integrator.Δt * oc.measperiod * nt)
  display(dw)
  heatmap!(ax,1:length(path),dw * fftshift(fftfreq(nt,nt)),map(x -> (abs.(x)),hmdat))

  is_interpolated = intensities_interpolated(sc, path, new_formula_sc;interpolation = :round);
  ax2 = Axis(f[2,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
  heatmap!(ax2,1:length(path),available_energies(sc),log10.(abs.(is_interpolated)))

  display(f)

  for l = 1:nt
    walk_online!(oc)
  end
  oc.data .= 0
  oc.nsamples = 0

  # Mocked up interpolation
  for (j,q) in enumerate(path)
    m = round.(Int, oc.sys.latsize .* q)
    ixqs[j] = map(i -> mod(m[i], oc.sys.latsize[i])+1, (1, 2, 3)) |> CartesianIndex{3}
    ks[j] = Sunny.Vec3(oc.sys.crystal.recipvecs * q)
  end
  l = 0
  function()
    l = l + 1
    walk_online!(oc)
    if mod(l,10) == 0
      fft!(oc.data,1)
      for (j,q) in enumerate(path)
        for t = 1:nt
          hmdat[][j,t] = new_formula.calc_intensity(oc, ks[j], ixqs[j], t)
        end
      end
      hmdat[] .= fftshift(hmdat[],2)
      ifft!(oc.data,1)
      notify(hmdat)
      sleep(0.001)
    end
  end

  #calc_intensity = function(oc::OnlineCorrelations, q_absolute::Vec3, ix_q::CartesianIndex{3}, ix_ω::Int64)
end

function streaming_afm()
  crystal = Crystal(I(3), [[0,0,0]], 1)
  units = Sunny.Units.theory
  sys_large = System(crystal, (20,20,1), [SpinInfo(1; S=1, g=2)], :dipole; units)
  randomize_spins!(sys_large)
  J = -0.3
  h = 1.0
  D = 0.0#1.0
  set_exchange!(sys_large, J, Bond(1, 1, [1, 0,0]))
  set_exchange!(sys_large, J, Bond(1, 1, [0, 1,0]))
  set_external_field!(sys_large, [0, 0, h])
  S = spin_matrices(1)
  #set_onsite_coupling!(sys_large, D*S[3]^2, 1)

  Δt = 0.05#/max(abs(J),abs(D))
  kT = 0.2  
  λ = 0.1  
  langevin = Langevin(Δt; kT, λ);

  nt = 240
  langevin.Δt = 0.03
  langevin.kT = 0.001#0.12
  langevin.λ = 0.01

  println("Thermalize")
  @time for _ in 1:10_000
    step!(sys_large, langevin)
  end

  oc = mk_oc(sys_large; measperiod = 4,nt, integrator = langevin, observables = nothing, correlations = nothing)

  dw = 2π / (oc.integrator.Δt * oc.measperiod * nt)
  ωmax = dw * nt

  sc = dynamical_correlations(sys_large; Δt = langevin.Δt, nω = (nt-1)÷2, ωmax)
  add_sample!(sc,sys_large;alg = :window)
  display(sc)

  points = [[1/2,   0, 0],  # List of wave vectors that define a path
            [0,   1/2, 0],
            [1/2,   1/2, 0],
            [0, 0, 0],
            [1/2,   0, 0]]
  density = 4
  path, xticks = reciprocal_space_path(oc.sys.crystal, points, density);

  formfactors = [FormFactor("Fe2"; g_lande=3/2)]
  new_formula = intensity_formula(oc, :trace; kT=Inf, formfactors)
  new_formula_sc = intensity_formula(sc, :trace; kT=Inf, formfactors)

  ixqs = Vector{CartesianIndex{3}}(undef,length(path))
  ks = Vector{Sunny.Vec3}(undef,length(path))

  hmdat = Observable(zeros(Float64,length(path),nt))

  f = Figure();
  ax = Axis(f[1,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
  heatmap!(ax,1:length(path),dw * fftshift(fftfreq(nt,nt)),map(x -> log10.(abs.(x)),hmdat))

  zeroDat_full = Observable(zeros(Float64,nt))
  halfDat_full = Observable(zeros(Float64,nt))
  #zeroDat_half = map(zeroDat_full) do zdf
    #ifft(zdf)
    #Observable(zeros(Float64,nt÷2))
  #zeroDat_qtr = Observable(zeros(Float64,nt÷4))
  ax = Axis(f[1,2],xlabel = "meV")
  lines!(ax,dw * fftshift(fftfreq(nt,nt)),map(x -> real.(x),zeroDat_full), color = :blue)
  lines!(ax,dw * fftshift(fftfreq(nt,nt)),map(x -> real.(x),halfDat_full), color = :orange)
  #lines!(ax,2dw * fftshift(fftfreq(nt÷2,nt÷2)),zeroDat_half)
  #lines!(ax,4dw * fftshift(fftfreq(nt÷4,nt÷4)),zeroDat_qtr)

  is_interpolated = intensities_interpolated(sc, path, new_formula_sc;interpolation = :round);
  ax2 = Axis(f[2,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
  heatmap!(ax2,1:length(path),available_energies(sc),log10.(abs.(is_interpolated)))

  display(f)

  for l = 1:nt
    walk_online!(oc)
  end
  oc.data .= 0
  oc.nsamples = 0

  # Mocked up interpolation
  for (j,q) in enumerate(path)
    m = round.(Int, oc.sys.latsize .* q)
    ixqs[j] = map(i -> mod(m[i], oc.sys.latsize[i])+1, (1, 2, 3)) |> CartesianIndex{3}
    ks[j] = Sunny.Vec3(oc.sys.crystal.recipvecs * q)
  end
  l = 0
  function()
    l = l + 1
    walk_online!(oc)
    if mod(l,10) == 0
      fft!(oc.data,1)
      for (j,q) in enumerate(path)
        for t = 1:nt
          hmdat[][j,t] = new_formula.calc_intensity(oc, ks[j], ixqs[j], t)
        end
      end
      m = round.(Int, oc.sys.latsize .* [0,0,0.])
      ix_q = map(i -> mod(m[i], oc.sys.latsize[i])+1, (1, 2, 3)) |> CartesianIndex{3}
      for t = 1:nt
        zeroDat_full[][t] = new_formula.calc_intensity(oc, Sunny.Vec3([0,0,0.]), ix_q, t)
      end
      zeroDat_full[] .= fftshift(zeroDat_full[])
      notify(zeroDat_full)

      m = round.(Int, oc.sys.latsize .* [0.5,0.5,0.])
      ix_q = map(i -> mod(m[i], oc.sys.latsize[i])+1, (1, 2, 3)) |> CartesianIndex{3}
      for t = 1:nt
        halfDat_full[][t] = new_formula.calc_intensity(oc, Sunny.Vec3(oc.sys.crystal.recipvecs * [0.5,0.5,0.]), ix_q, t)
      end
      halfDat_full[] .= fftshift(halfDat_full[])
      notify(halfDat_full)

      hmdat[] .= fftshift(hmdat[],2)
      ifft!(oc.data,1)
      notify(hmdat)
      sleep(0.001)
    end
  end

  #calc_intensity = function(oc::OnlineCorrelations, q_absolute::Vec3, ix_q::CartesianIndex{3}, ix_ω::Int64)
end

