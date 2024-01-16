using Statistics

function z_series(kT,λ;Δt = 0.05)
  cryst = Crystal(I(3),[[0,0,0]],1)
  sys = System(cryst,(1,1,1), [SpinInfo(1,S=1,g=2)], :dipole_large_S, units = Units.theory)

  # Prefer z = 0
  set_onsite_coupling!(sys, S -> S[3]^2,1)

  sys.dipoles[1] = [1,0,0]

  langevin = Langevin(Δt;kT,λ)

  # Burn in
  for i = 1:100
    step!(sys,langevin)
  end

  # Record
  n_record = 10000#0
  zvals = Vector{Float64}(undef,n_record)
  #std_est = 0.
  for j = 1:n_record
    step!(sys,langevin)
    this_z = sys.dipoles[1][3]
    zvals[j] = this_z
    #std_est = std_est + this_z
  end
  zvals
end

function measure_width(kT,λ;Δt = 0.05)
  zvals = z_series(kT,λ;Δt)
  stdm(zvals,0.)
end

function sweep_width_range(T,λ)
  w = [measure_width(kT, λ1) for kT = T, λ1 = λ]
  f = Figure(); ax = Axis(f[1,1], xlabel = "T", ylabel = "λ")
  hm = heatmap!(ax,log10.(T),log10.(λ),w ./ sqrt.(T); colorrange = (0,2))
  Colorbar(f[1,2],hm)
  #contour!(ax,log10.(T),log10.(λ),w ./ sqrt.(T); levels = [sqrt(2)],color = :black)
  #contour!(ax,log10.(T),log10.(λ),w ./ sqrt.(T); levels = [1/sqrt(2)],color = :red)
  display(f)
  w
end

function sweep_width(;kT0 = 1., λ0 = 0.1, dkT = 0.01, dλ = 0.001)
  irange = -20:20
  jrange = -20:21
  w = [measure_width(kT0 + i * dkT, λ0 + j * dλ) for i = irange, j = jrange]
  λs = [λ0 + j * dλ for j = jrange]
  kTs = [kT0 + i * dkT for i = irange]
  kTs, λs, w
end

