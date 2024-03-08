using Sunny, GLMakie, ComplexPortraits

function show_phase_portrait(dsc,zl,zh,corr_ix;kwargs...)
  corr_dat = real.(fftshift(ifft(dsc.data[corr_ix,1,1,1,1,1,:])))
  ts = fftshift(fftfreq(size(dsc.data,7),1/dsc.Δω)) * 2π
  image((imag(zl),imag(zh)),(real(zl),real(zh)),ComplexPortraits.portrait(zl,zh, z-> sum(corr_dat .* exp.(-(real(z) + im * round(imag(z) / dsc.Δω) * dsc.Δω) .* ts) .* dsc.Δω * π);kwargs...))
end

function show_direct_time_corr(dsc, ci)
  f = Figure();
  display(f);
  ax = Axis(f[1,1])
  show_direct_time_corr!(ax, dsc, ci)
end

function show_direct_time_corr!(ax,dsc, ci)
  ts = fftshift(fftfreq(size(dsc.data,7),1/dsc.Δω)) * 2π
  for (ni,i) = enumerate(ci)
    lines!(ax,ts,real.(fftshift(ifft(dsc.data[i,1,1,1,1,1,:]))),color = [:blue, :orange, :green][ni])
  end
end

function show_last_sample(dsc)
  f = Figure();
  display(f);
  ax = Axis(f[1,1])
  show_last_sample!(ax,dsc)
end

function show_last_sample!(ax,dsc)
  ts_sb = range(0,step = dsc.Δt * dsc.measperiod,length = size(dsc.samplebuf,6))
  for i = 1:3
    lines!(ax,ts_sb,real(dsc.samplebuf[i,1,1,1,1,:]),color = [:blue, :orange, :green][i])
  end
end

function use_rSt_show_reverse_time_corr(dsc,ci_to_show)
  f = Figure();
  display(f);
  ax = Axis(f[1,1])

  ts = fftshift(fftfreq(size(dsc.data,7),1/dsc.Δω)) * 2π
  ix_t_pos = findall(ts .> 0)
  ix_t_neg = reverse(findall(ts .< 0))
  if length(ix_t_neg) > length(ix_t_pos)
    ix_t_neg = ix_t_neg[1:end-1]
  end
  ix_t_zero = findall(.!((ts .> 0) .|| (ts .< 0)))

  observables = dsc.observables
  o_names = Sunny.all_observable_names(observables)
  for (ci, c) in observables.correlations  
    if c ∉ ci_to_show
      continue
    end
    α, β = ci.I
    swapped_corr = Sunny.lookup_correlations(observables,[(o_names[β],o_names[α])])[1]
    #sfwd = dsc.data[c,1,1,1,1,1,:]
    sfwd_swap = real.(fftshift(ifft((dsc.data[swapped_corr,1,1,1,1,1,:]))))
    srev = zero(sfwd_swap)

    # The (t > 0) part of the S_ab^{rev}(t) is given
    # by S_ba(-t), using [rSt] symmetry:
    srev[ix_t_pos] .= sfwd_swap[ix_t_neg]

    # The (t = 0) part S_ab^{rev}(0) coincides with the S_ba(0)
    srev[ix_t_zero] .= sfwd_swap[ix_t_zero]

    # same for (t < 0)
    srev[ix_t_neg] .= sfwd_swap[ix_t_pos]

    lines!(ts,srev)
  end
end

function reverse_time_dat(dsc)

  ts = fftfreq(size(dsc.data,7),1/dsc.Δω) * 2π
  ix_t_pos = findall(ts .> 0)
  ix_t_neg = reverse(findall(ts .< 0))
  if length(ix_t_neg) > length(ix_t_pos)
    ix_t_neg = ix_t_neg[2:end]
  end
  ix_t_zero = findall(.!((ts .> 0) .|| (ts .< 0)))

  observables = dsc.observables
  o_names = Sunny.all_observable_names(observables)
  dat_rev = zero(dsc.data)
  for (ci, c) in observables.correlations  
    α, β = ci.I
    swapped_corr = Sunny.lookup_correlations(observables,[(o_names[β],o_names[α])])[1]
    sfwd_swap = ifft(dsc.data[swapped_corr,:,:,:,:,:,:],6)

    # The (t > 0) part of the S_ab^{rev}(t) is given
    # by S_ba(-t), using [rSt] symmetry:
    dat_rev[c,:,:,:,:,:,ix_t_pos] .= sfwd_swap[:,:,:,:,:,ix_t_neg]

    # The (t = 0) part S_ab^{rev}(0) coincides with the S_ba(0)
    dat_rev[c,:,:,:,:,:,ix_t_zero] .= sfwd_swap[:,:,:,:,:,ix_t_zero]

    # same for (t < 0)
    dat_rev[c,:,:,:,:,:,ix_t_neg] .= sfwd_swap[:,:,:,:,:,ix_t_pos]
  end
  fft!(dat_rev,7)
  dat_rev
end

function reverse_dsc(dsc)
  dat_new = copy(reverse_time_dat(dsc))
  dsc.data .= dat_new
  nothing
end

function mk_dsc(;mode = :dipole)
  cryst = Crystal(diagm([1,1,3]), [[0,0,0]],1)
  sys = System(cryst, (100,1,1), [SpinInfo(1;S=1,g=2)],mode)
  set_exchange!(sys,1.,Bond(1,1,[1,0,0]))
  set_onsite_coupling!(sys, S -> -1.5 * S[3]^2,1)
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 500)

  temp = 0.3
  langevin = Langevin(0.03;λ = 0.1, kT=temp)
  for i = 1:10000
    step!(sys,langevin)
  end
  suggest_timestep(sys,langevin;tol = 1e-2)

  dsc = if mode == :SUN
    S = spin_matrices(1)
    obs = [S[1],S[2],S[3]]
    dynamical_correlations(sys,Δt = 0.03, nω = 120, ωmax = 12.0,observables = [obs;map(x-> x',obs)])
  else
    #obs = [[1,0,0]',[0,1,0]',[0,0,1]']
    #dynamical_correlations(sys,Δt = 0.03, nω = 60, ωmax = 8.0,observables = [obs;map(x-> conj.(x),obs)])
    dynamical_correlations(sys,Δt = 0.03, nω = 120, ωmax = 12.0)
  end

  weak_langevin = ImplicitMidpoint(0.03;λ = 1e-4, kT=temp)
  for l = 1:80; for j = 1:1000; step!(sys,langevin); end; add_sample!(dsc,sys;integrator = weak_langevin,max_lag_frac = 1.0); end;

  dsc
end

function kramers_kronig_matrix(ωs)
  dω = ωs[2] - ωs[1]
  n = length(ωs)
  [dω ./ (i == j ? Inf : (ωs[i] - ωs[j])) for i = 1:n, j = 1:n] ./ (im * π)
end
