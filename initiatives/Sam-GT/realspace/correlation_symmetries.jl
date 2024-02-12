using Test, LinearAlgebra, FFTW, Sunny

function mk_dsc(;mode = :SUN)
  cryst = Sunny.cubic_crystal()
  sys = System(cryst, (100,1,1), [SpinInfo(1;S=1/2,g=2)],mode)
  set_exchange!(sys,1.,Bond(1,1,[1,0,0]))
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 500)

  langevin = Langevin(0.05;λ = 0.1, kT=0.3)
  for i = 1:10000
    step!(sys,langevin)
  end

  dsc = if mode == :SUN
    S = spin_matrices(1/2)
    obs = [S[1],S[2],S[3]]
    dynamical_correlations(sys,Δt = 0.05, nω = 80, ωmax = 4.0,observables = [obs;map(x-> x',obs)])
  else
    obs = [[1,0,0]',[0,1,0]',[0,0,1]']
    dynamical_correlations(sys,Δt = 0.05, nω = 80, ωmax = 4.0,observables = [obs;map(x-> conj.(x),obs)])
  end

  for l = 1:20; for j = 1:1000; step!(sys,langevin); end; add_sample!(dsc,sys); end;

  dsc
end

# Swaps x_-q y_q to y_-q x_q
function sym_swap_operators(dat)
  p = [1,4,7,2,5,8,3,6,9] # Swaps x_-q y_q to y_q x_-q
  sym_q(dat[p,:,:,:,:,:,:]) # y_q x_-q to y_-q x_q
end

function sym_q(dat)
  for j = [4,5,6]
    dat = reverse_frequency_dimension(dat,j)
  end
  dat
end

sym_omega(dat) = reverse_frequency_dimension(dat,7)


function reverse_frequency_dimension(M,d)
  l = fftshift(M,d)
  l = if iseven(d)
    circshift(l,ntuple(i -> i == d ? -1 : 0,length(size(M))))
  else
    l
  end
  ifftshift(reverse(l,dims=d),d)
end

function assert_symmetries(dsc)
  dat = dsc.data
  
  obss = dsc.observables.observables
  dagger_map = [findfirst([Matrix(B) ≈ Matrix(A)' && i != j for (j,B) in enumerate(obss)]) for (i,A) in enumerate(obss)]
  println(dagger_map)

  # For each correlation, find its dagger partner based on the dagger map
  corrs = dsc.observables.correlations
  dagger_map_corr = zeros(Int64,length(corrs))
  swap_map_corr = zeros(Int64,length(corrs))
  for (ci,i) in corrs
    dagger_map_corr[i] = corrs[CartesianIndex(dagger_map[ci[1]],dagger_map[ci[2]])]
    swap_map_corr[i] = corrs[CartesianIndex(ci[2],ci[1])]
  end

  # The [†] symmetry takes S_{A,B} to S_{A†,B†}.
  # But in our case, the observables in question are S_{A_q,B_-q}, and
  # the dagger takes this to S_{A†_-q,B†_q}. In other words, it splits
  # into [†] ≡ [dq] where [d] means "apply dagger to the original (non-FT)
  # operators".
  #
  # The `dagger_map_corr` implements [d], so this is an implementation of [†]:
  sym_dagger_non_FT_operators(dat) = dat[dagger_map_corr,:,:,:,:,:,:] # [d]
  sym_dagger_operators(dat) = sym_q(sym_dagger_non_FT_operators(dat)) # [†] = [dq]

  # The `swap_map_corrs` implements:
  #
  #   S′ ∼ [x_-q y_q to y_-q x_q]
  # 
  # but what we actually want is swapping of the A_q operators:
  #
  #   S ∼ [x_-q y_q → y_q x_-q]
  #
  # which we construct as S = qS′ where
  #
  #   q ∼ [y_q x_-q → y_-q x_q]
  #
  sym_swap_operators(dat) = sym_q(dat[swap_map_corr,:,:,:,:,:,:])

  # This should always hold (due to bilateral fourier transform)
  @test conj.(sym_dagger_operators(sym_swap_operators(dat))) ≈ dat # [C†S]

  real_obs = typeof(dsc).parameters[1] == 0 ? all(isreal.(obss)) : all(ishermitian.(obss))
  if real_obs
    println("All observables are real! Verifying additional identities")

    # Most directly, real observables shouldn't care when you dagger them!
    # In our case, daggering swaps the q label,
    #
    #              (apply †)
    #   A_q, B_-q ----------→ [A_q]†, [B_-q]† = A†_-q, B†_q
    #
    # so we have to swap it back.
    @test sym_q(sym_dagger_operators(dat)) ≈ dat # [q†] (for real observables)
    # A simpler way to state this is to use [†] ≡ [qd]:
    @test sym_dagger_non_FT_operators(dat) ≈ dat # [d] (for real observables)

    # Next, the spectra of correlations between real observables should look like
    # real spectra: the -(q,ω) part should be obtainable from +(q,ω) by conjugation.
    #
    # N.B.: this doesn't apply to ω or to q individually, only to the combined spatiotemporal index!
    #
    @test conj.(sym_q(sym_omega(dat))) ≈ dat # [Cqω]

    # Test the derived symmetries for completeness:

    # Combine 2/3 of the previous symmetries:
    @test conj.(sym_swap_operators(sym_q(dat))) ≈ dat # [q†] + [C†S] → [CSq]
    @test sym_dagger_operators(sym_q(sym_omega(sym_swap_operators(dat)))) ≈ dat # [Cqω] + [C†S] → [†qωS]
    @test conj.(sym_dagger_operators(sym_omega(dat))) ≈ dat # [Cqω] + [q†] → [C†ω]

    # Combine 3/3 of the previous symmetries:
    # (equivalent to Detailed Balance [exp(βω)Sω] at infinite temperature β = 0)
    @test sym_swap_operators(sym_omega(dat)) ≈ dat # [C†S] + [C†ω] → [Sω]
  else
    println("Non-real observables detected. Not verifying identities exclusive to real observables")
  end
  println("Success!")
end

function detail_balance(dsc;dat = dsc.data)
  # Detailed Balance. We would like for this to hold, but it may not
  ω = Sunny.axes_bincenters(unit_resolution_binning_parameters(dsc;negative_energies=true))[4]
  ω = reshape(ω,1,1,1,1,1,1,length(ω))
  β = 1/0.3
  detailed_balance_dat = exp.(β .* ω) .* sym_swap_operators(sym_omega(dat))
end


function to_chi_prime_prime(dsc;β)
  dat = dsc.data
  ω = Sunny.axes_bincenters(unit_resolution_binning_parameters(dsc;negative_energies=true))[4]
  ω = reshape(ω,1,1,1,1,1,1,length(ω))
  dat .* (1 .- exp.(-β .* ω)) ./ 2
end
