using Sunny, LinearAlgebra, ProgressMeter, GLMakie, Statistics

cryst = Crystal(I(3), [[0,0,0]], 1)
spin_S = 3/2
g_factor = 2.3
sys = System(cryst, (1,1,1), [SpinInfo(1; S = spin_S, g = g_factor)], :dipole, units = Units.theory)

Bz = 1
set_external_field!(sys,[0,0,Bz])

n_res = 1000
thetas = range(-pi,pi,length = n_res)

beta = 1.0

es = zeros(n_res)
boltzmann_weight = zeros(n_res)

es_h = zeros(n_res)
boltzmann_weight_h = zeros(n_res)

e0 = -Bz * spin_S * g_factor

for i = 1:n_res
  set_dipole!(sys,[sin(thetas[i]),0,cos(thetas[i])],(1,1,1,1))
  es[i] = energy(sys)
  boltzmann_weight[i] = exp(-beta * (es[i] - e0))

  # Harmonic approximation
  es_h[i] = Bz * spin_S * g_factor * (- 1 + thetas[i]^2 / 2)
  boltzmann_weight_h[i] = exp(-beta * (es_h[i] - e0))
end
partition_function = sum(boltzmann_weight)
partition_function_h = sum(boltzmann_weight_h)

f, ax, l = lines(thetas,es,color = :blue)
display(f)
l_h = lines!(thetas,es_h,color = :blue, linestyle = :dash)

l_b = lines!(thetas,boltzmann_weight ./ maximum(boltzmann_weight),color = :orange)
l_b_h = lines!(thetas,boltzmann_weight_h ./ maximum(boltzmann_weight),color = :orange, linestyle = :dash)
ax.xlabel = "θ"
ax.ylabel = "meV"
Legend(f[1,2],[l,l_h,l_b,l_b_h],["Energy","Energy (harmonic approx)","P(x), fraction of maximum","P(x) (harmonic), same scale"])
ylims!(ax,(minimum(es)*1.1,maximum(es)*1.1))

expected_Sx_sq = spin_S^2 * sum(boltzmann_weight .* sin.(thetas).^2) ./ partition_function
expected_Sx_sq_h = spin_S^2 * sum(boltzmann_weight_h .* sin.(thetas).^2) ./ partition_function_h

expected_Sz = spin_S * sum(boltzmann_weight .* cos.(thetas)) ./ partition_function
expected_Sz_h = spin_S * sum(boltzmann_weight_h .* cos.(thetas)) ./ partition_function_h

langevin = Langevin(0.04,λ = 0.1, kT = 1/beta)

sys = repeat_periodically(sys,(4,4,4))

dsc = dynamical_correlations(sys;Δt = 0.04, nω = 200, ωmax = 30.0,apply_g = false)

nsample = 1
prog = Progress(nsample,"Sampling")
for l = 1:nsample
  for j = 1:1000
    step!(sys,langevin)
  end
  add_sample!(dsc,sys;integrator = langevin)
  next!(prog)
end
finish!(prog)

params = unit_resolution_binning_parameters(dsc;negative_energies = true)

formula = intensity_formula(dsc,:full)#,kT = 1/beta)

isfull, counts = intensities_binned(dsc,params,formula)

sum_rules = real(sum(map(diag,isfull)))

lineshape = real(sum(map(x -> x[1,2],isfull),dims = [1,2,3])[:])
bcs = axes_bincenters(params)
ax = Axis(f[2,1])
oscillator_frequency = Bz * g_factor
ix_peak = findmin(abs.(bcs[4] .- oscillator_frequency))[2]
ixs = (ix_peak - 10):(ix_peak + 10)
ixs_negative = length(bcs[4]) .- ixs .+ 1

@assert bcs[4][ixs] ≈ - bcs[4][ixs_negative]

pv = plot!(ax,bcs[4][ixs],lineshape[ixs],color = :blue,marker = 'o')
nv = plot!(ax,bcs[4][ixs],lineshape[ixs_negative],color = :red, marker = 'x')
vlines!(ax,oscillator_frequency)
ax.xlabel = "ω [meV]"
ax.ylabel = "Sxy"
Legend(f[2,2],[pv,nv],["ω > 0", "ω < 0"])

nss = Sunny.number_to_simple_string
println("β = $(nss(beta,digits = 3)), kT = $(nss(1/beta,digits=3))")
println("Sxx and Syy sum rules similar to within $(nss(100 * abs(sum_rules[1] - sum_rules[2])/sum_rules[1],digits = 4))%")
println("Sxx ≈ Syy = $((sum_rules[1] + sum_rules[2])/2)")
println("Szz = $(sum_rules[3])")
println("Computed: $(nss(sum(sum_rules),digits = 3))")
println("True:     $(nss(spin_S*(spin_S+1),digits = 3)) = S(S+1)")
println()
println("From stat mech: ⟨Sx²⟩ → $(nss(expected_Sx_sq,digits = 4))   ($(expected_Sx_sq_h > expected_Sx_sq ? "+" : "")$(nss(100*(expected_Sx_sq_h - expected_Sx_sq)/expected_Sx_sq,digits = 4))% for harmonic)")
println("                ⟨Sz⟩ →  $(nss(expected_Sz,digits = 4))   ($(expected_Sz_h > expected_Sz ? "+" : "")$(nss(100*(expected_Sz_h - expected_Sz)/expected_Sz,digits = 4))% for harmonic)")
println("Oscillator frequency: ω₀ = $(nss(oscillator_frequency,digits = 2))")
c2q_plus = Sunny.classical_to_quantum(oscillator_frequency,1/beta)
c2q_minus = Sunny.classical_to_quantum(-oscillator_frequency,1/beta)
println("c2q factor: g(βω₀) = $(nss(c2q_plus,digits = 3)) and g(-βω₀) = $(nss(c2q_minus,digits = 3)); bias Z = $(nss(log(c2q_minus/c2q_plus),digits = 3)))")
corrected_var = (c2q_plus + c2q_minus)/2 * expected_Sx_sq
println("[g(βω₀) + g(-βω₀)]/2 * ⟨Sx²⟩ = $(nss(corrected_var,digits = 5))")

 #f = Figure(); display(f); ax = Axis(f[1,1]); for i = 1:3; lines!(real(dsc.samplebuf[i,1,1,1,1,:])); end



