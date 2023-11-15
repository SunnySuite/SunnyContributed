using Sunny, Test, GLMakie, StaticArrays

a = b = c = 1
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
na = 12
positions = Vector{Float64}[]
spinfo = Sunny.SpinInfo[]
for i = 1:na
  push!(positions,[rand(),rand(),rand()])
  push!(spinfo,SpinInfo(i;S=3/2,g = randn()))
end
crystal = Crystal(latvecs,positions,1)

latsize = (1,1,1)
sys = System(crystal, latsize, spinfo, :dipole; seed=5)
for i = 1:800
  ii = rand(1:na)
  jj = rand(1:na)
  if ii == jj
    continue
  end
  set_exchange!(sys, randn(3,3),  Bond(ii, jj, [rand(0:1),rand(0:1),rand(0:1)]))
end
randomize_spins!(sys)
minimize_energy!(sys; maxiters = 3000)

swt = SpinWaveTheory(sys)
formula_delta_perp = intensity_formula(swt, :perp; kernel=Sunny.delta_function_kernel)
formula = intensity_formula(swt, :full; kernel=Sunny.delta_function_kernel)
q = [0.44,0.3,0.]
disp, intensities = intensities_bands(swt, [k * q for k = range(-1,1,length=30)], formula_delta_perp)
plot_band_intensities(disp,intensities)

ωvals = range(-100,100,length = 1000)
σ = 0.8
lorentz_width = 2.0
kT = 0.0
dipole_factor = Sunny.DipoleFactor(swt.observables)
formula = intensity_formula(swt, Sunny.required_correlations(dipole_factor); kernel=lorentzian(lorentz_width)) do k,ω,corrs
  Sunny.contract(corrs,k,dipole_factor) .* (1 .+ 1 ./ (exp.(ω ./ kT) .- 1))
end
lswt_intensities = intensities_broadened(swt,[q],ωvals,formula)
f = Figure(); ax = Axis(f[1,1]);

modified_bose = Sunny.regularization_function.(ωvals,σ) .* (1 .+ Sunny.bose_function.(kT,ωvals)) .* (ωvals .< σ)
#lines!(ax,ωvals,modified_bose,color = :yellow)
#scatter!(ax,ωvals,sign.(ωvals) .* (1 .+ Sunny.bose_function.(kT,ωvals)),color = :yellow)

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT = 0.,broadening, kernel = nothing, regularization_style = :cubic)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = 'x')
end

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :susceptibility)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  #kpmis_old = kpm_intensities(swt, [q], ωvals,P,kT,σ,broadening; kernel = nothing, regularization_style = :cubic)
  lines!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k])
  #lines!(ax,ωvals,kpmis_old[:],color = [:red,:green,:blue][k])
  if k == 3
    lines!(ax,-ωvals,-kpmis[:],color = :blue,linestyle = :dash)
  end
end
lines!(ax,ωvals,lswt_intensities[:], color = :black)



#=
begin
  P = 300
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,σ)
  kpmis = kpm_intensities(swt, [q], ωvals,P,kT,σ,broadening; kernel = nothing, regularization_style = :cubic)
  scatter!(ax,ωvals,kpmis[:],color = :orange)
end
=#
#scatter!(ax,ωvals,(lswt_intensities[:] ./ kpmis[:]) ./ pi,color = :pink)

display(f)

