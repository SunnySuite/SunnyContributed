using Sunny, Test, GLMakie, StaticArrays

a = b = 8.539
c = 5.2414
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
crystal = Crystal(latvecs,[[0.,0,0]],1)
latsize = (2,1,1)
sys = System(crystal, latsize, [SpinInfo(1; S=5/2, g=2)], :dipole; seed=5)
set_exchange!(sys, 0.85,  Bond(1, 1, [1,0,0]))   # J1
set_onsite_coupling!(sys, S -> 0.3 * S[3]^2,1)

#sys.dipoles[1] = SVector{3}([0,0,1])
#sys.dipoles[2] = SVector{3}([0,0,-1])
randomize_spins!(sys)
minimize_energy!(sys)

swt = SpinWaveTheory(sys)
formula_delta_perp = intensity_formula(swt, :perp; kernel=Sunny.delta_function_kernel)
formula = intensity_formula(swt, :full; kernel=Sunny.delta_function_kernel)
q = [0.44,0.,0.]
disp, intensities = intensities_bands(swt, [q], formula)

ωvals = range(-10,10,length = 100)
σ = 10.0
lorentz_width = 0.8
kT = 0.0
dipole_factor = Sunny.DipoleFactor(swt.observables)
formula = intensity_formula(swt, Sunny.required_correlations(dipole_factor); kernel=lorentzian(lorentz_width)) do k,ω,corrs
  Sunny.contract(corrs,k,dipole_factor) .* sign(ω) .* (1 .+ 1 ./ (exp.(ω ./ kT) .- 1))
end
lswt_intensities = intensities_broadened(swt,[q],ωvals,formula)
f = Figure(); ax = Axis(f[1,1]);

#temperature_factor = 1 .+ 1 ./ (exp.(ωvals ./ kT) .- 1)

#lswt_sym = (reverse(lswt_intensities[:]) + lswt_intensities[:])
#plot!(ax,ωvals,lswt_sym, color = :black)

modified_bose = Sunny.regularization_function.(ωvals,σ) .* (1 .+ Sunny.bose_function.(kT,ωvals)) .* (ωvals .< σ)
lines!(ax,ωvals,modified_bose,color = :yellow)
scatter!(ax,ωvals,sign.(ωvals) .* (1 .+ Sunny.bose_function.(kT,ωvals)),color = :yellow)
ylims!(ax,-15,15)

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :srf)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = 'o')
end

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT = 0.,broadening, kernel = nothing, regularization_style = :none)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = 'x')
end

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :cubic)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  kpmis_old = kpm_intensities(swt, [q], ωvals,P,kT,σ,broadening; kernel = nothing, regularization_style = :cubic)
  lines!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k])
  #lines!(ax,ωvals,kpmis_old[:],color = [:red,:green,:blue][k])
end

for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :susceptibility)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = '+')
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

