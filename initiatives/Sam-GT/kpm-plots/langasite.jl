using Sunny, Test, GLMakie

a = b = 8.539
c = 5.2414
latvecs = lattice_vectors(a, b, c, 90, 90, 120)

types = ["Fe","Nb","Ba","Si","O","O","O"]
positions = [[0.24964,0,0.5],[0,0,0],[0.56598,0,0],[2/3,1/3,0.5220],[2/3,1/3,0.2162],[0.5259,0.7024,0.3536],[0.7840,0.9002,0.7760]]
langasite = Crystal(latvecs, positions, 150; types)
crystal = subcrystal(langasite, "Fe")
#crystal = Crystal(latvecs, [[0.24964,0,0.5]], 150)
latsize = (1,1,7)
sys = System(crystal, latsize, [SpinInfo(1; S=5/2, g=2)], :dipole; seed=5)
set_exchange!(sys, 0.85,  Bond(3, 2, [1,1,0]))   # J1
set_exchange!(sys, 0.24,  Bond(1, 3, [0,0,0]))   # J2
set_exchange!(sys, 0.017, Bond(1, 1, [0,0,1]))   # J4

ϵD = +1
ϵH = +1
ϵT = ϵD * ϵH

if ϵT == -1
    set_exchange!(sys, 0.053, Bond(2, 3, [-1,-1,1]))
    set_exchange!(sys, 0.24, Bond(3, 2, [1,1,1]))
elseif ϵT == 1
    set_exchange!(sys, 0.24, Bond(2, 3, [-1,-1,1]))
    set_exchange!(sys, 0.053, Bond(3, 2, [1,1,1]))
end

for i in 1:3
    θ = -2π*(i-1)/3
    set_spiral_order_on_sublattice!(sys, i; q=[0,0,1/7], axis=[0,0,1], S0=[cos(θ),sin(θ),0])
end

randomize_spins!(sys)
minimize_energy!(sys)
minimize_energy!(sys)
minimize_energy!(sys)
minimize_energy!(sys)

swt = SpinWaveTheory(sys)
formula_delta_perp = intensity_formula(swt, :perp; kernel=Sunny.delta_function_kernel)
formula = intensity_formula(swt, :full; kernel=Sunny.delta_function_kernel)
q = [0.41568,0.56382,0.76414]
disp, intensities = intensities_bands(swt, [q], formula)
SpinW_energies = [2.6267,2.6541,2.8177,2.8767,3.2458,3.3172,3.4727,3.7767,3.8202,3.8284,3.8749,3.9095,3.9422,3.9730,4.0113,4.0794,4.2785,4.4605,4.6736,4.7564,4.7865]
SpinW_intensities = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2999830079, -0.2999830079im, 0,0.2999830079im, 0.2999830079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3591387785, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5954018134, -0.5954018134im, 0,0.5954018134im, 0.5954018134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3708506016,1.3708506016im, 0, -1.3708506016im, 1.3708506016, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0511743697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0734875342, 0.0 + 0.0734875342im, 0, 0.0 - 0.0734875342im, 0.0734875342, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0577275935, -0.0577275935im, 0,0.0577275935im, 0.0577275935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.1733740706, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0338873034,0.0338873034im, 0, -0.0338873034im, 0.0338873034, 0, 0, 0, 0]

#@test isapprox(disp[:], reverse(SpinW_energies); atol=1e-3)

intensities_reshaped = reinterpret(reshape, ComplexF64, intensities)[:]
#@test isapprox(SpinW_intensities/Sunny.natoms(crystal), intensities_reshaped; atol=1e-7)

ωvals = range(-10,10,length = 1000)
σ = 0.4
lorentz_width = 0.1
kT = 0.
dipole_factor = Sunny.DipoleFactor(swt.observables)
formula = intensity_formula(swt, Sunny.required_correlations(dipole_factor); kernel=lorentzian(lorentz_width)) do k,ω,corrs
  Sunny.contract(corrs,k,dipole_factor) .* sign(ω) .* (1 .+ 1 ./ (exp.(ω ./ kT) .- 1))
end
#q = [0.,0.01,0.0]
lswt_intensities = intensities_broadened(swt,[q],ωvals,formula)
f = Figure(); ax = Axis(f[1,1]);

#temperature_factor = 1 .+ 1 ./ (exp.(ωvals ./ kT) .- 1)

#lswt_sym = (reverse(lswt_intensities[:]) + lswt_intensities[:])
#plot!(ax,ωvals,lswt_sym, color = :black)

modified_bose = Sunny.regularization_function.(ωvals,σ) .* (1 .+ Sunny.bose_function.(kT,ωvals))
lines!(ax,ωvals,modified_bose,color = :yellow)
scatter!(ax,ωvals,sign.(ωvals) .* (1 .+ Sunny.bose_function.(kT,ωvals)),color = :yellow)
ylims!(ax,-1,4)


for k = 1:3
  P = [300,1000,3000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :susceptibility)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = 'o')
end

for k = 1:2
  P = [300,1000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT = 0.,broadening, kernel = nothing, regularization_style = :none)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  scatter!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k],marker = 'x')
end

for k = 1:2
  P = [300,1000][k]
  broadening = (ω,xγ,σ) -> lorentzian(xγ-ω,lorentz_width)
  kpm_formula = Sunny.intensity_formula_kpm(swt,:perp;P,σ,kT,broadening, kernel = nothing, regularization_style = :cubic)
  kpmis = intensities_broadened(swt,[q],ωvals,kpm_formula)
  kpmis_old = kpm_intensities(swt, [q], ωvals,P,kT,σ,broadening; kernel = nothing, regularization_style = :cubic)
  lines!(ax,ωvals,kpmis[:],color = [:red,:green,:blue][k])
  #lines!(ax,ωvals,kpmis_old[:],color = [:red,:green,:blue][k])
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

