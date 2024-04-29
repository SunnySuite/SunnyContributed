using LsqFit

a = b = 4.05012  # Lattice constants for triangular lattice
c = 6.75214      # Spacing in the z-direction
latvecs = lattice_vectors(a, b, c, 90, 90, 120) 
positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]] 
types = ["Fe", "I", "I"]
FeI2 = Crystal(latvecs, positions; types)
cryst = subcrystal(FeI2, "Fe")
sys = System(cryst, (4, 4, 4), [SpinInfo(1, S=1, g=2)], :SUN)

J1pm   = -0.236
J1pmpm = -0.161
J1zpm  = -0.261
J2pm   = 0.026
J3pm   = 0.166
J′0pm  = 0.037
J′1pm  = 0.013
J′2apm = 0.068

J1zz   = -0.236
J2zz   = 0.113
J3zz   = 0.211
J′0zz  = -0.036
J′1zz  = 0.051
J′2azz = 0.073

J1xx = J1pm + J1pmpm
J1yy = J1pm - J1pmpm
J1yz = J1zpm

set_exchange!(sys, [J1xx   0.0    0.0;
                    0.0    J1yy   J1yz;
                    0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
set_exchange!(sys, [J2pm   0.0    0.0;
                    0.0    J2pm   0.0;
                    0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
set_exchange!(sys, [J3pm   0.0    0.0;
                    0.0    J3pm   0.0;
                    0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
set_exchange!(sys, [J′0pm  0.0    0.0;
                    0.0    J′0pm  0.0;
                    0.0    0.0    J′0zz], Bond(1,1,[0,0,1]))
set_exchange!(sys, [J′1pm  0.0    0.0;
                    0.0    J′1pm  0.0;
                    0.0    0.0    J′1zz], Bond(1,1,[1,0,1]))
set_exchange!(sys, [J′2apm 0.0    0.0;
                    0.0    J′2apm 0.0;
                    0.0    0.0    J′2azz], Bond(1,1,[1,2,1]))

D = 2.165
set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)

#randomize_spins!(sys)
#minimize_energy!(sys)

sys_min = reshape_supercell(sys, [1 0 0; 0 2 1; 0 -2 1])
randomize_spins!(sys_min)
minimize_energy!(sys_min)

sys = repeat_periodically(sys_min, (2,1,1))

swt = SpinWaveTheory(sys_min)
path = [[0,0,0.],[0.5,0.0,0.0]]
disp = dispersion(swt, path);
disps_sw = [disp[1,8],disp[2,4]]

nw = 1000

langevin = Langevin(0.05,λ = 0.3,kT = 0.01)

f = Figure()
ax1 = Axis(f[2,1],ylabel = "Intensity",xlabel = "ω")
vlines!(ax1,disps_sw[1])

ax2 = Axis(f[2,2],xlabel = "ω")
vlines!(ax2,disps_sw[2])
display(f)

logTimesteps = range(-4,-2,length = 16)
view_range = 0.05
axf1 = Axis(f[1,1],title = "q = $(path[1])",xlabel = "peak ω ± linewidth")
vlines!(axf1,disps_sw[1])
xlims!(axf1,disps_sw[1] - view_range,disps_sw[1] + view_range)
ylims!(axf1,extrema(logTimesteps))
axf2 = Axis(f[1,2],title = "q = $(path[2])",xlabel = "peak ω ± linewidth")
vlines!(axf2,disps_sw[2])
xlims!(axf2,disps_sw[2] - view_range,disps_sw[2] + view_range)
ylims!(axf2,extrema(logTimesteps))
Colorbar(f[1,3],colorrange = extrema(logTimesteps),label = "log10 Δt")

#plot(fftfreq(1200,2pi/(dsc.Δt * dsc.measperiod)),abs.(fft(real(dsc.samplebuf[1,1,1,1,1,:]))))
@. lorentzian_model(x, p) = p[3] * (1 / (pi * p[1] * (1 + ((x - p[2]) / p[1])^2)) + 1 / (pi * p[1] * (1 + ((x + p[2]) / p[1])^2)))
function lorentzian_params(xs,ys;params0)
  curve_fit(lorentzian_model,xs,ys,params0)
end

timesteps = 10 .^ logTimesteps
for l = reverse(1:length(timesteps))
  println("dt = $(timesteps[l])")
  dsc = dynamical_correlations(sys;Δt = timesteps[l], nω = nw, ωmax = 5.0)
  @time for j = 1:1
    for i = 1:round(Int64,((100 * 0.05) / langevin.Δt))
      step!(sys,langevin)
    end
    add_sample!(dsc,sys)
  end

  is, counts = intensities_binned(dsc,unit_resolution_binning_parameters(dsc;negative_energies = true),intensity_formula(dsc,:trace;kT = langevin.kT))

  if any(isnan.(is))
    println("Warning: NaN value")
  end

  energies = sort(fftfreq(size(dsc.data,7),2pi/(dsc.Δt * dsc.measperiod)))
  near_low_freq_range = findall(disps_sw[1] - view_range .< energies .< disps_sw[1] + view_range)
  near_hi_freq_range = findall(disps_sw[2] - view_range .< energies .< disps_sw[2] + view_range)

  xx = energies[near_low_freq_range]
  yy = abs.(is[1,1,1,:])[near_low_freq_range]
  lines!(ax1,xx,yy,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  f = lorentzian_params(xx,yy;params0 = [0.001,disps_sw[1],1.0])
  println(f.param)
  xx_fine = range(minimum(xx),maximum(xx),length = 250)
  #lines!(ax1,xx_fine,lorentzian_model(xx_fine,f.param),color = :red)
  scatter!(ax1,f.param[2],1.0,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  xlims!(ax1,disps_sw[1] - view_range, disps_sw[1] + view_range)

  scatter!(axf1,f.param[2],log10(dsc.Δt),color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf1,f.param[2] + f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf1,f.param[2] - f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))

  ts = Sunny.suggest_timestep_aux(sys,ImplicitMidpoint(timesteps[l]);tol = abs(disps_sw[1] - f.param[2]))
  println(ts)
  scatter!(axf1,f.param[2],log10(ts),color = :red)

  xx = energies[near_hi_freq_range]
  yy = abs.(is[2,2,1,:])[near_hi_freq_range]
  lines!(ax2,xx,yy,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  f = lorentzian_params(xx,yy;params0 = [0.001,disps_sw[2],1.0])
  scatter!(ax2,f.param[2],1.0,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  xlims!(ax2,disps_sw[2] - view_range,disps_sw[2] + view_range)
  scatter!(axf2,f.param[2],log10(dsc.Δt),color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf2,f.param[2] + f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf2,f.param[2] - f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))

  ts = Sunny.suggest_timestep_aux(sys,ImplicitMidpoint(timesteps[l]);tol = abs(disps_sw[2] - f.param[2]))
  println(ts)
  scatter!(axf2,f.param[2],log10(ts),color = :red)
  sleep(0.01)
end


