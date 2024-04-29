using LsqFit

cryst = Crystal(I(3),[[0,0,0]],1)
#cryst = Crystal(I(3),[[0,0,0],[0.5,0.5,0.5]],1)
sys = System(cryst,(1,1,2),[SpinInfo(1,S=1,g=1)],:dipole)
#sys = System(cryst,(1,1,2),[SpinInfo(1,S=1,g=1),SpinInfo(2,S=1,g=1)],:dipole)

B_z_ext = 1.0 / sys.units.μB
set_external_field!(sys,[0,0,B_z_ext])
set_exchange!(sys,-1/4,Bond(1,1,[0,0,1]))
#set_exchange!(sys,-1.,Bond(1,2,[0,0,0]))

nw = 2400

langevin = Langevin(0.05,λ = 0.3,kT = 0.0001)

f = Figure()
ax1 = Axis(f[2,1],ylabel = "Intensity",xlabel = "ω")
vlines!(ax1,1.0)

ax2 = Axis(f[2,2],xlabel = "ω")
vlines!(ax2,2.0)
display(f)

view_range = 0.02
axf1 = Axis(f[1,1],title = "Bottom of band",xlabel = "peak ω ± linewidth")
xlims!(axf1,1.0 - view_range,1.0 + view_range)
ylims!(axf1,extrema(logTimesteps))
axf2 = Axis(f[1,2],title = "Top of band",xlabel = "peak ω ± linewidth")
xlims!(axf2,2.0 - view_range,2.0 + view_range)
ylims!(axf2,extrema(logTimesteps))
Colorbar(f[1,3],colorrange = extrema(logTimesteps),label = "log10 Δt")

#plot(fftfreq(1200,2pi/(dsc.Δt * dsc.measperiod)),abs.(fft(real(dsc.samplebuf[1,1,1,1,1,:]))))
@. lorentzian_model(x, p) = p[3] * (1 / (pi * p[1] * (1 + ((x - p[2]) / p[1])^2)) + 1 / (pi * p[1] * (1 + ((x + p[2]) / p[1])^2)))
function lorentzian_params(xs,ys;params0)
  curve_fit(lorentzian_model,xs,ys,params0)
end

logTimesteps = range(-2.5,-0.5,length = 20)
timesteps = 10 .^ logTimesteps
for l = reverse(1:length(timesteps))
  println("dt = $(timesteps[l])")
  dsc = dynamical_correlations(sys;Δt = timesteps[l], nω = nw, ωmax = 3.0)
  for j = 1:250
    for i = 1:round(Int64,((1000 * 0.05) / langevin.Δt))
      step!(sys,langevin)
    end
    add_sample!(dsc,sys)
  end

  is, counts = intensities_binned(dsc,unit_resolution_binning_parameters(dsc;negative_energies = false),intensity_formula(dsc,:trace;kT = langevin.kT))

  if any(isnan.(is))
    println("Warning: NaN value")
  end

  energies = fftfreq(size(dsc.data,7),2pi/(dsc.Δt * dsc.measperiod))
  near_low_freq_range = findall(1.0 - view_range .< energies .< 1.0 + view_range)
  near_hi_freq_range = findall(2.0 - view_range .< energies .< 2.0 + view_range)

  xx = energies[near_low_freq_range]
  yy = abs.(is[1,1,1,:])[near_low_freq_range]
  lines!(ax1,xx,yy,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  f = lorentzian_params(xx,yy;params0 = [0.001,1.0,1.0])
  println(f.param)
  xx_fine = range(minimum(xx),maximum(xx),length = 250)
  #lines!(ax1,xx_fine,lorentzian_model(xx_fine,f.param),color = :red)
  scatter!(ax1,f.param[2],1.0,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  xlims!(ax1,1.0 - view_range, 1.0 + view_range)

  scatter!(axf1,f.param[2],log10(dsc.Δt),color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf1,f.param[2] + f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf1,f.param[2] - f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))

  xx = energies[near_hi_freq_range]
  yy = abs.(is[1,1,2,:])[near_hi_freq_range]
  lines!(ax2,xx,yy,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  f = lorentzian_params(xx,yy;params0 = [0.001,2.0,1.0])
  scatter!(ax2,f.param[2],1.0,color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  xlims!(ax2,2.0 - view_range,2.0 + view_range)
  scatter!(axf2,f.param[2],log10(dsc.Δt),color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf2,f.param[2] + f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  scatter!(axf2,f.param[2] - f.param[1],log10(dsc.Δt),marker = '|',color = log10(dsc.Δt),colorrange = extrema(logTimesteps))
  sleep(0.01)
end


