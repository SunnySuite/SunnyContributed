using Sunny, LinearAlgebra, StaticArrays, GLMakie, Statistics

cryst = Crystal(I(3),[[0,0,0]],1)
sys = System(cryst,(1,1,1),[SpinInfo(1,S=1/2,g=2)],:dipole)
set_external_field!(sys,[0,0,1/sys.units.μB])

# Establish the common set of binning parameters to use for all histograms
dsc = dynamical_correlations(sys;Δt = 0.005,nω = 60,ωmax = 4.0)
common_params = unit_resolution_binning_parameters(dsc)#;negative_energies=true)
common_params.binwidth[4] = 0.38

swt = SpinWaveTheory(sys)
formula = intensity_formula(swt,:full;kernel = delta_function_kernel)
band_structure = formula.calc_intensity(swt,@SVector[0.,0,0])
lswt_bin = 1 + floor(Int64,(band_structure.dispersion[1] - common_params.binstart[4]) / common_params.binwidth[4])

# Compute static structure factor
isc = instant_correlations(sys)
add_sample!(isc,sys)
params_instant = unit_resolution_binning_parameters(isc)
is_static, counts = intensities_binned(isc,params_instant,intensity_formula(isc,:full))
elastic_bin = 1 + floor(Int64,(0.0 - common_params.binstart[4]) / common_params.binwidth[4])
@assert elastic_bin != lswt_bin

function get_classical_intensities_at_temperature(kT)
  println(kT)
  dsc = dynamical_correlations(sys;Δt = 0.005,nω = 60,ωmax = 4.0)
  formula = intensity_formula(dsc,:full;kT = kT)

  langevin = Langevin(0.05,λ=0.3,kT=kT)
  for j = 1:250
    for i = 1:1000
      step!(sys,langevin)
    end
    add_sample!(dsc,sys)
  end
  is, counts = intensities_binned(dsc,common_params,formula)
  is # Ignore counts (computing S², not S² per Hz)
end

# Evaluate classical intensities at successively lower temperatures
temps = 10 .^ range(1,-6,length = 20)
is_classicals = map(get_classical_intensities_at_temperature,temps)

is_sw = 0 .* is_classicals[1]
is_sw[1,1,1,lswt_bin] = band_structure.intensity[1] # 1-magnon part
is_sw[1,1,1,elastic_bin] = only(is_static) # 0-magnon part

# Compare and plot
bcs = axes_bincenters(common_params)
f = Figure()
ax1 = Axis(f[1,1],xlabel = "E [meV]",ylabel = "Intensity",title = "Spin Wave")
ax2 = Axis(f[2,1],xlabel = "E [meV]",ylabel = "Intensity", title = "Low Temp LL")
ax3 = Axis(f[3,1],xlabel = "E [meV]",ylabel = "Intensity", title = "Difference")
for i = 1:3#, j = 1:3
  j = i
  color = [:blue,:green,:red][i]
  marker = 'x'#['o','x','+'][j]
  markersize = 50
  sw = real(map(x->x[i,j],is_sw))[1,1,1,:]
  cl = real(map(x->x[i,j],is_classicals[end]))[1,1,1,:]
  scatter!(ax1,bcs[4],sw;color,marker,markersize)
  scatter!(ax2,bcs[4],cl;color,marker,markersize)
  scatter!(ax3,bcs[4],sw-cl;color,marker,markersize)
end
ax4 = Axis(f[4,1],xlabel = "log10 Temperature", ylabel = "log10 χ² SWT vs LL")
scatter!(ax4,log10.(temps),[log10(norm(diag.(is_sw .- is_classicals[i]))) for i = 1:length(temps)])
f

