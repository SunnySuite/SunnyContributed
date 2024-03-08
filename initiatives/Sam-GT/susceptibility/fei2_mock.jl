using Sunny, LinearAlgebra, GLMakie, FFTW

a = b = 4.05012#hide
c = 6.75214#hide
latvecs = lattice_vectors(a, b, c, 90, 90, 120)#hide
positions = [[0,0,0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]#hide
types = ["Fe", "I", "I"]#hide
FeI2 = Crystal(latvecs, positions; types)#hide
cryst = subcrystal(FeI2, "Fe")#hide
sys = System(cryst, (4,4,4), [SpinInfo(1,S=1,g=2)], :SUN, seed=2)#hide
J1pm   = -0.236#hide
J1pmpm = -0.161#hide
J1zpm  = -0.261#hide
J2pm   = 0.026#hide
J3pm   = 0.166#hide
J′0pm  = 0.037#hide
J′1pm  = 0.013#hide
J′2apm = 0.068#hide
J1zz   = -0.236#hide
J2zz   = 0.113#hide
J3zz   = 0.211#hide
J′0zz  = -0.036#hide
J′1zz  = 0.051#hide
J′2azz = 0.073#hide
J1xx = J1pm + J1pmpm#hide
J1yy = J1pm - J1pmpm#hide
J1yz = J1zpm#hide
set_exchange!(sys, [J1xx 0.0 0.0; 0.0 J1yy J1yz; 0.0 J1yz J1zz], Bond(1,1,[1,0,0]))#hide
set_exchange!(sys, [J2pm 0.0 0.0; 0.0 J2pm 0.0; 0.0 0.0 J2zz], Bond(1,1,[1,2,0]))#hide
set_exchange!(sys, [J3pm 0.0 0.0; 0.0 J3pm 0.0; 0.0 0.0 J3zz], Bond(1,1,[2,0,0]))#hide
set_exchange!(sys, [J′0pm 0.0 0.0; 0.0 J′0pm 0.0; 0.0 0.0 J′0zz], Bond(1,1,[0,0,1]))#hide
set_exchange!(sys, [J′1pm 0.0 0.0; 0.0 J′1pm 0.0; 0.0 0.0 J′1zz], Bond(1,1,[1,0,1]))#hide
set_exchange!(sys, [J′2apm 0.0 0.0; 0.0 J′2apm 0.0; 0.0 0.0 J′2azz], Bond(1,1,[1,2,1]))#hide
D = 2.165#hide
S = spin_operators(sys, 1)#hide
set_onsite_coupling!(sys, -D*S[3]^2, 1)#hide
sys

Δt = 0.05/D    # Should be inversely proportional to the largest energy scale
               # in the system. For FeI2, this is the easy-axis anisotropy,
               # `D = 2.165` (meV). The prefactor 0.05 is relatively small,
               # and achieves high accuracy.
kT = 0.2       # Temperature of the thermal bath (meV).
λ = 0.1        # This value is typically good for Monte Carlo sampling,
               # independent of system details.

langevin = Langevin(Δt; kT, λ);

randomize_spins!(sys)
for _ in 1:20_000
    step!(sys, langevin)
end

plot_spins(sys; color=[s[3] for s in sys.dipoles])

sys_large = resize_supercell(sys, (16,16,4)) # 16x16x4 copies of the original unit cell
plot_spins(sys_large; color=[s[3] for s in sys_large.dipoles])

kT = 3.5 * meV_per_K     # 3.5K ≈ 0.30 meV
langevin.kT = kT
println("Thermalize")
@time for _ in 1:10_000
    step!(sys_large, langevin)
end

sc = dynamical_correlations(sys_large; Δt=2Δt, nω=120, ωmax=7.5)

weak_langevin = ImplicitMidpoint(0.03;λ = 1e-7, kT=langevin.kT)
#add_sample!(sc, sys_large; alg = :window)        # Accumulate the sample into `sc`
add_sample!(sc, sys_large; alg = :no_window, integrator = weak_langevin)        # Accumulate the sample into `sc`

for _ in 1:8
  println("Sampling...")
    for _ in 1:1000               # Enough steps to decorrelate spins
        step!(sys_large, langevin)
    end
    #add_sample!(sc, sys_large; alg = :window)    # Accumulate the sample into `sc`
    add_sample!(sc, sys_large; alg = :no_window, integrator = weak_langevin)    # Accumulate the sample into `sc`
end

display(sc)

formula = intensity_formula(sc, :trace; kT)

qs = [[0, 0, 0], [0.5, 0.5, 0.5]]
is = intensities_interpolated(sc, qs, formula; interpolation = :round)

ωs = available_energies(sc)
fig = lines(ωs, is[1,:]; axis=(xlabel="meV", ylabel="Intensity"), label="(0,0,0)")
lines!(ωs, is[2,:]; label="(π,π,π)")
axislegend()
fig

formfactors = [FormFactor("Fe2"; g_lande=3/2)]
new_formula = intensity_formula(sc, :trace; kT, formfactors = formfactors)

points = [[0,   0, 0],  # List of wave vectors that define a path
          [1,   0, 0],
          [0,   1, 0],
          [1/2, 0, 0],
          [0,   1, 0],
          [0,   0, 0]]
density = 40
path, xticks = reciprocal_space_path(cryst, points, density);

is_interpolated = intensities_interpolated(sc, path, new_formula;
    interpolation = :linear,       # Interpolate between available wave vectors
);
# Add artificial broadening
is_interpolated_broadened = broaden_energy(sc, is, (ω, ω₀)->lorentzian(ω-ω₀, 0.05));

cut_width = 0.3
density = 15
paramsList, markers, ranges = reciprocal_space_path_bins(sc,points,density,cut_width);

total_bins = ranges[end][end]
energy_bins = paramsList[1].numbins[4]
is_binned = zeros(Float64,total_bins,energy_bins)
integrated_kernel = integrated_lorentzian(0.05) # Lorentzian broadening
for k in eachindex(paramsList)
    bin_data, counts = intensities_binned(sc,paramsList[k], new_formula;
        integrated_kernel = integrated_kernel
    )
    is_binned[ranges[k],:] = bin_data[:,1,1,:] ./ counts[:,1,1,:]
end

fig = Figure()
ax_top = Axis(fig[1,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
ax_bottom = Axis(fig[2,1],ylabel = "meV",xticks = (markers, string.(points)),xticklabelrotation=π/8,xticklabelsize=12)

heatmap!(ax_top,1:size(is_interpolated,1), ωs, is_interpolated;
    colorrange=(0.0,0.07),
)

heatmap!(ax_bottom,1:size(is_binned,1), ωs, is_binned;
    colorrange=(0.0,0.05),
)

fig

ff = Figure()
scdata = copy(sc.data);
real_data = ifft(sc.data,7);
time_lag = abs.(fftfreq(size(sc.data,7),size(sc.data,7)))
time_T = maximum(time_lag)
fracs = range(0.01,1,length=12)
for j = 1:12
  partial_data = copy(real_data)
  #lin = ones(Float64,size(sc.data,7))
  #lin .*= cos.((1/fracs[j])*pi*time_lag/(2time_T)).^2
  #lin[(time_lag/time_T) .> fracs[j]] .= 0
  partial_data .*= reshape(cos.((1/fracs[j])*pi*time_lag/(2time_T)).^2,(1,1,1,1,1,1,length(time_lag)))
  partial_data[:,:,:,:,:,:,(time_lag/time_T) .> fracs[j]] .= 0
  #partial_data[:,:,:,:,:,:,(time_lag/time_T) .< fracs[j]] .= real_data[:,:,:,:,:,:,(time_lag/time_T) .< fracs[j]]
  sc.data .= fft(partial_data,7)
  #is_interpolated = intensities_interpolated(sc, path, new_formula;
    #interpolation = :round,       # Interpolate between available wave vectors
  #);

  is_binned = zeros(Float64,total_bins,energy_bins)
  integrated_kernel = integrated_lorentzian(0.05) # Lorentzian broadening
for k = eachindex(paramsList)
      bin_data, counts = intensities_binned(sc,paramsList[k], new_formula;integrated_kernel = integrated_kernel)
      is_binned[ranges[k],:] = bin_data[:,1,1,:] ./ counts[:,1,1,:]
end

  i,k = divrem(j-1,4)
  ax = Axis(ff[i+1,k+1])
  heatmap!(ax,1:size(is_binned,1), ωs, is_binned;
      colorrange=(0.0,0.07),
  )
  #lines!(ax,lin)
end
sc.data .= scdata;
ff

#plot(abs.(ifft(sc.data[:,1,1,1,1,1,:],2)))
