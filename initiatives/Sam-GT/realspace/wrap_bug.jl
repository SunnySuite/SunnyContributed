using Sunny, GLMakie, LinearAlgebra, FFTW, StaticArrays

cryst = Crystal(I(3), [[0.,0,0]],1)
sys = System(cryst,(64,1,1),[SpinInfo(1;S=1/2,g=2)],:dipole)
set_exchange!(sys,-1.0,Bond(1,1,[1,0,0]))

nw = 500
dsc = dynamical_correlations(sys;Δt = 0.005, nω = nw, ωmax = 30.0)
dsc_energy_conserving = dynamical_correlations(sys;Δt = 0.005, nω = nw, ωmax = 30.0)
bug_data = zeros(ComplexF64,6,1,1,64,1,1,nw)

langevin = Langevin(0.05;λ = 0.1, kT = 0.001)

for j = 1:150
  for l = 1:1000
    step!(sys,langevin)
  end

  integrator = Sunny.ImplicitMidpoint(dsc.Δt)
  Sunny.new_sample!(dsc, sys,() -> nothing,integrator)

  # Compute bugged spectrum!
  FFTsampbuf = fft(dsc.samplebuf,(2,3,4,6))
  for comp = 1:3
    sampA = view(FFTsampbuf,comp,:,:,:,1,:)
    databuf = view(bug_data,[6,4,1][comp],1,1,:,:,:,:)
    for k = eachindex(databuf)
      diff = (sampA[k] * conj(sampA[k]) / (prod(sys.latsize) * size(dsc.samplebuf,6))) - databuf[k]
      databuf[k] += diff/j
    end
  end

  Sunny.accum_sample!(dsc)
end

params = unit_resolution_binning_parameters(dsc)
is, counts = intensities_binned(dsc,params,intensity_formula(dsc,:trace;kT = langevin.kT))
bcs = axes_bincenters(params)
#ix = round(Int64,4.0 / dsc.Δω)

is_filter = fft(ifft(is,4) .* reshape(cos.((pi) .* range(0,1,length = nw+1)[1:end-1]).^2,1,1,1,nw),4)


is_bug = real(sum([bug_data[comp,1,1,:,:,:,:] for comp = [1,4,6]])) ./ (params.numbins[4] * det(params.covectors[1:3,1:3]) / prod(params.binwidth[1:3]))
for j = 1:size(is_bug,4)
  is_bug[:,:,:,j] *= Sunny.classical_to_quantum(bcs[4][j],langevin.kT)
end

f = Figure()
ax1 = Axis(f[1,1],title = "New",xlabel = "q", ylabel = "E")
hm = heatmap!(ax1,bcs[1],bcs[4],log10.(abs.(is[:,1,1,:])),colorrange=(-9,-1))
ylims!(ax1,0,4.0)
Colorbar(f[1,2],hm)

ax2 = Axis(f[2,1],title = "Bug",xlabel = "q", ylabel = "E")
heatmap!(ax2,bcs[1],bcs[4],log10.(abs.(is_bug[:,1,1,1:nw÷2])),colorrange=(-9,-1))
ylims!(ax2,0,4.0)

ax3 = Axis(f[3,1],title = "New + Cosine Squared filter",xlabel = "q", ylabel = "E")
heatmap!(ax3,bcs[1],bcs[4],log10.(abs.(is_filter[:,1,1,:])),colorrange=(-9,-1))
ylims!(ax3,0,4.0)

sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
swt = SpinWaveTheory(sys_collapsed)
formula = intensity_formula(swt,:trace,kernel = delta_function_kernel)
function Sunny.intensities_binned(swt::SpinWaveTheory, params::BinningParameters, formula::Sunny.SpinWaveIntensityFormula)
    if any(params.covectors[1:3,4] .!= 0.) || any(params.covectors[4,1:3] .!= 0.)
      error("Complicated binning parameters not supported by intensities_bin_centers")
    end
    bin_centers = axes_bincenters(params)

    # coords = covectors * (q,ω)
    coords_to_q = inv(params.covectors[1:3,1:3])

    is = zeros(Float64,params.numbins...)

    # Loop over qs
    for ci in CartesianIndices(params.numbins.data[1:3])
        x_center = bin_centers[1][ci[1]]
        y_center = bin_centers[2][ci[2]]
        z_center = bin_centers[3][ci[3]]

        q = SVector{3}(coords_to_q * [x_center;y_center;z_center])
        ωvals = bin_centers[4]

        
        bs = formula.calc_intensity(swt,q)
        for i = 1:length(bs.dispersion)
          eBin = 1 + floor(Int64,(params.covectors[4,4] * bs.dispersion[i] - params.binstart[4]) ./ params.binwidth[4])
          is[ci,eBin] += bs.intensity[i]
        end
    end
    is
end
isswt = prod(params.binwidth) * Sunny.intensities_binned(swt,params,formula)

ax4 = Axis(f[4,1],title = "Spin Wave",xlabel = "q", ylabel = "E")
heatmap!(ax4,bcs[1],bcs[4],log10.(abs.(isswt[:,1,1,:])),colorrange=(-9,-1))
ylims!(ax4,0,4.0)


f2 = Figure()
ax1 = Axis(f2[1,1],title = "New",xlabel = "q", ylabel = "E")
heatmap!(ax1,bcs[1],bcs[4],abs.(is[2:end-3,1,1,:]))
ylims!(ax1,0,4.0)

ax2 = Axis(f2[2,1],title = "Bug",xlabel = "q", ylabel = "E")
heatmap!(ax2,bcs[1],bcs[4],abs.(is_bug[2:end-3,1,1,1:nw÷2]))
ylims!(ax2,0,4.0)

ax3 = Axis(f2[3,1],title = "New + Cosine Squared filter",xlabel = "q", ylabel = "E")
heatmap!(ax3,bcs[1],bcs[4],(abs.(is_filter[2:end-3,1,1,:])))
ylims!(ax3,0,4.0)

ax4 = Axis(f2[4,1],title = "Spin Wave",xlabel = "q", ylabel = "E")
heatmap!(ax4,bcs[1],bcs[4],(abs.(isswt[2:end-3,1,1,:])))
ylims!(ax4,0,4.0)

f3 = Figure()
ax1 = Axis(f3[1,1],xlabel = "Time", ylabel = "XX Correlation @ q = 0.375")
ts = range(0,step = dsc.Δt * dsc.measperiod,length = length(bcs[4]))
lines!(ax1,ts,real(fft(is[25,1,1,:])))
lines!(ax1,ts./2,real(fft(is_bug[25,1,1,:])))
lines!(ax1,ts,real(fft(is_filter[25,1,1,:])))


