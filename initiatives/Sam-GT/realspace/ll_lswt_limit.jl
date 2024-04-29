using Sunny, LinearAlgebra, StaticArrays, GLMakie, Statistics

# Setup system
#cryst = Crystal(I(3),[[0,0,0]],1)
cryst = Crystal(I(3),[[0,0,0],[0.5,0.5,0.5]],1)
sys = System(cryst,(1,1,2),[SpinInfo(1,S=5/2,g=2),SpinInfo(2,S=5/2,g=2)],:dipole)
#sys = System(cryst,(1,1,2),[SpinInfo(1,S=5/2,g=2)],:dipole)

B_z_ext = 2.3
set_external_field!(sys,[0,0,B_z_ext])
set_exchange!(sys,-1.,Bond(1,1,[0,0,1]))
set_exchange!(sys,-1.,Bond(1,2,[0,0,0]))

#set_exchange!(sys,1.,Bond(2,2,[0,0,1]))
#set_onsite_coupling!(sys,S -> -0.1 * S[3]^2,1)
#set_onsite_coupling!(sys,S -> -0.1 * S[3]^2,2)

randomize_spins!(sys)
minimize_energy!(sys)

# Compute spin wave theory on a finite system size
function sunny_swt_spectrum(sys;polyatomic = true)
  all_repeats = allequal([sys.dipoles[i,:] for i = Sunny.eachcell(sys)]) && allequal([sys.coherents[i,:] for i = Sunny.eachcell(sys)])
  if !all_repeats
    @warn "Not all unit cells in the system have the same spin data! Collapsing to first unit cell anyway"
  end

  sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
  swt = SpinWaveTheory(sys_collapsed;apply_g = false)

  na = Sunny.natoms(sys.crystal)
  nf = 1 #nf = sys.Ns[1] - 1

  if polyatomic
    # Find polyatomic required number of unit cells:
    nbzs = polyatomic_bzs(sys.crystal)
  else
    nbzs = [1,1,1]
  end

  comm_ixs = CartesianIndices(ntuple(i -> sys.latsize[i] .* nbzs[i],3))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  formula = intensity_formula(swt,:full;kernel = delta_function_kernel)

  Nx,Ny,Nz = sys.latsize .* nbzs
  S = zeros(ComplexF64,Nx,Ny,Nz,na*nf,3,3) # kx, ky, kz, band, Sμ, Sν
  disps = zeros(Float64,size(formula.calc_intensity.H,1)÷2,comm_ixs.indices...)
  for ix = 1:Nx, iy = 1:Ny, iz = 1:Nz
    band_structure = formula.calc_intensity(swt,ks_comm[ix,iy,iz])
    for band = 1:length(band_structure.dispersion)
      disps[band,ix,iy,iz] = band_structure.dispersion[band]
      S[ix,iy,iz,band,:,:] .= band_structure.intensity[band]
    end
  end
  disps, S
end

# Automatically choose enough area in reciprocal space
# to capture all phase-averaging interference phenomena exactly once
function polyatomic_bzs(crystal)
  na = Sunny.natoms(crystal)
  iszero_symprec(x) = abs(x) < crystal.symprec
  ΔRs = [map(x -> iszero_symprec(x) ? Inf : x,abs.(crystal.positions[i] - crystal.positions[j])) for i = 1:na, j = 1:na]
  round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))
end

# Establish the common set of binning parameters to use for all histograms
dsc = dynamical_correlations(sys;Δt = 0.005,nω = 200,ωmax = 20.0,apply_g = false)
common_params = unit_resolution_binning_parameters(dsc;negative_energies=true)
common_params.binend[1:3] .+= 1
common_params.binwidth[4] = 1.0
#common_params.binstart[4] -= 0.2

# Compute static structure factor
isc = instant_correlations(sys;apply_g = false)
add_sample!(isc,sys)
params_instant = unit_resolution_binning_parameters(isc)
params_instant.binend[1:3] .+= 1
is_static, counts = intensities_binned(isc,params_instant,intensity_formula(isc,:trace))

function get_classical_intensities_at_temperature(kT)
  dsc = dynamical_correlations(sys;Δt = 0.005,nω = 200,ωmax = 20.0,apply_g = false)
  formula = intensity_formula(dsc,:full;kT = kT)

  langevin = Langevin(0.05,λ=0.3,kT=kT)
  for j = 1:250
    for i = 1:1000
      step!(sys,langevin)
    end
    add_sample!(dsc,sys)
  end
  is, counts = intensities_binned(dsc,common_params,formula)
  # Fix the intensities_binned algorithm:
  # It currently divides by numbins in the histogram as a proxy for number of
  # modes in the classical calculation; but actually it should just divide by it directly.
  # This is a problem when they are different numbers, like here!!
  is .* common_params.numbins[4] ./ length(available_energies(dsc;negative_energies=true))
end

# Evalate spin wave theory
disps_sw, is_bands_sw = sunny_swt_spectrum(sys)
is_sw = zeros(Float64,common_params.numbins...)

# Bin along energy axis
for band = 1:size(disps_sw,1)
  for a = 1:common_params.numbins[1], b = 1:common_params.numbins[2], c = 1:common_params.numbins[3]
    bin_ix = 1 .+ floor.(Int64,(disps_sw[band,a,b,c] .- common_params.binstart[4]) ./ common_params.binwidth[4])
    for q = 1:3
      is_sw[a,b,c,bin_ix] += real(is_bands_sw[a,b,c,band,q,q]) / 2 # Why this factor 2?
    end
  end
end

# Evaluate classical intensities at successively lower temperatures
temps = 10 .^ range(1,-6,length = 20)
is_classicals = Array{Float64,4}[]
for i = 1:length(temps)
  dat = get_classical_intensities_at_temperature(temps[i])
  is_classical = zeros(Float64,common_params.numbins...)
  for a = 1:common_params.numbins[1], b = 1:common_params.numbins[2], c = 1:common_params.numbins[3], d = 1:common_params.numbins[4]
    for q = 1:3
      is_classical[a,b,c,d] += real(dat[a,b,c,d][q,q]) 
    end
  end
  push!(is_classicals,is_classical)
end

# Add 0-magnon term (static structure factor) back into spin wave theory
# Still missing: Reduced ordered moment correction
bin_ix_zero = 1 .+ floor.(Int64,(0.0 .- common_params.binstart[4]) ./ common_params.binwidth[4])
is_sw[:,:,:,bin_ix_zero] .+= is_static

# Compare and plot
bcs = axes_bincenters(common_params)
f = Figure()
ax = Axis(f[1,1],xlabel = "Q",ylabel = "E",title = "Spin Wave")
hm = heatmap!(ax,1:16,bcs[4],reshape(is_sw,:,common_params.numbins[4]))
Colorbar(f[1,2],hm)
ax = Axis(f[2,1],xlabel = "Q",ylabel = "E", title = "Low Temp LL")
hm = heatmap!(ax,1:16,bcs[4],reshape(is_classicals[end],:,common_params.numbins[4]))
Colorbar(f[2,2],hm)
ax = Axis(f[3,1],xlabel = "Q",ylabel = "E", title = "Difference")
hm = heatmap!(ax,1:16,bcs[4],reshape(is_sw .- is_classicals[end],:,common_params.numbins[4]))
Colorbar(f[3,2],hm)
ax = Axis(f[4,1],xlabel = "log10 Temperature", ylabel = "log10 χ² SWT vs LL")
scatter!(ax,log10.(temps),[log10(norm(is_sw .- is_classicals[i])) for i = 1:length(temps)])
f

