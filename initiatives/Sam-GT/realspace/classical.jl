using Sunny

function get_fei2_sc()

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

sys_large = resize_supercell(sys, (16,16,4)) # 16x16x4 copies of the original unit cell

kT = 3.5 * meV_per_K     # 3.5K ≈ 0.30 meV
langevin.kT = kT
for _ in 1:10_000
    step!(sys_large, langevin)
end

sc = dynamical_correlations(sys_large; Δt=2Δt, nω=120, ωmax=7.5)

add_sample!(sc, sys_large; alg = :window)        # Accumulate the sample into `sc`

for _ in 1:8
  println("Sampling...")
    for _ in 1:1000               # Enough steps to decorrelate spins
        step!(sys_large, langevin)
    end
    add_sample!(sc, sys_large; alg = :window)    # Accumulate the sample into `sc`
end

sys_large, sc
end

#sys, sc = get_fei2_sc()
scdata = sc.data
scnw = length(available_energies(sc))
upscale_factor = 2
sys_rep = repeat_periodically(sys,(16,1,1))
sc_super = dynamical_correlations(sys_rep; Δt=sc.Δt, nω=upscale_factor*scnw, ωmax=7.5)
Lx,Ly,Lz = sys.latsize
LX,LY,LZ = sys_rep.latsize
Le = size(sc.data,7)
LE = size(sc_super.data,7)
ix1 = LX÷2 + 1 .+ (1:Lx) .- (Lx÷2+1)
ix2 = LY÷2 + 1 .+ (1:Ly) .- (Ly÷2+1)
ix3 = LZ÷2 + 1 .+ (1:Lz) .- (Lz÷2+1)
ix4 = LE÷2 + 1 .+ (1:Le) .- (Le÷2+1)
sc_super.data[:,:,:,ix1,ix2,ix3,ix4] .= fftshift(ifft(sc.data,(4,5,6,7)))
@time sc_super.data .= ifftshift(sc_super.data)
@time fft!(sc_super.data,(4,5,6,7))

new_formula = intensity_formula(sc_super, :perp; kT, formfactors = formfactors)

points = [[0,   0, 0],  # List of wave vectors that define a path
          [1,   0, 0],
          [0,   1, 0],
          [1/2, 0, 0],
          [0,   1, 0],
          [0,   0, 0]]
density = 40
path, xticks = reciprocal_space_path(cryst, points, density);

@time is_interpolated = intensities_interpolated(sc_super, path, new_formula;
    interpolation = :round,       # Interpolate between available wave vectors
);

cut_width = 0.3
density = 15
paramsList, markers, ranges = reciprocal_space_path_bins(sc_super,points,density,cut_width);

#=
total_bins = ranges[end][end]
energy_bins = paramsList[1].numbins[4]
is_binned = zeros(Float64,total_bins,energy_bins)
integrated_kernel = integrated_lorentzian(0.05) # Lorentzian broadening
for k in eachindex(paramsList)
    @time bin_data, counts = intensities_binned(sc_super,paramsList[k], new_formula;
        integrated_kernel = integrated_kernel
    )
    is_binned[ranges[k],:] = bin_data[:,1,1,:] ./ counts[:,1,1,:]
end
=#

fig = Figure()
ax_top = Axis(fig[1,1],ylabel = "meV",xticklabelrotation=π/8,xticklabelsize=12;xticks)
#ax_bottom = Axis(fig[2,1],ylabel = "meV",xticks = (markers, string.(points)),xticklabelrotation=π/8,xticklabelsize=12)

ωs = available_energies(sc)
heatmap!(ax_top,1:size(is_interpolated,1), ωs, is_interpolated;
    colorrange=(0.0,0.07),
)

#heatmap!(ax_bottom,1:size(is_binned,1), ωs, is_binned;
    #colorrange=(0.0,0.05),
#)

fig


