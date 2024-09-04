# To execute this file, first create a new environment at your favorite directory. Then enter the julia REPL and run the following commands:
# ]
# activate .
# add https://github.com/Hao-Phys/Sunny.jl/tree/non-perturbative2
# add GLMakie
# add ProgressMeter
# add JLD2
# add DelimitedFiles

using Sunny
using GLMakie
using LinearAlgebra
using ProgressMeter
using Base.Threads
using DelimitedFiles
using JLD2

# In this file, we run all simulations for the triangular lattice UUD problem in a single-file.
# The parameters are chosen to reproduce the results in 
# [Ref]: npj Quantum Materials 8.1 (2023): 48.
Δ     = 100
Lmag  = 9

# `commons.jl` includes some auxilary functions
include("tri_uud_aux.jl")

J = 0.395
gab = 3.25
gc  = 0.2
s = 1/2
μB = 5.788381806e-2

# For the isotropic limit, we can only perform the calculation at B_eff = 3Js at this moment.
# Here we want to reproduce
B = Δ == 1.0 ? 3J*s/(gab*μB) : 4.0

# Create the `Crystal`, `System`, and `SpinWaveTheory` for non-perturbative calculations.
units = Units(:meV, :angstrom)
a = b = 1.0
c = 10a
lat_vecs = lattice_vectors(a, b, c, 90, 90, 120)
basis_vecs = [[0, 0, 0]]

L = 3
cryst = Crystal(lat_vecs, basis_vecs)
# In [Ref], the magnetic field lies in the x-y plane. 
# To maintain consistency with our calculations, we simply swap the definitions of the in-plane and out-of-plane g-factors.
sys = System(cryst, [1 => Moment(; s, g = -diagm([gc, gc, gab]))], :dipole; dims=(L, L, 1))

set_exchange!(sys, J*diagm([1.0,1.0,Δ]), Bond(1,1,[1,0,0]))
set_field!(sys, [0,0,B*units.T])

randomize_spins!(sys)
minimize_energy!(sys)
emin = energy_per_site(sys)

# Set the UUD phase by hand, then check the `energy_per_site` from the Optim minimization
set_dipole!(sys, ( 0,0, 1), (1,1,1,1))
set_dipole!(sys, ( 0,0, 1), (2,1,1,1))
set_dipole!(sys, ( 0,0,-1), (2,2,1,1))
set_dipole!(sys, ( 0,0, 1), (3,2,1,1))
set_dipole!(sys, ( 0,0,-1), (3,1,1,1))
set_dipole!(sys, ( 0,0, 1), (2,3,1,1))
set_dipole!(sys, ( 0,0,-1), (1,3,1,1))
set_dipole!(sys, ( 0,0, 1), (1,2,1,1))
set_dipole!(sys, ( 0,0, 1), (3,3,1,1))

eset = energy_per_site(sys)
@assert emin ≈ eset

# Create the `SpinWaveTheory` object
sys_min = reshape_supercell(sys, [1 0 0; -1 3 0; 0 0 1])
swt = SpinWaveTheory(sys_min; regularization=1e-5, measure=ssf_perp(sys_min))

# Then create the object for `NonPerturbativeTheory`
npt = Sunny.NonPerturbativeTheory(swt, (Lmag, Lmag, 1));

result_path = joinpath(@__DIR__, "results")
isdir(result_path) || mkdir(result_path)

# Get the renormalized single-particle energies
if Δ == 1.0
    npt′ = generate_renormalized_npt(npt; single_particle_correction=true, atol=1e-4);
else
    npt′ = npt;
end

# Save the results for future
resultname = joinpath(result_path, "npt′_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".jld2")
jldsave(resultname; npt′)

# Here we focus on the [H,H,0] cut
Hs  = [i/Lmag for i in 0:Lmag-1]
qs  = [[Hs[i], Hs[i], 0] for i in 1:Lmag]
com_indices = get_reshaped_cartesian_index(npt, qs)

# Get the two-particle eigenstates from exact diagonalization
num_2ps = length(npt′.two_particle_states[1,1,1])
E2ps = zeros(num_2ps, Lmag)

pm = Progress(Lmag; desc="Calculating two-particle energies")
Threads.@threads for i in 1:Lmag
    E = calculate_two_particle_energies(npt′, num_2ps, com_indices[i])
    E2ps[:, i] = E[1:num_2ps]
    next!(pm)
end
writedlm(joinpath(result_path, "triuud_E2ps_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"), E2ps)

δω = 0.025
ωs = collect(198:δω:202) * J
η  = 2δω

# Get the free two-particle intensities
Szz_free_continuum = zeros(Lmag, length(ωs));
pm = Progress(Lmag; desc="Calculating the free continuum intensities")
Threads.@threads for i in 1:Lmag
    ret = Sunny.dssf_free_two_particle_continuum_component(swt, qs[i], ωs, η, 3, 3; atol=1e-5)
    Szz_free_continuum[i, :] = ret
    next!(pm)
end

writedlm(joinpath(result_path, "triuud_Szz_free_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"), Szz_free_continuum)

# Get the two-particle intensities from continued fraction
Szz_cf = zeros(Lmag, length(ωs));
pm = Progress(Lmag; desc="Calculating intensities using continued fraction")
n_iters = 13
Threads.@threads for i in 1:Lmag
    Szz_cf[i, :] = Sunny.dssf_continued_fraction_two_particle(npt′, qs[i], ωs, η, n_iters)
    next!(pm)
end
writedlm(joinpath(result_path, "triuud_Szz_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"), Szz_cf)


# Plot the results
result_path = joinpath(@__DIR__, "results")
E2ps = readdlm(joinpath(result_path, "triuud_E2ps_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"))
@. E2ps /= J
is_cf = readdlm(joinpath(result_path, "triuud_Szz_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"))
is_fr = readdlm(joinpath(result_path, "triuud_Szz_free_HH0_L"*string(Lmag)*"Δ_"*string(round(Δ, digits=4))*".dat"))

# Plot the one-particle and two-particle energies
num_2ps = size(E2ps, 1)

@. ωs = ωs / J
fig = Figure()
ax  = Axis(fig[1, 1]; xlabel="(H, H, 0)", ylabel="Energy", title="Interacting continuum")
heatmap!(ax, Hs, ωs, is_cf, colorrange=(0, 5e-3))
for i in 1:num_2ps
    scatter!(ax, Hs, E2ps[i, :], color=:blue, marker=:rect, label="Two-particle")
end
xlims!(ax, 0.0, 0.5)
ylims!(ax, 198, 202)
fig

fig = Figure()
ax  = Axis(fig[1, 1]; xlabel="(H, H, 0)", ylabel="Energy", title="Free continuum")
heatmap!(ax, Hs, ωs, is_fr, colorrange=(0, 5e-3))
# for i in 1:num_2ps
#     scatter!(ax, Hs, E2ps[i, :], color=:blue, marker=:rect, label="Two-particle")
# end
xlims!(ax, 0.0, 0.5)
ylims!(ax, 198, 202)
fig
