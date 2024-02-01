# # AFM J1 Pyrochlore
#
# The pyrochlore lattice consists of tetrahedra with sites on each vertex, and which share corners with each other:

using Sunny, LinearAlgebra, Statistics, SparseArrays

pyrochlore = Crystal(I(3), [[1/2,0,0]],227,setting = "2")
view_crystal(pyrochlore)

# This unit cell contains 16 sites, forming 5 full tetrahedra.
#
# We will place one classical spin `S = 1/2` on each site, and couple them (Heisenberg) anti-ferromagnetically to the nearest neighbors, $J \vec S_i\cdot \vec S_j$, with $J = 1$ (meV).

sys = System(pyrochlore, (1,1,1), [SpinInfo(1,S=1/2,g=2)], :dipole)
J1 = 1.
set_exchange!(sys, J1, Bond(1,2,[0,0,0])) # Symmetry propagates to all nearest neighbor bonds

## Print couplings by site:
include("../susceptibility/support.jl") # Import detailed show instances for Sunny.Interactions
sys.interactions_union

# The coordination number is 6, since there are 3 neighbors from each tetrahedra, and each site is shared by two tetrahedra.
# If every bond were fully satisfied, then the energy per site would be $-(6\div 2) \times JS^2 = -3/4$ meV.
# Instead, the lowest possible energy is $-1/4$ meV:

randomize_spins!(sys)
minimize_energy!(sys;maxiters = 3000)

println("Ground state energy per site: $(Sunny.number_to_math_string(energy_per_site(sys),digits=3)) meV")

# This is due to frustration; the geometry of the crystal dictates that not all bonds can be simultaneously satisfied.
# Here's what this particular (highly non-unique) ground state looks like:

plot_spins(sys)

# In general, all states where all tetrahedra have spins summing to zero are ground states.

# This frustration persists even when we consider just *one* tetrahedron in isolation.
# The lowest classically obtainable energy per site is $-1/8$ meV in this case, obtained by any configuration of spins summing to zero, compared to $-(6\div 4) JS^2 = -3/8$ meV if every bond were satisfied.

# This system is nice because we can solve the case of *one* tetrahedron in full quantum generality:

## Spin-1/2 matrices
S = spin_matrices(1/2)
dimS = size(S[1],1)
Iden = I(dimS)

## Sx,Sy,Sz for each site A,B,C,D
Sa = [kron(S[i],Iden,Iden,Iden) for i = 1:3]
Sb = [kron(Iden,S[i],Iden,Iden) for i = 1:3]
Sc = [kron(Iden,Iden,S[i],Iden) for i = 1:3]
Sd = [kron(Iden,Iden,Iden,S[i]) for i = 1:3]
S_sites = [Sa,Sb,Sc,Sd]

## AFM Heisenberg nearest neighbors
H = sum([(l > r ? 1 : 0) * S_sites[l][i] * S_sites[r][i] for l = 1:4, r = 1:4, i = 1:3])
sparse(H)

# Since the hilbert space dimension is small, we can explicitly diagonalize to find the spectrum:

F = eigen(H)
eigenenergies = round.(F.values,digits = 12)
eigenenergies / 4 # Energy per site

# In the quantum case, the energy per site is as if every bond were satisfied!
# We can also inspect the ground state wavefunction:
# the two columns below are the two degenerate ground state wavefunctions in components of |↑↑↑↑⟩, |↑↑↑↓⟩, |↑↑↓↑⟩, etc:

gs1 = round.(F.vectors[:,1],digits = 12)
gs2 = round.(F.vectors[:,2],digits = 12)
[gs1 gs2]

# To understand this better, we can look at the reduced density matrices obtained by treating any pair of sites as isolated from the "environment" of the two other sites.

rho_gs = (1/2) * gs1 * gs1' + (1/2) * gs2 * gs2'
rho_gs_tensor = reshape(rho_gs,dimS,dimS,dimS,dimS,dimS,dimS,dimS,dimS)

function trace_over(M,d)
  ci1 = CartesianIndices(ntuple(i -> size(M,i),d-1))
  ci2 = CartesianIndices(ntuple(i -> size(M,i+d),3))
  ci3 = CartesianIndices(ntuple(i -> size(M,i+d+4),8 - (d+4)))
  sum(M[ci1,[i],ci2,[i],ci3] for i = 1:dimS)
end

println("Reduced density matrix for sites (1,2):")
rho_reduced = trace_over(trace_over(rho_gs_tensor,3),4)
display(reshape(rho_reduced,4,4))
println()

println("Reduced density matrix for sites (2,3):")
rho_reduced = trace_over(trace_over(rho_gs_tensor,1),4)
display(reshape(rho_reduced,4,4))
println()

println("... other pairs similar ...")

# The eigendecomposition of this reduced density matrix (which is the same for every site) tells us the ground state for the reduced system is a 1/6, 1/6, 1/6, 1/2 mixture of the four pure states: |↑↑⟩, |↓↓⟩, (|↑↓⟩ + |↓↑⟩)/√2, and most probably the singlet state (|↑↓⟩ - |↓↑⟩)/√2:

Fr = eigen(reshape(rho_reduced,4,4))

# The variance of the sum of the spins, $\vec S_1 + \vec S_2$ can be used to characterize the quantum fluctuations:

Ssum = [kron(S[i],Iden) + kron(Iden,S[i]) for i = 1:3]
exp_sum(z) = [round(real(z' * Ssum[i] * z),digits=12) for i = 1:3]
var_sum(z) = [round(real(z' * Ssum[i]^2 * z) - exp_sum(z)[i]^2,digits=12) for i = 1:3]

# Notice that for the 1/6,1/6,1/6 states, the variance (number after the ±) is nonzero:
up_up = [1,0,0,0]
down_down = [0,0,0,1]
triplet_mixed = [0,1,1,0]/sqrt(2)

println("(S1 + S2) = $(exp_sum(up_up)) ± $(var_sum(up_up)) for |↑↑⟩")
println("(S1 + S2) = $(exp_sum(down_down)) ± $(var_sum(down_down)) for |↓↓⟩")
println("(S1 + S2) = $(exp_sum(triplet_mixed)) ± $(var_sum(triplet_mixed)) for (|↑↓⟩ + |↓↑⟩)/√2")

# Meanwhile, for the most probable (singlet) state, we get exactly zero variance, since the two spins are entangled to always be opposite:
singlet = [0,1,-1,0]/sqrt(2)
println("(S1 + S2) = $(exp_sum(singlet)) ± $(var_sum(singlet)) for (|↑↓⟩ - |↓↑⟩)/√2")

# Thus, it's no surprise that the variance of $\vec S_1 + \vec S_2$ in the whole ground state is $\frac{2}{6}\times 0.5 + \frac{1}{6}\times 1.0 + \frac{1}{2} \times 0 = \frac{1}{3}$: (needs fact check)

## TODO: fact check this is actually related somehow??
tr(rho_gs * (Sa + Sb)[3]^2)

# The entanglement entropy gives a more detailed story:

function von_entropy(rho)
  ps = round.(real.(eigvals(rho)),digits = 12)

  # Neglect very improbable (or negative probability) states
  ps[ps .< 1e-12] .= 0

  entropy_spectrum = -ps .* log.(ps)

  ## Convention for entropy: 0 * log(0) = 0
  entropy_spectrum[isnan.(entropy_spectrum)] .= 0

  sum(entropy_spectrum)
end

entropy_vs_sites = zeros(Float64,4)
for n_sites = 1:4
  rho = copy(rho_gs_tensor)
  for i = 1:(4-n_sites)
    rho = trace_over(rho,i)
  end
  entropy_vs_sites[n_sites] = von_entropy(reshape(rho,dimS^n_sites,dimS^n_sites))
  println(entropy_vs_sites)
end

plot(1:4,entropy_vs_sites ./ log(dimS) ./ (1:4); axis = (xlabel = "Number of Sites Retained", ylabel = "[Entropy per Site] / ln(2)"))

# The full tetrahedron with 4 sites has a finite specific entropy $\frac{1}{4}\ln L$ because it's in a so-called "resonating valence bond (RVB) state" with both `gs1` and `gs2` allowed, and this total entropy of $\ln L$ is divided among the four sites.
# With 3 sites, there is entropy per site $\frac{1}{3}\ln (L \times L)$, (where $L = 2S+1$) essentially because the single "environment" spin has $L$ possible states, in addition to the $L$ RVB states.
# With 2 sites, there is entropy per site $\frac{1}{4}\ln 12$, which is complicated due to the relatively complicated 1/6,1/6,1/6,1/2 mixture mentioned before.
# With just 1 site, there is entropy $\ln L$, since spin up and down are equiprobable at this level.



