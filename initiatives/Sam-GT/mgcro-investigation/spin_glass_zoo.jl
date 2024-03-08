using Sunny, GLMakie, LinearAlgebra, StatsBase, ProgressMeter

# A crystal (lattice with defined nearest neighbors) is called bipartite if its sites can be given labels A and B, such that 
# all nearest neighbors of every A site are B sites, and vice versa. This splits the lattice into two parts: the A part
# and the B part, with neither part being directly connected to itself by a nearest neighbor bond.
#
# cite for parent/medial lattice: https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-070909-104138
#
# One way to witness the bipartiteness of a crystal is to explicitly label the sites with 0 (A) and 1 (B).
# Then we can verify that every nearest neighbor bond connects an A-site to a B-site.

## dist must be at least the nearest neighbor distance, but otherwise doesn't matter
function get_reference_nearest_neighbor_bond(cryst;dist = 3.0)
  ref_bonds = reference_bonds(cryst,dist)
  ref_bonds[1] # Site bonded to itself
  return ref_bonds[2] # Nearest neighbor bond
end

function get_all_nearest_neighbor_bonds(cryst)
  nn_ref = get_reference_nearest_neighbor_bond(cryst)
  Sunny.all_symmetry_related_bonds(cryst,nn_ref)
end

function witness_bipartite(cryst,labelling::Vector{Int64})
  if any((labelling .!= 1) .&& (labelling .!= 0)) || length(labelling) != Sunny.natoms(cryst)
    error("Invalid labelling! Each atom must be labelled 0 or 1.")
  end

  nn_bonds = get_all_nearest_neighbor_bonds(cryst)
  for b = nn_bonds
    if labelling[b.i] == labelling[b.j]
      error("Atoms $(b.i) (labelled $(labelling[b.i])) and $(b.j) (also labelled $(labelling[b.j])) connected by bond $b. Crystal is not bipartite!")
    end
  end
  println("Verified crystal is bipartite! ✓")
end

# For example, the diamond crystal,

diamond = Sunny.diamond_crystal()

# is bipartite with the following labelling:

witness_bipartite(diamond,[0,0,1,1,0,0,1,1])

# We can also brute-force construct a labelling if we don't feel like providing one explicitly.
# This algorithm works because the labelling is unique once we label a single site; in this case we
# take the first atom in the unit cell (by Sunny numbering) to be a 0 (A) site.

function label_bipartite(cryst)
  labelling = -1 * ones(Int64,Sunny.natoms(cryst))
  labelling[1] = 0

  it_count = 0
  while any(labelling .== -1)
    it_count += 1
    if it_count > 100
      error("Got stuck trying to label crystal and gave up :( [This shouldn't happen]")
    end
    for b = get_all_nearest_neighbor_bonds(cryst)
      for (m,n) = [(b.i,b.j),(b.j,b.i)]
        if labelling[m] == -1 && labelling[n] != -1
          labelling[m] = 1 - labelling[n]
        end
      end
    end
  end
  labelling
end

# Another interesting bipartite crystal is the laves graph,

laves = Crystal(I(3),[[0,0,0],[1/4,2/4,3/4],[2/4,3/4,1/4],[3/4,1/4,2/4],[2/4,2/4,2/4],[3/4,0,1/4],[0,1/4,3/4],[1/4,3/4,0]])

# which is still bipartite,

laves_label = label_bipartite(laves)
witness_bipartite(laves,laves_label)

# but has a lower coordination number than the diamond crystal:
coordination_number(x) = length(get_all_nearest_neighbor_bonds(x)) / Sunny.natoms(x)
println("Diamond coordination = $(coordination_number(diamond)), Laves graph coordination = $(coordination_number(laves))")

# Bipartite crystals are nice because the center points of their nearest neighbor bonds also form a lattice--the so-called
# medial lattice--where each site is a corner in a network of corner-sharing polyhedra. We can explicitly construct
# the medial lattice by making a crystal with the sites at the center of each bond:

function medial_lattice(cryst)
  medial_locations = Vector{Float64}[]
  for b = get_all_nearest_neighbor_bonds(cryst)
    if b.j > b.i # Create one site for each *directed* bond. No need to double up
      bond_start_rlu = cryst.positions[b.i]
      bond_end_rlu = cryst.positions[b.j] .+ b.n
      medial_location_rlu = mod.((bond_start_rlu + bond_end_rlu)/2,1)
      push!(medial_locations,medial_location_rlu)
    end
  end
  Crystal(cryst.latvecs,medial_locations)
end

function medial_bond_vectors(cryst)
  bond_vectors = Vector{Float64}[]
  for b = get_all_nearest_neighbor_bonds(cryst)
    if b.j > b.i # Create one site for each *directed* bond. No need to double up
      bond_start_rlu = cryst.positions[b.i]
      bond_end_rlu = cryst.positions[b.j] .+ b.n
      bond_vector_rlu = (bond_end_rlu - bond_start_rlu)
      push!(bond_vectors,bond_vector_rlu)
    end
  end
  bond_vectors
end

# Using this we can see that for medial lattice of the diamond crystal (known as the pyrochlore crystal),
# the polyhedra in question is a tetrahedron:

pyrochlore = medial_lattice(diamond)
view_crystal(pyrochlore)

# While for the laves graph it's corner-sharing triangles:

view_crystal(medial_lattice(laves))

# This feature of having a lattice of corner-sharing shapes enables interesting frustrated magnetic behavior to take place.

# ## Frustrated Magnetism on Corner-Sharing Networks
#
# What interactions are symmetry-allowed at the nearest neighbor level for this type of crystal?
# We can answer this on a case-by-case basis by checking the symmetry table:

print_symmetry_table(medial_lattice(diamond), 0.5)

# Here, we see that there is a two parameter space of allowed g-tensors, a one parameter space of allowed
# anisotropy (for $S < 2$, the higher stevens operators `O[4,x]` are zero), and a four parameter space of allowed
# J1 bilinear exchange matrix. (Total dimension = 7)
# We could also consider biquadratic exchanges, but Sunny doesn't (yet) provide a-priori symmetry analysis for them (it only checks
# they are symmetry allowed when setting the exchange).
#
# The Laves graph is slightly more complicated:

print_symmetry_table(medial_lattice(laves), 0.5)

# with three parameters of allowed g-tensor, two parameters of anisotropy, and six parameters of J1 matrix. (Total dimension = 11)
#
# ## G-factor
#
# For now, we only consider zero field, so the g-factor is immaterial.
# This makes our parameter space is a bit smaller.
#
# ## Anisotropy
#
# To understand the allowed anisotropy, we can use these functions to write them in the basis of spin matrices:

function dipolar_spin_matrix_coeffs(M;S)
  s = spin_matrices(S)
  [tr(S[i]' * M) / tr(S[i]' * S[i]) for i = 1:3]
end

function quadrupolar_spin_matrix_coeffs(M;S)
  s = spin_matrices(S)
  [tr((s[i] * s[j])' * M) / tr((s[i] * s[j])' * s[i] * s[j]) for i = 1:3, j = 1:3]
end

# For example, the allowed aniostropy (calculated for $S=1$) for the diamond crystal at the first site is $S^xS^y + S^yS^x - S^xS^z - S^zS^x - S^yS^z-S^zS^y$:

O = stevens_matrices(1)
quadrupolar_spin_matrix_coeffs(O[2,-2] -2O[2,-1] -2O[2,1],S=1)

# Comparing with the direction,

medial_bond_vectors(diamond)[1]

# of the bond on the diamond lattice which this site is placed on,
# we see that the anisotropy (which is nematic because it's quadrupolar) is aligned along the symmetry axis of the bond.
# If we set the sign of the anisotropy appropriately, we find that the ground state is Ising-like, with spins pointing either in or
# out of each tetrahedron:

sys_pyrochlore = System(medial_lattice(diamond),(1,1,1),[SpinInfo(1,S=1,g=2)],:dipole)
set_onsite_coupling!(sys_pyrochlore,-(O[2,-2] - 2O[2,-1] -2O[2,1]),1)
randomize_spins!(sys_pyrochlore)
minimize_energy!(sys_pyrochlore)
plot_spins(sys_pyrochlore)

# The opposite sign of anisotropy would make it easy-plane (in the reflection plane of the bond) instead of easy-axis (along the bond), as can be seen from the degenerate eigenvectors of the anisotropy matrix:

eigen(O[2,-2] - 2O[2,-1] - 2O[2,1])

# For the system derived from the Laves graph, on the site with bond vector

medial_bond_vectors(laves)[1]

# we can see that one of the flavors of anisotropy is either easy-plane in a plane containing the bond vector,
# or easy-axis in a direction perpendicular to the bond vector:

eigen(O[2,0] + 3O[2,2])

# and the other flips between two easy axes:

eigen(O[2,1])

# ## Bilinear exchange
# 
# So far, all sites were independent, so the ground states were just the easy-axis or easy-plane states, copied to every site.
# This means that there are a very large number ($2^N$ for the easy-axis case) of ground states.
# By introducing an exchange coupling between sites, we either select, modify, or hybridize between these non-interacting ground states, changing the ground state entropy.
# For simplicitly, we will mainly consider isotropic (Heisenberg) exchanges.
#
# For example, turning on a ferromagnetic J1 on top of the easy-axis anisotropy changes the entropy (number of possible ground states) as measured by the entropy of a single component of a spin on one site.
# Specifically, consider the ensemble of ground states of a unit cell of a pyrochlore with some fixed $J_1$ obtained by randomizing the spin configuration, and then "quenching" it by iteratively minimizing the energy:

function gs_dist(sys)
  sx = Float64[]
  sy = Float64[]
  sz = Float64[]
  for j = 1:2000
    randomize_spins!(sys)
    minimize_energy!(sys,maxiters = 2000)
    push!(sx,sys.dipoles[1][1])
    push!(sy,sys.dipoles[1][2])
    push!(sz,sys.dipoles[1][3])
  end
  sx,sy,sz
end

function gs_dist_x(sys)
  nsamp = 2000
  sxs = ntuple(i -> zeros(Float64,nsamp),16)
  for j = 1:nsamp
    randomize_spins!(sys)
    @suppress minimize_energy!(sys,maxiters = 3)
    for i = 1:16
      sxs[i][j] = sys.dipoles[1,1,1,i][1]
    end
  end
  sxs
end

# Using this ensemble, the spin on a given site is a random variable distributed according to the ensemble distribution.
# The entropy of the random variable can then be approxmated using histograms to discretize the spin configuration.
# Here's how the entropy of the "$S^x$ on the first site" variable varies with $J_1$:

J1s = range(-0.5,0.5,length=100)
entropies_one_site = zeros(length(J1s))
entropies = zeros(length(J1s))
nnb = get_reference_nearest_neighbor_bond(sys_pyrochlore.crystal;dist = 3.0)
prog = Progress(length(J1s),"Sweeping J1")
num_bins = 40
for (i,J) = enumerate(J1s)
  set_exchange!(sys_pyrochlore,J,nnb)
  sxs = gs_dist_x(sys_pyrochlore)

  prob_sx = normalize(fit(Histogram,sxs[1];nbins=num_bins);mode = :probability).weights
  log_prob_sx = log.(prob_sx)
  log_prob_sx[iszero.(prob_sx)] .= 0
  entropies_one_site[i] = sum(-prob_sx .* log_prob_sx)

  prob_sx = normalize(fit(Histogram,(sxs[1],sxs[8],sxs[11],sxs[12]);nbins=num_bins);mode = :probability).weights
  log_prob_sx = log.(prob_sx)
  log_prob_sx[iszero.(prob_sx)] .= 0
  entropies[i] = sum(-prob_sx .* log_prob_sx)
  
  next!(prog)
end
finish!(prog)
p = plot(J1s,entropies_one_site; axis = (;xlabel = "J1/[anisotropy strength]",ylabel = "Single Component Entropy (nats)"))
hlines!(log.([2,4]))
hlines!(log(num_bins),linestyle = :dash)
display(p)

# At $J_1 = 0$, there are just the two easy-axis states (bottom line), whereas at large ferromagnetic $J_1 < 0$, the four symmetry directions of the tetrahedra
# are all fully polarized ground states (middle line).
# Extending to antiferromagnetism ($J_1 > 0$) we find that at large $J_1$, the entropy is so large that it approaches the maximum entropy resolvable with our choice of histogram bins (dashed line).
# 
# The increase in entropy as $J_1$ is made slightly nonzero in either direction can be understood intuitively.
# In this limit, $J_1$ is much smaller than the anisotropy, so the energy landscape $E(\theta)$ experienced by 
# the angle $\theta$ away from the easy axis is mainly a quadratic minimum (due to the anisotropy).
# There is a small perturbation due to $J_1$, but the shift in the location of the minimum is proportional to the small
# quantity $J_1$/[anisotropy strength]. 
# Thus, the system decouples into the gross structure (frozen Ising spins with entropy $\log 2$) and the small interacting
# perturbations.
# The amplitude (and therefore the available phase space) of the perturbations is controlled by $\lvert J_1\rvert$, so increasing $J_1$ in either direction provides some entropy.
# As $J_1$ is increased further, the decoupling breaks down, and the entropy curve becomes nonlinear.
#
# To summarize: in the Ising limit $J_1\approx 0$, each site has entropy $\log 2$, with hybridization between sites (proportional to $\lvert J_1\rvert$) increasing the entropy experienced at any one site.
#
# ## Finite Temperature
#
# As is often the case in physics, it turns out that we are spending a fair bit
# of computational effort to get strongly into a limit which we don't care that strongly
# about in practice.
# In particular, much of the difficulty in computing entropy using the above method comes from
# sampling strictly over very many zero temperature "ground states" of the spin configuration.
# But who cares about zero temperature!? And who cares about a representative sampling of
# all possible ground state configurations, anyway?
#
# This is especially the case in the Ising limit, where the ground states are disconnected from each other by energy barriers. 
# Instead of doing that complicated sampling, let's just put the system in one ground state (arbitrarily selected) and let it cook:
#

langevin = ImplicitMidpoint(0.04,λ = 0.1, kT = 1e-3)

#randomize_spins!(sys_pyrochlore)
#minimize_energy!(sys_pyrochlore)

function gs_dist_thermal(f,sys,integrator;nsamp = 20,kTrange = range(1e-1,step = 1e-1,length = nsamp))
  nsamp = 20
  e_vars = zeros(Float64,nsamp)
  ncook = 8000
  nthermalize = 25
  e_buf = zeros(Float64,ncook)
  for j = 1:nsamp
    integrator.kT = kTrange[j]
    for k = 1:nthermalize
      step!(sys,integrator)
    end
    for k = 1:ncook
      step!(sys,integrator)
      e_buf[k] = f(sys) #sys.dipoles[1][3] #energy(sys)
    end
    e_vars[j] = empirical_entropy(e_buf,range(-1,1,length = 35))#var(e_buf)
  end
  e_vars # ./ (kTrange .^ 2)
end

function empirical_entropy(xs,edges)
  h = fit(Histogram,xs,weights(ones(size(xs))),edges)
  normalize!(h,mode = :probability)
  log_prob = log.(h.weights)
  log_prob[iszero.(h.weights)] .= 0
  sum(-h.weights .* log_prob)
end

J1s = range(-0.5,0.5,length=10)
nnb = get_reference_nearest_neighbor_bond(sys_pyrochlore.crystal;dist = 3.0)
prog = Progress(length(J1s),"Sweeping J1")
num_bins = 40
f = Figure()
display(f)
ax = Axis(f[1,1])
for (i,J) = enumerate(J1s)
  set_exchange!(sys_pyrochlore,J,nnb)
  randomize_spins!(sys_pyrochlore)
  minimize_energy!(sys_pyrochlore)
  for j = 1:100; step!(sys_pyrochlore,langevin); end; # thermalize
  kTs = range(1e-4,step = 1e-4,length = 20)
  sx_empirical_entropy_vs_T = gs_dist_thermal(x -> x.dipoles[1][1], sys_pyrochlore,langevin,kTrange = kTs)
  scatter!(J * ones(20),sx_empirical_entropy_vs_T,color = kTs)
  next!(prog)
end
finish!(prog)
p = plot(J1s,var_one_site; axis = (;xlabel = "J1/[anisotropy strength]",ylabel = "Variance Sx (one site)"))
#hlines!(log.([2,4]))
#hlines!(log(num_bins),linestyle = :dash)
display(p)


