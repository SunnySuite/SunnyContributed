# Analyzes the onsite anisotropy matrix of an SU(N) system in order
# to determine a low energy manifold (defined as the part of the spectrum
# that is protected by a gap at least gap_clearance*Δ, where Δ is the energy gap
# to the first excited state. The first excited state is defined as the first one
# not degenerate with the ground state according to the degeneracy_tol[erance]).
#
# The intention is that, once the low energy part is identified, the interactions
# can be projected and the entire system can be simulated using only the few low-energy
# degrees of freedom :). Example systems are TMGO or YMGO.
function project_system(sys;gap_clearance = 10.,degeneracy_tol = 1e-4)
  Na = length(sys.crystal.positions)
  anisos = [sys.interactions_union[i].onsite for i = 1:Na]
  if !allequal(anisos)
    error("Different anisotropies on different sites! How to handle this is not yet implemented")
  end
  aniso = anisos[1]
  F = eigen(aniso)
  @assert isreal(F.values)
  @assert issorted(F.values)
  E0_solver = F.values[1]
  rel_levels = F.values .- E0_solver
  energies = Float64[]
  multiplicities = Int64[]
  j = 1
  while j <= length(rel_levels)
    lowest_energy = rel_levels[j]
    degeneracy = findlast(x -> x <= degeneracy_tol,rel_levels[j:end] .- lowest_energy)
    push!(energies,lowest_energy)
    push!(multiplicities,degeneracy)
    j += degeneracy
  end

  E0 = energies[1]
  E1 = energies[2]
  Δ = E1 - E0

  projection_cutoff = findfirst(diff(energies) .> Δ * gap_clearance)

  printstyled("Anisotropy spectrum:",bold = true,underline = true)
  println()
  println("/")
  n(x) = Sunny.number_to_simple_string(x;digits = Int64(round(x == 0 ? 0 : log10(x)) - round(log10(degeneracy_tol)) + 1))
  for k = 1:length(energies)
    ordinal = k-1
    suffix = 1 <= ordinal <= 3 ? ["st","nd","rd"][ordinal] : "th"
    print("| ∘ ")
    printstyled("$(n(energies[k])) meV",color = :blue)
    if multiplicities[k] > 1
      print(" ")
      printstyled("×$(multiplicities[k])",underline = true)
    end
    printstyled(" ($ordinal$suffix)",color = :light_black)
    println()
    if k < length(energies)
      println("|")
      if k == 1
        println("|    ↕ [Δ = $(n(energies[k+1] - energies[k])) meV gap]")
      elseif k == projection_cutoff
        print("\\")
        printstyled("=========================",color = :red)
        println()
        println()
        print("     ")
        gap = energies[k+1] - energies[k]
        printstyled("↕ [$(n(gap)) meV gap = $(Sunny.number_to_simple_string(gap/Δ,digits = 4))×Δ]",color = :red)
        println()
        println()
        print("/")
        printstyled("=========================",color = :red)
        println()
      else
        println("|    ↕ [$(n(energies[k+1] - energies[k])) meV gap]")
      end
      println("|")
    end
  end
  println("\\")

  dimension_before_cutoff = sum(multiplicities)
  dimension_after_cutoff = sum(multiplicities[1:projection_cutoff])
  S = spin_matrices((dimension_before_cutoff - 1)/2)
  Π = F.vectors[:,1:projection_cutoff]
  Sproj = [Π' * s * Π for s = S]

  println()
  println("Projection fidelity: ")
  print(" ")
  println(repeat('=',2dimension_before_cutoff))
  ramp = reverse("\$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
  projection_approx = abs.(Π * Π')
  for i = 1:dimension_before_cutoff
    print("|")
    for j = 1:dimension_before_cutoff
      c = ramp[1 + Int64(floor(projection_approx[i,j] * length(ramp)))]
      print(c)
      print(c)
    end
    print("|")
    println()
  end
  print(" ")
  println(repeat('=',2dimension_before_cutoff))

  ### Hacked version of clone_system which does the projection
  (; origin, mode, crystal, latsize, Ns, gs, κs, extfield, interactions_union, ewald, dipoles, coherents, units, rng) = sys
  origin_clone = isnothing(origin) ? nothing : clone_system(origin)
  empty_dipole_buffers = Array{eltype(sys.dipoles), 4}[]
  empty_coherent_buffers = Array{eltype(sys.coherents), 4}[]

  # Project the interactions
  interactions_clone = map(x -> proj_interaction(x,sys,F,Sproj,projection_cutoff), interactions_union)

  ret = System(origin_clone, mode, crystal, latsize, Ns, copy(κs), copy(gs),
                 interactions_clone, nothing, copy(extfield), copy(dipoles), copy(coherents),
                 empty_dipole_buffers, empty_coherent_buffers, units, copy(rng))

  # Re-enable long range dipole if it was there before
  if !isnothing(ewald)
    enable_dipole_dipole!(ret)
  end
end

function proj_interaction(interaction,sys,Faniso,Sproj,cutoff)
  if interaction.onsite isa Sunny.StevensExpansion
    error("Projection of StevensExpansion is not yet supported!")
  end
  @assert interaction.onsite isa Sunny.HermitianC64
  onsite_matrix = Matrix(interaction.onsite)
  Π = Faniso.vectors[:,1:cutoff]
  onsite_projected = Π' * onsite_matrix * Π

  for i = 1:length(interaction.pair)
    (;isculled, bond, scalar, bilin, biquad, general) = interaction.pair[i]

    display(TensorDecomposition(Sproj,Sproj,Sunny.svd_tensor_expansion(Matrix(I(3) * bilin),cutoff,cutoff)))
    
    #bilin_proj = Π' * 

  #scalar   :: Float64              # Constant shift
    #bilin    :: Union{Float64, Mat3} # Bilinear
    #biquad   :: Union{Float64, Mat5} # Biquadratic

    ## General pair interactions, only valid in SU(N) mode
    #general  :: TensorDecomposition


  end

  Sunny.Interactions(onsite_projected)
end

