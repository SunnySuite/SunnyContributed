using StaticArrays, LinearAlgebra, Sunny, GLMakie, BenchmarkTools, Random

function FastSystem(crystal::Crystal, latsize::NTuple{3,Int}, infos::Vector{SpinInfo}, mode::Symbol;
                units=Units.meV, seed=nothing)
    if !in(mode, (:SUN, :dipole, :dipole_large_S))
        error("Mode must be `:SUN`, `:dipole`, or `:dipole_large_S`.")
    end

    # The lattice vectors of `crystal` must be conventional (`crystal` cannot be
    # reshaped).
    if !isnothing(crystal.root)
        @assert crystal.latvecs == crystal.root.latvecs
    end
    
    na = Sunny.natoms(crystal)

    infos = Sunny.propagate_site_info(crystal, infos)
    Ss = [si.S for si in infos]
    gs = [si.g for si in infos]

    # TODO: Label SU(2) rep instead
    Ns = @. Int(2Ss+1)

    if mode == :SUN
        allequal(Ns) || error("Currently all spins S must be equal in SU(N) mode.")
        N = first(Ns)
        κs = fill(1.0, na)
    elseif mode in (:dipole, :dipole_large_S)
        N = 0 # marker for :dipole mode
        κs = copy(Ss)
    end

    # Repeat such that `A[:]` → `A[cell, :]` for every `cell`
    repeat_to_lattice(A) = permutedims(repeat(A, 1, latsize...), (2, 3, 4, 1))

    Ns = repeat_to_lattice(Ns)
    κs = repeat_to_lattice(κs)
    gs = repeat_to_lattice(gs)

    interactions = Sunny.empty_interactions(mode, na, N)
    ewald = nothing

    extfield = MArray{Tuple{latsize[1],latsize[2],latsize[3],na}}(zeros(Sunny.Vec3, latsize..., na))
    dipoles = MArray{Tuple{latsize[1],latsize[2],latsize[3],na}}(fill(zero(Sunny.Vec3), latsize..., na))
    coherents = MArray{Tuple{latsize[1],latsize[2],latsize[3],na}}(fill(zero(Sunny.CVec{N}), latsize..., na))
    dipole_buffers = AbstractArray{Sunny.Vec3, 4}[]
    coherent_buffers = AbstractArray{Sunny.CVec{N}, 4}[]
    rng = isnothing(seed) ? Random.Xoshiro() : Random.Xoshiro(seed)

    ret = Sunny.System(nothing, mode, crystal, latsize, Ns, κs, gs, interactions, ewald,
                 extfield, dipoles, coherents, dipole_buffers, coherent_buffers, units, rng)
    Sunny.polarize_spins!(ret, (0,0,1))
    return ret
end


cryst = Crystal(I(3),Sunny.diamond_crystal().positions,1)
sys = System(cryst,(4,4,4),[SpinInfo(i,S=1/2,g=2) for i = 1:8],:dipole,seed = 0)
fastsys = FastSystem(cryst,(4,4,4),[SpinInfo(i,S=1/2,g=2) for i = 1:8],:dipole,seed = 0)

for inters = 1:300
  J = randn(3,3)
  i = rand(1:8)
  j = rand(setdiff(1:8,[i]))
  dx,dy,dz = rand(-1:1,3)
  set_exchange!(sys,J,Bond(i,j,[dx,dy,dz]))
  set_exchange!(fastsys,J,Bond(i,j,[dx,dy,dz]))
end

langevin = Langevin(0.005,λ = 0.3, kT = 1.0)

display(@benchmark step!(fastsys,langevin))

display(@benchmark step!(sys,langevin))


