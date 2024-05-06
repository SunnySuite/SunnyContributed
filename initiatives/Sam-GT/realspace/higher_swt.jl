using LinearAlgebra
using OffsetArrays
using StaticArrays
using FFTW
using SparseArrays
using ProgressMeter
using Sunny, GLMakie

bose(x) = 1/(exp(x) - 1)

function empty_y_matrix(;n_order = 2)
  #NaN .* OffsetArray(Array{Float64}(undef,n_order+1,2n_order + 1,2n_order + 1, 2n_order + 1),-1,-1,-1,-(n_order+1))
  NaN .* OffsetArray(Array{Float64}(undef,n_order+1,2n_order + 1,2n_order + 1),-1,-1,-1)
end

eulerian_polys = Vector{Int64}[[1],[1,1],[1,4,1],[1,11,11,1],[1,26,66,26,1],[1,57,302,302,57,1]]
function eulerian_poly(n)
  while n > length(eulerian_polys)
    this_N = length(eulerian_polys) + 1
    poly = [1]
    prev_poly = [eulerian_polys[this_N - 1];0]
    for this_k = 1:(this_N-1)
      # https://oeis.org/wiki/Eulerian_numbers,_triangle_of
      push!(poly,(this_N - this_k) * prev_poly[1+(this_k-1)] + (this_k + 1) * prev_poly[1+this_k])
    end
    push!(eulerian_polys,poly)
  end
  p = eulerian_polys[n] :: Vector{Int64} # Required for type stability :(
  p
end

# A "Y quantity" is a normal-ordered correlator between harmonic oscillator
# bosons b and b† of the form:
#
#                        M operators on left
#                          ┌─────┴─────┐
#                          │  m† of b† │
#                          ┌──┴──┐     │
#   Y{n,M,m†} = ∑_{μ,k} <μ|b†...b† b...b|k><k|b†...b† b...b|μ>
#                          └───────────────────┬──────────┘
#                                        2n total operators
#
#                        × exp(-β Eμ)/Z × δ_{ω = ω₀(M-2m†)}
#
# where Eμ is the energy of the base state |μ>, and ω₀ is the energy
# of one b† boson excitation. Unless there are (n-m†) many b† operators
# and (n-M-m†) b operators on the right, the Y-quantity will be zero due to
# not returning |μ> to itself by the time <μ| is reached.
#
# [[ N.B.: If interactions are allowed, then the density matrix would no
# longer be diagonal in the occupation number basis, and additional
# considerations would be needed. ]]
#
# This function returns a list indexed by the integer (M-2m†) containing
# the coefficient multiplying the delta-function. This result only depends
# on the three integers n, M, m†, and the product βω₀ (because Eμ = ω₀<μ|b†b|μ>).
y_quantity_harmonic(n,M,mDagger) = M - 2mDagger
function y_quantity(stored,n::Int64,M::Int64,mDagger::Int64;betaOmega::Float64)

  # Check if already computed
  if isnan(stored[n,M,mDagger])
    # If not, fill in the whole lookup table
    if n == 0
      @assert M == 0
      @assert mDagger == 0
      stored[n,M,mDagger] = 1
    end

    if n > 0
      for M0 = 0:(2n)
        for md = 0:M0
          #println()
          #println("Calculating: Y$n$M0$md")

          # These are not valid/out of bounds and need to be skipped
          if (md > n) || ((M0 - md) > n)
            #println("[[invalid Y object]]")
            #println()
            continue
          end

          # The string b†...b†b...b|k><k|b†...b†b...b, when acting on an
          # occupation number state |n>, gives rise to a sequence of intermediate
          # states with varying occupation numbers. The eigenvalue of |n> depends
          # on these intermediate occupation numbers, so we trace out the trajectory
          # of intermediate n values (relative to the starting value) here.
          #
          # More specifically, for each operator making
          # a transition |n> → |n±1>, we record max(n,n±1), since that
          # operator contributes a factor √max(n,n±1) to the overall eigenvalue.
          curr_n = 0
          n_traj = Int64[]
          for (sz_create,sz_destroy) = [(n-md,n-M0+md),(md,M0-md)]
            #println()
            #println("Going down!")
            for j = 1:sz_destroy
              #println("(n + $curr_n)")
              push!(n_traj,curr_n)
              curr_n = curr_n - 1
            end

            #println()
            #println("Going up!")
            for j = 1:sz_create
              curr_n = curr_n + 1
              #println("(n + $curr_n)")
              push!(n_traj,curr_n)
            end
          end

          # The n-trajectory has two of each entry, since it
          # needs to return back to the original number by the end.
          # Thus, when we sort, it will have the form:
          #
          #   n_traj = [A,A,B,B,C,C,...]
          #
          # and we get all intermediate occupation numbers
          # in sorted order by only picking every other element
          # of n_traj.
          sort!(n_traj)
          factors = view(n_traj,1:2:(2n))

          # Iteratively build the eigenvalue starting from `p(n) = 1' using:
          # 
          #   p(n) → (n + a) p(n)
          #
          # where `a' is one of the intermediate occupation numbers
          # relative to n. Notice that if the state is ever annihilated,
          # we naturally get a factor (n - n) = 0, so those terms
          # are avoided!
          poly = zeros(n+1)
          poly[1] = 1 # Constant 1 to start with
          for j = 1:n # Iteratively include factors
            for l = (n+1):-1:2
              poly[l] = factors[j] * poly[l] + poly[l-1]
            end
            poly[1] = factors[j] * poly[1]
          end

          # It can be shown that the Y-quantity equals:
          #
          #   ∑_p cp nB^p * (Ap0 + Ap1 exp(βω) + Ap2 exp(2βω) + ... + Ap{p-1} exp((p-1)βω))
          # 
          # where Apk is the k-th coefficient of the Eulerian polynomial A_p.
          # However, exp((p-1)βω) can be very large, and nB^p very small,
          # so we instead compute
          #
          #   ∑_p cp ∑_{k=0,…,p-1} Apk [exp(kβω) nB^p]
          #
          # iwth the term in brackets being actually computed as
          #
          #   exp log [exp(kβω) nB^p] = exp(kβω - p log(exp(βω) - 1))
          #
          # to avoid dramatic loss of precision.
          tot = 0.0
          for p = 0:n
            cp = poly[p+1]
            if p == 0
              tot += cp
            else
              Ap = eulerian_poly(p)
              for k = 0:(p-1)
                term = exp(k*betaOmega - p * log(exp(betaOmega) - 1))
                Apk = Ap[k+1]
                tot += cp * Apk * term
              end
            end
          end
          stored[n,M0,md] = tot
        end
      end
    end
  end

  if any(isnan.(stored[n,M,mDagger]))
    display(stored[n,M,mDagger])
    error("Y quantity computation failed! n = $n, M = $M, m† = $mDagger")
  end

  return stored[n,M,mDagger]
end


# Convert a bit string (1 -> creation, 0 -> deletion) to a
# wick-normal-form matrix by appending one bit at a time
function wickify(bit_string_A)
  wick_A = empty_wick()
  for bit = reverse(bit_string_A)
    wick_A = wick_append_left(bit,wick_A)
  end

  wick_A
end

# A wick-normal-form matrix is a linear combination of several
# small normal-ordered operators. Given two such matrices, the
# correlation between the two total operators is a linear combination
# of correlations of normal-ordered operators. A "Y object" is a correlation
# between normal-order operators. So, we compute the total correlation
# in terms of Y objects, and this is what the function does
function wick_to_y_object(stored,wick_A,wick_B;betaOmega, verbose = false)
  a_terms = findall(.!iszero.(wick_A))
  b_terms = findall(.!iszero.(wick_B))
  tot = zeros(Float64,-2:2) #0 * y_quantity(stored,0,0,0;betaOmega)
  for at = a_terms, bt = b_terms
    n_create_a = at.I[1] - 1
    n_destroy_a = at.I[2] - 1
    n_create_b = bt.I[1] - 1
    n_destroy_b = bt.I[2] - 1

    this_wick_A = 0 .* wick_A
    this_wick_A[at] = 1

    this_wick_B = 0 .* wick_B
    this_wick_B[bt] = 1

    # For a <μ|...|k><k|...|μ> type correlation, we need to
    # return to the ground state at the end of the day, so
    # the number of creation and deletion events had better match!
    #@assert n_create_a + n_create_b == n_destroy_a + n_destroy_b
    if !(n_create_a + n_create_b == n_destroy_a + n_destroy_b)
      if verbose
        print("Skipping forbidden Y-object (")
        show_wick(this_wick_A)
        print(", ")
        show_wick(this_wick_B)
        println(") which doesn't conserve magnon number!")
      end
      continue
    end

    m_dagger = n_create_a
    n = n_create_b + m_dagger
    M = n_destroy_a + m_dagger

    #=
    print("Retreiving: Y$n$M$m_dagger ~ (")
    show_wick(this_wick_A)
    print(", ")
    show_wick(this_wick_B)
    println(")")
    =#

    harmonic = y_quantity_harmonic(n,M,m_dagger)
    tot[harmonic] += wick_A[at] * wick_B[bt] * y_quantity(stored,n,M,m_dagger;betaOmega)
    if any(isnan.(tot))
      display(stored)
      display(tot)
      show_wick(wick_A)
      println()
      show_wick(wick_B)
      error("NaN value for wick_to_y_object at n = $n, M = $M, m† = $m_dagger")
    end
  end
  tot
end


# The martix entry Xij describes the normal ordered operator
# with i creation and j deletion operators (zero-indexed)
empty_wick() = ones(Float64,1,1)
function wick_append_left(new_b, X)
  if new_b == 1 # Left append creation operator; trivial
    [zeros(Float64,1,size(X,2)); X]
  else
    X_new = 0 * [zeros(Float64,size(X,1),1) ;; X]
    for r = 1:size(X,1)
      for c = 1:size(X,2)
        n_create = r - 1
        if n_create > 0
          X_new[r-1,c] += X[r,c] * n_create
        end
        X_new[r,c+1] += X[r,c]
      end
    end
    X_new
  end
end

function show_bitstring(X;b = 'b')
  for i = 1:length(X)
    print(b)
    X[i] == 1 && print("†")
  end
end

function show_wick(X;b = 'b')
  leading = true
  for r = 1:size(X,1)
    for c = 1:size(X,2)
      a = X[r,c]
      if !iszero(a)
        if a > 0
          !leading && print(" + ")
        else
          print(" - ")
        end
        if (r == 1 && c == 1) || a != 1.0
          print("$(Sunny.number_to_math_string(abs(a);digits = 2))")
        end
        for i = 1:(r-1)
          print(b)
          print("†")
        end
        for i = 1:(c-1)
          print(b)
        end
        leading = false
      end
    end
  end
end

function finite_spin_wave_Vmats(sys; polyatomic = true)
  # Assess if all unit cells have the same data
  all_repeats = allequal([sys.dipoles[i,:] for i = Sunny.eachcell(sys)]) && allequal([sys.coherents[i,:] for i = Sunny.eachcell(sys)])
  if !all_repeats
    @warn "Not all unit cells in the system have the same spin data! Collapsing to first unit cell anyway"
  end

  sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
  swt = SpinWaveTheory(sys_collapsed)

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

  Hs = zeros(ComplexF64,size(formula.calc_intensity.H)...,comm_ixs.indices...)
  Vs = zeros(ComplexF64,na,nf,2,na*nf,2,comm_ixs.indices...)
  disps = zeros(Float64,size(formula.calc_intensity.H,1)÷2,comm_ixs.indices...)

  for i in comm_ixs
    k_comm = Sunny.Vec3((i.I .- 1) ./ sys.latsize)
    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, k_comm)
    Hmat = formula.calc_intensity.H
    if sys.mode == :SUN
        Sunny.swt_hamiltonian_SUN!(Hmat, swt, q_reshaped)
    elseif sys.mode == :dipole
        Sunny.swt_hamiltonian_dipole!(Hmat, swt, q_reshaped)
    end
    Hs[:,:,i] .= Hmat

    try
      disps[:,i] .= Sunny.bogoliubov!(formula.calc_intensity.V,Hmat)
    catch e
      println("Failed to perform Bogoliubov transform at $k_comm")
      display(e)
    end

    Vs[:,:,:,:,:,i] .= copy(reshape(formula.calc_intensity.V,na,nf,2,na*nf,2))
  end

  Hs, Vs, disps, swt.data.local_rotations
end

function polyatomic_bzs(crystal)
  na = Sunny.natoms(crystal)
  iszero_symprec(x) = abs(x) < crystal.symprec
  ΔRs = [map(x -> iszero_symprec(x) ? Inf : x,abs.(crystal.positions[i] - crystal.positions[j])) for i = 1:na, j = 1:na]
  round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))
end

function blit_one_over_S_spectrum(sys,dE,npos;beta = 1.0,two_magnon = false)
  Hs, Vs, disps, Us = finite_spin_wave_Vmats(sys;polyatomic = false)
  ssf = zero_magnon_sector(Hs,Vs,sys.crystal;betaOmega = beta * 0.0)
  one_magnon_canvas = blit_one_magnon(Hs,Vs,sys.crystal,disps,dE,npos;betaOmega = beta * disps)

  if two_magnon
    two_magnon_canvas = blit_two_magnon(Hs,Vs,sys.crystal,disps,dE,npos;betaOmega = beta * disps)
  end

  # Find polyatomic required number of unit cells:
  nbzs = polyatomic_bzs(sys.crystal)

  # The limited range of wave vectors that is suitable for
  # each individual sublattice
  nx,ny,nz = sys.latsize

  # The full range of wave vectors that will be in the output!
  Nx,Ny,Nz = sys.latsize .* nbzs

  comm_ixs = CartesianIndices((Nx,Ny,Nz))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  # TODO: treat frequencies correctly; not "harmonics", bin it instead!
  # 1/S expansion
  s = (sys.Ns[1]-1) / 2

  na = size(Vs,1)
  nf = size(Vs,2)

  # kx, ky, kz; m,n (sublattice); xyz, xyz ; energy
  S_local = zeros(ComplexF64,Nx,Ny,Nz,na,na,3,3,2npos+1)

  for m = 1:na, n = 1:na, bzx = 1:nbzs[1], bzy = 1:nbzs[2], bzz = 1:nbzs[3]
    # The indices of the BZ being computed right now
    xs = (1:nx) .+ (bzx - 1) * nx
    ys = (1:ny) .+ (bzy - 1) * ny
    zs = (1:nz) .+ (bzz - 1) * nz

    ks = [([x,y,z] .- 1) ./ [nx,ny,nz] for x = xs, y = ys, z = zs]

    # The sublattice distance offset for this (m,n) flavor pair
    atom_m = m
    atom_n = n
    δ = sys.crystal.positions[atom_n] - sys.crystal.positions[atom_m]
    #println("m = $m, n = $n, δ = $δ")
    phases = exp.(-(2pi * im) .* map(k -> δ⋅k,ks))

    @assert size(one_magnon_canvas,5) == 1
    @assert size(one_magnon_canvas,8) == 1

    # Sxx
    S_local[xs,ys,zs,m,n,1,1,:] .+= 2 * s * sum([one_magnon_canvas[:,:,:,m,1,i,n,1,j,2,:] for i = 1:2, j = 1:2])/4

    # Sxy
    S_local[xs,ys,zs,m,n,1,2,:] .+= 2 * s * sum([[1,-1][j] * one_magnon_canvas[:,:,:,m,1,i,n,1,j,2,:] for i = 1:2, j = 1:2])/(4 * im)

    # Syx
    S_local[xs,ys,zs,m,n,2,1,:] .+= 2 * s * sum([[1,-1][i] * one_magnon_canvas[:,:,:,m,1,i,n,1,j,2,:] for i = 1:2, j = 1:2])/(4 * im)

    # Syy
    S_local[xs,ys,zs,m,n,2,2,:] .+= 2 * s * sum([[1,-1][i] * [1,-1][j] * one_magnon_canvas[:,:,:,m,1,i,n,1,j,2,:] for i = 1:2, j = 1:2])/(-4)

    # SSF is at zero energy!
    S_local[xs,ys,zs,m,n,3,3,npos+1] .= s * s * prod([nx,ny,nz]) * sum(ssf,dims=4)

    S_local[xs,ys,zs,m,n,3,3,:] .+= -s * one_magnon_canvas[:,:,:,m,1,2,n,1,1,1,:]
    S_local[xs,ys,zs,m,n,3,3,:] .+= -s * one_magnon_canvas[:,:,:,m,1,2,n,1,1,3,:]

    if two_magnon
      # Sxx
      # i       j
      # (a + a†)(a†aa + a†a†a) → 
      S_local[xs,ys,zs,m,n,1,1,:] .+= -sum([two_magnon_canvas[:,:,:,m,n,1,i,1,[2,2][j],1,j,1,[1,1][j],2,:] for i = 1:2, j = 1:2])/8
      # i             j
      # (a†aa + a†a†a)(a + a†) → 
      S_local[xs,ys,zs,m,n,1,1,:] .+= -sum([two_magnon_canvas[:,:,:,m,n,1,[2,2][i],1,i,1,[1,1][i],1,j,4,:] for i = 1:2, j = 1:2])/8

      # Sxy
      # i       j
      # (a + a†)(a†aa - a†a†a) → 
      S_local[xs,ys,zs,m,n,1,2,:] .+= -sum([[1,-1][j] * two_magnon_canvas[:,:,:,m,n,1,i,1,[2,2][j],1,j,1,[1,1][j],2,:] for i = 1:2, j = 1:2])/(8im)
      # i             j
      # (a†aa + a†a†a)(a - a†) → 
      S_local[xs,ys,zs,m,n,1,2,:] .+= -sum([[1,-1][j] * two_magnon_canvas[:,:,:,m,n,1,[2,2][i],1,i,1,[1,1][i],1,j,4,:] for i = 1:2, j = 1:2])/(8im)

      # Syx
      # i       j
      # (a - a†)(a†aa + a†a†a) → 
      S_local[xs,ys,zs,m,n,2,1,:] .+= -sum([[1,-1][i] * two_magnon_canvas[:,:,:,m,n,1,i,1,[2,2][j],1,j,1,[1,1][j],2,:] for i = 1:2, j = 1:2])/(8im)
      # i             j
      # (a†aa - a†a†a)(a + a†) → 
      S_local[xs,ys,zs,m,n,2,1,:] .+= -sum([[1,-1][i] * two_magnon_canvas[:,:,:,m,n,1,[2,2][i],1,i,1,[1,1][i],1,j,4,:] for i = 1:2, j = 1:2])/(8im)

      # Syy
      # i       j
      # (a - a†)(a†aa - a†a†a) → 
      S_local[xs,ys,zs,m,n,2,2,:] .+= -sum([[1,-1][i] * [1,-1][j] * two_magnon_canvas[:,:,:,m,n,1,i,1,[2,2][j],1,j,1,[1,1][j],2,:] for i = 1:2, j = 1:2])/(-8)
      # i             j
      # (a†aa - a†a†a)(a - a†) → 
      S_local[xs,ys,zs,m,n,2,2,:] .+= -sum([[1,-1][i] * [1,-1][j] * two_magnon_canvas[:,:,:,m,n,1,[2,2][i],1,i,1,[1,1][i],1,j,4,:] for i = 1:2, j = 1:2])/(-8)

      # Szz
      # i     j
      # (-a†a)(-a†a)
      S_local[xs,ys,zs,m,n,3,3,:] .+= two_magnon_canvas[:,:,:,m,n,1,2,1,1,1,2,1,1,3,:]
    end

    S_local[xs,ys,zs,m,n,:,:,:] .*= phases
  end

  S_global = 0 .* copy(S_local)
  #display(Us[1])
  #display(Us[2])

  for i = 1:3, j = 1:3, k = 1:3, l = 1:3, m = 1:na, n = 1:na
    S_global[:,:,:,m,n,i,j,:] .+= Us[m][i,k] * Us[n][j,l] * S_local[:,:,:,m,n,k,l,:]
  end

  S_local, S_global
end


function render_one_over_S_spectrum(sys;beta = 1.0,two_magnon = false)
  Hs, Vs, disps, Us = finite_spin_wave_Vmats(sys;polyatomic = false)
  ssf = zero_magnon_sector(Hs,Vs,sys.crystal;betaOmega = beta * 0.0)
  #corr_a_mat, c_1_aa, c_aa_1 = one_magnon_sector(Hs,Vs;betaOmega = beta * disps)
  one_magnon_c = tabulate_one_magnon(Hs,Vs,sys.crystal;betaOmega = beta * disps)

  if two_magnon
    two_magnon_c = two_magnon_sector(Hs,Vs,sys.crystal;betaOmega = beta * disps)
  end


  # Find polyatomic required number of unit cells:
  nbzs = polyatomic_bzs(sys.crystal)

  # The limited range of wave vectors that is suitable for
  # each individual sublattice
  nx,ny,nz = sys.latsize

  # The full range of wave vectors that will be in the output!
  Nx,Ny,Nz = sys.latsize .* nbzs

  comm_ixs = CartesianIndices((Nx,Ny,Nz))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  # TODO: treat frequencies correctly; not "harmonics", bin it instead!
  # 1/S expansion
  s = (sys.Ns[1]-1) / 2

  na = size(Vs,1)
  nf = size(Vs,2)

  # kx, ky, kz; [band]; m,n (sublattice); harmonic; xyz, xyz
  S_local = zeros(ComplexF64,Nx,Ny,Nz,na*nf,na,na,5,3,3)

  for m = 1:na, n = 1:na, bzx = 1:nbzs[1], bzy = 1:nbzs[2], bzz = 1:nbzs[3]
    # The indices of the BZ being computed right now
    xs = (1:nx) .+ (bzx - 1) * nx
    ys = (1:ny) .+ (bzy - 1) * ny
    zs = (1:nz) .+ (bzz - 1) * nz

    ks = [([x,y,z] .- 1) ./ [nx,ny,nz] for x = xs, y = ys, z = zs]

    # The sublattice distance offset for this (m,n) flavor pair
    atom_m = m
    atom_n = n
    δ = sys.crystal.positions[atom_n] - sys.crystal.positions[atom_m]
    #println("m = $m, n = $n, δ = $δ")
    phases = exp.(-(2pi * im) .* map(k -> δ⋅k,ks))

    @assert size(one_magnon_c,6) == 1
    @assert size(one_magnon_c,9) == 1

    # Sxx
    S_local[xs,ys,zs,:,m,n,:,1,1] .+= 2 * s * sum([one_magnon_c[:,:,:,:,m,1,i,n,1,j,:,2] for i = 1:2, j = 1:2])/4

    # Sxy
    S_local[xs,ys,zs,:,m,n,:,1,2] .+= 2 * s * sum([[1,-1][j] * one_magnon_c[:,:,:,:,m,1,i,n,1,j,:,2] for i = 1:2, j = 1:2])/(4 * im)

    # Syx
    S_local[xs,ys,zs,:,m,n,:,2,1] .+= 2 * s * sum([[1,-1][i] * one_magnon_c[:,:,:,:,m,1,i,n,1,j,:,2] for i = 1:2, j = 1:2])/(4 * im)

    # Syy
    S_local[xs,ys,zs,:,m,n,:,2,2] .+= 2 * s * sum([[1,-1][i] * [1,-1][j] * one_magnon_c[:,:,:,:,m,1,i,n,1,j,:,2] for i = 1:2, j = 1:2])/(-4)

    # For c_{1,aa} and c_{aa,1}, our inference of the total momentum k will always have placed everything
    # in k = 0:
    @assert abs(sum(one_magnon_c[:,:,:,:,:,:,:,:,:,:,:,1]) - sum(one_magnon_c[1,1,1,:,:,:,:,:,:,:,:,1])) < 1e-12
    @assert abs(sum(one_magnon_c[:,:,:,:,:,:,:,:,:,:,:,3]) - sum(one_magnon_c[1,1,1,:,:,:,:,:,:,:,:,3])) < 1e-12

    # SSF is always band #1 ???
    S_local[xs,ys,zs,1,m,n,:,3,3] .= s * s * prod([nx,ny,nz]) * ssf[:,:,:,:]

    S_local[xs,ys,zs,:,m,n,:,3,3] .+= -s * one_magnon_c[:,:,:,:,m,1,2,n,1,1,:,1]
    S_local[xs,ys,zs,:,m,n,:,3,3] .+= -s * one_magnon_c[:,:,:,:,m,1,2,n,1,1,:,3]

    if two_magnon
      for dm = 1:(na*nf) # temporary hack for stacked two magnon
        # Sxx

        # i       j
        # (a + a†)(a†aa + a†a†a) → 
        S_local[xs,ys,zs,dm,m,n,:,1,1] .+= -sum([two_magnon_c[:,:,:,dm,dm,m,1,i,n,1,[2,2][j],n,1,j,n,1,[1,1][j],:,2] for i = 1:2, j = 1:2])/8

        # i             j
        # (a†aa + a†a†a)(a + a†) → 
        S_local[xs,ys,zs,dm,m,n,:,1,1] .+= -sum([two_magnon_c[:,:,:,dm,dm,m,1,[2,2][i],m,1,i,m,1,[1,1][i],n,1,j,:,4] for i = 1:2, j = 1:2])/8

        # TODO: other than Sxx
      end
    end

    S_local[xs,ys,zs,:,m,n,:,:,:] .*= phases
  end

  S_global = 0 .* copy(S_local)
  #display(Us[1])
  #display(Us[2])

  for i = 1:3, j = 1:3, k = 1:3, l = 1:3, m = 1:na, n = 1:na
    S_global[:,:,:,:,m,n,:,i,j] .+= Us[m][i,k] * Us[n][j,l] * S_local[:,:,:,:,m,n,:,k,l]
  end

  S_local, S_global
end

function sunny_swt_spectrum(sys;polyatomic = true)
  all_repeats = allequal([sys.dipoles[i,:] for i = Sunny.eachcell(sys)]) && allequal([sys.coherents[i,:] for i = Sunny.eachcell(sys)])
  if !all_repeats
    @warn "Not all unit cells in the system have the same spin data! Collapsing to first unit cell anyway"
  end

  sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
  swt = SpinWaveTheory(sys_collapsed)

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


  # kx, ky, kz; [band]; m,n (sublattice); harmonic; xyz, xyz
  Nx,Ny,Nz = sys.latsize .* nbzs
  S = zeros(ComplexF64,Nx,Ny,Nz,na*nf,3,3)
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

function validate_one_magnon_sector(sys;beta = 85.0)
  disps_sunny, S_sunny = sunny_swt_spectrum(sys)
  S = render_one_over_S_spectrum(sys;beta)
  S_summed = sum(S[2],dims=[5,6])[:,:,:,:,1,1,4,:,:] # perform sublattice sum and extract one-magnon band

  is_valid = true
  max_deviation = 0
  for nx = 1:size(S_sunny,1), ny = 1:size(S_sunny,2), nz = 1:size(S_sunny,3), band = 1:size(S_sunny,4), i = 1:3, j = 1:3
    spec_sunny = real(S_sunny[nx,ny,nz,band,i,j])
    spec_one_magnon = 2real(S_summed[nx,ny,nz,band,i,j])
    diff = spec_one_magnon-spec_sunny
    if abs(diff) > 1e-12
      is_valid = false
      println()
      printstyled("Deviation",color = :red)
      println(" from Sunny LSWT detected at C$("xyz"[i])$("xyz"[j]){k → [$nx,$ny,$nz]} band $band:")
      println("  Expected: $spec_sunny")
      deviation = (spec_one_magnon-spec_sunny)/spec_sunny
      println("    Actual: $spec_one_magnon\t ($(Sunny.number_to_simple_string(deviation*100,digits = 5))% off from expected)")
      if abs(deviation) > abs(max_deviation)
        max_deviation = deviation
      end
    end
  end

  if is_valid
    printstyled("Unqualified pass!",color = :green)
    println()
  else
    if abs(max_deviation) < 0.01
      printstyled("Qualifed pass; maximum deviation <1%",color = :yellow)
      println()
      println("  Worst deviation: $(Sunny.number_to_simple_string(max_deviation*100,digits = 5))% off from expected")
    else
      printstyled("Substantial deviation from Sunny LSWT:",color = :red)
      println()
      println("  Worst deviation: $(Sunny.number_to_simple_string(max_deviation*100,digits = 5))% off from expected")
    end
  end

  swt = SpinWaveTheory(Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys)))
  mag_correction = Sunny.magnetization_lswt_correction(swt;atol = 1e-6)
  na = size(S[2],5)
  n_cell = prod(sys.latsize)
  println("Magnetization correction from Sunny: $mag_correction (per site)")
  println("                         Calculated: $((sum(S[1][1,1,1,1,:,:,3,3,3]) - na * na * n_cell) / (na*n_cell))) (per site)")
end

function raster_spec(spec,disps,dE,npos,i,j)
  nx,ny,nz = size(disps)[2:4]
  Nx,Ny,Nz = size(spec)[1:3]
  N = size(spec,5)
  raster = zeros(ComplexF64,Nx,Ny,Nz,2npos + 1)
  n_harmonic = size(spec,7)
  for ix = 1:Nx, iy = 1:Ny, iz = 1:Nz
    for band = 1:size(disps,1), h = 1:n_harmonic, m = 1:N, n = 1:N
      harmonic = h - (n_harmonic ÷ 2) - 1
      E = harmonic * disps[band,mod1(ix,nx),mod1(iy,ny),mod1(iz,nz)]
      bin = 1 + floor(Int64,E/dE + (npos + 1/2))
      if bin < 1 || bin > 2npos + 1
        println("Outside bin range!")
        continue
      end
      raster[ix,iy,iz,bin] += spec[ix,iy,iz,band,m,n,h,i,j]
    end
  end
  raster
end

function grid_it_2d(sys;beta = 8.0)
  Sloc, Sglob = render_one_over_S_spectrum(sys; beta)
  Hs, Vs, disps = finite_spin_wave_Vmats(sys;polyatomic = false)
  f = Figure()
  display(f)
  for i = 1:3, j = 1:3
    s = log10.(abs.(raster_spec(Sglob,disps,0.4,30,i,j)[:,1,1,:]))
    ax = Axis(f[i,(j - 1) * 2 + 1])
    hm = heatmap!(ax,s,colorrange = (-8,2))
    Colorbar(f[i,(j-1)*2+2],hm)
  end
end

# Returns the index of the wavevector which completes the given
# list of wavevector indices in a momentum-conserving way.
# The ks should be a list [[ix,iy,iz],[jx,jz,jz],…], and
# daggers should be a bit string [0,1,1,0,…] with 1 for a†
function momentum_conserving_index(ks::Vector{Vector{Int64}},daggers::Vector{Int64},ns::Tuple{Int64,Int64,Int64})
  nx,ny,nz = ns
  dagger_signs = (-1) .^ daggers
  momentum_so_far = Int64[0,0,0]
  for i = 1:length(ks)
    momentum_so_far .+= dagger_signs[i] * (ks[i] .- 1)
  end
  #momentum_so_far = sum(dagger_signs[1:end-1] .* map(k -> k .- 1,ks);init = [0,0,0])
  I = mod1.(1 .- momentum_so_far * dagger_signs[end],[nx,ny,nz])

  #=
  freqs = fftfreq.(ns)
  push!(ks,I)
  println("Momentum conservation:")
  for i = 1:length(ks)
    if i == length(ks)
      print(" => k$i → [")
    else
      print("  | k$i → [")
    end
    for j = 1:3
      print(Sunny.number_to_math_string(freqs[j][ks[i][j]],digits=2))
      if j < 3
        print(", ")
      end
    end
    print("] ")
    if daggers[i] == 1
      println("(†)")
    else
      println("( )")
    end
    #if i == length(ks)
      #println(" [Inferred]")
    #else
      #println()
    #end
  end
  =#
  CartesianIndex(ntuple(i -> I[i],3))
end

function zero_magnon_sector(Hs,Vs,cryst;betaOmega)
  # This is correlators with exactly *zero* boson operators
  # Namely, it's the C_{1,1} correlator.
  yy = empty_y_matrix()
  yobj = wick_to_y_object(yy,wickify([]),wickify([]);betaOmega)
  nx,ny,nz = size(Vs)[6:8]
  ssf = zeros(ComplexF64,nx,ny,nz,length(yobj))
  ssf[1,1,1,:] .= yobj[-2:2]
  ssf
end

function one_magnon_sector(f,Hs,Vs,cryst;betaOmega)
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # Fix phases! This implements "commutes with dagger" property
  Vs_fix = phase_fix_Vs(Vs)

  # One Y matrix for each oscillator; only need ω > 0 to cover all oscillators
  Y_storage = Array{OffsetArray{ComplexF64}}(undef,nx,ny,nz,na*nf)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz, i = 1:na*nf
    Y_storage[ix,iy,iz,i] = empty_y_matrix()
  end

  # The correlations are labelled by:
  # - The total momentum (transfer), labelled by [nx,ny,nz]
  # - The band number of the specific eigenmode (there is exactly one)
  # - The site, flavor and dagger configuration of the two a-bosons being correlated
  # - The harmonic of the eigenmode
  # - The "partition number" which tells the position of the comma: ,aa ; a,a ; aa,
  correlator = zeros(ComplexF64,nx,ny,nz,na*nf,na,nf,2,na,nf,2,size(empty_y_matrix(),4),3)

  # Loop over the momentum of the first operator in the correlation
  for x1 = 1:nx, y1 = 1:ny, z1 = 1:nz
    #println("=== Working on k1 = $((x1,y1,z1)) ===")

    # Loop over a-boson dagger configuration (0 = no dagger, 1 = dagger)
    for iDag = [0,1], jDag = [0,1]
      if any(isnan.(correlator))
        error("Nan")
      end

      # Infer k2 based on momentum conservation for a-bosons.
      # This requires translational symmetry between unit cells, and
      # the constraint is specifically: ∑ ±ki = 0, with signs
      # depending on the dagger configuration:
      #   - sign for non-dagger a-boson or
      #   + sign for dagger a-boson.
      ks = [[x1,y1,z1]]
      x2,y2,z2 = momentum_conserving_index(ks,[iDag,jDag],(nx,ny,nz)).I

      # Perform the Bogoliubov expansion, which writes a-bosons in terms
      # of b-bosons. Schemtically:
      #
      #   a{k} = α b{k} + β [b{-k}]†
      # 
      # The coefficients in the expansion are stored in tables. There
      # is one table for each flavor/sublattice of a-boson and for
      # each momentum k of the lattice of unit cells. All operators
      # appearing in the same table have the same fourier transform coefficients
      # (in this example, they all expand as X{k} ∼ exp(ik⋅Rj) X{j}).
      #
      # exp(ikR)|  b{k}  | [b{-k}]†
      # --------+--------+----------+
      #   a{k}  |   α    |    β
      # --------+--------+----------+
      # [a{-k}]†|   γ    |    δ
      #
      # In order to expand each of a{±k} and [a{±k}]†, the table for exp(-ikR) is
      # also needed. It can be obtained for free by conjugating the existing table:
      #
      # exp(-ikR)|  b{-k} | [b{k}]†
      # ---------+--------+----------+
      #   a{-k}  |   δ*   |    γ*
      # ---------+--------+----------+
      # [a{k}]†  |   β*   |    α*
      #
      # or by directly computing the Bogoliubov transform at -k, as is done here.
      #
      # Which table we need to consult depends on the dagger configuration [iDag,jDag].
      # For example, a{k} needs the exp(ikR) table, but [a{k}]† needs the exp(i(-k)R) table:
      table_for_k1 = iDag == 1 ? negate_momentum((x1,y1,z1),(nx,ny,nz)) : (x1,y1,z1)
      table_for_k2 = jDag == 1 ? negate_momentum((x2,y2,z2),(nx,ny,nz)) : (x2,y2,z2)

      # Explicitly for reference, these are the coefficients in the expansion
      #
      #   ak1{site i,flavor f} = ∑_m Vk1[i,f,1,m,1] b{k1,mode m} + Vk1[i,f,1,m,2] [b{-k1,mode m}]†
      #
      # for non-dagger a-bosons, or
      #
      #   [ak1{site i,flavor f}]† = ∑_m Vk1[i,f,2,m,1] [b{k1,mode m}]† + Vk1[i,f,2,m,2] b{-k1,mode m}
      #
      # for dagger a-bosons.
      Vk1 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k1))
      Vk2 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k2))

      # Loop over the possible partitions of the operators in the correlator:
      #
      #   C{1,aa} ; C{a,a} ; C{aa,1}
      #
      for partition = 1:3
        # Infer total k based on momenta left of the comma.
        # This constraint arises from the sum inside the fourier transform C(Δ) → C(k):
        #
        #   ∑_iΔ exp(ikΔ) C{(…)i,(…)i+Δ} = C_{(…)-k,(…)k}
        #                                     └───┘
        #                           this part has momentum -k

        # Momenta and dagger configuration of that part:
        ks_part = [[x1,y1,z1],[x2,y2,z2]][1:partition-1]
        daggers_part = [iDag,jDag][1:partition-1]

        # Infer what momentum should be added to the left of the comma to reach zero
        push!(daggers_part,0) # Gives the correct sign for k
        total_k = momentum_conserving_index(ks_part,daggers_part,(nx,ny,nz))

        # Loop over dagger configuration of bb and mode label m
        for left = [0,1], right = [0,1], m = 1:(na*nf)

          # Given that a{k} expands to include both b{k} and [b{-k}]† (see tables above),
          # we need to compute which eigenmode (i.e. which k label) we end up with in each term.
          # This is important because the one-magnon-gas-correlator only applies to b-bosons
          # which *belong to the same eigenmode*.
          #
          # The answer is that we need to negate the momentum one time each time we encounter
          # a dagger in the expansion (at either the a-boson or b-boson level).
          bk1 = xor(left,iDag) == 1 ? negate_momentum((x1,y1,z1),(nx,ny,nz)) : (x1,y1,z1)
          bk2 = xor(right,jDag) == 1 ? negate_momentum((x2,y2,z2),(nx,ny,nz)) : (x2,y2,z2)

          # Optimization: Only contributes if they come from the same eigenmode!
          if bk1 != bk2
            continue
          end

          left_mode = (bk1[1],bk1[2],bk1[3],m)
          right_mode = (bk2[1],bk2[2],bk2[3],m)
          omgc = one_magnon_gas_correlator(left_mode,right_mode,left,right,partition;betaOmega = betaOmega[m,x1,y1,z1], Y = Y_storage[bk1[1],bk1[2],bk1[3],m])
          @assert sum(abs.(imag(omgc))) < 1e-12


          #=
          val = omgc[a,b,:,partition]* Vk1[i,1,iDag+1,m,left+1] * Vk2[j,1,jDag+1,n,right+1]
          if sum(abs.(val)) > 1e-12 && !iszero(omgc[a,b,4,partition]) && m == 2 && left == 0 && right == 1 && total_k.I[1] == 3 && i == 2 && j == 1
            println("!! Finite contribution: eigenmodes ($m,$left) and ($n,$right) to HP bosons ($i,$iDag) and ($j,$jDag)")
            println("   Vk1: $(Vk1[i,1,iDag+1,m,left+1]) ($i 1 $iDag $m $left)")
            println("   Vk2: $(Vk2[j,1,jDag+1,n,right+1]) ($j 1 $jDag $n $right)")
            println("   omgc: $(omgc[a,b,:,partition])")
            #println("   Sublattice phase: $sublattice_phase")
            kt = [0,0,0.]
            for L = 1:3
              kt[L] = fftfreq([nx,ny,nz][L],1)[total_k.I[L]]
            end
            println(kt)
            gp = exp.((2pi * im) .* (cryst.positions[j] - cryst.positions[i])⋅kt)
            println("   Greater phase: $gp")

            println("   term: $(sublattice_phase * Vk1[i,1,iDag+1,m,left+1] * Vk2[j,1,jDag+1,n,right+1] * omgc[a,b,:,partition] * gp)")
          end
          =#

          n = m
          for i = 1:na, j =1:na
            contrib = Vk1[i,1,iDag+1,m,left+1] * Vk2[j,1,jDag+1,n,right+1] * omgc
            f(contrib, total_k, m, i, iDag, j, jDag, partition)
            #view(correlator,total_k,m,i,1,iDag+1,j,1,jDag+1,:,partition) .+= Vk1[i,1,iDag+1,m,left+1] * Vk2[j,1,jDag+1,n,right+1] * omgc
          end
        end
      end
    end
  end

  correlator
end

function tabulate_one_magnon(Hs,Vs,cryst;betaOmega)
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # The correlations are labelled by:
  # - The total momentum (transfer), labelled by [nx,ny,nz]
  # - The band number of the specific eigenmode (there is exactly one)
  # - The site, flavor and dagger configuration of the two a-bosons being correlated
  # - The harmonic of the eigenmode
  # - The "partition number" which tells the position of the comma: ,aa ; a,a ; aa,
  correlator = zeros(ComplexF64,nx,ny,nz,na*nf,na,nf,2,na,nf,2,size(empty_y_matrix(),4),3)

  one_magnon_sector(Hs,Vs,cryst;betaOmega) do contribution, total_k, m, i, iDag, j, jDag, partition
    view(correlator,total_k,m,i,1,iDag+1,j,1,jDag+1,:,partition) .+= contribution
  end
  correlator
end

function blit_one_magnon(Hs,Vs,cryst,disps,dE,npos;betaOmega)
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # The canvas only retains information about:
  # - The total momentum [nx,ny,nz]
  # - The a-bosons being correlated
  # - The partition number
  # - The energy *bin* the contribution should end up in
  canvas = zeros(ComplexF64,nx,ny,nz,na,nf,2,na,nf,2,3,2npos+1)

  n_warn = 0

  one_magnon_sector(Hs,Vs,cryst;betaOmega) do contribution, total_k, m, i, iDag, j, jDag, partition
    n_harmonic = length(contribution)
    for h = 1:n_harmonic
      harmonic = h - (n_harmonic ÷ 2) - 1
      E = harmonic * disps[m,total_k]
      bin = 1 + floor(Int64,E/dE + (npos + 1/2))
      if bin < 1 || bin > 2npos + 1
        if n_warn < 15
          n_warn = n_warn + 1
          println("Outside bin range!")
        elseif n_warn == 15
          n_warn = n_warn + 1
          println("Supressing further warnings...")
        end
        continue
      end
      canvas[total_k,i,1,iDag+1,j,1,jDag+1,partition,bin] += contribution[h]
    end
  end
  canvas
end
function blit_two_magnon(Hs,Vs,cryst,disps,dE,npos;betaOmega)
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # The canvas only retains information about:
  # - The total momentum [nx,ny,nz]
  # - The TWO sites and FOUR flavors of a-bosons being correlated
  # - The partition number
  # - The energy *bin* the contribution should end up in
  canvas = zeros(ComplexF64,nx,ny,nz,na,na,nf,2,nf,2,nf,2,nf,2,5,2npos+1)

  n_warn = 0

  two_magnon_sector(Hs,Vs,cryst;betaOmega) do contribution, total_k, m, n, table_1, table_2, i, iDag, j, jDag, k, kDag, l, lDag, partition
    if !allequal([i,j,k,l][1:partition-1]) || !allequal([i,j,k,l][partition:end])
      # This is a non-2 point correlator!!
      return
    end

    # Open Q: what if partition is 1 or 5??
    left_site = i
    right_site = l

    n_harmonic = size(contribution,1)
    @assert n_harmonic == size(contribution,2)
    for h1 = 1:n_harmonic, h2 = 1:n_harmonic
      harmonic1 = h1 - (n_harmonic ÷ 2) - 1
      harmonic2 = h2 - (n_harmonic ÷ 2) - 1
      E = harmonic1 * disps[m,CartesianIndex(table_1)] 
      E += harmonic2 * disps[n,CartesianIndex(table_2)] 
      bin = 1 + floor(Int64,E/dE + (npos + 1/2))
      if bin < 1 || bin > 2npos + 1
        if n_warn < 15
          n_warn = n_warn + 1
          println("Outside bin range!")
        elseif n_warn == 15
          n_warn = n_warn + 1
          println("Supressing further warnings...")
        end
        continue
      end
      canvas[total_k,left_site,right_site,1,iDag+1,1,jDag+1,1,kDag+1,1,lDag+1,partition,bin] += contribution[h1,h2]
    end
  end
  canvas
end


function blit_four_point_correlator(Hs,Vs,cryst,disps,dE,npos;betaOmega)
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # The canvas only retains information about:
  # - The total momentum [nx,ny,nz]
  # - The a-bosons being correlated
  # - The partition number
  # - The energy *bin* the contribution should end up in
  canvas = zeros(ComplexF64,nx,ny,nz,na,nf,2,na,nf,2,na,nf,2,na,nf,2,5,2npos+1)

  n_warn = 0

  two_magnon_sector(Hs,Vs,cryst;betaOmega) do contribution, total_k, m, n, table_1, table_2, i, iDag, j, jDag, k, kDag, l, lDag, partition
    n_harmonic = size(contribution,1)
    @assert n_harmonic == size(contribution,2)
    for h1 = 1:n_harmonic, h2 = 1:n_harmonic
      harmonic1 = h1 - (n_harmonic ÷ 2) - 1
      harmonic2 = h2 - (n_harmonic ÷ 2) - 1
      E = harmonic1 * disps[m,CartesianIndex(table_1)] 
      E += harmonic2 * disps[n,CartesianIndex(table_2)] 
      bin = 1 + floor(Int64,E/dE + (npos + 1/2))
      if bin < 1 || bin > 2npos + 1
        if n_warn < 15
          n_warn = n_warn + 1
          println("Outside bin range!")
        elseif n_warn == 15
          n_warn = n_warn + 1
          println("Supressing further warnings...")
        end
        continue
      end
      canvas[total_k,i,1,iDag+1,j,1,jDag+1,k,1,kDag+1,l,1,lDag+1,partition,bin] += contribution[h1,h2]
    end
  end
  canvas
end



function two_magnon_sector(f::Function,Hs::Array{ComplexF64,5},Vs::Array{ComplexF64,8},cryst::Sunny.Crystal;betaOmega::Array{Float64,4})
  nx,ny,nz = size(Vs)[6:8]

  na = size(Vs,1)
  nf = size(Vs,2)

  # Fix phases! This implements "commutes with dagger" property
  Vs_fix = phase_fix_Vs(Vs)

  # One Y matrix for each oscillator; only need ω > 0 to cover all oscillators
  Y_storage = Array{OffsetArray{ComplexF64}}(undef,nx,ny,nz,na*nf)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz, i = 1:na*nf
    Y_storage[ix,iy,iz,i] = empty_y_matrix()
  end

  # The correlations are labelled by:
  # - The total momentum (transfer), labelled by [nx,ny,nz]
  # - The band number of the specific eigenmodes (there are always two but they may coincide)
  # - The site, flavor and dagger configuration of the four a-bosons being correlated
  # - The harmonic of the eigenmode
  # - The "partition number" which tells the position of the comma: ,aa ; a,a ; aa,
  #correlator = zeros(ComplexF64,nx,ny,nz,na*nf,na*nf,na,nf,2,na,nf,2,na,nf,2,na,nf,2,size(empty_y_matrix(),4),size(empty_y_matrix(),4),5)

  prog = Progress((nx*ny*nz)^2,"Wavevectors")
  # Loop over the known momenta
  for x1 = 1:nx, y1 = 1:ny, z1 = 1:nz, x2 = 1:nx, y2 = 1:ny, z2 = 1:nz
    next!(prog)

    # Optimization: there can only be a maximum of two classes of momenta overall in the correlator!
    # So if k1 and k2 are in different classes,
    k1_k2_different_class = ((x1,y1,z1) != (x2,y2,z2)) && ((x1,y1,z1) != negate_momentum((x2,y2,z2),(nx,ny,nz)))
    # then the possible values for k3 are
    possible_k3 = if k1_k2_different_class
      # the possible momenta ±k1 and ±k2 in the existing classes
      unique([(x1,y1,z1),(x2,y2,z2),negate_momentum((x1,y1,z1),(nx,ny,nz)),negate_momentum((x2,y2,z2),(nx,ny,nz))])
    else
      # or if k1 and k2 are in the same class, then k3 can be anything
      [(x3,y3,z3) for x3 = 1:nx for y3 = 1:ny for z3 = 1:nz]
    end

    for (x3,y3,z3) in possible_k3
      # Loop over a-boson dagger configuration
      for iDag = [0,1], jDag = [0,1], kDag = [0,1], lDag = [0,1]
        #if any(isnan.(correlator))
          #error("Nan")
        #end

        # Infer remaining momenta based on momentum conservation for a-bosons.
        ks = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
        x4,y4,z4 = momentum_conserving_index(ks,[iDag,jDag,kDag,lDag],(nx,ny,nz)).I

        # Which table we need to consult depends on the dagger configuration
        # [iDag,jDag,kDag,lDag]
        table_for_k1 = iDag == 1 ? negate_momentum((x1,y1,z1),(nx,ny,nz)) : (x1,y1,z1)
        table_for_k2 = jDag == 1 ? negate_momentum((x2,y2,z2),(nx,ny,nz)) : (x2,y2,z2)
        table_for_k3 = kDag == 1 ? negate_momentum((x3,y3,z3),(nx,ny,nz)) : (x3,y3,z3)
        table_for_k4 = lDag == 1 ? negate_momentum((x4,y4,z4),(nx,ny,nz)) : (x4,y4,z4)

        Vk1 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k1))
        Vk2 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k2))
        Vk3 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k3))
        Vk4 = view(Vs_fix,:,:,:,:,:,CartesianIndex(table_for_k4))

        # Loop over the possible partitions of the operators in the correlator:
        #
        #   C{1,aaaa} ; C{a,aaa} ; C{aa,aa} ; C{aaa,a} ; C{aaaa,1}
        #
        for partition = 1:5
          # Infer total k based on momenta left of the comma.
          ks_part = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]][1:partition-1]
          daggers_part = [iDag,jDag,kDag,lDag][1:partition-1]
          push!(daggers_part,0) # Gives the correct sign for k
          total_k = momentum_conserving_index(ks_part,daggers_part,(nx,ny,nz))

          # Loop over dagger configuration of bb
          for bd1 = [0,1], bd2 = [0,1], bd3 = [0,1], bd4 = [0,1]

            # Given that a{k} expands to include both b{k} and [b{-k}]† (see tables above),
            # we need to compute which eigenmode (i.e. which k label) we end up with in each term.
            # This is important because the one-magnon-gas-correlator only applies to b-bosons
            # which *belong to the same eigenmode*.
            #
            # The answer is that we need to negate the momentum one time each time we encounter
            # a dagger in the expansion (at either the a-boson or b-boson level).
            bk1 = xor(bd1,iDag) == 1 ? negate_momentum((x1,y1,z1),(nx,ny,nz)) : (x1,y1,z1)
            bk2 = xor(bd2,jDag) == 1 ? negate_momentum((x2,y2,z2),(nx,ny,nz)) : (x2,y2,z2)
            bk3 = xor(bd3,kDag) == 1 ? negate_momentum((x3,y3,z3),(nx,ny,nz)) : (x3,y3,z3)
            bk4 = xor(bd4,lDag) == 1 ? negate_momentum((x4,y4,z4),(nx,ny,nz)) : (x4,y4,z4)

            bks = [bk1,bk2,bk3,bk4]
            bds = [bd1,bd2,bd3,bd4]

            # Evaluate mode label coincidences:
            class_1_count = 1
            class_1_rep = bk1
            class_1_mask = 0b0001

            class_2_count = 0
            class_2_rep = nothing
            class_2_mask = 0b0000
            for j = 2:4
              bkj = bks[j]
              if bkj == class_1_rep
                class_1_mask |= 0x1 << (j-1)
                class_1_count += 1
                continue
              end
              if isnothing(class_2_rep)
                class_2_rep = bkj
              end
              if bkj == class_2_rep
                class_2_mask |= 0x1 << (j-1)
                class_2_count += 1
              end
            end

            # Optimization/selection rule: Can skip any time there is an odd
            # number in a particular class (can't return to ground state):
            if isodd(class_1_count) || isodd(class_2_count)
              #println("Rejecting: ")
              #println("class_1: $(bitstring(class_1_mask))")
              #println("class_2: $(bitstring(class_2_mask))")
              continue
            end

            if (class_1_mask | class_2_mask) != 0b1111
              # Selection rule: Just by looking at k, we find there
              # are more than two modes participating (since one of them remained unsorted),
              # but this is verboten!
              continue
            end
            #=
            println()
            println("Term: bks = $bks")
            println("      bds = $bds")
            println("      aks = $([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]])")
            println("      ads = $([iDag,jDag,kDag,lDag])")
            println("class_1: $(bitstring(class_1_mask))")
            println("class_2: $(bitstring(class_2_mask))")
            =#
             

            # Two cases
            if class_2_count == 0
              # Case: Everything was class_1

              #=
            println()
            println("Term: bks = $bks")
            println("      bds = $bds")
            println("      aks = $([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]])")
            println("      ads = $([iDag,jDag,kDag,lDag])")
            println("class_1: $(bitstring(class_1_mask))")
            println("class_2: $(bitstring(class_2_mask))")
              println("  ^ All class 1; class_1_k = $class_1_rep")
              =#
              # Loop over mode identity for class_1
              for m = 1:(na*nf)
                L = bds[1:partition-1]
                R = bds[partition:end]
                y = wick_to_y_object(Y_storage[class_1_rep...,m], wickify(L), wickify(R); betaOmega = betaOmega[m,class_1_rep...])[-2:2]
                for i = 1:na, j = 1:na, k = 1:na, l = 1:na
                  y *= Vk1[i,1,iDag+1,m,bd1+1]
                  y *= Vk2[j,1,jDag+1,m,bd2+1]
                  y *= Vk3[k,1,kDag+1,m,bd3+1]
                  y *= Vk4[l,1,lDag+1,m,bd4+1]
                  f(y * y', total_k, m, m, class_1_rep, class_1_rep, i, iDag, j, jDag, k, kDag, l, lDag, partition)
                end
              end
            else
              # Case: Both class_1 and class_2 present
              
              # Loop over mode identities for the two classes
              for m = 1:(na*nf), n = 1:(na*nf)
                # Class 1
                L1 = Ref(empty_wick())
                R1 = Ref(empty_wick())
                L2 = Ref(empty_wick())
                R2 = Ref(empty_wick())
                #=
                println()
                println()
                println("Before:")
                println("L1: $L1")
                println("R1: $R1")
                println("L2: $L2")
                println("R2: $R2")
                =#
                for j = 4:-1:1
                  #println()
                  #println("j = $j")
                  is_left = j < partition
                  #println(is_left ? "Left" : "Right")
                  #println(1 == 1 & (class_1_mask >> (j-1)))
                  target = 1 == 1 & (class_1_mask >> (j-1)) ? (is_left ? L1 : R1) : (is_left ? L2 : R2)
                  #println(target[])
                  #println(bds[j])
                  target[] = wick_append_left(bds[j],target[])
                  #println(target[])
                end
                #=
                println()
                println("After:")
                println("L1: $L1")
                println("R1: $R1")
                println("L2: $L2")
                println("R2: $R2")
                =#

                #=
                L = bds[1:partition-1][class_1[class_1 .< partition]]
                R = bds[partition:end][class_1[class_1 .>= partition] .- (partition-1)]
                wL = wickify(L)
                wR = wickify(R)
                =#
                y_1 = wick_to_y_object(Y_storage[class_1_rep...,m], L1[], R1[]; betaOmega = betaOmega[m,class_1_rep...])[-2:2]

                # Class 2
                #L = bds[1:partition-1][class_2[class_2 .< partition]]
                #R = bds[partition:end][class_2[class_2 .>= partition] .- (partition-1)]
                y_2 = wick_to_y_object(Y_storage[class_1_rep...,n], L2[], R2[]; betaOmega = betaOmega[n,class_1_rep...])[-2:2]
                for i = 1:na, j = 1:na, k = 1:na, l = 1:na
                  y  = Vk1[i,1,iDag+1,(0b0001 == class_1_mask & 0b0001) ? m : n,bd1+1]
                  y *= Vk2[j,1,jDag+1,(0b0010 == class_1_mask & 0b0010) ? m : n,bd2+1]
                  y *= Vk3[k,1,kDag+1,(0b0100 == class_1_mask & 0b0100) ? m : n,bd3+1]
                  y *= Vk4[l,1,lDag+1,(0b1000 == class_1_mask & 0b1000) ? m : n,bd4+1]
                  f(y * y_1 * y_2', total_k, m, n, class_1_rep, class_2_rep, i, iDag, j, jDag, k, kDag, l, lDag, partition)
                  #view(correlator,total_k,m,n,i,1,iDag+1,j,1,jDag+1,k,1,kDag+1,l,1,lDag+1,:,:,partition) .+= y * y_1 * y_2'
                end
              end
            end
          end
        end
      end
    end
  end
  finish!(prog)

  #correlator
  nothing
end



function two_magnon_sector(Hs,Vs;betaOmega)
  nx,ny,nz = size(Vs)[3:5]
  N = size(Vs,1)÷2
  It = diagm([repeat([1],N); repeat([-1],N)])

  # Fix phases! This implements "commutes with dagger" property
  Vs_fix = copy(Vs)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz
    V = view(Vs_fix,:,:,ix,iy,iz)
    for fix = 1:N
      # Make first entry in each mode real in first half
      view(V,:,fix) .*= exp(-im * angle(V[1,fix]))

      # Make first entry in *second half* of each mode real in second half
      view(V,:,N+fix) .*= exp(-im * angle(V[N+1,N+fix]))
    end
  end

  # Two-magnon correlators have exactly four boson operators.
  # This means that they can have at most two distinct wavevectors (at the level
  # of a-bosons) and at most two distinct eigenmodes (at the level of b-bosons)
  #
  # If there are two distinct wavectors, then this situation can be handled as a
  # convolution of things in the one-magnon sector. Let the two wavevectors be k1 and k2
  # then we can organize by the one-magnon partition of each wavevector. For example:
  #
  #   C_{ak1 ak2 ak1 ak2,1}
  #
  # follows partition (3,3) whereas
  #
  #   C_{ak1 , ak2 ak1 ak2}
  #
  # follows partition (2,1). Since operators commute off-site, that last
  # term will be the same as these:
  #
  #   C_{ak1 , ak1 ak2 ak2} \
  #   C_{ak1 , ak2 ak1 ak2} | all equal to C_{ak1,ak1} * C_{1,ak2 ak2}
  #   C_{ak1 , ak2 ak2 ak1} /
  #
  # On the other hand, for the case with only one distinct wavevector,
  #
  #   C_{ak, ak ak ak}
  #
  # we can organize by the usual partition ∈ [1,5].

  # One Y matrix for each oscillator; only need ω > 0 to cover all oscillators
  Y_storage = Array{OffsetArray{ComplexF64}}(undef,nx,ny,nz,N)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz, i = 1:N
    Y_storage[ix,iy,iz,i] = empty_y_matrix()
  end

  # Five cases:
  # C_{aaaa,1}
  # C_{aaa,a}
  # C_{aa,aa}
  # C_{a,aaa}
  # C_{aaaa,1}
  correlator = zeros(ComplexF64,nx,ny,nz,N,N,N,N,2N,2N,2N,2N,size(empty_y_matrix(),4),size(empty_y_matrix(),4),5)
   
  # For distinct wave vectors, these are the two.
  for x1 = 1:nx, y1 = 1:ny, z1 = 1:nz
  for x2 = 1:nx, y2 = 1:ny, z2 = 1:nz
  for x3 = 1:nx, y3 = 1:ny, z3 = 1:nz

    Vk1 = Vs_fix[:,:,x1,y1,z1]
    Vk2 = Vs_fix[:,:,x2,y2,z2]
    Vk3 = Vs_fix[:,:,x3,y3,z3]

    # C_{ak1 ak2 ak3 ak4}

    # Now, do convolution of one-magnons:
    #
    # C_{ak1 ak2 ak1 ak2} 
    # C_{ak1 ak1} * C_{ak2 ak2}

    # Loop over configurations of eigenmodes
    for m = 1:N, n = 1:N, o = 1:N, p = 1:N
      if any(isnan.(correlator))
        error("Nan")
      end

      # Loop over dagger configuration for bbbb
      for bDag1 = [0,1], bDag2 = [0,1], bDag3 = [0,1], bDag4 = [0,1]
        for partition = 1:5
          L = [bDag1,bDag2,bDag3,bDag4][1:partition-1]
          R = [bDag1,bDag2,bDag3,bDag4][partition:end]

          # Break by site distinction
          if (x1,y1,z1,m)
          end
          y1 = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])
          y2 = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])
          y3 = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])
          # Loop over a-boson flavor
          for i = 1:N, j = 1:N, k = 1:N, l = 1:N, iDag = [0,1], jDag = [0,1], kDag = [0,1], lDag = [0,1]
            # Infer k4 based on daggers
            ks = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
            x4,y4,z4 = momentum_conserving_index(ks,[iDag,jDag,kDag,lDag],(nx,ny,nz)).I
            Vk4 = Vs_fix[:,:,x4,y4,z4]

            # Infer total k based all momenta
            ks = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]
            total_k = momentum_conserving_index(ks,[iDag,jDag,kDag,lDag,1],(nx,ny,nz))

            i_ix = (iDag * N) + i
            j_ix = (jDag * N) + j
            k_ix = (kDag * N) + k
            l_ix = (lDag * N) + l
            correlator[total_k,m,n,o,p,i_ix,j_ix,k_ix,l_ix,:,partition] .+= Vk1[i_ix,(bDag1 * N) + m] * Vk2[j_ix,(bDag2 * N) + n] * Vk3[k_ix,(bDag3 * N) + o] * Vk4[l_ix,(bDag4 * N) + p] * y[-2:2]
          end
        end
      end
    end
  end
  end
  end

  correlator
end

# Computes C_{bi bj}
function one_magnon_gas_correlator(label_i,label_j,left,right,partition;betaOmega, Y = nothing, classical = false)

  ix,iy,iz,m = label_i
  jx,jy,jz,n = label_j
  #nx,ny,nz = ns

  # This function computes C_{bb}. There is at most one distinct eigenmode in play
  # because there are two boson operators. So if we are asked for something else,
  # that correlator is zero:
  if !(label_i == label_j)
    println("Verboten! because more than one distinct eigenmode (max 1 allowed)")
    println("  requested: $label_i and $label_j")
    return [0,0,0,0,0]
  end

  if classical
    # TODO: verify
    if partition == 2
      if left == 0 && right == 1
        [0,0,0,1/betaOmega,0]
      elseif left == 1 && right == 0
        [0,1/betaOmega,0,0,0]
      else
        [0,0,0,0,0]
      end
    else
      if left == 0 && right == 1
        [0,0,1/betaOmega,0,0]
      elseif left == 1 && right == 0
        [0,0,1/betaOmega,0,0]
      else
        [0,0,0,0,0]
      end
    end
  else
    Y_storage = isnothing(Y) ? empty_y_matrix() : Y

    L = [left,right][1:partition-1]
    R = [left,right][partition:end]
    wick_to_y_object(Y_storage, wickify(L), wickify(R); betaOmega)[-2:2]
  end
end



function to_lab_displacements(swt;q = [0.1,0,0])
  formula = intensity_formula(swt, :full; kernel=delta_function_kernel)
  
  Sunny.swt_hamiltonian_SUN!(formula.calc_intensity.H,swt,Sunny.Vec3(q))

  display(Sunny.bogoliubov!(formula.calc_intensity.V,formula.calc_intensity.H))
  V = formula.calc_intensity.V

  f = Figure()
  for band = 1:size(V,2)
    ax = Axis(f[1 + (band-1) ÷ 8,mod1(band,8)])
    println()
    println("band = $band")
    println("----")
    eigmodes_by_site = reshape(V[:,band],2,4,2)
    U_transverse = swt.data.local_unitaries[:,1:2,:]

    # First atom:
    println("b setting:")
    b_set = U_transverse[:,:,1] * eigmodes_by_site[:,1,1]
    display(b_set)

    println("b† setting:")
    bd_set = U_transverse[:,:,1] * eigmodes_by_site[:,1,2]
    display(bd_set)

    scatter!(ax,real(b_set),imag(b_set))
    scatter!(ax,real(bd_set),imag(bd_set))
    xlims!(ax,-1,1)
    ylims!(ax,-1,1)
  end
  display(f)
end

function speak_bogo_coeffs(Vs)
  num_site = size(Vs,1)
  nf = size(Vs,2)
  @assert num_site * nf == size(Vs,4)
  for ix = CartesianIndices((size(Vs,6),size(Vs,7),size(Vs,8)))
    for a_site = 1:num_site, a_flavor = 1:nf, aDag = 1:2
      print("$(nf > 1 ? a_flavor : "")a$a_site$(aDag == 2 ? "†" : " ") = ")
      started = false
      for band = 1:(num_site * nf), bDag = 1:2
        x = Vs[a_site,a_flavor,aDag,band,bDag,ix]
        if iszero(round(x;digits = 12))
          continue
        end
        if started
          print(" + ")
        end
        print("(")
        print(Sunny.number_to_simple_string(real(x);digits = 3))
        print(" + ")
        print(Sunny.number_to_simple_string(imag(x);digits = 3))
        print("i)b$band$(bDag == 2 ? "†" : " ")")
        started = true
      end
      println()
    end
    println()
  end
end

function guide_dispersion!(disps;blue = true)
  for j = 1:size(disps,1)
    lines!(disps[j,:,1,1],color = :black,linestyle = :dash)
    lines!(-disps[j,:,1,1],color = :black,linestyle = :dash)
  end
  if !blue
    return
  end
  for j = 1:size(disps,1), l = 1:size(disps,1)
    lines!(disps[j,:,1,1] + disps[l,:,1,1], color = :blue,linestyle = :dash)
    lines!(disps[j,:,1,1] - disps[l,:,1,1], color = :blue,linestyle = :dash)
    lines!(-disps[j,:,1,1] - disps[l,:,1,1], color = :blue,linestyle = :dash)
  end
end

negate_momentum(k,n) = ntuple(i -> mod1((n[i] + 1) - (k[i] - 1),n[i]),3)
function phase_fix_Vs(Vs)
  nx = size(Vs,6)
  ny = size(Vs,7)
  nz = size(Vs,8)

  na = size(Vs,1)
  nf = size(Vs,2)

  Vs_fix = copy(Vs)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz, mode = 1:na*nf
    Vhere = view(Vs_fix,:,:,:,:,:,ix,iy,iz)

    mirror_ix = mod1(nx-(ix-1)+1,nx)
    mirror_iy = mod1(ny-(iy-1)+1,ny)
    mirror_iz = mod1(nz-(iz-1)+1,nz)
    mirror_mode = (na*nf)-(mode-1)

    Vthere = view(Vs_fix,:,:,:,:,:,mirror_ix,mirror_iy,mirror_iz)
    for fix_a = 1:na, fix_f = 1:nf
      # Replace the negative energy modes at the opposite Q with equivalent
      # ones that are phase-matched to the positive energy modes at this Q
      if !is_phase_rotation(view(Vthere,:,:,2,mirror_mode,2),conj.(Vhere[:,:,1,mode,1]))
        @warn "Making non-simple change of variables; do you have degenerate bands?"
      end
      view(Vthere,:,:,2,mirror_mode,2) .= conj.(Vhere[:,:,1,mode,1]) # Copy a  = ... b to a† = ... b†
      if !is_phase_rotation(view(Vthere,:,:,1,mirror_mode,2),conj.(Vhere[:,:,2,mode,1]))
        @warn "Making non-simple change of variables; do you have degenerate bands?"
      end
      view(Vthere,:,:,1,mirror_mode,2) .= conj.(Vhere[:,:,2,mode,1]) # Copy a† = ... b to a  = ... b†
    end
  end

  # Reverse the order of the † eigenmodes so that they are in the same order as the non-† modes:
  Vs_fix[:,:,:,1:(na*nf),2,:,:,:] .= Vs_fix[:,:,:,(na*nf):-1:1,2,:,:,:]

  Vs_fix
end

function is_phase_rotation(a,b)
  finite_a_ix = abs.(a) .> 1e-8
  angles = angle.(b[finite_a_ix] ./ a[finite_a_ix])
  all(isapprox.(angles,sum(angles)/length(angles);atol = 1e-8))
end

function interactive_example()
  cryst = Crystal(I(3),[[0,0,0],[1/2,0,0]],1)
  n_cell = 16
  na = 2
  sys = System(cryst,(n_cell,1,1),[SpinInfo(1,S=1,g=2),SpinInfo(2,S=1,g=2)],:dipole)
  set_exchange!(sys,1.0,Bond(2,1,[1,0,0]))
  set_exchange!(sys,1.0,Bond(1,2,[0,0,0]))
  set_onsite_coupling!(sys, S -> -S[3]^2, 1)
  for i = 1:n_cell
    sys.dipoles[i,1,1,2] = [0,0,-1]
  end
  
  S0 = render_one_over_S_spectrum(sys;beta = 85.0)
  Sloc = S0[1]
  Sglob = S0[2]

  Hs, Vs, disps = finite_spin_wave_Vmats(sys;polyatomic = false)

  f = Figure()
  display(f)
  nE = 30
  dE = 0.2
  trS = Observable(zeros(ComplexF64,size(Sglob,1),2nE+1))
  for i = 1:3
    trS[] .+= raster_spec(Sglob,disps,dE,nE,i,i)[:,1,1,:]
  end
  qs = ((1:size(Sglob,1)) .- 1) ./ sys.latsize[1]
  es = -(nE*dE):dE:(nE*dE)
  ax = Axis(f[1,1],xlabel = "Momentum",ylabel = "Energy")
  hm = heatmap!(ax,qs,es,map(x -> log10.(abs.(x)),trS),colorrange = (-8,2))
  function filter_it(x)
    if !(real(x) < 0 || abs(imag(x)) > 1e-5)
      NaN
    else
      1.0
    end
  end
  heatmap!(ax,qs,es,map(x -> filter_it.(x),trS),colorrange = (0,1),colormap = :jet1)
  Colorbar(f[1,2],hm)

  sg = SliderGrid(f[2,1],
    (label = "kT", range = -4:0.01:2, startvalue = -2, format = x -> "$(Sunny.number_to_simple_string(10. ^ x; digits = 2))"),
    (label = "Frame", range = 0:1e-3:1, startvalue = 1.0, format = x -> "$(Sunny.number_to_simple_string(x; digits = 3))")
   )

  kT = Observable(1/85)
  hlines!(ax,map(x -> -x,kT),color = :black)
  frame = Observable(1.)
  tgs_sublat = Matrix{Observable{Bool}}(undef,na,na)
  tgs_tr = Vector{Observable{Bool}}(undef,3)

  Hs, Vs, disps, Us = finite_spin_wave_Vmats(sys;polyatomic = false)
  function redraw()
    Sloc, Sglob = blit_one_over_S_spectrum(sys,dE,nE;beta = 1/kT[])
    Sglob .= 0
    for m = 1:na, n = 1:na
      if !tgs_sublat[min(m,n),max(m,n)][]
        continue
      end
      Um = exp(frame[] * log(Us[m]))
      Un = exp(frame[] * log(Us[n]))
      for i = 1:3, j = 1:3, k = 1:3, l = 1:3
        Sglob[:,:,:,m,n,i,j,:] .+= Um[i,k] * Un[j,l] * Sloc[:,:,:,m,n,k,l,:]
      end
    end
    trS[] .= 0
    for i = 1:3
      if !tgs_tr[i][]
        continue
      end
      # Sublattice sum and trace
      trS[] .+= sum(Sglob,dims=[4,5])[:,1,1,1,1,i,i,:]
    end
    notify(trS)
  end

  gl = GridLayout(f[3,1],tellwidth = false)
  gl1 = GridLayout(gl[1,1])
  for m = 1:na, n = 1:na
    if n >= m
      tog = Toggle(gl1[m,n])
      tgs_sublat[m,n] = tog.active
      on(tog.active) do e
        redraw()
      end
    end
  end

  gl2 = GridLayout(gl[1,2])
  for m = 1:3
    tog = Toggle(gl2[m,m];buttoncolor = :red)
    tgs_tr[m] = tog.active
    on(tog.active) do e
      redraw()
    end
  end



  on(sg.sliders[1].value;update = true) do logkT
    kT[] = 10^logkT
    redraw()
  end

  on(sg.sliders[2].value;update = true) do fr
    frame[] = fr
    redraw()
  end


  nothing
end

