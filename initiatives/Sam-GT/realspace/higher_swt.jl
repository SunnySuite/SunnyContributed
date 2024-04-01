using LinearAlgebra
using OffsetArrays

bose(x) = 1/(exp(x) - 1)

function empty_y_matrix(;n_order = 2)
  NaN .* OffsetArray(Array{Float64}(undef,n_order+1,2n_order + 1,2n_order + 1, 2n_order + 1),-1,-1,-1,-(n_order+1))
end

eulerian_polys = [[1],[1,1],[1,4,1],[1,11,11,1],[1,26,66,26,1],[1,57,302,302,57,1]]

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
  eulerian_polys[n]
end

function y_quantity(stored,n,M,mDagger;betaOmega)
  n_order = size(stored,1) - 1
  if any(isnan.(stored[n,M,mDagger,-n_order:n_order]))
    if n == 0
      @assert M == 0
      @assert mDagger == 0
      stored[n,M,mDagger,-n_order:-1] .= 0
      stored[n,M,mDagger,0] = 1
      stored[n,M,mDagger,1:n_order] .= 0
    end

    if n > 0
      for M0 = 0:(2n)
        for md = 0:M0
          println()
          println("Y$n$M0$md")
          #println("M = $M0")
          #println("m† = $md")

          # These are not valid
          if (md > n) || ((M0 - md) > n)
            continue
          end

          harmonic = M0 - 2md
          println("harmonic = $harmonic")

          # Don't need to worry about explicitly skipping any early
          # terms in the sum; the polynomial caclulation below will kill
          # those terms automatically!
          #
          #n_skip_sum = n - M0 - md
          #println("n_skip_sum = $n_skip_sum (NYI)")

          max_deg = 8
          poly = zeros(max_deg)

          poly[1] = 1 # Constant 1 to start with
          incld = 0
          curr_n = 0
          n_traj = Int64[]

          # Build up the polynomial (in the occupation number of the current site)
          # from the string b†...b†b...b|k><k|b†...b†b...b
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

          println(n_traj)
          factors = sort(n_traj)[1:2:end]
          for j = 1:length(factors)
            poly .= factors[j] .* poly .+ [0 ; poly[1:max_deg-1]]
            #println("poly = $poly")
          end

          nB = bose(betaOmega)
          xvar = exp(betaOmega)
          xPows = OffsetArray([xvar ^ k for k = 0:maximum(length(eulerian_poly(max_deg-1)))],-1)

          println("Using poly = $poly")
          println("xPows = $xPows")

          tot = 0
          for p = 0:(max_deg-1)
            #println("p = $p")
            if p == 0 # Constant term
              tot += poly[p+1]
            else # Series sum of n^p
              #println("  Euler = $(eulerian_poly(p))")

              # Only requires up to x^(p-1)
              e_sum = sum(eulerian_poly(p) .* xPows[0:(p-1)])
              #println(e_sum)
              tot += poly[p+1] * nB^p * e_sum
            end
            #println("tot = $tot")
          end
          stored[n,M0,md,-n_order:n_order] .= 0
          stored[n,M0,md,harmonic] = tot

        end
      end
    end
  end

  if any(isnan.(stored[n,M,mDagger,-n_order:n_order]))
    display(stored[n,M,mDagger,:])
    error("Y quantity computation failed! n = $n, M = $M, m† = $mDagger")
  end

  return stored[n,M,mDagger,:]
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
function wick_to_y_object(stored,wick_A,wick_B;betaOmega)
  a_terms = findall(.!iszero.(wick_A))
  b_terms = findall(.!iszero.(wick_B))
  tot = 0 * y_quantity(stored,0,0,0;betaOmega)
  for at = a_terms, bt = b_terms
    n_create_a = at.I[1] - 1
    n_destroy_a = at.I[2] - 1
    n_create_b = bt.I[1] - 1
    n_destroy_b = bt.I[2] - 1

    # For a <μ|...|k><k|...|μ> type correlation, we need to
    # return to the ground state at the end of the day, so
    # the number of creation and deletion events had better match!
    #@assert n_create_a + n_create_b == n_destroy_a + n_destroy_b
    if !(n_create_a + n_create_b == n_destroy_a + n_destroy_b)
      println("Skipping term which doesn't conserve magnon number!")
      continue
    end

    m_dagger = n_create_a
    n = n_create_b + m_dagger
    M = n_destroy_a + m_dagger
    println("Y$n$M$m_dagger")
    tot += wick_A[at] * wick_B[bt] * y_quantity(stored,n,M,m_dagger;betaOmega)
    if any(isnan.(tot))
      display(stored)
      display(tot)
      show_wick(wick_A)
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

function show_wick(X)
  for r = 1:size(X,1)
    for c = 1:size(X,2)
      if !iszero(X[r,c])
        print(" + ")
        print(X[r,c])
        for i = 1:(r-1)
          print("b†")
        end
        for i = 1:(c-1)
          print("b")
        end
      end
    end
  end
  println()
end

function finite_spin_wave_Vmats(sys)
  # Assess if all unit cells have the same data
  all_repeats = allequal([sys.dipoles[i,:] for i = Sunny.eachcell(sys)]) && allequal([sys.coherents[i,:] for i = Sunny.eachcell(sys)])
  if !all_repeats
    @warn "Not all unit cells in the system have the same spin data! Collapsing to first unit cell anyway"
  end

  sys_collapsed = Sunny.reshape_supercell_aux(sys, (1,1,1), Sunny.cell_shape(sys))
  swt = SpinWaveTheory(sys_collapsed)

  # Find polyatomic required number of unit cells:
  na = Sunny.natoms(sys.crystal)
  ΔRs = [map(x -> iszero(x) ? Inf : x,abs.(sys.crystal.positions[i] - sys.crystal.positions[j])) for i = 1:na, j = 1:na]
  nbzs = round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))

  comm_ixs = CartesianIndices(ntuple(i -> sys.latsize[i] .* nbzs[i],3))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  formula = intensity_formula(swt,:full;kernel = delta_function_kernel)

  Hs = zeros(ComplexF64,size(formula.calc_intensity.H)...,comm_ixs.indices...)
  Vs = zeros(ComplexF64,size(formula.calc_intensity.V)...,comm_ixs.indices...)
  disps = zeros(ComplexF64,size(formula.calc_intensity.H,1)÷2,comm_ixs.indices...)

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

    disps[:,i] .= Sunny.bogoliubov!(formula.calc_intensity.V,Hmat)

    Vs[:,:,i] .= formula.calc_intensity.V

    #formula.calc_intensity(swt,Sunny.Vec3(k_comm))
  end

  #=
  sys_periodic = repeat_periodically(sys_collapsed,sys.latsize)

  isc = instant_correlations(sys_periodic)
  add_sample!(isc,sys_periodic)

  params = unit_resolution_binning_parameters(isc)
  params.binend[1:3] .+= nbzs .- 1

  is_static = intensities_binned(isc,params,intensity_formula(isc,:full))[1]

  enhanced_nbands = Sunny.nbands(swt) + 1
  is_type = typeof(bs[1]).parameters[2]
  enhanced_bs = Array{Sunny.BandStructure{enhanced_nbands,is_type}}(undef,size(bs)...)
  for i = comm_ixs
    x = bs[i]
    enhanced_dispersion = SVector{enhanced_nbands,Float64}([0., x.dispersion...])
    enhanced_intensity = SVector{enhanced_nbands,is_type}([is_static[i], (x.intensity ./ prod(sys.latsize))...])
    enhanced_bs[i] = Sunny.BandStructure{enhanced_nbands,is_type}(enhanced_dispersion,enhanced_intensity)
  end

  is_swt = map(x -> sum(x.intensity),bs) / prod(sys.latsize)

  is_all = is_static + is_swt

  enhanced_bs, is_all, sys_periodic
  =#

  Hs, Vs, disps
end

function render_one_over_S_spectrum(sys;beta = 1.0)
  Hs, Vs, disps = finite_spin_wave_Vmats(sys)
  ssf = zero_magnon_sector(Hs,Vs;betaOmega = beta * 0.0)
  #corr_a_mat, c_1_aa, c_aa_1 = one_magnon_sector(Hs,Vs;betaOmega = beta * disps)
  one_magnon_c = one_magnon_sector(Hs,Vs;betaOmega = beta * disps)

  # TODO: treat frequencies correctly; not "harmonics", bin it instead!
  # 1/S expansion
  s = (sys.Ns[1]-1) / 2

  nx,ny,nz = size(Vs)[3:5]
  N = size(Vs,1)÷2

  Sxx = zeros(ComplexF64,nx,ny,nz,N,5)
  Syy = zeros(ComplexF64,nx,ny,nz,N,5)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz
    ix_k1 = CartesianIndex(ix,iy,iz)
    ix_mk1 = CartesianIndex(nx-(ix-1),ny-(iy-1),nz-(iz-1))
    for i = 1:2, j = 1:2
      # Infer total k from k1:
      ix_k = i == j ? ix_mk1 : ix_k1
      Sxx[ix_k,:,:] .+= 2 * s * one_magnon_c[ix_k,:,i,j,:,2] / 4
      Syy[ix_k,:,:] .+= 2 * s * [1,-1][i] * [1,-1][j] * one_magnon_c[ix_k,:,i,j,:,2] / (-4)
    end
  end
  #Sxx = 2 * s * sum([corr_a_mat[:,:,:,:,i,j,:] for i = 1:2, j = 1:2])/4
  #Syy = 2 * s * sum([[1,-1][i] * [1,-1][j] * corr_a_mat[:,:,:,:,i,j,:] for i = 1:2, j = 1:2])/(-4)

  Szz = zeros(ComplexF64,nx,ny,nz,N,5)
  Szzk0 = view(Szz,1,1,1,:,:) # ??? which band does ssf go in?; bin it

  # Integrate the internal index on c_{1,aa} and c_{aa,1}
  int_terms = sum(one_magnon_c,dims = [1,2,3])[:,:,:,:,2,1,:,1] + sum(one_magnon_c,dims = [1,2,3])[:,:,:,:,2,1,:,3]
  Szzk0[1,:] .= s * s * prod([nx,ny,nz]) * ssf[1,1,1,:]
  Szzk0 .-= s * int_terms[1,1,1,:,:]

  Sxx + Syy + Szz
end

# Returns the index of the wavevector which completes the given
# list of wavevector indices in a momentum-conserving way.
# The ks should be a list [[ix,iy,iz],[jx,jz,jz],…], and
# daggers should be a bit string [0,1,1,0,…] with 1 for a†
function momentum_conserving_index(ks,daggers,ns)
  nx,ny,nz = ns
  dagger_signs = (-1) .^ daggers
  momentum_so_far = sum(dagger_signs[1:end-1] .* map(k -> k .- 1,ks))
  I = mod1.(1 .- momentum_so_far * dagger_signs[end],[nx,ny,nz])

  # TODO: verify
  #=
  freqs = fftfreq.(ns)
  push!(ks,I)
  for i = 1:length(ks)
    for j = 1:3
      print(freqs[j][ks[i][j]])
      print(", ")
    end
    if daggers[i] == 1
      println("dagger")
    else
      println("")
    end
  end
  =#
  CartesianIndex(tuple(I...))
end

function zero_magnon_sector(Hs,Vs;betaOmega)
  # This is correlators with exactly *zero* boson operators
  # Namely, it's the C_{1,1} correlator.
  yy = empty_y_matrix()
  yobj = wick_to_y_object(yy,wickify([]),wickify([]);betaOmega)
  nx,ny,nz = size(Vs)[3:5]
  N = size(Vs,1)÷2
  ssf = zeros(ComplexF64,nx,ny,nz,length(yobj))
  ssf[1,1,1,:] .= yobj[-2:2]
  ssf
end

function one_magnon_sector(Hs,Vs;betaOmega)
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

  # One-magnon correlators have exactly two boson operators.
  # This means that they can have at most one distinct wavevector (at the level
  # of a-bosons) and at most one distinct eigenmode (at the level of b-bosons)

  # One Y matrix for each oscillator; only need ω > 0 to cover all oscillators
  Y_storage = Array{OffsetArray{ComplexF64}}(undef,nx,ny,nz,N)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz, i = 1:N
    Y_storage[ix,iy,iz,i] = empty_y_matrix()
  end

  # Correlations C_{...ai...aj...} broken down by:
  # - The ix,iy,iz,m label of the one distinct eigenmode
  # - The i,j flavors of a-boson being correlated
  # - The harmonic of the eigenmode
  # - The partition number
  correlator = zeros(ComplexF64,nx,ny,nz,N,2N,2N,size(empty_y_matrix(),4),3)
  correlator_a_matrix = zeros(ComplexF64,nx,ny,nz,N,2N,2N,size(empty_y_matrix(),4))
  correlator_1_aa = zeros(ComplexF64,nx,ny,nz,N,2N,2N,size(empty_y_matrix(),4))
  correlator_aa_1 = zeros(ComplexF64,nx,ny,nz,N,2N,2N,size(empty_y_matrix(),4))

  # The exactly one distinict wavevector of the one-magnon sector
  for ix = 1:nx, iy = 1:ny, iz = 1:nz

    Vk1 = Vs_fix[:,:,ix,iy,iz]
    #Vmk = Vs_fix[:,:,end - (ix-1),end - (iy-1),end - (iz-1)]
    #Vmk = Vs_fix[:,:,momentum_conserving_index([(ix,iy,iz)],(nx,ny,nz))]

    # We have that a1 = ∑ᵢT_1i bi.
    # So C_{ai,aj} = ∑_mn T_im T_jn C_{bm,bn}, but only m=n contributes

    # Loop over eigenmodes at this k
    for m = 1:N
      if any(isnan.(correlator))
        error("Nan")
      end
      # Loop over bb,b†b,bb†,b†b†
      for left = [0,1], right = [0,1]
        for partition = 1:3
          L = [left,right][1:partition-1]
          R = [left,right][partition:end]
          y = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])
          # Loop over a-boson flavor
          for i = 1:N, j = 1:N, iDag = [0,1], jDag = [0,1]
            # Infer k2 based on daggers
            ks = [[ix,iy,iz]]
            Vk2 = Vs_fix[:,:,momentum_conserving_index(ks,[iDag,jDag],(nx,ny,nz))]

            #Vk2 = iDag == jDag ? Vmk : Vk # Infer k2 based on daggers

            i_ix = (iDag * N) + i
            j_ix = (jDag * N) + j
            correlator[ix,iy,iz,m,i_ix,j_ix,:,partition] .+= Vk1[i_ix,(left * N) + m] * Vk2[j_ix,(right * N) + m] * y[-2:2]
          end
        end
      end
          #=
        Vk1 = Vk

        # Calculate C_{a,a}
        y = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify([left]),wickify([right]);betaOmega = betaOmega[m,ix,iy,iz])
        for i = 1:(2N), j = 1:(2N)
          Vk2 = (i > N) == (j > N) ? Vmk : Vk # Infer k2 based on daggers
          correlator_a_matrix[ix,iy,iz,m,i,j,:] .+= Vk1[i,(left * N) + m] * Vk2[j,(right * N) + m] * y[-2:2]
        end

        # Calculate C_{1,aa}
        y = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify([]),wickify([left,right]);betaOmega = betaOmega[m,ix,iy,iz])
        for i = 1:(2N), j = 1:(2N)
          Vk2 = (i > N) == (j > N) ? Vmk : Vk # Infer k2 based on daggers
          correlator_1_aa[ix,iy,iz,m,i,j,:] .+= Vk1[i,(left * N) + m] * Vk2[j,(right * N) + m] * y[-2:2]
        end

        # Calculate C_{aa,1}
        y = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify([left,right]),wickify([]);betaOmega = betaOmega[m,ix,iy,iz])
        for i = 1:(2N), j = 1:(2N)
          Vk2 = (i > N) == (j > N) ? Vmk : Vk # Infer k2 based on daggers
          correlator_aa_1[ix,iy,iz,m,i,j,:] .+= Vk1[i,(left * N) + m] * Vk2[j,(right * N) + m] * y[-2:2]
        end
      end
      =#
    end
  end

  #correlator_a_matrix, correlator_1_aa, correlator_aa_1
  correlator
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

  # This is correlators with exactly four boson operators, so at most
  # two distinct eigenmodes may participate.

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
  non_distinct_correlator = zeros(ComplexF64,nx,ny,nz,N,2N,2N,2N,2N,size(empty_y_matrix(),4),5)
  correlator = zeros(ComplexF64,nx,ny,nz,N,nx,ny,nz,N,nx,ny,nz,N,2N,2N,2N,2N,size(empty_y_matrix(),4),size(empty_y_matrix(),4),5)

  # Correlations C_{...ai...aj...} broken down by:
  # - The ix,iy,iz,m label of the first distinct eigenmode
  # - The ix,iy,iz,m label of the second distinct eigenmode
  # - The i,j,k,l flavors of a-boson being correlated
  # - The harmonics of the two eigenmodes

  # Loop over all values of (k1,k2,k3) (so we can go over pairs of eigenmodes)
  for x1 = 1:nx, y1 = 1:ny, z1 = 1:nz
  for x2 = 1:nx, y2 = 1:ny, z2 = 1:nz
  for x3 = 1:nx, y3 = 1:ny, z3 = 1:nz

    Vk1 = Vs_fix[:,:,x1,y1,z1]
    Vk2 = Vs_fix[:,:,x2,y2,z2]
    Vk3 = Vs_fix[:,:,x3,y3,z3]

    # So C_{ak1 ak2 ak3 ak4} = ∑_mn T_im T_jn C_{bm,bn}, but only m=n contributes

    # Loop over pairs of eigenmodes at these ki and kj
    for m = 1:N, n = 1:N, o = 1:N, p = 1:N
      if any(isnan.(correlator))
        error("Nan")
      end

      # Loop over possible daggers on aaaa
      for a = [0,1], b = [0,1], c = [0,1], d = [0,1]
        # TODO: verify this conserves momentum
        xx = mod1(-(-1)^a * x1 - (-1)^b * x2 - (-1)^c * x3,nx)
        yy = mod1(-(-1)^a * y1 - (-1)^b * y2 - (-1)^c * y3,ny)
        zz = mod1(-(-1)^a * z1 - (-1)^b * z2 - (-1)^c * z3,nz)
        Vk4 = Vs_fix[:,:,xx,yy,zz]
        for partition = 1:5
          L = [a,b,c,d][1:partition-1]
          R = [a,b,c,d][partition:end]
          y = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])
          for i = 1:(2N), j = 1:(2N), k = 1:(2N), l = 1:(2N), h = 1:5
            non_distinct_correlator[ix,iy,iz,m,i,j,k,l,h,total_partition] .+= V[i,(a * N) + m] * V[j,(b * N) + m] * V[k,(c * N) + m] * V[l,(d * N) + m] * y[-2:2][h]
          end
        end
      end
    end
  end

      #=
      else
        # Distinct eigenmodes Steve and Fred; compute individually and convolve

        # Loop over possible daggers on bb for each
        for steve_L = [0,1], steve_R = [0,1], fred_L = [0,1], fred_R = [0,1]
          for partition_steve = 1:3, partition_fred = 1:3
            L = [steve_L,steve_R][1:partition-1]
            R = [steve_L,steve_R][partition:end]
            y_steve = wick_to_y_object(Y_storage[ix,iy,iz,m],wickify(L),wickify(R);betaOmega = betaOmega[m,ix,iy,iz])

            L = [fred_L,fred_R][1:partition-1]
            R = [fred_L,fred_R][partition:end]
            y_fred = wick_to_y_object(Y_storage[jx,jy,jz,n],wickify(L),wickify(R);betaOmega = betaOmega[n,jx,jy,jz])

            total_partition = (parition_steve - 1) + (parition_fred - 1) + 1

            for i = 1:(2N), j = 1:(2N), k = 1:(2N), l = 1:(2N), h_steve = 1:5, h_fred = 1:5
              correlator[ix,iy,iz,m,jx,jy,jz,n,i,j,k,l,h_steve,h_fred,total_partition] .+= one_magnon_c[ix,iy,iz,m] * y_steve[-2:2][h_steve] * y_fred[-2:2][h_fred]
          correlator_a_matrix[ix,iy,iz,m,i,j,:] .+= V[i,(left * N) + m] * V[j,(right * N) + m] * y[-2:2]
              correlator[ix,iy,iz,m,jx,jy,jz,n,i,j,k,l,h_steve,h_fred,total_partition] .+= V[i,(a * N) + m] * V[j,(b * N) + m] * V[k,(c * N) + m] * V[l,(d * N) + m] * y_steve[-2:2][h_steve] * y_fred[-2:2][h_fred]
            end
          end
        end
      end
    end
    end
    =#
  end
  end

  correlator
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
