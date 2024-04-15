using LinearAlgebra
using OffsetArrays
using StaticArrays
using FFTW

RasterSpectrum{N} = SVector{N,ComplexF64}

struct BandSpectrum
  energy
  intensity
end

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
          println("Calculating: Y$n$M0$md")
          #println("M = $M0")
          #println("m† = $md")

          # These are not valid
          if (md > n) || ((M0 - md) > n)
            println("[[invalid Y object]]")
            println()
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

          println("n-trajectory: $n_traj")
          factors = sort(n_traj)[1:2:end]
          for j = 1:length(factors)
            poly .= factors[j] .* poly .+ [0 ; poly[1:max_deg-1]]
            #println("poly = $poly")
          end

          nB = bose(betaOmega)
          xvar = exp(betaOmega)
          xPows = OffsetArray([xvar ^ k for k = 0:maximum(length(eulerian_poly(max_deg-1)))],-1)

          print("Boson number (n) → ")
          leading = true
          for p = 0:(max_deg-1)
            ps = ["","n","n²","n³","n⁴","n⁵","n⁶","n⁷","n⁸","n⁹"][p+1]
            a = poly[p+1]
            if !iszero(a)
              if a > 0
                !leading && print(" + ")
              else
                print(" - ")
              end
              if p == 0 || a != 1.0
                print("$(Sunny.number_to_math_string(abs(a);digits = 2))")
              end
              print(ps)
              leading = false
            end
          end
          println()

          #println("xPows = $xPows")

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
function wick_to_y_object(stored,wick_A,wick_B;betaOmega, verbose = false)
  a_terms = findall(.!iszero.(wick_A))
  b_terms = findall(.!iszero.(wick_B))
  tot = 0 * y_quantity(stored,0,0,0;betaOmega)
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

    tot += wick_A[at] * wick_B[bt] * y_quantity(stored,n,M,m_dagger;betaOmega)
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
    ΔRs = [map(x -> iszero(x) ? Inf : x,abs.(sys.crystal.positions[i] - sys.crystal.positions[j])) for i = 1:na, j = 1:na]
    nbzs = round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))
  else
    nbzs = [1,1,1]
  end

  comm_ixs = CartesianIndices(ntuple(i -> sys.latsize[i] .* nbzs[i],3))
  ks_comm = [Sunny.Vec3((i.I .- 1) ./ sys.latsize) for i = comm_ixs]

  formula = intensity_formula(swt,:full;kernel = delta_function_kernel)

  Hs = zeros(ComplexF64,size(formula.calc_intensity.H)...,comm_ixs.indices...)
  Vs = zeros(ComplexF64,na,nf,2,na,nf,2,comm_ixs.indices...)
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

    disps[:,i] .= Sunny.bogoliubov!(formula.calc_intensity.V,Hmat)

    Vs[:,:,:,:,:,:,i] .= reshape(formula.calc_intensity.V,na,nf,2,na,nf,2)

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

  Hs, Vs, disps, swt.data.local_rotations
end

function render_one_over_S_spectrum(sys;beta = 1.0)
  Hs, Vs, disps, Us = finite_spin_wave_Vmats(sys;polyatomic = false)
  ssf = zero_magnon_sector(Hs,Vs,sys.crystal;betaOmega = beta * 0.0)
  #corr_a_mat, c_1_aa, c_aa_1 = one_magnon_sector(Hs,Vs;betaOmega = beta * disps)
  one_magnon_c = one_magnon_sector(Hs,Vs,sys.crystal;betaOmega = beta * disps)


  # Find polyatomic required number of unit cells:
  na = Sunny.natoms(sys.crystal)
  ΔRs = [map(x -> iszero(x) ? Inf : x,abs.(sys.crystal.positions[i] - sys.crystal.positions[j])) for i = 1:na, j = 1:na]
  nbzs = round.(Int64,max.([1,1,1],1 ./ minimum(ΔRs)))

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

    ks = [([x,y,z] .- 1) ./ sys.latsize for x = xs, y = ys, z = zs]

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

    #Sxx[xs,ys,zs,:,m,n,:] .+= 2 * s * sum([one_magnon_c[:,:,:,:,(i * N) + m,(j * N) + n,:,2] for i = 0:1, j = 0:1])/4
    #Syy[xs,ys,zs,:,m,n,:] .+= 2 * s * sum([[1,-1][i] * [1,-1][j] * one_magnon_c[:,:,:,:,(i - 1) * N + m,(j - 1) * N + n,:,2] for i = 1:2, j = 1:2])/(-4)


    # For c_{1,aa} and c_{aa,1}, our inference of the total momentum k will always have placed everything
    # in k = 0:
    @assert abs(sum(one_magnon_c[:,:,:,:,:,:,:,:,:,:,:,1]) - sum(one_magnon_c[1,1,1,:,:,:,:,:,:,:,:,1])) < 1e-12
    @assert abs(sum(one_magnon_c[:,:,:,:,:,:,:,:,:,:,:,3]) - sum(one_magnon_c[1,1,1,:,:,:,:,:,:,:,:,3])) < 1e-12

    # SSF is always band #1 ???
    S_local[xs,ys,zs,1,m,n,:,3,3] .= s * s * prod([nx,ny,nz]) * ssf[:,:,:,:]
    #Szz[xs,ys,zs,1,m,n,:] .= s * s * prod([nx,ny,nz]) * ssf[:,:,:,:] .* phases

    S_local[xs,ys,zs,:,m,n,:,3,3] .+= -s * one_magnon_c[:,:,:,:,m,1,2,n,1,1,:,1]
    S_local[xs,ys,zs,:,m,n,:,3,3] .+= -s * one_magnon_c[:,:,:,:,m,1,2,n,1,1,:,3]

    #Szz[xs,ys,zs,:,m,n,:] .+= -s * one_magnon_c[:,:,:,:,N+m,n,:,1]
    #Szz[xs,ys,zs,:,m,n,:] .+= -s * one_magnon_c[:,:,:,:,N+m,n,:,3]
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

function grid_it_2d(sys)
  Sloc, Sglob = render_one_over_S_spectrum(sys; beta = 8.0)
  Hs, Vs, disps = finite_spin_wave_Vmats(sys;polyatomic = false)
  f = Figure()
  display(f)
  for i = 1:3, j = 1:3
    s = log10.(abs.(raster_spec(Sloc,disps,0.2,50,i,j)[:,1,1,:]))
    ax = Axis(f[i,(j - 1) * 2 + 1])
    hm = heatmap!(ax,s,colorrange = (-8,2))
    Colorbar(f[i,(j-1)*2+2],hm)
  end
end

# Returns the index of the wavevector which completes the given
# list of wavevector indices in a momentum-conserving way.
# The ks should be a list [[ix,iy,iz],[jx,jz,jz],…], and
# daggers should be a bit string [0,1,1,0,…] with 1 for a†
function momentum_conserving_index(ks,daggers,ns)
  nx,ny,nz = ns
  dagger_signs = (-1) .^ daggers
  momentum_so_far = sum(dagger_signs[1:end-1] .* map(k -> k .- 1,ks);init = [0,0,0])
  I = mod1.(1 .- momentum_so_far * dagger_signs[end],[nx,ny,nz])

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
  CartesianIndex(tuple(I...))
end

function zero_magnon_sector(Hs,Vs,cryst;betaOmega)
  # This is correlators with exactly *zero* boson operators
  # Namely, it's the C_{1,1} correlator.
  yy = empty_y_matrix()
  yobj = wick_to_y_object(yy,wickify([]),wickify([]);betaOmega)
  nx,ny,nz = size(Vs)[7:9]
  ssf = zeros(ComplexF64,nx,ny,nz,length(yobj))
  ssf[1,1,1,:] .= yobj[-2:2]
  ssf
end

function one_magnon_sector(Hs,Vs,cryst;betaOmega)
  nx,ny,nz = size(Vs)[7:9]

  na = size(Vs,1)
  nf = size(Vs,2)

  It = diagm([repeat([1],N); repeat([-1],N)])

  # Fix phases! This implements "commutes with dagger" property
  #=
  Vs_fix = copy(Vs)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz
    V = view(Vs_fix,:,:,:,:,:,:,ix,iy,iz)
    for fix_a = 1:na, fix_f = 1:nf
      # Make first entry in each mode real in first half of columns
      view(V,:,:,:,fix_a,fix_f,1) .*= exp(-im * angle(V[1,1,1,fix_a,fix_f,1]))

      # Make first entry in *second half* of each mode real in second half
      view(V,:,:,:,fix_a,fix_f,2) .*= exp(-im * angle(V[1,1,2,fix_a,fix_f,2]))
    end
  end
  =#

  Vs_fix = copy(Vs)
  for ix = 1:nx, iy = 1:ny, iz = 1:nz
    Vhere = view(Vs_fix,:,:,:,:,:,1,ix,iy,iz)
    Vthere = view(Vs_fix,:,:,:,:,:,2,mod1(nx-(ix-1)+1,nx),mod1(ny-(iy-1)+1,ny),mod1(nz-(iz-1)+1,nz))
    for fix_a = 1:na, fix_f = 1:nf
      # Replace the negative energy modes at the opposite Q with equivalent
      # ones that are phase-matched to the positive energy modes at this Q
      #=
      println()
      println("Replacing:")
      display(view(Vthere,:,:,1,fix_a,fix_f))
      println("with")
      display(conj.(Vhere[:,:,2,fix_a,fix_f]))
      println("divisor::")
      display(angle.(view(Vthere,:,:,1,fix_a,fix_f) ./conj.(Vhere[:,:,2,fix_a,fix_f])))
      println("and")
      display(view(Vthere,:,:,2,fix_a,fix_f))
      println("with")
      display(conj.(Vhere[:,:,1,fix_a,fix_f]))
      println("divisor::")
      display(angle.(view(Vthere,:,:,2,fix_a,fix_f) ./conj.(Vhere[:,:,1,fix_a,fix_f])))
      =#
      view(Vthere,:,:,1,fix_a,fix_f) .= conj.(Vhere[:,:,2,fix_a,fix_f])
      view(Vthere,:,:,2,fix_a,fix_f) .= conj.(Vhere[:,:,1,fix_a,fix_f])
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
  # - The sublattice, flavor, and dagger configurations of a-boson being correlated
  # - The harmonic of the eigenmode
  # - The partition number
  correlator = zeros(ComplexF64,nx,ny,nz,na*nf,na,nf,2,na,nf,2,size(empty_y_matrix(),4),3)

  # The exactly one distinict wavevector of the one-magnon sector
  for x1 = 1:nx, y1 = 1:ny, z1 = 1:nz
    println("=== Working on k = $((x1,y1,z1)) ===")

    Vk1 = Vs_fix[:,:,:,:,:,:,x1,y1,z1]

    # We have that a1 = ∑ᵢT_1i bi.
    # So C_{ai,aj} = ∑_mn T_im T_jn C_{bm,bn}, but only m=n contributes

    # Loop over eigenmodes (flavor of b boson) at this k
    for m = 1:(na*nf)
      if any(isnan.(correlator))
        error("Nan")
      end
      # Loop over a-boson flavor and dagger configuration
      for i = 1:na, j = 1:na, iDag = [0,1], jDag = [0,1]

        print("::: C_")
        show_bitstring([iDag,jDag];b = 'a')
        println()

        # Infer k2 based on daggers
        ks = [[x1,y1,z1]]
        #println()
        println("Inferring k2 ...")
        x2,y2,z2 = momentum_conserving_index(ks,[iDag,jDag],(nx,ny,nz)).I
        Vk2 = Vs_fix[:,:,:,:,:,:,x2,y2,z2]

        k1 = (x1,y1,z1,m)
        k2 = (x2,y2,z2,m) # Only m = n contributes!
        omgc = one_magnon_gas_correlator((nx,ny,nz),N,k1,k2;betaOmega = betaOmega[m,x1,y1,z1], Y = Y_storage[x1,y1,z1,m])
        @assert sum(abs.(imag(omgc))) < 1e-12

        #println(cryst.positions[[i,j]])
        #ks = [[x1,y1,z1],[x2,y2,z2]]
        #println(ks)
        #println([iDag,jDag])
        #println([exp((-1)^(1-dag) * 2π * im * δ⋅((k .- 1)./[nx,ny,nz])) for (δ,k,dag) in zip(cryst.positions[[i,j]],ks,[iDag,jDag])])
        #sublattice_phase = prod([exp((-1)^(1-dag) * 2π * im * δ⋅((k .- 1)./[nx,ny,nz])) for (δ,k,dag) in zip(cryst.positions[[i,j]],ks,[iDag,jDag])])

        #println("Sublat phase:")
        #println(sublattice_phase)
        
        i_ix = (iDag * N) + i
        j_ix = (jDag * N) + j
        for partition = 1:3
          # Infer total k based on momenta left of the comma
          ks = [[x1,y1,z1],[x2,y2,z2]][1:partition-1]
          left_daggers = [iDag,jDag][1:partition-1]
          push!(left_daggers,1)
          #println(ks)
          #println(left_daggers)

          #println()
          println("Inferring total k ...")
          total_k = momentum_conserving_index(ks,left_daggers,(nx,ny,nz))
          #println(total_k)
          #println()

          # Compute sublattice phase, which depends on the partition:
          sublattice_phase = 1.0 + 0im
          for term = 1:2
            #if term < partition
              #println("left")
            #else
              #println("right")
            #end
            δ = cryst.positions[term < partition ? i : j]
            kraw = ([[x1,y1,z1],[x2,y2,z2]][term] .- 1) ./ [nx,ny,nz]
            ksigned = (-1)^(1 - [iDag,jDag][term]) * kraw
            sublattice_phase *= exp(2 * pi * im * δ⋅ksigned)
          end
          #println("Sublattice phase: $sublattice_phase")
              println(" k1 = $([x1,y1,z1]), k2 = $([x2,y2,z2]), i = $i, j = $j")

          # Loop over dagger configuration of bb
          for left = [0,1], right = [0,1]
            a = (left * N) + m
            b = (right * N) + m
            #println("Index: $((total_k,m,i_ix,j_ix)) with a = $a, b = $b, partition = $partition")
            #println("Vk1: $(Vk1[i_ix,a])")
            #println("Vk2: $(Vk2[j_ix,b])")
            #println("omgc: $(omgc[a,b,:,partition])")
            #println("Value: $(Vk1[i_ix,a] * Vk2[j_ix,b] * omgc[a,b,:,partition])")
            val = sublattice_phase * Vk1[i,1,iDag+1,m,1,left+1] * Vk2[j,1,jDag+1,m,1,right+1] * omgc[a,b,:,partition]
            if sum(abs.(val)) > 1e-12
              println("!! Finite contribution: eigenmodes $a and $b to HP bosons $i_ix and $j_ix")
              println("   Vk1: $(Vk1[i,1,iDag+1,m,1,left+1])")
              println("   Vk2: $(Vk2[j,1,jDag+1,m,1,right+1])")
              println("   Sublattice phase: $sublattice_phase")
            end
            correlator[total_k,m,i,1,iDag+1,j,1,jDag+1,:,partition] .+= sublattice_phase * Vk1[i,1,iDag+1,m,1,left+1] * Vk2[j,1,jDag+1,m,1,right+1] * omgc[a,b,:,partition]
          end
        end
      end
    end
    println()
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
function one_magnon_gas_correlator(ns,N,label_i,label_j;betaOmega, Y = nothing)
  ix,iy,iz,m = label_i
  jx,jy,jz,n = label_j
  nx,ny,nz = ns
  It = diagm([repeat([1],N); repeat([-1],N)])

  # This function computes C_{bb}. There is at most one distinct eigenmode in play
  # because there are two boson operators. So if we are asked for something else,
  # that correlator is zero:
  Y_storage = isnothing(Y) ? empty_y_matrix() : Y
  gas_correlator = zeros(ComplexF64,2N,2N,size(Y_storage,4),3)
  # Loop over dagger configuration
  for left = [0,1], right = [0,1]
    print("  OMGC $label_i $(left == 1 ? "(†)" : "( )") and $label_j $(right == 1 ? "(†)" : "( )") → ")
    label_j_reversed_k = collect(label_j)
    for i = 1:3
      label_j_reversed_k[i] = mod1((ns[i] + 1) - (label_j[i]-1),ns[i])
    end
    #println(collect(label_i))
    #println(label_j_reversed_k)
    #
    #
    # If they have the same dagger, e.g. (ak1,ak2), then it's verboten to have (k1 + k2) nonzero;
    # that is to say that (k1 = -k2) is required. If the daggers are opposite, e.g. (ak1†,ak2),
    # then instead (k1 = k2) is required.
    if left == right ? !(collect(label_i) == label_j_reversed_k) : !(label_i == label_j)
      println("Verboten! because more than one distinct eigenmode (max 1 allowed)")
      continue
    end

    println("Allowed!")

    # Loop over position of comma
    for partition = 1:3
      L = [left,right][1:partition-1]
      R = [left,right][partition:end]
      y = wick_to_y_object(Y_storage, wickify(L), wickify(R); betaOmega)

      ix_L = (left * N) + m
      ix_R = (right * N) + n
      gas_correlator[ix_L,ix_R,:,partition] = y[-2:2]
    end
  end

  gas_correlator
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
