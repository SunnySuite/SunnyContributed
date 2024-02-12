using Sunny, FFTW, LinearAlgebra, GLMakie, ProgressMeter, Statistics

function spatial_correlation(sc)
  # Because the data was already in units S^2/BZ/fs, we need to divide by the
  # number of lattice offsets in the BZ to produce S^2/fs. This gets done
  # by the factor living inside ifft()
  real_corr = real.(ifft(sc.data,(4,5,6)))


  # Displacements associated with realspace correlations
  ls = size(sc.data)[4:6]
  dri = []
  for i = 1:3
    push!(dri,fftfreq(ls[i],ls[i]))
  end

  R = Array{Sunny.Vec3,3}(undef,ls...)
  for cell in CartesianIndices(ls)
    R[cell] = sc.crystal.latvecs * [dri[1][cell[1]];dri[2][cell[2]];dri[3][cell[3]]]
  end
  atom_pos = map(p -> sc.crystal.latvecs * p, sc.crystal.positions)
  na = Sunny.natoms(sc.crystal)

  R_all = Vector{Sunny.Vec3}(undef,0)
  corr_all = Vector{Vector{Float64}}(undef,0)

  # Correlations from every offset Δx
  for cell in CartesianIndices(ls), a = 1:na, b = 1:na

    # The detailed Δx includes the lattice displacement and the sublattice displacement
    this_R = R[cell] .+ (atom_pos[a] .- atom_pos[b])
    correlation = sum(real_corr[:,a,b,cell,:],dims = 2)[:,1]
    @assert iszero(imag(correlation))
    push!(R_all,this_R)
    push!(corr_all,real(correlation))
  end

  R_all, corr_all
end

function show_falloff(sc;kwargs...)
  f = Figure(); ax = Axis(f[1,1]);
  show_falloff!(ax,sc;kwargs...)
  f
end

function show_falloff!(ax,sc; cont = x -> x[1] + x[5] + x[9],slope = 0.0, scale = 1.0, log_mode = true, color = 0.5, uncorr_line = false, logx = true, kwargs...)
  R_all, corr_all = spatial_correlation(sc)
  corr_all = map(cont,corr_all)

  logdist = log10.(norm.(R_all))
  logcorr = map(x -> scale * log10(abs(x)),corr_all)
  non_logcorr = map(x -> scale * x,corr_all)

  logdist[.!isfinite.(logdist)] .= NaN
  logcorr[.!isfinite.(logcorr)] .= NaN
  ix_equal = findall(norm.(R_all) .== 0)

  inferred_spin_squared_value = mean(non_logcorr[ix_equal])
  num_estimators = prod(size(sc.data)[4:6]) * sc.nsamples[1]
  uncorrelated_spin_std = sqrt((inferred_spin_squared_value^2/3) / num_estimators)

  if log_mode
    scatter!(ax,logdist,logcorr .+ slope .* logdist; color = color, colorrange = (0,1), kwargs...)
    if iszero(slope)
      hlines!(ax,logcorr[ix_equal],color = :black)
      if uncorr_line
        hlines!(ax,log10(uncorrelated_spin_std),color = :black,linestyle = :dash)
      end
    end
  else
    @assert iszero(slope)
    scatter!(ax,logx ? logdist : norm.(R_all),non_logcorr; color = color, colorrange = (0,1), kwargs...)
    hlines!(ax,non_logcorr[ix_equal],color = :black)
    if uncorr_line
      hlines!(ax,[-uncorrelated_spin_std,uncorrelated_spin_std],color = :black,linestyle = :dash)
    end
  end

  #println(logcorr[logdist .< 0.1]) # This number sometimes universal

  unit_cell_scale = abs(det(sc.crystal.latvecs))^(1/3)
  lattice_count_scale = prod(size(sc.data)[4:6])^(1/3)

  # One unit cell away, then half the lattice away, then the maximum corner point
  characteristic_distances = [unit_cell_scale, unit_cell_scale * lattice_count_scale/2, unit_cell_scale * lattice_count_scale *sqrt(3)/2]
  if logx
    vlines!(ax,log10.(characteristic_distances),color = :black)
  end
  R_all, non_logcorr
end


function example_falloff_system_afm_chain(;kT = 0.1)
  cryst = Sunny.cubic_crystal()
  sys = System(cryst, (100,1,1), [SpinInfo(1;S=1/2,g=2)],:dipole)
  set_exchange!(sys,1.,Bond(1,1,[1,0,0]))
  randomize_spins!(sys)
  minimize_energy!(sys;maxiters = 500)

  langevin = Langevin(0.05;λ = 0.1, kT)
  for i = 1:10000
    step!(sys,langevin)
  end

  isc = instant_correlations(sys)

  for j = 1:16
    for i = 1:10000
      step!(sys,langevin)
    end
    add_sample!(isc,sys)
  end
  isc
end

# Shows that, for low temperatures, the correlation falls off as ~ exp(-aT) as
# temperature is increased.
# 
# At temperatures >> J, the spin constraint |S| = 1/2 causes the (paramagnetic) spins to be
# much more correlated than they would be for unconstrained spins.
function sweep_afm_chain(;log_mode = true)
  Ts = 10 .^ range(-4,-1,length = 30)
  f = Figure(); ax = Axis(f[1,1])
  for (i,T) = enumerate(Ts)
    isc = example_falloff_system_afm_chain(;kT = T)
    show_falloff!(ax,isc; scale = log_mode ? 1/T : 1.0, log_mode, logx = false, color = i/length(Ts), marker = :hline, colormap = :darkrainbow)
    display(f)
    sleep(0.01)
  end
  f
end


function mk_J1_sys(;bilin = 0.0,latsize = (6,6,6), ising = false)
  local pyrochlore
  pyrochlore = Crystal(I(3), [[1/2,0,0]],227,setting = "2")
  local sys
  sys = System(pyrochlore, latsize, [SpinInfo(1;S=1,g=2)],:dipole)
  set_pair_coupling!(sys,(Si, Sj) -> Si' * 1. * Sj - (Si' * bilin * Sj)^2,Bond(1,2,[0,0,0]))
  # Global Sz^2 anisotropy
  if ising
    se = Sunny.StevensExpansion(2,[0],[0,0,-0.1,0,0],zeros(Float64,9),zeros(Float64,13))
    for i = 1:16; sys.interactions_union[i].onsite = se;end;
  end
  #to_inhomogeneous
  randomize_spins!(sys)
  sys
end



function mk_J1_closure(;kwargs...)
  sys = mk_J1_sys(;kwargs...)

  local isc
  isc = instant_correlations(sys)

  function(;kT = 0.1,nsamp,nwander = 1000)
    randomize_spins!(sys)
    minimize_energy!(sys;maxiters = 500)
    isc.data .= 0
    isc.nsamples .= 0

    langevin = Langevin(0.05;λ = 0.1, kT)
    for i = 1:10000
      step!(sys,langevin)
    end

    add_sample!(isc,sys) # First sample for free!
    prog = Progress(nsamp-1,"Sampling")
    for j = 1:(nsamp-1)
      for i = 1:nwander
        step!(sys,langevin)
      end
      add_sample!(isc,sys)
      next!(prog)
    end
    finish!(prog)
    isc
  end
end

example_falloff_system_J1_pyrochlore = mk_J1_closure(;latsize = (3,3,3), ising = false)

function mean_polarization(sys)
  # Make sure it's a pyrochlore crystal
  @assert length(sys.crystal.positions) == 16
  vs = []
  for i = 1:3
    sysd = to_polarization_system(sys;ix_polar = i)
    ts(x) = Sunny.number_to_simple_string(x;digits = 4)
    v = mean(sysd.dipoles)
    vv = var(sysd.dipoles)
    println("Mean for flavor $(["x","y","z"][i]) = [$(ts(v[1])), $(ts(v[2])), $(ts(v[3]))]; variance = [$(ts(vv[1])), $(ts(vv[2])), $(ts(vv[3]))]")
    push!(vs,v)
  end
  vs
end

function charge_stats(sys)
  # Make sure it's a pyrochlore crystal
  @assert length(sys.crystal.positions) == 16
  sys_charge = to_polarization_system(sys;get_charge = true)
  println("Vector charge: μ = $(mean(sys_charge.dipoles))")
  println("               σ = $(var(sys_charge.dipoles))")
  println("|Vector| charge: μ = $(mean(norm.(sys_charge.dipoles))), σ = $(var(norm.(sys_charge.dipoles)))")
  nothing
end

function to_polarization_system(sys_pyro;ix_polar = 3,get_charge = false, henley_mode = false)
  pyrochlore = Crystal(I(3), [[1/2,0,0]],227,setting = "2")
  diamond_chlore = Crystal(I(3), [[1/8,1/8,5/8]],227,setting = "2")
  neighbors = [diamond_chlore.positions[i] .- pyrochlore.positions[j] for i = 1:8, j = 1:16]
  cell_displace = map(y -> map(x -> isapprox(1-0.125,abs(x),atol = 1e-8) ? Int64(sign(x)) : Int64(0),y),neighbors)
  adjacency = map(y -> all(map(x -> isapprox(0.125,abs(x),atol = 1e-8) || isapprox(1-0.125,abs(x),atol = 1e-8),y)),neighbors)

  bond_vectors = Vector{Vector{Float64}}(undef,16)
  for j = 1:16
    bond_ends = findall(adjacency[:,j])
    bipartite_site_parity = [1,1,-1,-1,1,1,-1,-1]

    # bond_ends has an arbitrary direction 1→end for the bond.
    # Here we adjudicate the arbitrary direction with the bipartite nature of the lattice
    bond_sign = bipartite_site_parity[bond_ends[1]]
    raw_bond = diamond_chlore.positions[bond_ends[end]] .- diamond_chlore.positions[bond_ends[1]]
    closest_pass = diamond_chlore.latvecs * (mod.((diamond_chlore.latvecs \ raw_bond) .+ 0.5,1.) .- 0.5)
    bond_vectors[j] = bond_sign * normalize(closest_pass)
  end

  # Henley mode follows PhysRevB.71.014424 equation (2.1), which says to
  # perform a preliminary coarse-graining where the spins on the corners of each
  # tetrahedra are *schwoomped* [technical term] onto the center of the tetrahedra.
  # The centers of tetrahedra are represented by the sites of the diamond_chlore system.
  if henley_mode
    sys_d = System(diamond_chlore,sys_pyro.latsize,[SpinInfo(1;S=1,g=2)],:dipole)

    sys_d.dipoles .= Ref(0 .* sys_d.dipoles[1])
    for k = 1:8
      neighs = findall(adjacency[k,:])
      for ci = Sunny.eachcell(sys_pyro)
        for ni = 1:4
          neighbor_atom = neighs[ni]
          ci_grab = CartesianIndex(mod1.(ci.I .+ cell_displace[k,neighbor_atom],sys_pyro.latsize)...)
          if get_charge
            sys_d.dipoles[ci,k] += sys_pyro.dipoles[ci_grab,neighbor_atom]
          else
            sys_d.dipoles[ci,k] += sys_pyro.dipoles[ci_grab,neighbor_atom][ix_polar] * bond_vectors[neighbor_atom]
          end
        end
      end
    end
    sys_d

  # In non-Henley mode, we instead leave the individual polarizations on the sites
  # of the pyrochlore lattice (that is, we *don't* perform the sum in eqn 2.1)
  else
    sys_pyro_polar = Sunny.clone_system(sys_pyro)
    for j = 1:16
      for ci = Sunny.eachcell(sys_pyro)
        if get_charge
          # Getting the charge in non-Henley mode is a no-op because coarse-graining
          # the original spins is exactly just computing the charge!
          sys_pyro_polar.dipoles[ci,j] = sys_pyro.dipoles[ci,j]
        else
          sys_pyro_polar.dipoles[ci,j] = sys_pyro.dipoles[ci,j][ix_polar] * bond_vectors[j]
        end
      end
    end
    sys_pyro_polar
  end
end

function hyperplot_spins(sys;kwargs...)
  f = Figure(); ax = LScene(f[1,1])
  hyperplot_spins!(ax,sys;kwargs...)
  f
end

function hyperplot_spins!(ax,sys;sigma = 1.0,n_spin = 1000,ntracer = 10,phase_mode = false,flux_color = :black, flux_width = 1.5,kwargs...)
  supervecs = sys.crystal.latvecs * diagm(Vec3(sys.latsize))
  max_eig = maximum(eigvals(sys.crystal.latvecs))
  three_sigma = round(Int64,(3 * sigma) / max_eig)
  wrap_ranges = min.(sys.latsize,2 * three_sigma + 1)
  pts = supervecs * rand(3,n_spin)
  atom_pos = map(p -> sys.crystal.latvecs * p, sys.crystal.positions)
  println(three_sigma)
  println(sys.latsize)

  f_vec = function(pt)
    v = [0.,0.,0.]
    z = 0
    nominal_cell = round.(Int64,sys.crystal.latvecs \ pt)
    for disp_cell in CartesianIndices(wrap_ranges), a = 1:size(sys.dipoles,4)
      cell = mod1.(disp_cell.I .- ((wrap_ranges .- 1).÷2 .+ 1) .+ nominal_cell,sys.latsize)
      this_R = sys.crystal.latvecs * [cell[1] ; cell[2]; cell[3]] .+ atom_pos[a]
      this_spin = sys.dipoles[cell[1],cell[2],cell[3],a]
      closest_pass = supervecs * (mod.((supervecs \ (this_R .- pt)) .+ 0.5,1.) .- 0.5)
      v .+= this_spin .* exp(-norm(closest_pass)^2/(2sigma^2)) / sigma
      z += exp(-norm(closest_pass)^2/(2sigma^2)) / sigma
    end
    v
  end
  vv = [normalize(f_vec(pts[:,i]))/7 for i = 1:n_spin]
  vx = map(x -> x[1],vv)
  vy = map(x -> x[2],vv)
  vz = map(x -> x[3],vv)

  acobs = Observable(copy(vz))
  c3d = ax.scene.camera_controls
  on(c3d.eyeposition) do ey
    empty!(acobs[])
    if phase_mode
      v_ref = normalize(Float64.(c3d.lookat[] .- c3d.eyeposition[]))
      v_z = [0,0,1.]
      v_right = normalize(cross(v_ref,v_z))
      v_up = normalize(cross(v_right,v_ref))
      for i = eachindex(vx)
        v_arrow = normalize([vx[i];vy[i];vz[i]])
        ph = atan(v_up ⋅ v_arrow,v_right ⋅ v_arrow)
        push!(acobs[], ph)
      end
    else
      v_ref = normalize(Float64.(c3d.lookat[] .- c3d.eyeposition[]))
      for i = eachindex(vx)
        push!(acobs[], v_ref ⋅ normalize([vx[i];vy[i];vz[i]]))
      end
    end
    notify(acobs)
  end
  arrows!(ax,pts[1,:],pts[2,:],pts[3,:],vx,vy,vz;arrowcolor = acobs, colorrange = phase_mode ? (-pi,pi) : (0,1),colormap = phase_mode ? :phase : :bluesreds, color = acobs, arrowsize = 0.1,linewidth = 0.03,kwargs...)

  prog = Progress(ntracer,"Tracing")
  for n = 1:ntracer
    nstep = 1000
    trail = zeros(Float64,3,nstep)
    x0 = supervecs * rand(3)
    for t = 1:nstep
      trail[:,t] .= x0
      rel = supervecs \ x0
      if any(rel .> 1.1) || any(rel .< -0.1)
        trail[:,t] .= NaN
        x0 = supervecs * mod.(rel,1)
        continue
      end
      v0 = f_vec(x0)
      x0 = x0 + v0 * 1e-2
    end
    lines!(ax,trail[1,:],trail[2,:],trail[3,:],linewidth = flux_width,color = flux_color)
    next!(prog)
  end
  finish!(prog)
  f
end

function sweep_J1_pyrochlore()
  Ts = 10 .^ range(-8,4,length = 30)
  #Ts = 10 .^ range(3,4,length = 2)
  f = Figure(); ax = Axis(f[1,1],xlabel = "log_{10} |R|",ylabel = "Correlation")
  ax2 = Axis(f[2,1],xlabel = "Nearest Neighbor Correlation")
  ax3 = Axis(f[3,1],xlabel = "Correlation (beyond unit cell distance only)")
  ax2a = Axis(f[2,3],xlabel = "log_{10} Temperature",ylabel = "Mean NN Correlation")
  ax3a = Axis(f[3,3],xlabel = "log_{10} Temperature",ylabel = "Estimated pop. Variance (Correlation b.u.c.d.o.)")
  hlines!(ax2a,0.)
  vlines!(ax2a,0.)
  vlines!(ax3a,0.)
  Colorbar(f[1,2];colormap = :lightrainbow, colorrange = (log10(Ts[1]),log10(Ts[end])))
  for (i,T) = enumerate(Ts)
    nsamp = 16
    isc = example_falloff_system_J1_pyrochlore(;kT = T,nsamp)
    R_all, corr_all = show_falloff!(ax,isc; scale = 1.0, log_mode = false, color = i/length(Ts), marker = :hline, colormap = :lightrainbow)

    # Nearest neighbors only
    ix_nn = 0.1 .< norm.(R_all) .< 0.5
    stephist!(ax2,corr_all[ix_nn], color = i/length(Ts),colormap = :lightrainbow,colorrange = (0,1))
    m = mean(corr_all[ix_nn])
    scatter!(ax2a,log10(T),m,color = i/length(Ts), colormap = :lightrainbow, colorrange = (0,1))

    ix_outside_cell = norm.(R_all) .> 1.0
    stephist!(ax3,corr_all[ix_outside_cell], color = i/length(Ts),colormap = :lightrainbow,colorrange = (0,1))
    v = var(corr_all[ix_outside_cell])
    num_estimators = prod(size(isc.data)[4:6]) * nsamp
    scatter!(ax3a,log10(T),v*num_estimators,color = i/length(Ts), colormap = :lightrainbow, colorrange = (0,1))

    println("log10 T = $(log10(T)), m = $m, v*#est = $(v * num_estimators), #est = $num_estimators")

    display(f)
    sleep(0.01)
  end
  f
end

function glass_transition_J1_pyrochlore()
  Ts = 10 .^ range(-4,-1.5,length = 30)
  wander_times = round.(Int64,10 .^ range(1.5,3,step = 0.25))
  pop_variance_est = zeros(Float64,length(wander_times),length(Ts))

  f = Figure(); ax = Axis(f[1,1],xlabel = "log_{10} |R|",ylabel = "Correlation")
  ax3 = Axis(f[2,1],xlabel = "Correlation (beyond unit cell distance only)")
  ax3a = Axis(f[2,3],xlabel = "log_{10} Temperature",ylabel = "Estimated pop. Variance (Correlation b.u.c.d.o.)")

  Colorbar(f[1,2];colormap = :lightrainbow, colorrange = (log10(Ts[1]),log10(Ts[end])))
  Colorbar(f[2,2];colormap = :viridis, colorrange = (log10(wander_times[1]),log10(wander_times[end])))
  for (i,T) = enumerate(Ts)
    println("log10 T = $(log10(T))")
    for (i_t,tw) = enumerate(wander_times)
      nsamp = 16

      isc = example_falloff_system_J1_pyrochlore(;kT = T,nsamp,nwander = tw)
      R_all, corr_all = if i_t == length(wander_times)
        show_falloff!(ax,isc; scale = 1.0, log_mode = false, color = i/length(Ts), marker = :hline, colormap = :lightrainbow)
      else
        R_all, corr_all = spatial_correlation(isc)
        non_logcorr = map(x -> (x[1] + x[5] + x[9]),corr_all)
        R_all, non_logcorr
      end


      ix_outside_cell = norm.(R_all) .> 1.0
      if i_t == length(wander_times)
        stephist!(ax3,corr_all[ix_outside_cell], color = i/length(Ts),colormap = :lightrainbow,colorrange = (0,1))
      end
      v = var(corr_all[ix_outside_cell])
      num_estimators = prod(size(isc.data)[4:6]) * nsamp
      pop_variance_est[i_t,i] = v * num_estimators
      scatter!(ax3a,log10(T),v*num_estimators,color = i_t/length(wander_times), colormap = :viridis, colorrange = (0,1))
      println("  log10 tw = $(log10(tw)), v*#est = $(v * num_estimators), #est = $num_estimators")
      display(f)
      sleep(0.01)
    end
  end
  display(f)
  pop_variance_est
end

function anneal_schedule!(sys; step_each = 1000)
  Ts = reverse(10 .^ range(-4,1,length = 100))
  langevin = Langevin(0.05;λ = 0.1, kT = 0)
  f = plot_spins(sys)
  ax_temp = Axis(f.figure[1,2])
  spin_temps = Observable(log10.(copy(Ts)))
  scatter!(ax_temp,log10.(Ts),spin_temps)
  lines!(ax_temp,log10.(Ts),log10.(Ts))
  spin_temps[] .= NaN
  display(f)
  sleep(0.01)
  for i = 1:length(Ts)
    println(Ts[i])
    langevin.kT = Ts[i]
    for j = 1:step_each
      step!(sys,langevin)
    end
    sT = classical_spin_temperature(sys)
    spin_temps[][i] = sT > 0 ? log10(sT) : NaN
    notify(spin_temps)
    println("Spin temperature = $(sT)")
    notify(f)
    sleep(0.01)
  end
end

function classical_spin_temperature(sys)
  dHdS = -Sunny.energy_grad_dipoles(sys)
  numer = sum([norm(cross(sys.dipoles[i],dHdS[i]))^2 for i = eachindex(sys.dipoles)])
  denom = 2 * sum([dot(sys.dipoles[i],dHdS[i]) for i = eachindex(sys.dipoles)])
  numer / denom
end
