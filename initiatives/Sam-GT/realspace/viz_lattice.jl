using Sunny, GLMakie, LinearAlgebra, Observables

include("higher_swt.jl")
include("viz_hist.jl")

function type_to_color(site,type)
  try
    Sunny.PlottingExt().type_to_color(type)
  catch e
    site
  end
end

function view_lattice(cryst;params = nothing,magnetic_structure = nothing,latsize = (1,1,1),crystal_for_recip_size = cryst)
  if !isnothing(params)
    @assert iszero(params.covectors[1:3,4]) && iszero(params.covectors[4,1:3])
  end
  if !isnothing(magnetic_structure)
    @assert magnetic_structure.latsize == latsize
  end
  na = length(cryst.positions)
  rs = abs(det(cryst.latvecs))^(1/3)

  f = Figure()
  controls = GridLayout(f[3,1],tellwidth = false)
  gl = GridLayout(controls[1,1])
  txt_tog = Toggle(gl[1,1])
  Label(gl[1,2];text = "Toggle labels")
  sg = SliderGrid(controls[2,1],(label = "Font Size",range = range(1,16,length = 200),startvalue = 10),width = 300)
  fontsize = sg.sliders[1].value
  Label(controls[3,1];text = "rₛ=$(Sunny.number_to_simple_string(rs;digits = 4))Å")

  instructs = GridLayout(f[3,2],tellwidth = false)
  Label(instructs[1,1];text = "Left/Right click any view to Rotate/Pan")
  Label(instructs[2,1];text = "Ctrl+click any view to reset it if you get lost!")

  gl = GridLayout(f[1,1])
  ax_real_lattice = LScene(gl[2,1];show_axis = false)
  Label(gl[1,1];text = "Lattice (Lab Frame)",tellwidth=false)

  lat_points = Point3f[]
  lat_points_rlu = Point3f[]
  point_sites = Int64[]
  nbzs = polyatomic_bzs(crystal_for_recip_size)

  recip_lat_bin_ix = CartesianIndex{3}[]
  recip_lat_comm_points = Point3f[]
  recip_lat_abs_comm_points = Point3f[]

  for lat_loc = CartesianIndices(latsize)
    for i = 1:na
      push!(lat_points,Point3f(cryst.latvecs * (collect(lat_loc.I .- 1) + cryst.positions[i])))
      push!(lat_points_rlu,Point3f(collect(lat_loc.I .- 1) + cryst.positions[i]))
      push!(point_sites,i)
    end
    for bz = CartesianIndices(ntuple(i -> nbzs[i],3))
      q = collect(bz.I .- 1) + collect((lat_loc.I .- 1) ./ latsize)
      push!(recip_lat_bin_ix,CartesianIndex(ntuple(i -> latsize[i] * (bz.I[i] - 1) + (lat_loc.I[i] - 1) + 1,3)))
      push!(recip_lat_comm_points,Point3f(q))
      push!(recip_lat_abs_comm_points,Point3f(cryst.recipvecs * q))
    end
  end

  scatter!(ax_real_lattice,lat_points,color = map(i -> type_to_color(i,cryst.types[i]),point_sites))
  n(y) = Sunny.number_to_math_string(y;atol = cryst.symprec)
  m(y) = Sunny.number_to_simple_string(y;digits = 4)
  text!(ax_real_lattice,lat_points;color = :black,text = map(x -> "($(m(x[1])),$(m(x[2])),$(m(x[3])))",lat_points),align = (:center,:top),visible = txt_tog.active,fontsize)
  text!(ax_real_lattice,lat_points;color = :black,text = map(x -> cryst.types[x],point_sites),align = (:center,:bottom),visible = txt_tog.active,fontsize)

  linesegments!(ax_real_lattice,[(Point3f(-1,0,0),Point3f(1,0,0)),(Point3f(0,-1,0),Point3f(0,1,0)),(Point3f(0,0,-1),Point3f(0,0,1))],color = :black)
  linesegments!(ax_real_lattice,[(Point3f(0,0,0),Point3f(cryst.latvecs[:,j]...)) for j = 1:3],color = [:red,:orange,:magenta],linewidth = 2.5)
  if !isnothing(params)
    linesegments!(ax_real_lattice,[(Point3f(0,0,0),Point3f((cryst.latvecs * params.covectors[j,1:3])...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
  end
  if !isnothing(magnetic_structure)
    Sunny.Plotting.plot_spins!(ax_real_lattice,magnetic_structure,color = [ix.I[4] for ix = eachsite(sys)])
  end

  scatter!(ax_real_lattice,cryst.latvecs[:,1]...,color = :red,marker = 'o')
  scatter!(ax_real_lattice,cryst.latvecs[:,2]...,color = :orange,marker = 'o')
  scatter!(ax_real_lattice,cryst.latvecs[:,3]...,color = :deeppink,marker = 'o')
  text!(ax_real_lattice,cryst.latvecs[:,1]...;text ="a",align = (:right,:bottom),color = :red)
  text!(ax_real_lattice,cryst.latvecs[:,2]...;text ="b",align = (:right,:bottom),color = :orange)
  text!(ax_real_lattice,cryst.latvecs[:,3]...;text ="c",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_real_lattice,rs,0,0,color = :black,marker = 'o')
  scatter!(ax_real_lattice,0,rs,0,color = :black,marker = 'o')
  scatter!(ax_real_lattice,0,0,rs,color = :black,marker = 'o')
  text!(ax_real_lattice,rs,0,0;text ="x (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_lattice,0,rs,0;text ="y (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_lattice,0,0,rs;text ="z (rₛ,Å)",align = (:left,:bottom))

  cam0 = Makie.cam3d!(ax_real_lattice.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  gl = GridLayout(f[2,1])
  ax_real_rlu = LScene(gl[2,1];show_axis = false)
  Label(gl[1,1];text = "Lattice (Crystal-adapted frame)",tellwidth=false)

  M = inv(cryst.latvecs)

  scatter!(ax_real_rlu,lat_points_rlu,color = map(i -> type_to_color(i,cryst.types[i]),point_sites))
  text!(ax_real_rlu,lat_points_rlu;color = :black,text = map(x -> "($(n(x[1])),$(n(x[2])),$(n(x[3])))",lat_points_rlu),align = (:center,:top),visible = txt_tog.active,fontsize)
  text!(ax_real_rlu,lat_points_rlu;color = :black,text = map(x -> cryst.types[x],point_sites),align = (:center,:bottom),visible = txt_tog.active,fontsize)

  linesegments!(ax_real_rlu,[(Point3f(-rs * M[:,j]...),Point3f(rs * M[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_real_rlu,[(Point3f(0,0,0),Point3f(I(3)[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  scatter!(ax_real_rlu,1,0,0;color = :red,marker = 'o')
  scatter!(ax_real_rlu,0,1,0;color = :orange,marker = 'o')
  scatter!(ax_real_rlu,0,0,1;color = :deeppink,marker = 'o')
  text!(ax_real_rlu,1,0,0;text ="a",align = (:right,:bottom),color = :red)
  text!(ax_real_rlu,0,1,0;text ="b",align = (:right,:bottom),color = :orange)
  text!(ax_real_rlu,0,0,1;text ="c",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_real_rlu,rs * M[:,1]...,color = :black,marker = 'o')
  scatter!(ax_real_rlu,rs * M[:,3]...,color = :black,marker = 'o')
  scatter!(ax_real_rlu,rs * M[:,2]...,color = :black,marker = 'o')
  text!(ax_real_rlu,rs * M[:,1]...;text ="x (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_rlu,rs * M[:,2]...;text ="y (rₛ,Å)",align = (:left,:bottom))
  text!(ax_real_rlu,rs * M[:,3]...;text ="z (rₛ,Å)",align = (:left,:bottom))

  if !isnothing(params)
    linesegments!(ax_real_rlu,[(Point3f(0,0,0),Point3f(params.covectors[j,1:3]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
  end
  if !isnothing(magnetic_structure)
    # This doesn't really work because the spin orientations are tied to the physical coordinates
    # in a complicated, pseudo-spin way... need to think more about this!
    #Sunny.Plotting.plot_spins!(ax_real_rlu,magnetic_structure,color = [ix.I[4] for ix = eachsite(sys)])
  end

  cam1 = Makie.cam3d!(ax_real_rlu.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  gl = GridLayout(f[2,2])
  ax_recip_rlu = LScene(gl[2,1];show_axis = false)
  Label(gl[1,1];text = "Reciprocal Lattice (Crystal-adapted frame)",tellwidth=false)
  N = cryst.latvecs'

  scatter!(ax_recip_rlu,recip_lat_comm_points,color = [all(isinteger.(x)) ? :red : :black for x in recip_lat_comm_points])
  text!(ax_recip_rlu,recip_lat_comm_points;color = :black,text = map(x -> "($(n(x[1])),$(n(x[2])),$(n(x[3])))",recip_lat_comm_points),align = (:center,:top),visible = txt_tog.active,fontsize)

  linesegments!(ax_recip_rlu,[(Point3f(-(1/rs) * N[:,j]...),Point3f((1/rs) * N[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_recip_rlu,[(Point3f(0,0,0),Point3f(I(3)[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  text!(ax_recip_rlu,1,0,0;text ="qx [rlu]",align = (:right,:bottom),color = :red)
  text!(ax_recip_rlu,0,1,0;text ="qy [rlu]",align = (:right,:bottom),color = :orange)
  text!(ax_recip_rlu,0,0,1;text ="qz [rlu]",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_recip_rlu,(1/rs) * N[:,1]...,color = :black,markersize = 12,marker = 'o')
  scatter!(ax_recip_rlu,(1/rs) * N[:,3]...,color = :black,markersize = 12,marker = 'o')
  scatter!(ax_recip_rlu,(1/rs) * N[:,2]...,color = :black,markersize = 12,marker = 'o')
  text!(ax_recip_rlu,(1/rs) * N[:,1]...;text ="x (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_rlu,(1/rs) * N[:,2]...;text ="y (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_rlu,(1/rs) * N[:,3]...;text ="z (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))

  if !isnothing(params)
    mantid_axes_as_columns = inv(params.covectors[1:3,1:3])
    linesegments!(ax_recip_rlu,[(Point3f(0,0,0),Point3f(mantid_axes_as_columns[:,j]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
    viz_qqq_path!(ax_recip_rlu,params;bin_colors = [:blue,:green,:purple])#,line_alpha = 1.0,bin_line_width = 1.5)
  end

  cam2 = Makie.cam3d!(ax_recip_rlu.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  gl = GridLayout(f[1,2])
  ax_recip_abs = LScene(gl[2,1];show_axis = false)
  Label(gl[1,1];text = "Reciprocal Lattice (Lab frame)",tellwidth=false)
  L = 2π * I(3)

  if !isnothing(magnetic_structure)
    isc = instant_correlations(magnetic_structure)
    add_sample!(isc,magnetic_structure)
    p_isc = unit_resolution_binning_parameters(isc)
    p_isc.binend[1:3] .+= nbzs .- 1
    #display(cryst)
    #display(magnetic_structure)
    #display(nbzs)
    #display(p_isc)
    i,c = intensities_binned(isc,p_isc,intensity_formula(isc,:trace))
    scatter!(ax_recip_abs,recip_lat_abs_comm_points,color = [i[ix,1] for ix in recip_lat_bin_ix],markersize = 18)
  else
    # Default: Just color integer R.L.U. points red
    scatter!(ax_recip_abs,recip_lat_abs_comm_points,color = [all(isinteger.(x)) ? :red : :black for x in recip_lat_comm_points])
  end
  text!(ax_recip_abs,recip_lat_abs_comm_points;color = :black,text = map(x -> "($(m(x[1])),$(m(x[2])),$(m(x[3])))",recip_lat_abs_comm_points),align = (:center,:top),visible = txt_tog.active,fontsize)

  linesegments!(ax_recip_abs,[(Point3f(-(1/rs) * L[:,j]...),Point3f((1/rs) * L[:,j]...)) for j = 1:3],color = :black)
  linesegments!(ax_recip_abs,[(Point3f(0,0,0),Point3f(cryst.recipvecs[:,j]...)) for j = 1:3],color = [:red,:orange,:deeppink])

  text!(ax_recip_abs,cryst.recipvecs[:,1]...;text ="qx [rlu]",align = (:right,:bottom),color = :red)
  text!(ax_recip_abs,cryst.recipvecs[:,2]...;text ="qy [rlu]",align = (:right,:bottom),color = :orange)
  text!(ax_recip_abs,cryst.recipvecs[:,3]...;text ="qz [rlu]",align = (:right,:bottom),color = :deeppink)

  scatter!(ax_recip_abs,(1/rs) * L[:,1]...,color = :black,marker = 'o',markersize = 12)
  scatter!(ax_recip_abs,(1/rs) * L[:,3]...,color = :black,marker = 'o',markersize = 12)
  scatter!(ax_recip_abs,(1/rs) * L[:,2]...,color = :black,marker = 'o',markersize = 12)
  text!(ax_recip_abs,(1/rs) * L[:,1]...;text ="x (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_abs,(1/rs) * L[:,2]...;text ="y (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))
  text!(ax_recip_abs,(1/rs) * L[:,3]...;text ="z (rₛ⁻¹,Å⁻¹)",align = (:left,:bottom))

  if !isnothing(params)
    mantid_axes_as_columns = cryst.recipvecs * inv(params.covectors[1:3,1:3])
    linesegments!(ax_recip_abs,[(Point3f(0,0,0),Point3f(mantid_axes_as_columns[:,j]...)) for j = 1:3],color = [:blue,:green,:purple],linewidth = 5.5,overdraw = true)
    params_abs = copy(params)
    Sunny.bin_absolute_units_as_rlu!(params_abs,cryst)
    viz_qqq_path!(ax_recip_abs,params_abs;bin_colors = [:blue,:green,:purple])#,line_alpha = 1.0,bin_line_width = 1.5)
  end

  cam3 = Makie.cam3d!(ax_recip_abs.scene;projectiontype = Makie.Orthographic,clipping_mode = :static)

  cams = [cam0,cam1,cam2,cam3]
  axs = [ax_real_lattice,ax_real_rlu,ax_recip_rlu,ax_recip_abs]
  mats = Matrix{Matrix}(undef,4,4)
  mats[1,2] = cryst.latvecs
  mats[2,1] = M

  mats[3,1] = M
  mats[1,3] = cryst.latvecs
  mats[2,3] = I(3)
  mats[3,2] = I(3)

  mats[1,4] = cryst.latvecs * inv(cryst.recipvecs)
  mats[4,1] = cryst.recipvecs * inv(cryst.latvecs)
  mats[2,4] = inv(cryst.recipvecs)
  mats[4,2] = cryst.recipvecs
  mats[3,4] = inv(cryst.recipvecs)
  mats[4,3] = cryst.recipvecs
  function update_A_from_B(a,b)
    if a == b
      return # Already up to date!
    end
    camA = cams[a+1]
    camB = cams[b+1]
    mAB = mats[a+1,b+1]
    Observables.setexcludinghandlers!(camA.lookat, mAB * camB.lookat[])
    Observables.setexcludinghandlers!(camA.eyeposition, mAB * camB.eyeposition[])
    Observables.setexcludinghandlers!(camA.upvector, mAB * camB.upvector[])
    update_cam!(axs[a+1].scene)
  end
  function sync_cameras(source)
    for dest = (1:length(cams)) .- 1
      update_A_from_B(dest,source)
    end
  end

  on(x -> sync_cameras(0), cam0.lookat)
  on(x -> sync_cameras(0), cam0.eyeposition)
  on(x -> sync_cameras(1), cam1.lookat)
  on(x -> sync_cameras(1), cam1.eyeposition)
  on(x -> sync_cameras(2), cam2.lookat)
  on(x -> sync_cameras(2), cam2.eyeposition)
  on(x -> sync_cameras(3), cam3.lookat)
  on(x -> sync_cameras(3), cam3.eyeposition)



  display(f)

  # Returns the labelled points used in each view
  out = Matrix{Any}(undef,2,2)
  out[1,1] = lat_points
  out[1,2] = recip_lat_abs_comm_points
  out[2,1] = lat_points_rlu
  out[2,2] = recip_lat_comm_points
  out
end

