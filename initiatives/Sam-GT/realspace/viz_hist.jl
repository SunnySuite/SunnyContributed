using Sunny, GLMakie, StaticArrays

function viz_qqq_path(params; kwargs...)
  f = Figure()
  ax = LScene(f[1,1];show_axis = false)
  Makie.cam3d!(ax.scene;projectiontype = Makie.Orthographic)
  viz_qqq_path!(ax,params;kwargs...)

  aabb_lo, aabb_hi = binning_parameters_aabb(params)
  lo = min.(0,floor.(Int64,aabb_lo))
  hi = max.(0,ceil.(Int64,aabb_hi))
  scatter!(ax,map(x -> Point3f(lo .+ x.I .- 1),CartesianIndices(ntuple(i -> 1 + hi[i] - lo[i],3)))[:],color = :black)
  linesegments!(ax,[(Point3f(-1,0,0),Point3f(1,0,0)),(Point3f(0,-1,0),Point3f(0,1,0)),(Point3f(0,0,-1),Point3f(0,0,1))],color = :black)
  text!(1.1,0,0;text ="qx [R.L.U.]")
  text!(0,1.1,0;text ="qy [R.L.U.]")
  text!(0,0,1.1;text ="qz [R.L.U.]")
  display(f)
  ax
end

function viz_qqq_path!(ax,params; background = nothing, line_alpha = 0.3,color = nothing,colorrange = nothing,bin_colors = [:red,:blue,:green],bin_line_width = 0.5)
  @assert iszero(params.covectors[1:3,4]) && iszero(params.covectors[4,1:3])
  bcs = Sunny.axes_bincenters(params)
  bes = Sunny.axes_binedges(params)
  M = inv(params.covectors[1:3,1:3])
  for dir = 1:3
    ix = [2,3,1][dir]
    iy = [3,1,2][dir]
    iz = dir

    # The grid of q points making up the lowest side of the histogram
    # along the iz direction
    grid = Vector{Float64}[]
    grid_sparse = Vector{Float64}[]
    for i = 1:length(bes[ix]), j = 1:length(bes[iy])
      is_i_edge = (i == 1 || i == length(bes[ix]))
      is_j_edge = (j == 1 || j == length(bes[iy]))
      grid_point = [bes[ix][i],bes[iy][j],bes[iz][1]][invperm([ix,iy,iz])]
      if is_i_edge && is_j_edge # Corner case; render special for outline
        push!(grid_sparse,grid_point)
        continue
      end
      push!(grid,grid_point)
    end
    offset = [0,0,bes[iz][end] - bes[iz][1]][invperm([ix,iy,iz])]

    if !isempty(grid)
      segs = map(x -> (Point3f(M * x),Point3f(M * (x .+ offset))),grid[:])
      linesegments!(ax,segs,color = bin_colors[dir],linewidth = bin_line_width,alpha = line_alpha)
    end

    segs = map(x -> (Point3f(M * x),Point3f(M * (x .+ offset))),grid_sparse[:])
    linesegments!(ax,segs;color = isnothing(color) ? :black : color,linewidth = 2.5,colorrange)
  end
end

# Find an axis-aligned bounding box containing the histogram
function binning_parameters_aabb(params)
    (; binstart, binend, covectors) = params
    bin_edges = Sunny.axes_binedges(params)
    first_edges = map(x -> x[1],bin_edges)
    last_edges = map(x -> x[end],bin_edges)
    bin_edges = [first_edges last_edges]
    this_corner = MVector{4,Float64}(undef)
    q_corners = MMatrix{4,16,Float64}(undef)
    for j = 1:16 # The sixteen corners of a 4-cube
        for k = 1:4 # The four axes
            this_corner[k] = bin_edges[k,1 + (j >> (k-1) & 1)]
        end
        this_corner[.!isfinite.(this_corner)] .= 0
        q_corners[:,j] = covectors \ this_corner
    end
    lower_aabb_q = minimum(q_corners,dims=2)[1:3]
    upper_aabb_q = maximum(q_corners,dims=2)[1:3]
    return lower_aabb_q, upper_aabb_q
end


