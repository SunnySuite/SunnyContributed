using Sunny, GLMakie
include("viz_hist.jl")
cryst = Sunny.hexagonal_crystal()
path = q_space_path(cryst,[[0,0,0],[0,1,0]],14)

# Two different possible transverse binning configurations
params1 = Sunny.specify_transverse_binning(path,[1,1,0],[0,0,1],0.2,0.3)
params2 = Sunny.specify_transverse_binning(path,[1,0,0],[0,0,1],0.5,0.1)
ax = viz_qqq_path(params1;color = :blue)
viz_qqq_path!(ax,params2;color = :orange)

# Plot the original Q path for comparison
scatter!(map(x -> Point3f(x...),path.qs),color = :red,marker = 'x',markersize = 25)
