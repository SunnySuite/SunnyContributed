using Sunny

# Load experiment data
histogram_parameters, data = load_nxs("experiment_data_normalized.nxs")
display(histogram_parameters)

# Demonstrate that the data is binned correctly
@assert collect(size(data)) == histogram_parametres.numbins

# Plot the data
using GLMakie

heatmap(data[:,:,1,1])

heatmap(data[:,:,1,3])

bin_centers = axes_bincenters(histogram_parameters)
heatmap(bin_centers[1],bin_centers[2],data[:,:,1,1])

# Export to ParaView (uncomment to try it)
# using WriteVTK
# export_vtk("example_vtk",histogram_parameters,data)

# Q-resolution effects
slice_thin  =     data[13   ,:,:,:]
slice_thick = sum(data[12:14,:,:,:], dims=1)

heatmap(slice_thin[1,:,1,3:end])
heatmap(slice_thick[1,:,1,3:end])
