using Sunny, GLMakie, LinearAlgebra, FFTW

cryst = Crystal(I(3),[[0,0,0],[1/3,0,0],[1/2,0,0]])

sys = System(cryst,(20,1,1),[SpinInfo(1,S=1,g=1),SpinInfo(2,S=1,g=1),SpinInfo(3,S=1,g=1)],:dipole)

# This sets the state to:
#
#  x=0   x=1/3
#   ^      7
#   |     /    -->   ....
#             x=1/2
#
for j = 1:20; set_dipole!(sys,[0,1,0],(j,1,1,1)); set_dipole!(sys,[1,1,0],(j,1,1,2)); set_dipole!(sys,[1,0,0],(j,1,1,3)); end;

isc = instant_correlations(sys); add_sample!(isc,sys);

params = unit_resolution_binning_parameters(isc)
params.binend[1] += 5
bcs = axes_bincenters(params)

is = intensities_binned(isc,params,intensity_formula(isc,:full))[1]

println("Sum rule:")
println("  sum(is) =")
display(sum(is))
println()
println("  NBZ = 6")
println("  tr(sum(is))/NBZ = $(tr(sum(is))/6)")

f = Figure(); axs = [Axis(f[i,j], title = "S$("xyz"[i])$("xyz"[j])", xlabel = "Qx [R.L.U.]", ylabel = "Re & Im") for i = 1:3, j = 1:3]; display(f)

for i = 1:3, j = 1:3; plot!(axs[i,j],bcs[1],map(x->real(x[i,j]),is[:,1,1,1])); end;

for i = 1:3, j = 1:3; plot!(axs[i,j],bcs[1],map(x->imag(x[i,j]),is[:,1,1,1])); end;


ax = Axis(f[4,1],title = "Real space",xlabel = "Δx", ylabel = "Correlation")

# Sunny uses the spatial fourier transform convention B(k) ~ exp(+ikR) B(R):
sunny_convention_spatial_ifft(is) = fft(is)#/length(is)

corr_range = [x/6 for x = 0:9]
pxy = plot!(ax,corr_range,real.(sunny_convention_spatial_ifft(map(x -> x[1,2],is[:,1,1,1])))[1:10])

pyx = plot!(ax,corr_range,real.(sunny_convention_spatial_ifft(map(x -> x[2,1],is[:,1,1,1])))[1:10],marker = 'x')
Legend(f[4,2],[pxy,pyx],["Sxy","Syx"],tellwidth = false)

# Using | to represent [0,1,0], / for [1,1,0], - for [1,0,0], the pairs of
# spins contributing to real-space correlations at each range are:
#
# Range 0/6: ||  +  //  +  --
# Range 1/6: /- (xy = 0, yx nonzero)
# Range 2/6: |/ (xy = 0, yx nonzero)
# Range 3/6: |-  +  -|
# Range 4/6: /| (xy nonzero, yx = 0)
# Range 5/6: -/ (xy nonzero, yz = 0)
#
# following the convention that Sab ~ a(x0) b(x0 + Δx).
