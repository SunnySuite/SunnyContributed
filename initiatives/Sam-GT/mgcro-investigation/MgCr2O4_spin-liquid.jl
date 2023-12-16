using Sunny # The main package
using GLMakie # Plotting package
using LinearAlgebra, StaticArrays, Statistics

cif = "MgCr2O4_160953_2009.cif"
xtal_mgcro = Crystal(cif; symprec=0.001)
xtal_mgcro = subcrystal(xtal_mgcro,"Cr1")

dims = (6, 6, 6)  # Supercell dimensions 
#dims = (20, 20, 20)  # Supercell dimensions 
spininfos = [SpinInfo(1, S=3/2, g=2)]  # Specify spin information, note that all sites are symmetry equivalent 
sys_mgcro = System(xtal_mgcro, dims, spininfos, :dipole); # Same on MgCr2O4 crystal

print_symmetry_table(xtal_mgcro, 6.0) 

view_crystal(xtal_mgcro, 5.9)
 
J1      = 3.27/(3/2)*5/2  # value of J1 in meV from Bai's PRL paper

isc = instant_correlations(sys_mgcro)
dsc = dynamical_correlations(sys_mgcro; Δt = 0.05, nω = 30, ωmax = 30.)

formula_d_mgcro = intensity_formula(dsc, :perp, kT = 1.8)
formula_i_mgcro = intensity_formula(isc, :perp)#, kT = 1.8)

params = unit_resolution_binning_parameters(isc)
params.binstart[1:2] .-= 4
params.binend[1:2] .+= 3
#params.binstart[1:2] .-= 2
#params.binend[1:2] .+= 1
params.binstart[3] -= 1
params.binend[3] += 1
L_binwidth = params.binwidth[3]

params_dyn = unit_resolution_binning_parameters(dsc)
params_dyn.binstart[1:2] .-= 4
params_dyn.binend[1:2] .+= 3
#params_dyn.binstart[1:2] .-= 2
#params_dyn.binend[1:2] .+= 1
params_dyn.binstart[3] -= 1
params_dyn.binend[3] += 1

nsample = 30
sample_resolved_is_i = Array{Float64,6}(undef,params.numbins...,nsample,2)
sample_resolved_is_d = Array{Float64,6}(undef,params_dyn.numbins...,nsample,2)
for model_num = 1:2
  J_mgcro = if model_num == 1
    [1.00,0.0815,0.1050,0.0085]*J1; # PRL model
  elseif model_num == 2
    [0.794,0.00631,0.0947,0.0211]*J1; # New model
  end


  ## === Assign exchange interactions to MgCr2O4 system ===
  set_exchange!(sys_mgcro, J_mgcro[1], Bond(1, 2, [0,0,0]))  # J1
  set_exchange!(sys_mgcro, J_mgcro[2], Bond(1, 7, [0,0,0]))  # J2
  set_exchange!(sys_mgcro, J_mgcro[3], Bond(1, 3, [1,0,0]))  # J3a -- Careful here!  
  set_exchange!(sys_mgcro, J_mgcro[4], Bond(1, 3, [0,0,0])); # J3b -- And here!

  randomize_spins!(sys_mgcro);


  Δt = 0.05  # Integration time step in meV^-1
  λ  = 0.1   # Phenomenological damping parameter
  kT = 1.8   # Desired temperature in meV
  langevin = Langevin(Δt; λ, kT); # Construct integrator

  # We can now thermalize our systems by running the integrator.
  println("Thermalizing")
  for _ in 1:2000
      step!(sys_mgcro, langevin)
  end


  #isf_mgcro = instant_correlations(sys_mgcro);

  println("Correlating")
  for i in 1:nsample
      ## Run dynamics to decorrelate
      @time for _ in 1:500
          step!(sys_mgcro, langevin)
      end
      ## Add samples
      @time begin
        isc.nsamples[1] = 0
        isc.data .= 0
        add_sample!(isc, sys_mgcro)#; alg = :window)
        Sq_mgcro, counts = intensities_binned(isc, params, formula_i_mgcro);
        sample_resolved_is_i[:,:,:,:,i,model_num] .= Sq_mgcro ./ counts
      end

      @time begin
        dsc.nsamples[1] = 0
        dsc.data .= 0
        add_sample!(dsc, sys_mgcro; alg = :window)
        Sq_mgcro, counts = intensities_binned(dsc, params_dyn, formula_d_mgcro);
        sample_resolved_is_d[:,:,:,:,i,model_num] .= Sq_mgcro ./ counts
      end
  end
end

#formula_mgcro = intensity_formula(scs[1], :perp)
#Sq_mgcro, counts = intensities_binned(scs[1], params, formula_mgcro);

#formula_mgcro = intensity_formula(scs[2], :perp)
#Sq_mgcro2, counts2 = intensities_binned(scs[2], params, formula_mgcro);

Sq_mgcro_i = sum(sample_resolved_is_i[:,:,:,:,:,1],dims=[5,4]) ./ nsample
Sq_mgcro2_i = sum(sample_resolved_is_i[:,:,:,:,:,2],dims=[5,4]) ./ nsample

Sq_mgcro_d = sum(sample_resolved_is_d[:,:,:,:,:,1],dims=[5,4]) ./ nsample * dsc.Δω
Sq_mgcro2_d = sum(sample_resolved_is_d[:,:,:,:,:,2],dims=[5,4]) ./ nsample * dsc.Δω

Sq_mgcro_i_var = sum((sample_resolved_is_i[:,:,:,:,:,1] .- Sq_mgcro_i[:,:,:,:,1]) .^ 2,dims=[5,4]) ./ (nsample - 1);
Sq_mgcro2_i_var = sum((sample_resolved_is_i[:,:,:,:,:,2] .- Sq_mgcro2_i[:,:,:,:,1]) .^ 2,dims=[5,4]) ./ (nsample - 1);

Sq_mgcro_d_var = sum((sample_resolved_is_d[:,:,:,:,:,1] .- Sq_mgcro_d[:,:,:,:,1]) .^ 2,dims=[5,4]) ./ (nsample - 1) * dsc.Δω;
Sq_mgcro2_d_var = sum((sample_resolved_is_d[:,:,:,:,:,2] .- Sq_mgcro2_d[:,:,:,:,1]) .^ 2,dims=[5,4]) ./ (nsample - 1) * dsc.Δω;

bcs = axes_bincenters(params)
h = bcs[1]
k = bcs[2]
pn(x) = Sunny.number_to_simple_string(x;digits = 3)

fig = Figure(; resolution=(1500,1000))
axparams = (aspect = true, xticks=-4:4, yticks=-4:4, titlesize=20,
    xlabel = "H", ylabel = "K", xlabelsize = 18, ylabelsize=18,)

target_L = [0,1/2,1]
Lbinixs = 1 .+ floor.(Int64,(target_L .- params.binstart[3]) ./ params.binwidth[3])
for ix = 1:3
  bin_ix = Lbinixs[ix]

  ax_mgcro = Axis(fig[1,ix]; title="PRL model (T=1.8K), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_d[:,:,bin_ix])

  ax_mgcro = Axis(fig[2,ix]; title="PRL model (instant), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_i[:,:,bin_ix])

  ax_mgcro = Axis(fig[3,ix]; title="New model (instant), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro2_i[:,:,bin_ix])

  ax_mgcro = Axis(fig[4,ix]; title="New model (T=1.8K), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro2_d[:,:,bin_ix])
end


fig

var_fig = Figure(; resolution=(1500,1000))
axparams = (aspect = true, xticks=-4:4, yticks=-4:4, titlesize=20,
    xlabel = "H", ylabel = "K", xlabelsize = 18, ylabelsize=18,)

for ix = 1:3
  bin_ix = Lbinixs[ix]

  ax_mgcro = Axis(var_fig[1,ix]; title="Var PRL model (T=1.8K), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_d_var[:,:,bin_ix])

  ax_mgcro = Axis(var_fig[2,ix]; title="Var PRL model (instant), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_i_var[:,:,bin_ix])

  ax_mgcro = Axis(var_fig[3,ix]; title="Var New model (instant), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro2_i_var[:,:,bin_ix])

  ax_mgcro = Axis(var_fig[4,ix]; title="Var New model (T=1.8K), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro2_d_var[:,:,bin_ix])
end
var_fig



fig2 = Figure(); ax = Axis(fig2[1,1],xlabel = "μ", ylabel = "σ");
scatter!(ax,Sq_mgcro2_i[:],sqrt.(Sq_mgcro2_i_var[:]),color = :orange,marker = 'o')
scatter!(ax,Sq_mgcro_i[:],sqrt.(Sq_mgcro_i_var[:]),color = :blue,marker = 'o')
scatter!(ax,Sq_mgcro2_d[:],sqrt.(Sq_mgcro2_d_var[:]),color = :orange,marker = 'x')
scatter!(ax,Sq_mgcro_d[:],sqrt.(Sq_mgcro_d_var[:]),color = :blue,marker = 'x')
println("Instant mean σ/μ = $(pn(mean(sqrt.(Sq_mgcro_i_var[:]) ./ Sq_mgcro_i[:])))")
println("Dynamic mean σ/μ = $(pn(mean(sqrt.(Sq_mgcro_d_var[:]) ./ Sq_mgcro_d[:])))")
fig2

figQE = Figure(); 
bcs = axes_bincenters(params_dyn)
q = bcs[1]
e = bcs[4]
Sqw = sum(sample_resolved_is_d[:,:,:,:,:,1],dims=5) ./ nsample

ax = Axis(figQE[1,1])
heatmap!(ax,q,e,log10.(max.(0,Sqw[:,1,Lbinixs[1],:])))

ax = Axis(figQE[1,2])
heatmap!(ax,q,e,log10.(max.(0,Sqw[:,1,Lbinixs[2],:])))

ax = Axis(figQE[1,3])
heatmap!(ax,q,e,log10.(max.(0,Sqw[:,1,Lbinixs[3],:])))
#heatmap!(ax,q,e,Sqw[:,1,Lbinixs[3],:])

figQE

figSlices = Figure()
k = bcs[2]
eixs = round.(Int64,range(1.,length(e),length=8))
for i = 1:8
  eix = eixs[i]
  ax = Axis(figSlices[1,i])
  #heatmap!(ax,q,k,log10.(max.(0,Sqw[:,:,Lbinixs[1],eix])))
  heatmap!(ax,q,k,Sqw[:,:,Lbinixs[1],eix])
end

figSlices

scga_fig = Figure()
axparams = (aspect = true, xticks=-4:4, yticks=-4:4, titlesize=20,
    xlabel = "H", ylabel = "K", xlabelsize = 18, ylabelsize=18,)

params_scga = copy(params)
params_scga.binwidth[1:2] ./= 2
is_scga = scga_bincenters(params_scga,sys_mgcro,1/1.8)
Lbinixs_scga = 1 .+ floor.(Int64,(target_L .- params_scga.binstart[3]) ./ params_scga.binwidth[3])

bcs_scga = axes_bincenters(params_scga)
h_scga = bcs_scga[1]
k_scga = bcs_scga[2]

for ix = 1:3
  bin_ix = Lbinixs_scga[ix]

  ax_mgcro = Axis(scga_fig[1,ix]; title="PRL model (T=1.8K), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_d[:,:,bin_ix])

  ax_mgcro = Axis(scga_fig[2,ix]; title="PRL model (instant), L = $(pn(bcs[3][bin_ix])) ± $(pn(L_binwidth))",  axparams...)
  hm = heatmap!(ax_mgcro, h, k, Sq_mgcro_i[:,:,bin_ix])

  ax_mgcro = Axis(scga_fig[3,ix]; title="PRL model (SCGA at bincenters), L = $(pn(bcs[3][bin_ix]))",  axparams...)
  hm = heatmap!(ax_mgcro, h_scga, k_scga, is_scga[:,:,bin_ix])
end

scga_fig





function time_traj_stats(dsc)
  pn(x) = Sunny.number_to_simple_string(x;digits = 6)
  dt = dsc.Δt
  nsamps = size(dsc.samplebuf,6)
  nskip = dsc.measperiod
  T = dt * nsamps * nskip
  println("Settings (nω = $nsamps, ωmax = $(pn(dsc.Δω * nsamps))) so Δω = $(pn(dsc.Δω))")
  println()
  println("In the time-domain, the trajectories are:")
  println("  length T = $(pn(T)) = $nsamps * $(pn(dt * nskip)) = $(nsamps * nskip) * $(pn(dt))")
  println("  integration resolution = $(pn(dt)) = $(pn(T / nsamps)) / $nskip")
end
