using Sunny, LinearAlgebra, GLMakie, Optim

# Load experiment data (so we have histogram_parameters)
histogram_parameters, data = load_nxs("experiment_data_normalized.nxs")

# Crystallography & Chemistry
cryst = Crystal("example_cif.cif"; symprec=1e-4)
sys_chemical = System(subcrystal(cryst,"Cr"), (1,1,1), [SpinInfo(1,S=3/2,g=2)], :SUN)
sys = reshape_supercell(sys_chemical, [1 1 0; 1 -1 0; 0 0 1]) # Neel state

plot_spins(sys)

function get_Z(is)
  sum(is[7:14,7:14,:,4]) # Qx, Qy, and E range of magnetic bragg peak
end

# Multi-sampling magic numbers
msaa4 = [[0.625, 0.625, 0.125]
        ,[0.875, 0.125, 0.375]
        ,[0.375, 0.375, 0.875]
        ,[0.125, 0.875, 0.625]]
energy_multisample = [(n + 0.5)/5 for n = 1:5]

# Known J2
J2 = 0.16
set_exchange!(sys,J2,Bond(1,1,[1,1,0]))

# Unknown J1, A
function forward_problem(J1,A)
  set_exchange!(sys,J1,Bond(1,1,[1,0,0]))
  
  Sz = spin_operators(sys,1)[3]
  set_onsite_coupling!(sys,A*Sz^2,1)

  # Standard calculation:
  randomize_spins!(sys)
  minimize_energy!(sys)

  swt = SpinWaveTheory(sys)
  
  formula = intensity_formula(swt,:perp;
    # TODO: instrument-adapted broadening
    kernel = lorentzian(2.)
    ,mode_fast = true
    ,formfactors = [FormFactor("Cr3")]
    )

  
  intensity, counts = Sunny.intensities_bin_multisample(swt
                                                       ,histogram_parameters
                                                       ,msaa4
                                                       ,energy_multisample
                                                       ,formula)
  return intensity ./ counts
end

function loss_function(experiment_data,simulation_data)
  Z_experiment = get_Z(experiment_data)
  normalized_exp_data = experiment_data ./ Z_experiment

  Z_sunny = get_Z(simulation_data)
  normalized_sim_data = simulation_data ./ Z_sunny

  weights = 1.

  # Compute squared error over every histogram bin
  squared_errors = (normalized_exp_data .- normalized_sim_data).^2
  squared_errors[isnan.(squared_errors)] .= 0 # Filter out missing experiment data
  squared_errors[:,:,:,1:2] .= 0 # Filter out elastic line
  sqrt(sum(weights .* squared_errors))
end

function get_loss(parameters)
  J1,A = parameters
  simulation_data = forward_problem(J1,A)
  return loss_function(data, simulation_data)
end

# Parameter sweep to generate loss landscape
do_sweep = false
Js, As, loss_landscape = if !do_sweep
  include("pre_render_landscape.jl")
  Js, As, loss_landscape
else
  nJ = 30
  nA = 30
  loss_landscape = zeros(Float64,nJ,nA)
  Js = range(9.6,10.6,length=nJ)
  As = range(0.06,0.09,length=nA)
  for (ij,J) in enumerate(Js)
    for (id,A) in enumerate(As)
      @time loss_landscape[ij,id] = get_loss([J,A])
    end
  end
  Js, As, loss_landscape
end

println("Got landscape")

fig = Figure()
ax = Axis(fig[1,1],xlabel = "J [meV]", ylabel = "A [meV]")
heatmap!(ax,Js,As,loss_landscape)

# Example use of Optim.jl to find minimum iteratively
#=
x0 = [10.6,0.075]
opt_result = optimize(get_loss
                     ,x0
                     ,method=GradientDescent(alphaguess=1e-3)
                     ,store_trace=true
                     ,extended_trace = true
                     ,time_limit=120.)

lines!(ax,Point2f.(Optim.x_trace(opt_result)))
scatter!(ax,-1,10)
=#
display(fig)

# Plot plausible model result
is = forward_problem(10.1,0.075)
is ./= get_Z(is)
is[isnan.(data)] .= NaN

fig = Figure()
heatmap!(Axis(fig[1,1]),sum(is[12:14,:,1,3:end],dims=1)[1,:,:])
heatmap!(Axis(fig[1,2]),sum(data[12:14,:,1,3:end],dims=1)[1,:,:] ./ get_Z(data))
display(fig)
