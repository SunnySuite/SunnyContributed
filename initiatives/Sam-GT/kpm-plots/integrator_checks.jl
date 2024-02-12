cryst = Crystal(I(3), [[0,0,0]], 1)
sys = System(cryst, (1,1,1), [SpinInfo(1; S=3/2, g=2)], :dipole, units = Units.theory)
set_external_field!(sys,[0,0,1])

set_dipole!(sys,[2,0,1],(1,1,1,1))
s0 = sys.dipoles[1]

function do_sim(dt)
  impl_midpoint = ImplicitMidpoint(dt)

  langevin = Langevin(dt,λ = 0.1,kT = 0.0)
  langevin.λ = 0.0 # Hack in a λ=0 Langevin integrator

  ts = range(0,length=1000,step=dt)

  c = 1
  traj_exact = zeros(3,1000)
  @assert iszero(s0[2])
  traj_exact[1,:] .= s0[1] * cos.(sys.units.μB * sys.gs[1][1] * ts * c)
  traj_exact[2,:] .= s0[1] * sin.(sys.units.μB * sys.gs[1][1] * ts * c)
  traj_exact[3,:] .= s0[3]

  set_dipole!(sys,s0,(1,1,1,1))
  traj = zeros(3,1000)
  traj[:,1] .= sys.dipoles[1]
  for i = 2:1000
    step!(sys,langevin)
    traj[:,i] .= sys.dipoles[1]
  end

  c = atan(traj[2,2],traj[1,2]) / (2 * dt) # 1.0016616488792511 for dt = 0.05
  println(c)
  traj_exact_fast = zeros(3,1000)
  @assert iszero(s0[2])
  traj_exact_fast[1,:] .= s0[1] * cos.(sys.units.μB * sys.gs[1][1] * ts * c)
  traj_exact_fast[2,:] .= s0[1] * sin.(sys.units.μB * sys.gs[1][1] * ts * c)
  traj_exact_fast[3,:] .= s0[3]

  set_dipole!(sys,s0,(1,1,1,1))
  traj_renorm = zeros(3,1000)
  traj_renorm[:,1] .= sys.dipoles[1]
  for i = 2:1000
    step!(sys,langevin; extra_normalize = true)
    traj_renorm[:,i] .= sys.dipoles[1]
  end

  set_dipole!(sys,s0,(1,1,1,1))
  traj_impl = zeros(3,1000)
  traj_impl[:,1] .= sys.dipoles[1]
  for i = 2:1000
    step!(sys,impl_midpoint)
    traj_impl[:,i] .= sys.dipoles[1]
  end

  set_dipole!(sys,s0,(1,1,1,1))
  traj_renorm_impl = zeros(3,1000)
  traj_renorm_impl[:,1] .= sys.dipoles[1]
  for i = 2:1000
    step!(sys,impl_midpoint)#; extra_normalize = true)
    traj_renorm_impl[:,i] .= sys.dipoles[1]
  end

  traj_exact, traj_exact_fast, traj, traj_renorm, traj_impl, traj_renorm_impl
end

function convergents(dt)
  traj_exact, traj_exact_fast, traj, traj_renorm, traj_impl, traj_renorm_impl = do_sim(dt)
  trajs = [traj_exact, traj_exact_fast, traj, traj_renorm, traj_impl, traj_renorm_impl]

  #[norm(trajs[i][:,end] .- trajs[j][:,end]) for i = 1:5, j = 1:5]
  #[norm(trajs[i][:,2] .- trajs[j][:,2]) for i = 1:5, j = 1:5]
  [norm(trajs[i][:,300] .- trajs[j][:,300]) for i = 1:6, j = 1:6]
end

dt_range = 10 .^ range(-5,-2,length = 50)
convs = [convergents(dt)[1,2:end] for dt = dt_range]

ldt = log10.(dt_range)
lerr = log10.(map(x->x[1],convs))

f = Figure()
ax = Axis(f[1,1],xticks = -5:-2,yticks = -12:-2)
p1 = plot!(ax,ldt,log10.(map(x->x[1],convs)),marker = 'o')
p2 = plot!(ax,ldt,log10.(map(x->x[2],convs)),marker = 'x')
p3 = plot!(ax,ldt,log10.(map(x->x[3],convs)))
p4 = plot!(ax,ldt,log10.(map(x->x[4],convs)))
#p5 = plot!(ax,ldt,log10.(map(x->x[5],convs)))
Legend(f[1,2],[p1,p2,p3,p4],["Exact (speed boost)", "Langevin (main)", "Langevin tangent_map", "ImplicitMidpoint (main)"])
#Legend(f[1,2],[p1,p2,p3,p4,p5],["Exact (speed boost)", "Langevin (main)", "Langevin tangent_map", "ImplicitMidpoint (main)", "ImplicitMidpoint tangent_map"])
f


