# This script illustrates the prototype eigenmode viewer by Sam Quinn. Used in
# conjunction with a basic spin-1/2 ferromagnet.

using Sunny, GLMakie

cryst = Sunny.square_crystal()
sys = System(cryst, [1 => Moment(s=1/2, g=2)], :SUN)
J = -1.0
set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
set_field!(sys, [0, 0, 0.01])
randomize_spins!(sys)
minimize_energy!(sys)

swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
q_points = [[0, 0, 0], [1, 1, 0]]
qpath = q_space_path(cryst, q_points, 400)

include("eigenmode_viewer.jl")
interact_eigenmodes(swt, qpath; super_size=(12, 12, 1), ndims=2)
