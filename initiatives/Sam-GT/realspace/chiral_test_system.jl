using Sunny

cryst = Crystal(I(3),[[0,0,0],[1/3,0,0],[1/2,0,0]],1)

sys = System(cryst,(1,1,1),[SpinInfo(1,S=1,g=2),SpinInfo(2,S=1,g=2),SpinInfo(3,S=1,g=2)],:dipole)

for i = 1:3, j = 1:3, l = 0:1
  if i == j && l == 0
    continue
  end
  set_exchange!(sys,randn(3,3),Bond(i,j,[l,0,0]))
end

#set_exchange!(sys,[0 1 0; 0 0 0; 0 0 0],Bond(1,2,[0,0,0]))

randomize_spins!(sys)
minimize_energy!(sys;maxiters = 3000)
swt = SpinWaveTheory(sys)
formula = intensity_formula(swt, :full; kernel=delta_function_kernel)
q0 = [0.415,0.56,0.76]
#q0 = 0 * q0
hh = formula.calc_intensity.H
vv = formula.calc_intensity.V

Sunny.swt_hamiltonian_dipole!(hh,swt,Sunny.Vec3(q0))
h_plus = copy(hh)
disp_plus = Sunny.bogoliubov!(vv,hh)
v_plus = copy(vv)

Sunny.swt_hamiltonian_dipole!(hh,swt,Sunny.Vec3(-q0))
h_minus = copy(hh)
disp_minus = Sunny.bogoliubov!(vv,hh)
v_minus = copy(vv)

It = diagm([repeat([1],3);repeat([-1],3)])

f = Figure()
display(f)
ax = Axis(f[1,1])
scatter!(ax,eigvals(It,h_plus))
scatter!(ax,eigvals(It,h_minus))

eigs_plus = eigvals(It,h_plus,sortby = x -> -1/x)
eigs_minus = eigvals(It,h_minus,sortby = x -> -1/x)

ax = Axis(f[1,2])
heatmap!(ax,abs.(vv))

Xmat = [0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]

v_band_plus = v_plus[:,1]
v_band_minus = conj.(Xmat * v_plus[:,1])
#v_band_minus /= angle(v_band_minus[end])
phases = [exp(-2π*im * dot(q0, cryst.positions[i])) for i = 1:3]
operator_to_disp = [1 1; -im im; 0 0]/2

#vp = (phases .* reshape(v_band_plus,3,2)) * transpose(operator_to_disp)
#vm = (phases .* reshape(v_band_minus,3,2)) * transpose(operator_to_disp)

#vp = (reshape(v_band_plus,3,2)) * transpose(operator_to_disp)
#vm = (reshape(v_band_minus,3,2)) * transpose(operator_to_disp)

displacements = zeros(ComplexF64,6,3,3)
displacementsalt = zeros(ComplexF64,6,3,3)

for atom = 1:3, mode = 1:6
  v_band_plus = v_plus[:,mode]
  v_band_minus = conj.(Xmat * v_plus[:,mode])
  displacements[mode,atom,:] = swt.data.local_rotations[atom] * operator_to_disp * reshape(v_band_plus,3,2)[atom,:]/2
  displacements[mode,atom,:] += swt.data.local_rotations[atom] * operator_to_disp * reshape(v_band_minus,3,2)[atom,:]/2
  displacementsalt[mode,atom,:] = swt.data.local_rotations[atom] * operator_to_disp * reshape(v_band_plus,3,2)[atom,:]/(2 * im)
  displacementsalt[mode,atom,:] -= swt.data.local_rotations[atom] * operator_to_disp * reshape(v_band_minus,3,2)[atom,:]/(2 *im)
end

