using Sunny, GLMakie, LinearAlgebra 

a =  10.6172 
b =  9.0777
c = 2.9661     
latvecs = lattice_vectors(a, b, c, 90, 90, 90) 
positions = [[0.6594, 0.7597,1/4], [0.6127, 0.4401, 1/4], [0.1005, 0.4165,1/4],
[0.1603, 0.2027,1/4],[0.4752,0.1172,1/4],[0.7852, 0.5270,1/4],[0.4273, 0.4174,1/4]] 
types = ["Ca", "Cr", "Cr","O","O","O","O"]
CaCrO = Crystal(latvecs, positions,62; types, setting = "cab")
cryst = subcrystal(CaCrO,"Cr")
dims = (1,1,37)
spinfos = [1 => Moment(s=3/2, g=2),5 => Moment(s=3/2, g=2)]
sys = System(cryst,spinfos, :dipole;dims)
Dyanis=0.2
Day=0.5
DMI = dmvec([0,Day,0])
J2=I(3)*8
J1 = I(3)*1.
J22 = J2+DMI 
J21 = J2+DMI
J11 = J1
J12 = J1
Jb = 0.0
Ja = -4*J1

set_exchange!(sys, J22, Bond(1, 1, [0, 0, 1]))
set_exchange!(sys, J21, Bond(5, 5, [0, 0, 1]))
set_exchange!(sys, J11, Bond(6, 7, [0, 1, 0]))
set_exchange!(sys, J12, Bond(1, 4, [0, 0, 0]))
set_exchange!(sys, Jb, Bond(3, 5, [0, 0, 0]))
set_exchange!(sys, Ja,Bond(4, 5, [0, 0, 0]))
set_onsite_coupling!(sys, S -> Dyanis*S[2]^2, 1)
set_onsite_coupling!(sys, S -> Dyanis*S[2]^2, 5)
randomize_spins!(sys)
minimize_energy!(sys)
print_wrapped_intensities(sys)
plot_spins(sys; color =[ S[3] for S in sys.dipoles ])
print_wrapped_intensities(sys)

# compare to analytical 
k=2acos(-norm(Ja)/(4norm(diag(J2))))
krlu = k/2π
1-krlu
rationalize(krlu;tol=0.001)
k=2acos(-norm(Ja)/(4norm(diag(J2))))
krlu = k/2π
1-krlu
kexp=rationalize(1-krlu;tol=0.001) # k_exp in 1BZ
rationalize(krlu;tol=0.001) 
kexp
round(Float64(kexp),digits=4)  === 0.4595

measure = ssf_perp(sys; )
swt = SpinWaveTheoryKPM(sys; measure,tol=0.01)

Γ = [0,0,0]
R = [1/2,1/2,1/2]
S = [1/2,1/2,0]
Tp = [0,1/2,1/2]
U = [1/2,0,1/2]
X = [1/2,0,0]
Y = [0,1/2,0]
Z = [0,0,1/2]
offset = [0.0,0.0,1.0]
nqs = 100 

q_points = [Γ,Z,U,X,Γ]
q_points_shift = [point + offset for point in q_points]
path = q_space_path(cryst, q_points_shift, 150)
function br(ω, x, σ)
    return (1/π) * (σ / ((x - ω)^2 + σ^2))
end
Emax = 35
σin= 0.025*Emax
kernel = lorentzian(fwhm=σin)

@time begin
    energies = range(0,Emax,150) 
    res = intensities(swt, path; energies, kernel)
    plot_intensities(res)
end
