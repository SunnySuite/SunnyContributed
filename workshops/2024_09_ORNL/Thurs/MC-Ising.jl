using Sunny, GLMakie

latvecs = lattice_vectors(1, 1, 10, 90, 90, 90)
crystal = Crystal(latvecs, [[0, 0, 0]])

L = 128
sys = System(crystal, [1 => Moment(s=1, g=-1)], :dipole; dims=(L, L, 1), seed=0)
polarize_spins!(sys, (0, 0, 1))

set_exchange!(sys, -1.0, Bond(1, 1, (1, 0, 0)))

B = 0
set_field!(sys, (0, 0, B))

Tc = 2/log(1+√2)

obs = Observable(zeros(L, L))
heatmap(obs; colorrange=(-1,1))

sampler = LocalSampler(kT=Tc, propose=propose_flip)

for i in 1:400
    for i in 1:10
        step!(sys, sampler)
    end

    obs[] = reshape([S[3] for S in sys.dipoles], (L, L))
    sleep(1/60)
end
