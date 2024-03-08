using Sunny, GLMakie, LinearAlgebra, StatsBase, StaticArrays

darboux_mode = false

if darboux_mode
  # Darboux
  bes_phi = range(-π,π,length = 60)
  bes_z = range(-1,-0.6,length = 70)
  bcs_phi = (bes_phi[2:end] + bes_phi[1:end-1])/2
  bcs_z = (bes_z[2:end] + bes_z[1:end-1])/2
else
  # Sterographic
  bes = range(-0.8,0.8,length = 60)
  bcs = (bes[2:end] + bes[1:end-1])/2
end


if darboux_mode
  # Darboux coordinates
  st_to_phi(st) = atan.(st[:,2],st[:,1])
  st_to_z(st) = st[:,3]
  make_hist(states) = fit(Histogram,(st_to_phi(states),st_to_z(states)),(bes_phi,bes_z)).weights
else
  # Stereographic projection of the spin-1 sphere
  make_hist(states) = fit(Histogram,(states[:,1] ./ (1 .- states[:,3]), states[:,2] ./ (1 .- states[:,3])),(bes,bes)).weights
end

# Take data
cryst = Crystal(I(3), [[0,0,0]], 1)
sys = System(cryst,(1,1,1),[SpinInfo(1,S=1,g=2)],:dipole)
set_external_field!(sys,[0,0,-1])
langevin = Langevin(0.01; λ = 0.1, kT = 0.005)
ntraj = 20000
states = zeros(ntraj,3)
gradients = zeros(ntraj,3)
run_hist = 0 * make_hist(states)
for k = 1:200; for j = 1:ntraj; step!(sys,langevin); states[j,:] .= sys.dipoles[1]; gradients[j,:] .= cross(cross(states[j,:],Sunny.energy_grad_dipoles(sys)[1]),states[j,:]); end; run_hist .+= make_hist(states); end;


if darboux_mode
  # Darboux
  dphi = bcs_phi[2] - bcs_phi[1]
  dphim = [abs(i - j) == 1 ? (j-i)/(2dphi) : 0. for i = 1:length(bcs_phi), j = 1:length(bcs_phi)]
  dphim[1,length(bcs)] = dphim[2,1]
  dphim[length(bcs),1] = dphim[1,2]

  dz = bcs_z[2] - bcs_z[1]
  dzm = [abs(i - j) == 1 ? (j-i)/(2dz) : 0. for i = 1:length(bcs_z), j = 1:length(bcs_z)]
else
  # Stereo
  dx = bcs[2] - bcs[1]
  dxm = [abs(i - j) == 1 ? (j-i)/(2dx) : 0. for i = 1:length(bcs), j = 1:length(bcs)]
  # Would make it periodic, which we aren't
  #dxm[1,length(bcs)] = dxm[2,1]
  #dxm[length(bcs),1] = dxm[1,2]
end



# Apply rotating (applied field) convection and diffusion
dt = 1e-2
diff_coeff = 1e-3
ω = 0.115768

if darboux_mode
  # Darboux
  do_step(rho) = rho + dt * (-dphim * ω * rho + diff_coeff * (dphim * dphim * rho + rho * dzm' * dzm'))
else
  # Stereo
  do_step(rho) = rho + dt * (-dxm * ω * (rho * diagm(bcs)) - ω * ((-diagm(bcs) * rho) * dxm') + diff_coeff * (dxm * dxm * rho + rho * dxm' * dxm'))
end

if darboux_mode
  # Darboux
  response_operators = [x -> - dphim * x, x -> -x * dzm']
  readout_operators = [x -> diagm(bcs_phi) * x, x -> x * diagm(bcs_z)]
else
  # Stereo
  response_operators = [x -> - dxm * x, x -> -x * dxm']
  readout_operators = [x -> diagm(bcs) * x, x -> x * diagm(bcs)]
end

f = Figure()
display(f)

rec_vals_corr = Array{Vector{Float64}}(undef,2,2)
rec_vals_resp = Array{Vector{Float64}}(undef,2,2)
n_timestep = 10000
for i = 1:2, j = 1:2

  # Compute a correlation: x ---t--> y
  rho_cur = readout_operators[j](run_hist);
  rec_vals_corr[i,j] = zeros(n_timestep)
  for k = 1:n_timestep;
    rho_cur = do_step(rho_cur);
    rec_vals_corr[i,j][k] = sum(readout_operators[i](rho_cur))/sum(run_hist);
  end;

  # Compute a response function: x ---t--> y
  rho_cur = response_operators[j](run_hist);
  rec_vals_resp[i,j] = zeros(n_timestep)
  for k = 1:n_timestep;
    rho_cur = do_step(rho_cur);
    rec_vals_resp[i,j][k] = sum(readout_operators[i](rho_cur))/sum(run_hist);
  end;

end


j0 = jacobian_dipoles(sys)[1] # Our system happens to have constant j0
jexps = [exp(dt * n * j0) for n = 0:(n_timestep - 1)]

dtm = [abs(i - j) == 1 ? (j-i)/(2dt) : 0. for i = 1:n_timestep, j = 1:n_timestep]
dtm[n_timestep,:] .= 0
dtm[1,:] .= 0
for i = 1:2, j = 1:2
  ax = Axis(f[i,j],title = "$j --t--> $i")
  #plot!(ax,rec_vals_corr[i,j])
  plot!(ax,rec_vals_resp[i,j])

  # FDT
  l = 1 + (1-(i-1))
  sig = i == 1 ? -1 : 1

  # Confusing factors:
  lines!(ax,dtm * (sig * rec_vals_corr[l,j])/(dt * ω), linestyle = :dash, color = :black)

  lines!(ax,map(x -> x[i,j],jexps), linestyle = :dash, color = :red)
end

#for j = 1:1000; rho_cur = do_step(rho_cur); end; heatmap(bcs,bcs,rho_cur)

#@. Δs = -s × (- Δt*∇E)

# Δsi[c] = si[a] × diE[b]
# Jac[i,j][c,:] = djΔsi[c] = djsi[a] × diE[b] + si[a] × djdiE[b]

function trajectory_jacobians(states)
  jac_traj = zeros(typeof(jacobian_dipoles(sys)[1]),size(states,1))
  jac_prod = I(3)
  for j = 1:size(states,1)
    sys.dipoles[1] = states[j,:]
    jac_traj[j] = jacobian_dipoles(sys)[1]
    jac_prod = (I(3) + dt * jac_traj[j]) * jac_prod
  end
  jac_prod
end

function jacobian_dipoles(sys)
  JacdE = energy_jacobian_dipoles(sys)
  #display(JacdE)
  dE = Sunny.energy_grad_dipoles(sys)
  #display(dE)
  na = Sunny.natoms(sys.crystal)
  Jac = zeros(MMatrix{3,3,Float64,9},sys.latsize...,na,sys.latsize...,na)
  for siteI = eachsite(sys), siteJ = eachsite(sys)
    Jac[siteI,siteJ] = zero(MMatrix{3,3,Float64,9})
  end
  for i = 1:3, siteI = eachsite(sys)
    a = [1,2,3][i]
    b = [2,3,1][i]
    c = [3,1,2][i]
    for siteJ = eachsite(sys)
      # s x ddE
      #
      Jac[siteI,siteJ][c,:] .+= JacdE[siteJ,siteI][b,:] * sys.dipoles[siteI][a]
      Jac[siteI,siteJ][c,:] .+= -JacdE[siteJ,siteI][a,:] * sys.dipoles[siteI][b]
    end
    # ds x dE
    Jac[siteI,siteI][c,a] += dE[siteI][b]
    Jac[siteI,siteI][c,b] += -dE[siteI][a]
    #display(siteI)
    #display(i)
    #display(Jac[1:4])
    #display(Jac[siteI,siteI])
  end
  Jac
end

function to_block_jac(jj)
  jjflat = reinterpret(reshape,Float64,SMatrix.(jj))
  nsite = prod(size(jj)[2:5])
  reshape(permutedims(reshape(jjflat,3,3,nsite,nsite),(3,1,4,2)), 3*nsite,3*nsite)
end

function energy_jacobian_dipoles(sys)
  na = Sunny.natoms(sys.crystal)
  Jac = zeros(Sunny.Mat3,sys.latsize...,na,sys.latsize...,na)
  set_energy_jacobian_dipoles!(Jac,sys.dipoles,sys)
  Jac
end

function set_energy_jacobian_dipoles!(J, dipoles::Array{Sunny.Vec3, 4}, sys::System{N}) where N

    # Anisotropies and exchange interactions
    for i in 1:Sunny.natoms(sys.crystal)
        if Sunny.is_homogeneous(sys)
            # Interactions for sublattice i (same for every cell)
            interactions = sys.interactions_union[i]
            set_energy_jacobian_dipoles_aux!(J, dipoles, interactions, sys, i, Sunny.eachcell(sys))
        else
            for cell in Sunny.eachcell(sys)
                # Interactions for sublattice i and a specific cell
                interactions = sys.interactions_union[cell, i]
                set_energy_jacobian_dipoles_aux!(J, dipoles, interactions, sys, i, (cell,))
            end
        end
    end

    if !isnothing(sys.ewald)
        error("Ewald jacobian not implemented")
        accum_ewald_grad!(∇E, dipoles, sys)
    end
end

function set_energy_jacobian_dipoles_aux!(Jac, dipoles::Array{Sunny.Vec3, 4}, ints::Sunny.Interactions, sys::System{N}, i::Int, cells) where N
    # Single-ion anisotropy only contributes in dipole mode. In SU(N) mode, the
    # anisotropy matrix will be incorporated directly into local H matrix.
    if sys.mode in (:dipole, :dipole_large_S)
        stvexp = ints.onsite :: Sunny.StevensExpansion
        for cell in cells
            s = dipoles[cell, i]
            #J[cell, i] += energy_and_gradient_for_classical_anisotropy(s, stvexp)[2]
        end
    end

    for pc in ints.pair
        (; bond, isculled) = pc
        isculled && break

        for cellᵢ in cells
            cellⱼ = Sunny.offsetc(cellᵢ, bond.n, sys.latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]

            # Bilinear
            J = pc.bilin
            Jac[cellᵢ, bond.i, cellⱼ, bond.j] += J
            Jac[cellⱼ, bond.j, cellᵢ, bond.i] += J'

            # Biquadratic for dipole mode only (SU(N) handled differently)
            if sys.mode in (:dipole, :dipole_large_S)
              error("Biquadratic jacobian not implemented")
                if !iszero(pc.biquad)
                    Qᵢ = quadrupole(sᵢ)
                    Qⱼ = quadrupole(sⱼ)
                    ∇Qᵢ = grad_quadrupole(sᵢ)
                    ∇Qⱼ = grad_quadrupole(sⱼ)

                    # In matrix case, energy is `Qᵢ' * biquad * Qⱼ`, and we are
                    # taking gradient with respect to either sᵢ or sⱼ.
                    if pc.biquad isa Float64
                        J = pc.biquad::Float64
                        ∇E[cellᵢ, bond.i] += J * (Qⱼ .* scalar_biquad_metric)' * ∇Qᵢ
                        ∇E[cellⱼ, bond.j] += J * (Qᵢ .* scalar_biquad_metric)' * ∇Qⱼ
                    else
                        J = pc.biquad::Mat5
                        ∇E[cellᵢ, bond.i] += (Qⱼ' * J') * ∇Qᵢ
                        ∇E[cellⱼ, bond.j] += (Qᵢ' * J)  * ∇Qⱼ
                    end
                end
            end
        end
    end
end


function energy_gradient_and_jacobian_for_classical_anisotropy(s::Sunny.Vec3, stvexp::Sunny.StevensExpansion)
    (; kmax, c0, c2, c4, c6) = stvexp

    E      = only(c0)
    dE_dz  = 0.0
    dE_dJp = 0.0 + 0.0im

    kmax == 0 && @goto exit
    error("Quadratic anisotropy jacobian not implemented!")

    # Quadratic contributions

    X = s⋅s
    Jp¹ = s[1] + im*s[2]
    Jz¹ = s[3]
    Jp² = Jp¹*Jp¹
    Jz² = Jz¹*Jz¹

    A = (3Jz²-X, Jz¹, 1)
    dA_dz = (6Jz¹, 1)
    E +=        (c2[1]*real(Jp²)+c2[5]*imag(Jp²))A[3] +
                (c2[2]*real(Jp¹)+c2[4]*imag(Jp¹))A[2] +
                c2[3]*A[1]
    dE_dz +=    (c2[2]*real(Jp¹)+c2[4]*imag(Jp¹))dA_dz[2] +
                c2[3]*dA_dz[1]
    dE_dJp +=   (2/2)*(c2[1]*Jp¹-im*c2[5]*Jp¹)A[3] +
                (1/2)*(c2[2]    -im*c2[4]    )A[2]

    kmax == 2 && @goto exit
    error("Quartic and above contributions not implemented!")

    # Unpack gradient components

    @label exit
    dE_dx = +2real(dE_dJp)
    dE_dy = -2imag(dE_dJp)
    return (E, Sunny.Vec3(dE_dx, dE_dy, dE_dz))
end
