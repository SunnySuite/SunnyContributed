using Sunny, LinearAlgebra

# N.B. 9/2/24: Probably not currently working. Intention was to include
# the static structure factor (aka Bragg peaks) into the Spin Wave caclculation.
# This is done better and more generally in the "higher spin wave theory" code.
function include_longitudinal_fluctuations(swt,q_res,formula)
  qrx,qry,qrz = q_res

  L = Sunny.num_bands(swt)
  Vmats = zeros(ComplexF64,2L,2L,qrx,qry,qrz)

  for ci = CartesianIndices((qrx,qry,qrz))
    ix,iy,iz = ci.I

    qx = ix / qrx
    qy = iy / qry
    qz = iz / qrz

    band_structure = formula.calc_intensity(swt,Sunny.Vec3([qx,qy,qz]))
    Vmats[:,:,ix,iy,iz] .= formula.calc_intensity.Vmat
  end


  Nm, Ns = length(swt.sys.dipoles), swt.sys.Ns[1]
  Nm_inv = 1.0 / Nm
  No = Sunny.num_observables(swt.observables)

  big_M = 1 # TODO: verify

  bases = swt.data.local_quantization_basis
  obs = swt.data.observable_operators
  δT = zeros(ComplexF64,Nm,No,qrx,qry,qrz)
  for ci = CartesianIndices(q_res)
    ix,iy,iz = ci.I
    qx = ix / qrx
    qy = iy / qry
    qz = iz / qrz
    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, [qx,qy,qz])
    for atom = 1:Nm
      phase = exp(-2π*im * dot(q_reshaped, swt.sys.crystal.positions[atom]))
      for μ = 1:No
        obs_mat = obs[:,:,μ,atom]
        ground_part = obs_mat[1,1]
        for k_ci = CartesianIndices(q_res)
          kix,kiy,kiz = ci.I
          # b at k
          b_at_k = Vmats[:,1:L,kix,kiy,kiz]
          # b† at (q-k)
          bd_at_qmk = Vmats[:,L .+ (1:L),mod1(ix-kix,qrx),mod1(iy-kiy,qry),mod1(iz-kiz,qrz)]
          for α = 2:Ns, β = 2:Ns
            factor = obs_mat[α,β]
            if α == β # Contribution from S - ∑ n
              factor = factor - ground_part
            end
            δT[atom,μ,ix,iy,iz] += factor * conj(bd_at_qmk[α]) * b_at_k[β]
          end
                            #Avec[μ] += Avec_pref[i] * (Obs[α, 1] * v[(i-1)*(Ns-1)+α-1+nmodes] + Obs[1, α] * v[(i-1)*(Ns-1)+α-1])
          #k_reshaped = Sunny.to_reshaped_rlu(swt.sys, [kx,ky,kz])
        end
      end
    end
  end

  Vmats,δT
end
