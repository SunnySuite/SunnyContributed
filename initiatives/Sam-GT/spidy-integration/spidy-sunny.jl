using SpiDy, Sunny, LinearAlgebra

mutable struct ColoredNoiseLangevin{N}
  dt::Float64
  noisebuffer
  noise_ix::Int64
  bath_v
  bath_w
  Jsd
  bath_coupling::Matrix
end


function shared_bath_cnl(sys,dt,Jsd,noise,matrix;buffer_size = 100000)
  shared_bath_coupling = zeros(Float64,length(sys.dipoles),1)
  ColoredNoiseLangevin(sys,dt,Jsd,noise,shared_bath_coupling,matrix;num_bath = 1,buffer_size)
end

function ColoredNoiseLangevin(sys::System{0},dt,Jsd,noise,coupling,matrix;buffer_size,num_bath)
  baths = zeros(Float64,num_bath,3,buffer_size)
  for b = 1:num_bath
    for s = 1:3
      baths[b,s,:] .= bfield(buffer_size,dt,Jsd[b],noise;interpolation = false)
    end
    for j = 1:buffer_size
      baths[b,:,j] .= matrix.C * baths[b,:,j]
    end
  end
  bath_v = zeros(Float64,num_bath,3)
  bath_w = zeros(Float64,num_bath,3)
  ColoredNoiseLangevin{0}(dt,baths,1,bath_v,bath_w,Jsd,coupling)
end

@inline function rhs_dipole_no_damp!(Δs, s, ξ, ∇E, integrator)
  (; dt) = integrator
  @. Δs = - s × (ξ + dt*∇E)
end

function step!(sys::System{0}, integrator::ColoredNoiseLangevin{0})
    (s′, Δs₁, Δs₂, ξ, ∇E) = Sunny.get_dipole_buffers(sys, 5)
    s = sys.dipoles

    for i = eachindex(ξ), j = 1:size(integrator.noisebuffer,1)
      ξ[i] += integrator.bath_coupling[i,j] * integrator.noisebuffer[j,:,integrator.noise_ix]
      += bath_v
    end
    integrator.noise_ix += 1

    # Euler prediction step
    Sunny.set_energy_grad_dipoles!(∇E, s, sys)
    rhs_dipole_no_damp!(Δs₁, s, ξ, ∇E, integrator)
    @. s′ = Sunny.normalize_dipole(s + Δs₁, sys.κs)

    # Correction step
    Sunny.set_energy_grad_dipoles!(∇E, s′, sys)
    rhs_dipole_no_damp!(Δs₂, s′, ξ, ∇E, integrator)
    @. s = Sunny.normalize_dipole(s + (Δs₁+Δs₂)/2, sys.κs)

    # Simulate bath variables
    for j = 1:size(integrator.noisebuffer,1)
      spec = integrator.Jsd[j]
      # Leapfrog!
      # First, compute half-way value of w:
      dw = - (spec.ω0^2)*bath_v[j,:] - spec.Γ*bath_w[j,:]
      for i = eachindex(s), j = 1:size(integrator.noisebuffer,1)
        dw += -spec.α * integrator.bath_coupling[i,j] * s[i]
      end
      w_halfway = bath_w[j,:] + (integrator.dt/2) * dw

      # Second, update v with the halfway value:
      bath_v[j,:] .+= integrator.dt * w_halfway

      # Lastly, finish updating w, using the new v:
      dw = - (spec.ω0^2)*bath_v[j,:] - spec.Γ*w_halfway[j,:]
      for i = eachindex(s), j = 1:size(integrator.noisebuffer,1)
        dw += -spec.α * integrator.bath_coupling[i,j] * s[i]
      end
      bath_w[j,:] = w_halfway + (integrator.dt/2) * dw
    end
    return
end

function step!(sys::System{N}, integrator::ColoredNoiseLangevin) where N
    (Z′, ΔZ₁, ΔZ₂, ζ, HZ) = get_coherent_buffers(sys, 5)
    Z = sys.coherents

    fill_noise!(sys.rng, ζ, integrator)

    # Euler prediction step
    set_energy_grad_coherents!(HZ, Z, sys)
    rhs_sun!(ΔZ₁, Z, ζ, HZ, integrator)
    @. Z′ = normalize_ket(Z + ΔZ₁, sys.κs)

    # Correction step
    set_energy_grad_coherents!(HZ, Z′, sys)
    rhs_sun!(ΔZ₂, Z′, ζ, HZ, integrator)
    @. Z = normalize_ket(Z + (ΔZ₁+ΔZ₂)/2, sys.κs)

    # Coordinate dipole data
    @. sys.dipoles = expected_spin(Z)

    return
end


