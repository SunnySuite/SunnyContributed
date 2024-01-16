# # S(q,ω) Normalization

using Sunny, GLMakie, LinearAlgebra, SparseArrays


# A good conceptual model for decorrelating signals is the pure sine wave with phase kicks.
# Consider the base signal $\cos(\omega t)$, but perturbed to $\cos(\omega t + \varphi(t))$ where $\varphi(t)$ is a random walk whose entire job is to decorrelate the signal from itself at long times.
# The $\varphi$ can be constructed at consecutive timesteps by `φ(t) = randn * σ * √Δt + ϕ(t - Δt)`.

function mk_signal(;Δt, Tmax, σ, ω)
  ## Expected time to walk 2π:
  ## <|ϕ(t)|>/√(t/Δt) = √(2/π) * √Δt * σ
  ## 4π²/(2/π) = σ² Tcorr
  ## Tcorr = 2π³/σ²
  Tcorr = π^2/σ^2
  println("Tcorr = $(Tcorr)")
  ts = range(0,Tmax,step=Δt)
  ϕ = cumsum(randn(length(ts)) .* sqrt(Δt)) * σ
  if Tcorr < Tmax
    println(ϕ[findmin(abs.(ts .- Tcorr))[2]])
  end
  #ts, cos.(ω .* ts .+ ϕ)
  ts, ϕ
end

f = Figure()
ax1 = Axis(f[1,1]);
ax2 = Axis(f[2,1]);
display(f)
function regen(;Δt = 0.01, Tmax = 80., σ = 0.1, ω = 1.)
  ts, signal = mk_signal(;Δt,Tmax,σ,ω);
  #ix_last = findlast((1e-3/sqrt(Δt)) .> (1 .- signal))
  #ts = ts[1:ix_last]
  #signal = signal[1:ix_last]

  lines!(ax1,ts,signal)

  tShift = fftshift(fftfreq(length(ts),length(ts) * Δt))
  lines!(ax2,tShift,fftshift(real.(ifft(conj.(fft(signal)) .* fft(signal)))))

end

