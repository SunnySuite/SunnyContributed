
using GLMakie, FFTW, LinearAlgebra

N = 100
ts = range(0,1,length=N)
As = exp.(-(ts .- 0.15).^2 ./ (2 * 0.05^2))
Bs = exp.(-(ts .- 0.35).^2 ./ (2 * 0.1^2))


As_extended = zeros(2N)
Bs_extended = zeros(2N)

As_extended[1:N] = As
Bs_extended[1:N] = Bs

f = Figure()
ax1 = Axis(f[1,1])
plot!(ax1,As_extended)
plot!(ax1,As,color = :blue)

ax2 = Axis(f[1,2])
plot!(ax2,Bs_extended)
plot!(ax2,Bs,color = :blue)

function cross_correlation(u, v)
    ifft(conj.(fft(u)) .* fft(v))
end

C1 = real(cross_correlation(As_extended, As_extended))
C2 = real(cross_correlation(As, As))

ax3 = Axis(f[2,1])
lines!(ax3,C1,color = :black)
lines!(ax3,C2,color = :blue)

function rescale_correlations!(C)
    T = Int(length(C)/2)

    factor = copy(C)
    for t in -(T-1):(T-1)
        i = mod(t, 2T) + 1
        f = cos(π*t/2T)^2
        factor[i] = T - abs(t)
        C[i] *= f / (T - abs(t))
    end

    # The shift t = ±T corresponds to the index i = T+1, and in this case the
    # scaling factor has the limit f/(T-|t|) → 0.
    C[T+1] *= 0

    factor
end


C3 = copy(C1)
factor = rescale_correlations!(C3)

ax4 = Axis(f[2,2])
lines!(ax4,C3)

f
