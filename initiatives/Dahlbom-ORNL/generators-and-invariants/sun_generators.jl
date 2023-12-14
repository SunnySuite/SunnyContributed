comm(A, B)     = A*B - B*A
anticomm(A, B) = A*B + B*A
δ(i,j) = i == j ? 1.0 : 0.0

function e_mat(j, h, n)
    @assert j <= n && h <= n
    mat = zeros(ComplexF64, n, n)
    mat[h, j] = 1.0
    mat
end

function β(j, h, n)
    @assert j <= n && h <= n
    -im*(e_mat(j, h, n) - e_mat(h, j, n))
end

function θ(j, h, n)
    @assert j <= n && h <= n
    e_mat(j, h, n) + e_mat(h, j, n)
end

function η(m, n)
    @assert 1 <= m < n
    √(2/(m*(m+1))) * (sum(e_mat(j, j, n) for j ∈ 1:m ) - m*e_mat(m+1, m+1, n))
end


"""
    sun_generators(n)
Create generators for SU(N), of which there will be N²-1.  See Appendix A of
Nemoto (2000), "Generalized coherent states for SU(n) systems." Recovers Pauli
matrices for N=2, Gell-Mann matrices for N=3.
"""
function sun_generators(N)
    Ts = []
    for j ∈ 2:N
        for h ∈ 1:j-1
            push!(Ts, θ(j, h, N))
            push!(Ts, β(j, h, N))
        end 
        push!(Ts, η(j-1, N))
    end
    Ts
end

classical_quadratic_invariant(N) = sum(T[1,1]^2 for T in sun_generators(N))
quantum_quadratic_invariant(N) = sum(T^2 for T in sun_generators(N))[1,1]