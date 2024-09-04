using LinearAlgebra

function custom_exponential(A)
    I + A + A^2/2 + A^3/6 + A^4/24

    # sum(A^n / factorial(n) for n in 0:20)
end
