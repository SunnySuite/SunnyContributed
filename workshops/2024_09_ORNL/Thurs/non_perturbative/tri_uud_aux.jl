using StaticArrays

function get_reshaped_cartesian_index(npt::Sunny.NonPerturbativeTheory, qs)
    (; clustersize) = npt
    Nu1, Nu2, Nu3 = clustersize
    all_qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    com_indices = CartesianIndex[]

    for q in qs
        @assert q in all_qs "The momentum is not in the grid."
        q_reshaped = Sunny.to_reshaped_rlu(npt.swt.sys, q)
        for i in 1:3
            (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
        end
        q_reshaped = mod.(q_reshaped, 1.0)
        qcom_carts_index = findmin(x -> norm(x - q_reshaped), all_qs)[2]
        push!(com_indices, qcom_carts_index)
    end

    return com_indices
end

function calculate_quartic_corrections(npt::Sunny.NonPerturbativeTheory, num_1ps::Int, qcom_index::CartesianIndex{3}; opts...)
    H1p = zeros(ComplexF64, num_1ps, num_1ps)
    Sunny.one_particle_hamiltonian!(H1p, npt, qcom_index; opts...)
    @assert Sunny.diffnorm2(H1p, H1p') < 1e-10
    hermitianpart!(H1p)
    return H1p
end

# For two-particle calculations
function generate_renormalized_npt(npt::Sunny.NonPerturbativeTheory; single_particle_correction::Bool=true, opts...)
    (; swt, clustersize, two_particle_states, qs, Es, Vps, real_space_quartic_vertices, real_space_cubic_vertices) = npt
    N1, N2, N3 = clustersize
    cart_indices = CartesianIndices((1:N1, 1:N2, 1:N3))
    num_1ps = Sunny.nbands(swt)
    Es′ = similar(Es)
    Es′ .= 0.0
    pm = Progress(N1*N2*N3, desc="Generating the renormalized non-perturbative theory")
    Threads.@threads for cart_index in cart_indices
        H1p = calculate_quartic_corrections(npt, num_1ps, cart_index; single_particle_correction, opts...)
        E, V = eigen(H1p)
        # sort the eigenvalues based on the overlap with the original 1-particle states. In this way, we should avoid the band-cross issue
        order = Int[]
        for i in 1:num_1ps
            max_index = argmax(abs.(V[:, i]))
            push!(order, max_index)
        end
        E[1:num_1ps] = E[order]
        Es′[:, cart_index] = E
        next!(pm)
    end

    npt′ = Sunny.NonPerturbativeTheory(swt, clustersize, two_particle_states, qs, Es′, Vps, real_space_quartic_vertices, real_space_cubic_vertices)
    return npt′
end

function calculate_two_particle_energies(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, qcom_index::CartesianIndex{3})
    H2p = zeros(ComplexF64, num_2ps, num_2ps)
    Sunny.two_particle_hamiltonian!(H2p, npt, qcom_index)
    @assert Sunny.diffnorm2(H2p, H2p') < 1e-12
    hermitianpart!(H2p)
    E, _ = eigen(H2p)
    return E
end

"""
    calculate_two_particle_intensities(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, q, qcom_index::CartesianIndex{3}, ωs, η)

Calculate the two-particle intensity using the Lehmann representation. I.e., get the eigenstates from exact diagonalization.
"""
function calculate_two_particle_ed_intensities(npt::Sunny.NonPerturbativeTheory, num_2ps::Int, q, qcom_index::CartesianIndex{3}, ωs, η)
    num_1ps = Sunny.nbands(npt.swt)
    H2p = zeros(ComplexF64, num_2ps, num_2ps)
    Sunny.two_particle_hamiltonian!(H2p, npt, qcom_index)
    @assert Sunny.diffnorm2(H2p, H2p') < 1e-12
    hermitianpart!(H2p)
    E, V = eigen(H2p; sortby = x -> -1/real(x))
    f0 = Sunny.continued_fraction_initial_states(npt, q, qcom_index)
    f0_3 = view(f0, num_1ps+1:num_1ps+num_2ps, 3)
    amps = zeros(num_2ps)
    for i in axes(V, 2)
        amps[i] = abs2(V[:, i] ⋅ f0_3)
    end
    
    ints = zeros(length(ωs))

    for (iω, ω) in enumerate(ωs)
        for j in 1:num_2ps
            ints[iω] += amps[j] * Sunny.lorentzian(ω - E[j], η)
        end
    end

    return ints
end