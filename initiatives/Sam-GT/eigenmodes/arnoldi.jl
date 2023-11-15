# Application of Arnoldi iteration to find the lowest energy modes of a Linear Spin Wave Theory
#
# Author: Sam Quinn
# Date: Nov 2, 2023

using Sunny, Arpack, SparseArrays

function bogoliubov_arnoldi(H::SparseMatrixCSC{ComplexF64,Int64};nev)
    L = div(size(H, 1), 2)
    @assert size(H) == (2L, 2L)

    # Initialize V to the para-unitary identity Ĩ = diagm([ones(L), -ones(L)])
    V = Diagonal([repeat([1],L);repeat([-1],L)])

    # Solve generalized eigenvalue problem, Ĩ t = λ H t, for columns t of V.
    # Eigenvalues are sorted such that quasi-particle energies will appear in
    # descending order.
    #λ, V0 = eigen!(Hermitian(V), Hermitian(H); sortby = x -> -1/real(x))
    display(V)
    display(H)
    λ, V, nconv, niter, nmult, resid = eigs(V,H;nev,which = :LR,ncv = 2*nev+1)
    println("nconv = $nconv")
    println("niter = $niter")
    println("nmult = $nmult")

    # Normalize columns of V so that para-unitarity holds, V† Ĩ V = Ĩ.
    for j in axes(V, 2)
        c = 1 / sqrt(abs(λ[j]))
        view(V, :, j) .*= c
    end

    # Relation between generalized eigenproblem and the actual modes
    display(λ)
    disp = 2 ./ real.(λ)

    return disp, V
end

function intensity_formula_arnoldi(swt::SpinWaveTheory, mode::Symbol; kwargs...)
    if mode == :trace
        contractor = Sunny.Trace(swt.observables)
        string_formula = "Tr S"
    elseif mode == :perp
        contractor = Sunny.DipoleFactor(swt.observables)
        string_formula = "∑_ij (I - Q⊗Q){i,j} S{i,j}\n\n(i,j = Sx,Sy,Sz)"
    elseif mode == :full
        contractor = Sunny.FullTensor(swt.observables)
        string_formula = "S{α,β}"
    end
    intensity_formula_arnoldi(swt,contractor;string_formula,kwargs...)
end

function intensity_formula_arnoldi(swt::SpinWaveTheory, contractor::Sunny.Contraction{T}; kwargs...) where T
    intensity_formula_arnoldi(swt,Sunny.required_correlations(contractor); return_type = T,kwargs...) do k,ω,correlations
        intensity = Sunny.contract(correlations, k, contractor)
    end
end



function intensity_formula_arnoldi(f::Function,swt::SpinWaveTheory,corr_ix::AbstractVector{Int64}; kernel::Union{Nothing,Function},
                           return_type=Float64, string_formula="f(Q,ω,S{α,β}[ix_q,ix_ω])",
                           formfactors=nothing,nev = 6)
    (; sys, data, observables) = swt
    Nm, Ns = length(sys.dipoles), sys.Ns[1] # number of magnetic atoms and dimension of Hilbert space
    S = (Ns-1) / 2
    nmodes = Sunny.num_bands(swt)
    sqrt_Nm_inv = 1.0 / √Nm
    sqrt_halfS  = √(S/2)

    # Preallocation
    H = sparse(zeros(ComplexF64, 2*nmodes, 2*nmodes))
    Avec_pref = zeros(ComplexF64, Nm)
    intensity = zeros(return_type, 2nev)

    # Expand formfactors for symmetry classes to formfactors for all atoms in
    # crystal
    ff_atoms = Sunny.propagate_form_factors_to_atoms(formfactors, swt.sys.crystal)

    # Upgrade to 2-argument kernel if needed
    kernel_edep = if isnothing(kernel)
        nothing
    else
        try
            kernel(0.,0.)
            kernel
        catch MethodError
            (ω,Δω) -> kernel(Δω)
        end
    end

    # In Spin Wave Theory, the Hamiltonian depends on momentum transfer `q`.
    # At each `q`, the Hamiltonian is diagonalized one time, and then the
    # energy eigenvalues can be reused multiple times. To facilitate this,
    # `I_of_ω = calc_intensity(swt,q)` performs the diagonalization, and returns
    # the result either as:
    #
    #   Delta function kernel --> I_of_ω = (eigenvalue,intensity) pairs
    #
    #   OR
    #
    #   Smooth kernel --> I_of_ω = Intensity as a function of ω
    #
    calc_intensity = function(swt::SpinWaveTheory, q::Sunny.Vec3)
        # This function, calc_intensity, is an internal function to be stored
        # inside a formula. The unit system for `q` that is passed to
        # formula.calc_intensity is an implementation detail that may vary
        # according to the "type" of a formula. In the present context, namely
        # LSWT formulas, `q` is given in RLU for the original crystal. This
        # convention must be consistent with the usage in various
        # `intensities_*` functions defined in LinearSpinWaveIntensities.jl.
        # Separately, the functions calc_intensity for formulas associated with
        # SampledCorrelations will receive `q_absolute` in absolute units.
        q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)
        q_absolute = swt.sys.crystal.recipvecs * q_reshaped

        if sys.mode == :SUN
            Sunny.swt_hamiltonian_SUN!(H, swt, q_reshaped)
        else
            @assert sys.mode in (:dipole, :dipole_large_S)
            Sunny.swt_hamiltonian_dipole!(H, swt, q_reshaped)
        end

        Hsp = sparse(H)
        disp, V = try
            bogoliubov_arnoldi(Hsp;nev)
        catch e
            error("Bogoliubov failed at wavevector q = $q")
        end

        for i = 1:Nm
            @assert Nm == Sunny.natoms(sys.crystal)
            phase = exp(-2π*im * dot(q_reshaped, sys.crystal.positions[i]))
            Avec_pref[i] = sqrt_Nm_inv * phase

            # TODO: move form factor into `f`, then delete this rescaling
            Avec_pref[i] *= Sunny.compute_form_factor(ff_atoms[i], q_absolute⋅q_absolute)
        end

        # Fill `intensity` array
        for band = 1:nev, s = [1,-1]
            v = if s == 1
              V[:, band]
            else
              conj.([V[(nmodes+1):end,band];V[1:nmodes, band]])
            end
            corrs = if sys.mode == :SUN
                Avec = zeros(ComplexF64, Sunny.num_observables(observables))
                (; observable_operators) = data
                for i = 1:Nm
                    for μ = 1:Sunny.num_observables(observables)
                        @views O = observable_operators[:, :, μ, i]
                        for α = 2:Ns
                            Avec[μ] += Avec_pref[i] * (O[α, 1] * v[(i-1)*(Ns-1)+α-1+nmodes] + O[1, α] * v[(i-1)*(Ns-1)+α-1])
                        end
                    end
                end
                corrs = Vector{ComplexF64}(undef,Sunny.num_correlations(observables))
                for (ci,i) in observables.correlations
                    (α,β) = ci.I
                    corrs[i] = Avec[α] * conj(Avec[β])
                end
                corrs
            else
                @assert sys.mode in (:dipole, :dipole_large_S)
                Avec = zeros(ComplexF64, 3)
                (; R_mat) = data
                for i = 1:Nm
                    Vtmp = [v[i+nmodes] + v[i], im * (v[i+nmodes] - v[i]), 0.0]
                    Avec += Avec_pref[i] * sqrt_halfS * (R_mat[i] * Vtmp)
                end

                @assert observables.observable_ixs[:Sx] == 1
                @assert observables.observable_ixs[:Sy] == 2
                @assert observables.observable_ixs[:Sz] == 3
                corrs = Vector{ComplexF64}(undef,num_correlations(observables))
                for (ci,i) in observables.correlations
                    (α,β) = ci.I
                    corrs[i] = Avec[α] * conj(Avec[β])
                end
                corrs
            end

            intensity[band + nev * (s == 1 ? 0 : 1)] = f(q_absolute, s * disp[band], corrs[corr_ix])
        end

        # Return the result of the diagonalization in an appropriate
        # format based on the kernel provided
        if isnothing(kernel)
            # Delta function kernel --> (eigenvalue,intensity) pairs

            # If there is no specified kernel, we are done: just return the
            # BandStructure
            return Sunny.BandStructure{2*nev,return_type}([disp;-disp], intensity)
        else
            # Smooth kernel --> Intensity as a function of ω (or a list of ωs)
            disp_dup = [disp;-disp]
            return function(ω)
                is = Vector{return_type}(undef,length(ω))
                is .= sum(intensity' .* kernel_edep.(disp_dup',ω .- disp_dup'),dims=2)
                is
            end
        end
    end
    output_type = isnothing(kernel) ? Sunny.BandStructure{2*nev,return_type} : return_type
    Sunny.SpinWaveIntensityFormula{output_type}(string_formula,kernel_edep,calc_intensity)
end
