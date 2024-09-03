using Sunny

# N.B. 9/2/2024: This is an unfinished attempt at taking gradients through the KPM
# (Kernel Polynomial Method) of https://arxiv.org/abs/2312.08349 which would enable
# very quick fitting of experimental data by gradient descent :)

function get_all_coefficients(M, ωs, broadening, σ, kT,γ;η=0.05, regularization_style)
    f = (ω,x) -> broadening(ω, x*γ, σ)
    output = OffsetArray(zeros(M, length(ωs)), 0:M-1, 1:length(ωs))
    for i in eachindex(ωs)
        output[:, i] = Sunny.cheb_coefs(M, 2M, x -> f(ωs[i], x), (-1, 1))
    end
    return output
end

struct GradKPMIntensityFormula{T}
    P :: Int64
    kT :: Float64
    σ :: Float64
    broadening
    kernel
    string_formula :: String
    calc_intensity :: Function
end

function intensity_formula_gradkpm(f,swt::SpinWaveTheory,corr_ix::AbstractVector{Int64}; P =50, kT=Inf,σ=0.1,broadening, γ = nothing, gradient_direction = nothing , return_type = Float64, string_formula = "f(Q,ω,S{α,β}[ix_q,ix_ω])",regularization_style)
    # P is the max Chebyshyev coefficient
    (; sys, data) = swt
    Nm, Ns = length(sys.dipoles), sys.Ns[1] # number of magnetic atoms and dimension of Hilbert space
    Nf = sys.mode == :SUN ? Ns-1 : 1
    N=Nf+1
    nmodes = Nf*Nm
    M = sys.mode == :SUN ? 1 : (Ns-1) # scaling factor (=1) if in the fundamental representation
    sqrt_M = √M #define prefactors
    sqrt_Nm_inv = 1.0 / √Nm #define prefactors
    S = (Ns-1) / 2
    sqrt_halfS  = √(S/2) #define prefactors   
    sqrt_Nm_inv = 1.0 / √Nm #define prefactors
    S = (Ns-1) / 2
    sqrt_halfS  = √(S/2) #define prefactors   
    ii = spdiagm([ones(nmodes); -ones(nmodes)]) 
    n_iters = 50
    Hmat = zeros(ComplexF64, 2*nmodes, 2*nmodes)
    Hmatgrad = zeros(ComplexF64, 2*nmodes, 2*nmodes)
    Avec_pref = zeros(ComplexF64, Nm) # initialize array of some prefactors   
    chebyshev_moments = OffsetArray(zeros(ComplexF64,3,3,P),1:3,1:3,0:P-1)

    calc_intensity = function(swt::SpinWaveTheory,q::Vec3)
        q_reshaped = Sunny.to_reshaped_rlu(sys, q)
        q_absolute = swt.sys.crystal.recipvecs * q_reshaped
        u = zeros(ComplexF64,3,2*nmodes)
        if sys.mode == :SUN
            swt_hamiltonian_SUN!(Hmat, swt, q_reshaped)

            orig_sys = swt.sys
            swt.sys = gradient_direction
            swt_hamiltonian_SUN!(Hmatgrad, swt, q_reshaped)
            swt.sys = orig_sys
        else
            swt_hamiltonian_dipole!(Hmat, swt, q_reshaped)
        end

        D = 2.0*sparse(Hmat)
        Dgrad = 2.0*sparse(Hmatgrad)

        γ = if isnothing(γ)
          lo,hi = Sunny.eigbounds(ii*D,n_iters; extend=0.25) # calculate bounds

          γ=max(lo,hi) # select upper bound (combine with the preceeding line later)
        else
          γ
        end

        A = ii*D / γ
        Agrad = ii*Dgrad / γ

        # u(q) calculation
        for site = 1:Nm
            # note that d is the chemical coordinates
            chemical_coor = sys.crystal.positions[site] # find chemical coords
            phase = exp(2*im * π  * dot(q_reshaped, chemical_coor)) # calculate phase
            Avec_pref[site] = sqrt_Nm_inv * phase  # define the prefactor of the tS matrices
        end
        # calculate u(q)
        if sys.mode == :SUN
            for site=1:Nm
                @views tS_μ = data.dipole_operators[:, :, :, site]*Avec_pref[site] 
                for μ=1:3
                    for j=2:N
                        u[μ,(j-1)+(site-1)*(N-1) ]=tS_μ[μ,j,1]
                        u[μ,(N-1)*Nm+(j-1)+(site-1)*(N-1) ]=tS_μ[μ,1,j]
                    end
                end
            end
        elseif sys.mode == :dipole
            for site = 1:Nm
                R=data.R_mat[site]
                u[1,site]= Avec_pref[site] * sqrt_halfS * (R[1,1] + 1im * R[1,2])
                u[1,site+nmodes] = Avec_pref[site] * sqrt_halfS * (R[1,1] - 1im * R[1,2])
                u[2,site]= Avec_pref[site] * sqrt_halfS * (R[2,1] + 1im * R[2,2])
                u[2,site+nmodes] = Avec_pref[site] * sqrt_halfS * (R[2,1] - 1im * R[2,2])
                u[3,site]= Avec_pref[site] * sqrt_halfS * (R[3,1] + 1im * R[3,2])
                u[3,site+nmodes] = Avec_pref[site] * sqrt_halfS * (R[3,1] - 1im * R[3,2])
            end

        end
        # phi0 = ii * u
        # phi1 = A * phi0
        # phi(m+1) = 2 * A * phi(m) - phi(m-1)
        #
        # psi0 = 0
        # psi1 = Agrad * phi0
        # psi(m+1) = 2 * Agrad * phi(m) + 2 * A * psi(m) - psi(m-1)
        for β=1:3
            α0 = zeros(ComplexF64,2*nmodes)
            α1 = zeros(ComplexF64,2*nmodes)
            mul!(α0,ii,u[β,:]) # calculate α0
            mul!(α1,A,α0) # calculate α1
            for α=1:3
                chebyshev_moments[α,β,0] = dot(u[α,:],α0)
                chebyshev_moments[α,β,1] = dot(u[α,:],α1)
            end
            for m=2:P-1
                αnew = zeros(ComplexF64,2*nmodes)
                mul!(αnew,A,α1)
                @. αnew = 2*αnew - α0
                for α=1:3
                    chebyshev_moments[α,β,m] = (dot(u[α,:],αnew))
                end
                (α1, α0) = (αnew, α1)
            end
        end

        return function(ωlist)
            intensity = zeros(Float64,length(ωlist))
            ωdep = get_all_coefficients(P,ωlist,broadening,σ,kT,γ;regularization_style)
            apply_kernel(ωdep,kernel,P)
            Sαβ = Matrix{ComplexF64}(undef,3,3)
            corrs = Vector{ComplexF64}(undef,num_correlations(swt.observables))
            for (iω,ω) = enumerate(ωlist)
                for α=1:3
                    for β=1:3
                        Sαβ[α,β] = sum(chebyshev_moments[α,β,:] .* ωdep[:,iω])
                    end
                end

                for (ci,i) in swt.observables.correlations
                    (α,β) = ci.I
                    corrs[i] = Sαβ[α,β]
                end
                intensity[iω] = f(q_absolute,corrs[corr_ix])
            end
            return intensity
        end
    end
    KPMIntensityFormula{return_type}(P,kT,σ,broadening,kernel,string_formula,calc_intensity)
end
