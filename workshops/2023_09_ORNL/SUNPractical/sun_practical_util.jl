plot_spins! = Sunny.Plotting.plot_spins!

function BFSO(dims; J=1.028, B = [0.0 0.0 0.0], g = 1.93)
    mpK = Sunny.meV_per_K

    ## Set up crystal
    a = b =  8.3194; c =  5.3336
    α = β = γ = 90 
    latvecs = lattice_vectors(a, b, c, α, β, γ)
    types = ["Fe"]
    basisvecs = [[0.0, 0, 0]]
    cryst = Crystal(latvecs, basisvecs, 113; types)

    ## Set up system
    sys = System(cryst, dims, [SpinInfo(1; S=2,g)], :SUN)

    ## Nearest neighbor in-plane interactions
    J₁ = J * mpK
    set_exchange!(sys, J₁, Bond(1, 2, [0, 0, 0]))

    ## Next-nearest neighbor in-plane interactions
    J₂ = 0.1*J₁
    set_exchange!(sys, J₂, Bond(1, 1, [1, 0, 0]))

    ## Nearest out-of-plane interaction
    J′₁ = 0.1*J₁
    set_exchange!(sys, J′₁ , Bond(1, 1, [0, 0, 1]))

    ## Anisotropy
    A   = 1.16 * mpK   
    C   = -1.74 * mpK  
    D   = 28.65 * mpK 
    Sˣ, Sʸ, Sᶻ = spin_operators(sys, 1) 
    Λ = D*(Sᶻ)^2 + A*((Sˣ)^4 + (Sʸ)^4) + C*(Sᶻ)^4
    set_onsite_coupling!(sys, Λ, 1)

    ## External field
    set_external_field!(sys, B) 

    return sys, cryst
end