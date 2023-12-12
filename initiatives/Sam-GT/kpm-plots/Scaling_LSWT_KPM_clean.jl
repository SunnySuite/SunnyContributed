using Sunny, BenchmarkTools,GLMakie,LinearAlgebra,JLD2,FileIO, Optim, Colors, ColorSchemes

a = 1.0 # Lattice constants for cubic lattice
b =4.0 
c =5.0
latvecs = lattice_vectors(a, b, c, 90, 90, 90) # A 3x3 matrix of lattice vectors that
                                                # define the conventional unit cell
positions = [[0, 0, 0]]  # Positions of atoms in fractions
                                                           # of lattice vectors
cryst = Crystal(latvecs, positions,1;)
view_crystal(cryst, 1.0)
print_symmetry_table(cryst, 1.0)
function br(ω, x, σ)
    return (1/π) * (σ / ((x - ω)^2 + σ^2))
end

function init_system_biased(cryst,n)
    sys = System(cryst, (n, 1, 1), [SpinInfo(1, S=1, g=2)], :dipole; seed=2,units = Units.theory)
    set_exchange!(sys,1,Bond(1,1,[1,0,0]))
    D = 0.5
    set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)
    axis = [1,0,0]
    set_spiral_order_on_sublattice!(sys, 1; q=[1/2,0,0], axis, S0=[0, 0, 1])
    return SpinWaveTheory(sys)
end

Emax = 2.5*1.25
σin= 0.025*Emax
swt=init_system_biased(cryst,2)

function RunKPM(swt,qs,energies,kpmformula)
    isKPM = intensities_broadened(swt, qs, energies, kpmformula)
    return isKPM
end

function RunLSWT(swt,qs,energies,broadened_formula)
    is = intensities_broadened(swt, qs, energies, broadened_formula);
    return is
end

function TimeScaling(cryst,nlist,M,σin)
    # BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000
    nq=1
    ne=50
    energies = range(2σin,2.5,ne)
    qs = [rand(3) for _ ∈ 1:nq]
    tsLSWT = zeros(Float64,length(nlist))
    tsKPM = zeros(Float64,length(nlist))
    for (idx,n) ∈ enumerate(nlist)
        println(n)
        swt = init_system_biased(cryst,n)
        kpmformula = Sunny.intensity_formula_kpm(swt, :perp; P=M, kT= 2*meV_per_K, σ=σin,broadening = br,regularization_style = :cubic)
        #broadened_formula = intensity_formula(swt, :perp; kernel=lorentzian(σin), mode_fast = true)
        tKPM = @belapsed RunKPM($swt,$qs,$energies,$kpmformula)
        println(tKPM)
        #tLSWT = @belapsed RunLSWT($swt,$qs,$energies,$broadened_formula)
        #println(tLSWT)
        tsKPM[idx]=tKPM
        #tsLSWT[idx]=tLSWT
    end
    return tsKPM, tsLSWT
end

nlist = 2round.(Int64,10 .^ (0.5:0.35:3.7))
#nlist = 2:40:802
KPM,LSWT =  TimeScaling(cryst,nlist,350,σin)

# function for finding convergence
function Determine_Cheby_Max(swt,tol;σ=0.1)
    nq=50
    ne=50
    energies = range(2σ,2.5,ne)
    qs = [rand(3) for _ ∈ 1:nq]
    Mmax = 1000
    Mmin = 50
    dM = 25
    diffs = []
    kpmformula = Sunny.intensity_formula_kpm(swt, :perp; P=Mmin-dM, kT= 2*meV_per_K, σ=σin,broadening = br,regularization_style = :cubic)
    isKPM = intensities_broadened(swt, qs, energies, kpmformula)
    for M ∈ Mmin:dM:Mmax
        kpmformula = Sunny.intensity_formula_kpm(swt, :perp; P=M, kT= 2*meV_per_K, σ=σin,broadening = br,regularization_style = :cubic)
        isKPMnew = intensities_broadened(swt, qs, energies, kpmformula)
        diffr  = sum(norm.(isKPM - isKPMnew))/(nq*ne)
        push!(diffs,diffr)
        isKPM = isKPMnew
        if diffr < tol
            return M,diffs
        else
        end
    end
    if diffs[end] > tol
        println("not converged to within threshold")
        return Mmax, diffs
    else
    end
end

