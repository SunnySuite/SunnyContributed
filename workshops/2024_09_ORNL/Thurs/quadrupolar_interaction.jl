using Sunny, GLMakie, LinearAlgebra, CairoMakie
units = Units(:meV, :angstrom)
set_theme!(theme_latexfonts())
update_theme!(fontsize=20) 

# define crystal
lattvecs=lattice_vectors( 5.,  5.,  7.,  90.,  90., 120.)
positions = [[0, 0, 1/2]]
crystal = Crystal(lattvecs, positions)
GLMakie.activate!()
view_crystal(crystal)

# define system 
sys = System(crystal, [1 => Moment(s=1, g=-1)], :dipole; dims=(1,1,1), seed=1)
plot_spins(sys)

# compute LSWT spectrum
J = 1
D = 4
S = spin_matrices(1)
set_onsite_coupling!(sys, -D*S[3]^2, 1)

bond = Bond(1, 1, [1, 0, 0])

field = [0,0,10]
set_field!(sys, field)

# define interaction - method 1
set_exchange!(sys, J, bond)

# define interaction - method 2
set_pair_coupling!(sys, (Si, Sj) -> Si'*J*Sj, bond)

# define interaction - method 3
S = spin_matrices(1)
Si, Sj = to_product_space(S, S)
set_pair_coupling!(sys, Si'*J*Sj, bond)

# calculate LSWT spectrum
swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
path = q_space_path(crystal, qs, 400)
res = intensities_bands(swt, path)
fig = plot_intensities(res; units)

save("./figures/SU2_LSWT.png", fig; px_per_unit=4)

# SU3 B//c
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    field = [0,0,10]
    set_field!(sys, field)

    # define interaction - method 1
    set_exchange!(sys, J, bond)

    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units)

    save("./figures/SU3_LSWT.png", fig; px_per_unit=4)

end

# dipole tilted field
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :dipole; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    field = [0,1,10]
    set_field!(sys, field)

    # define interaction - method 1
    set_exchange!(sys, J, bond)

    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units)

    save("./figures/SU2_LSWT_tilted_field.png", fig; px_per_unit=4)

end

# SU(3) tilted field
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    field = [0,1,10]
    set_field!(sys, field)

    # define interaction - method 1
    set_exchange!(sys, J, bond)

    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units)

    save("./figures/SU3_LSWT_tilted_field.png", fig; px_per_unit=4)

end

# SU(3) lower field
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 10
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 1
    set_exchange!(sys, J, bond)

    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims=(0,17))

    save("./figures/SU3_LSWT_B=$(B).png", fig; px_per_unit=4)

end

# SU(3) lower field, low energy
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6.5
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 1
    set_exchange!(sys, J, bond)

    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims = (0,3))

    save("./figures/SU3_LSWT_B=6.5_low_energy.png", fig; px_per_unit=4)

end

# SU(3) lower field, low energy, effective quadrupolar interaction
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6.5
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 3
    S = spin_matrices(1)
    Si, Sj = to_product_space(S, S)
    Si⁺ = Si[1] + im*Si[2];
    Sj⁻ = Sj[1] - im*Sj[2];
    A = Si⁺^2 * Sj⁻^2;
    A = (A + A')/2;

    JQ = -0.05
    set_pair_coupling!(sys, JQ*A + J*Si'*Sj, bond)

    randomize_spins!(sys)
    minimize_energy!(sys)
    
    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims = (0,3))

    save("./figures/QQ_SU3_LSWT_B=6.5_low_energy.png", fig; px_per_unit=4)

end

# SU(3) quadrupolar phase
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 3
    S = spin_matrices(1)
    Si, Sj = to_product_space(S, S)
    Si⁺ = Si[1] + im*Si[2];
    Sj⁻ = Sj[1] - im*Sj[2];
    A = Si⁺^2 * Sj⁻^2;
    A = (A + A')/2;

    JQ = -0.05
    set_pair_coupling!(sys, JQ*A + J*Si'*Sj, bond)

    randomize_spins!(sys)
    minimize_energy!(sys)
    
    # calculate LSWT spectrum
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [[0,0,0], [1/2,0,0],[1/3,1/3,0],[0,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims = (0,3))

    save("./figures/QQ_SU3_LSWT_B=6.png", fig; px_per_unit=4)

end

# SU(3) quadrupolar phase, Sxx in the Blume-Maleev frame, transverse channel
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    #plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 3
    S = spin_matrices(1)
    Si, Sj = to_product_space(S, S)
    Si⁺ = Si[1] + im*Si[2];
    Sj⁻ = Sj[1] - im*Sj[2];
    A = Si⁺^2 * Sj⁻^2;
    A = (A + A')/2;

    JQ = -0.05
    set_pair_coupling!(sys, JQ*A + J*Si'*Sj, bond)

    randomize_spins!(sys)
    minimize_energy!(sys)
    
    # calculate LSWT spectrum
    measure = ssf_custom_bm(sys; u=[1, 0, 0], v=[0, 1, 0]) do q, ssf
        real(ssf[2,2])
    end
    swt = SpinWaveTheory(sys;measure )
    qs = [[0.001,0,0], [1/2,0,0],[1/3,1/3,0],[0.001,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims = (0,3))

    save("./figures/QQ_SU3_LSWT_B=6_Myy.png", fig; px_per_unit=4)

end


# SU(3) quadrupolar phase, Szz in the Blume-Maleev frame, longitudinal channel
begin
    # define system 
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    plot_spins(sys)

    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 3
    S = spin_matrices(1)
    Si, Sj = to_product_space(S, S)
    Si⁺ = Si[1] + im*Si[2];
    Sj⁻ = Sj[1] - im*Sj[2];
    A = Si⁺^2 * Sj⁻^2;
    A = (A + A')/2;

    JQ = -0.05
    set_pair_coupling!(sys, JQ*A + J*Si'*Sj, bond)

    randomize_spins!(sys)
    minimize_energy!(sys)
    
    # calculate LSWT spectrum
    measure = ssf_custom_bm(sys; u=[1, 0, 0], v=[0, 1, 0]) do q, ssf
        real(ssf[3,3])
    end
    swt = SpinWaveTheory(sys;measure )
    qs = [[0.001,0,0], [1/2,0,0],[1/3,1/3,0],[0.001,0,0]]
    path = q_space_path(crystal, qs, 400)
    res = intensities_bands(swt, path)
    fig = plot_intensities(res; units, ylims = (0,3))

    save("./figures/QQ_SU3_LSWT_B=6_Mzz.png", fig; px_per_unit=4)

end

# examine order parameters
sys.dipoles

S = spin_matrices(1)
Q⁺ = (S[1] + im*S[2])^2;
Q⁻ = (S[1] - im*S[2])^2;
sys.coherents[1]'*(Q⁺)*sys.coherents[1]

Qexp=real.([sys.coherents[1]'*(S[i]*S[j]+S[j]*S[i]-4/3*I[i,j]*I)*sys.coherents[1] for i=1:3, j=1:3])
eigen(Qexp)


# construct phase diagram
begin
    sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
    A = [2 -1 0; 1 1 0; 0 0 1]
    sys = reshape_supercell(sys, A)
    # compute LSWT spectrum
    J = 1
    D = 4
    set_onsite_coupling!(sys, -D*S[3]^2, 1)
    bond = Bond(1, 1, [1, 0, 0])

    B = 6
    field = [0,0,B]
    set_field!(sys, field)

    # define interaction - method 3
    S = spin_matrices(1)
    Si, Sj = to_product_space(S, S)
    Si⁺ = Si[1] + im*Si[2];
    Sj⁻ = Sj[1] - im*Sj[2];
    A = Si⁺^2 * Sj⁻^2;
    A = (A + A')/2;

    JQ = -0.05
    set_pair_coupling!(sys, JQ*A + J*Si'*Sj, bond)
end 

OP = Vector{Float64}[]
for field = 7:-0.005:5
    set_field!(sys, [0,0,field])
    randomize_spins!(sys)
    sys_old = Sunny.clone_system(sys)
    for _ = 1:10
        randomize_spins!(sys)
        step = 1
        while step ≠ 0
            step = minimize_energy!(sys)
            #println(step)
        end
        
        if energy(sys) < energy(sys_old)
            sys_old = Sunny.clone_system(sys)
        end
    end
    sys = Sunny.clone_system(sys_old)
    Q⁺exp = [sys.coherents[i]'*(Q⁺)*sys.coherents[i] for i =1:3]
    Q⁻exp = [sys.coherents[i]'*(Q⁻)*sys.coherents[i] for i =1:3]
    #oo=sortslices([norm.(sys.dipoles[:]) norm.(Q⁺exp)],dims=1,by=x->x[1],rev=false)
    push!(OP,[field, sort(norm.(sys.dipoles[:]))..., sort(norm.(Q⁺exp))...])
end
OP = vcat(OP'...)

begin
    CairoMakie.activate!()
    fig = Figure(resolution = (400,500));
    ax = Axis(fig[2,1])
    sc1=lines!(ax,OP[:,1],OP[:,7],linestyle = :dash)
    sc2=lines!(ax,OP[:,1],OP[:,2])
    ax.xticklabelsvisible = false
    
    #ax.xlabel = L"$H$ (T)"
    ax.ylabel = L"$i=1$"   
    ax = Axis(fig[3,1])
    lines!(ax,OP[:,1],OP[:,6],linestyle = :dash)
    lines!(ax,OP[:,1],OP[:,3])
    #ax.xlabel = L"$H$ (T)"
    ax.ylabel = L"$i=2$"
    ax.xticklabelsvisible = false
    ax = Axis(fig[4,1])
    lines!(ax,OP[:,1],OP[:,5],linestyle = :dash)
    lines!(ax,OP[:,1],OP[:,4])
    ax.xlabel = L"$B$ (T)"
    ax.ylabel = L"$i=3$"
    rowgap!(fig.layout,0)
    Legend(fig[1,1], [sc2, sc1], [ L"$|\langle {\mathbf{S}}_i\rangle|$", L"$|\langle {Q}^+_i\rangle|$"],  
    position = :t,    orientation = :horizontal, bgcolor=:white,
    margin = (0, 0, 10, 0))
    display(fig)
end
save("./figures/phase_diag.png",fig; px_per_unit=4)

sys = System(crystal, [1 => Moment(s=1, g=-1)], :SUN; dims=(1,1,1), seed=1)
A = [2 -1 0; 1 1 0; 0 0 1]
sys = reshape_supercell(sys, A)

GLMakie.activate!()
plot_spins(sys)