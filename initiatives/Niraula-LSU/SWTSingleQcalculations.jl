using Sunny

function cartesian(n) # create two vectors perpendicular to n.
        z = [0,0,1]
        y = [0,1,0]
        c = cross(n,z)
        if norm(c) < 1e-8
            vy = cross(n,y)
        else
            vy = c
        end
        vy = normalize(vy)
        vz = normalize(cross(n,vy))
        return([vy,vz])
end


# planar

function gm_planar!(sys::System,n,x)
    k = x[end-2:end]
    u,v = cartesian(n);
    nspin = length(x) - 3
    phi = x[1:nspin]
    for i in 1:nspin
            if length(sys.Ns) != nspin
                error("Error:The magnetic atoms are not correct")
            end
            sys.dipoles[i] = (u * cos(phi[i]) + v * sin(phi[i])) * sys.κs[i];
    end
    E = spiral_energy(sys,k,n)
    return E
end

function gm_spherical3d!(sys::System,n,x)
    nspin = Int64((length(x)-3)/2)
    k = x[end-2:end]
    MTheta  = x[(1:nspin) .* 2 .- 1]
    MPhi    = x[(1:nspin) .* 2]
	R = Sunny.rotation_between_vectors(n,[0.0,0.0,1])
    for i in 1:nspin
        if length(MTheta)!=length(sys.Ns)
            error("gm_spherical3d:NumberOfMoments','The number of fitting parameters doesn''t produce the right number of moments!")
        end
        sys.dipoles[i] = R * [sin(MTheta[i])*cos(MPhi[i]); sin(MTheta[i])*sin(MPhi[i]); cos(MTheta[i])] * sys.κs[i]
    end
    E = spiral_energy(sys,k,n)
    return E
end



function spiral_energy(sys::System{N}, k, axis) where N
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    E = 0
    L = natoms(sys.crystal)
    for i = 1:L
        (; onsite, pair) = sys.interactions_union[i]
        Si = sys.dipoles[i]

        # Pair coupling
        for coupling in pair
            (; isculled, bond, bilin, biquad) = coupling
            isculled && break

            (; j, n) = bond
            θ = 2π * dot(k, n)
            R = axis_angle_to_matrix(axis, θ)

            J = bilin
            Sj = sys.dipoles[j]
            E += Si' * (J * R) * Sj
            # Note invariance under global rotation R
            @assert J * R ≈ R * J

            @assert iszero(biquad) "Biquadratic interactions not supported"
        end

        # Onsite coupling
        E += energy_and_gradient_for_classical_anisotropy(Si, onsite)[1]

        # Zeeman coupling
        E -= sys.extfield[i]' * magnetic_moment(sys, i)
    end
    return E
end

function optimagstr(f::Function,xmin,xmax,x0,n) # optimizing function to get minimum energy and propagation factor.
    results = optimize(x->f(sys,n,x),xmin,xmax,x0,Fminbox(BFGS()));
	println("Ground state Energy(meV) = ", results.minimum);
	opt = Optim.minimizer(results)
    k = opt[end-2:end]
    return k
end

function construct_uniaxial_anisotropy(; n, c20=0., c40=0., c60=0., S)
    # Anisotropy operator in frame of `n`
    O = Sunny.stevens_matrices(S)
    op = c20*O[2, 0] + c40*O[4, 0] + c60*O[6, 0]
    # Anisotropy operator in global frame (rotates n to [0, 0, 1])
    R = Sunny.rotation_between_vectors(n, [0, 0, 1])
    return rotate_operator(op, R)
end
