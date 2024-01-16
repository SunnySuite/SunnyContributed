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
    E = energy(sys,k,n)
    return E
end

function gm_spherical3d!(sys::System,n,x)
    nspin = Int64((length(x)-3)/2)
    k = x[end-2:end]
    MTheta  = x[(1:nspin) .* 2 .- 1]
    MPhi    = x[(1:nspin) .* 2]
    R = [-sin(nphi) -cos(nphi)*cos(ntheta) cos(nphi)*sin(ntheta);
        cos(nphi) -sin(nphi)*cos(ntheta) sin(nphi)*sin(ntheta);
        0.0     sin(ntheta)           cos(ntheta)]
    for i in 1:nspin
        if length(MTheta)!=length(sys.Ns)
            error("gm_spherical3d:NumberOfMoments','The number of fitting parameters doesn''t produce the right number of moments!")
        end
        sys.dipoles[i] = R * [sin(MTheta[i])*cos(MPhi[i]); sin(MTheta[i])*sin(MPhi[i]); cos(MTheta[i])] * sys.κs[i]
    end
    E = energy(sys,k,n)
    return E
end


function energy(sys::System,k,n)  # calculate the energy of the system
    E = 0
    L  = Sunny.natoms(sys.crystal)
    A = n * n'
    for matom = 1:L
        ints = sys.interactions_union[matom]
        for c in ints.pair
            d = c.bond.n
            θ = (2*π * dot(k,d))
            R = [cos(θ)+(n[1]^2)*(1-cos(θ)) n[1]*n[2]*(1-cos(θ))-n[3]*sin(θ) n[1]*n[3]*(1-cos(θ))+n[2]*sin(θ);
                n[2]*n[1]*(1-cos(θ))+n[3]*sin(θ) cos(θ)+(n[2]^2)*(1-cos(θ)) n[2]*n[3]*(1-cos(θ))-n[1]*sin(θ);
                n[3]*n[1]*(1-cos(θ))-n[2]*sin(θ) n[2]*n[3]*(1-cos(θ))+n[1]*sin(θ) cos(θ)+(n[3]^2)*(1-cos(θ))]

            J = c.bilin *I
            Jij = (J * R + R * J) ./ 2
            sub_i, sub_j = c.bond.i, c.bond.j
            Si = sys.dipoles[sub_i]
            Sj = sys.dipoles[sub_j]
            E += (Si' * Jij * Sj)
        end
    end
    E = E/2
    for i in 1:L
        Si = sys.dipoles[i]
        E += (Si' * A * Si)
    end
    for i in 1:L
        Si = sys.dipoles[i]
        B = sys.units.μB * (Transpose(sys.extfield[1, 1, 1, i]) * sys.gs[1, 1, 1, i])  
        E += B * Si  
    end
    return real(E)
end

function optimagstr(f::Function,xmin,xmax,x0,n) # optimizing function to get minimum energy and propagation factor.
    f = (x->gm_planar!(sys,n, [x...,]))  
	results = optimize(f,xmin,xmax,x0,Fminbox(BFGS()));
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
