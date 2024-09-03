using Printf
function detail_sys(sys::System{N}) where N
    io = stdout
    modename = if sys.mode==:SUN
        "SU($N)"
    elseif sys.mode==:dipole
        "Dipole mode"
    elseif sys.mode==:large_S
        "Large-S classical limit"
    else
        error("Unreachable")
    end
    printstyled(io, "System [$modename]\n"; bold=true, color=:underline)
    println(io, "Lattice: $(sys.latsize)×$(Sunny.natoms(sys.crystal))")
    if !isnothing(sys.origin)
        #println(io, "Reshaped cell geometry $(cell_dimensions(sys))")
        is_enlarged = abs(det(sys.crystal.latvecs)) > abs(det(sys.origin.crystal.latvecs))
        println(io)
        println(io, "Unit cell has been $(is_enlarged ? "enlarged" : "reshaped") from original such that:")
        printstyled(io, "  [Original lattice vectors]";bold=true,color=:red)
        print(io," * $(cell_dimensions(sys)) = ")
        printstyled(io,"[Reshaped lattice vectors]\n";bold=true,color=:blue)
        println(io, "where")
        printstyled(io, "  [Original] ";bold=true,color=:red)
        show(io, sys.origin.crystal)
        printstyled(io, "\n  [Reshaped] ";bold=true,color=:blue)
        show(io, sys.crystal)
        println(io)
    else
        show(io, sys.crystal)
    end
    println(io)

    if Sunny.is_homogeneous(sys)
        ints = Sunny.interactions_homog(sys)
        if isempty(ints)
            println(io, "No interactions")
        else
            print(io, "Homogeneous interactions by atom:\n")
            for (i,int) in enumerate(ints)
                if sys.crystal.types[i] != ""
                    print(io,"  $i. '$(sys.crystal.types[i])' atom has ")
                else
                    print(io,"  Atom $i has ")
                end
                show(io,int)
                println(io)
            end
        end
    else
        print(io, "Inhomogeneous interactions (may differ at every site)")
    end

    if !isnothing(sys.ewald)
        println(io, "Long range dipole-dipole interactions enabled!")
    end

    if !iszero(sys.extfield)
        if allequal(sys.extfield)
            println(io, "Uniform magnetic field B = $(sys.extfield[1]) applied")
        else
            mean_field = sum(sys.extfield) ./ length(sys.extfield)
            rms_field = sqrt.(sum([(B .- mean_field) .^ 2 for B in sys.extfield]) ./ length(sys.extfield))

            mean_field_str = @sprintf "[%.4g %.4g %.4g]" mean_field[1] mean_field[2] mean_field[3]
            rms_field_str = @sprintf "[%.4g %.4g %.4g]" rms_field[1] rms_field[2] rms_field[3]

            println(io, "Spatially periodic magnetic field B = (mean $mean_field_str ± $rms_field_str RMS) applied")
        end
    end
end

function Base.show(io::IO, stvexp::Sunny.StevensExpansion)
    print(io,"StevensExpansion{c0=$(stvexp.c0),0,$(stvexp.c2),0,$(stvexp.c4),0,$(stvexp.c6)}")
end

function Base.show(io::IO, ::MIME"text/plain", stvexp::Sunny.StevensExpansion)
    print(io,show_stevens_expansion(stvexp))
end

function Base.iszero(s::Sunny.StevensExpansion)
  iszero(s.c0) && iszero(s.c2) && iszero(s.c4) && iszero(s.c6)
end

function Base.show(io::IO, ints::Sunny.Interactions)
    has_onsite = !iszero(ints.onsite)
    count_pair = length(ints.pair)
    if !has_onsite && count_pair == 0
        print(io,"[No Interactions]")
    else
        print(io,"Interactions($(has_onsite ? "Onsite Coupling, " : "")$(count_pair) Pair Couplings)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ints::Sunny.Interactions)
    if !iszero(ints.onsite)
      if ints.onsite isa Sunny.StevensExpansion
        println(io,"Onsite coupling stevens expansion: ",show_stevens_expansion(ints.onsite))
      else
        println(io,"Onsite coupling matrix: ",display(ints.onsite))
      end
    end

    if isempty(ints.pair)
        if iszero(ints.onsite)
           println(io,"No interactions")
        end
        return
    end
    println(io,"Pair couplings:")
    count_culled = 0
    for pair in ints.pair
        if pair.isculled
          count_culled += 1
          continue
        end
        print(io,"  ")
        show(io,pair)
        println(io)
    end
    if count_culled > 0
        println(io,"  + $(count_culled) culled couplings")
    end
end

function show_stevens_expansion(stvexp::Sunny.StevensExpansion)
    c = map(1:6) do k
        if k == 2
            stvexp.c2
        elseif k == 4
            stvexp.c4
        elseif k == 6
            stvexp.c6
        else
            zeros(Float64, 2k+1)
        end
    end

    terms = String[]
    for k in 1:6
        for (c_km, m) in zip(reverse(c[k]), -k:k)
            abs(c_km) < 1e-12 && continue
            push!(terms, *(Sunny.coefficient_to_math_string(c_km), "𝒪", Sunny.int_to_underscore_string.((k,m))...))
        end
    end

    # Linear shift c_00 is not included in StevensExpansion
    push!(terms, "trace")

    # Concatenate with plus signs
    str = join(terms, " + ")
    # Remove redundant plus signs and print
    str = replace(str, "+ -" => "- ")
    str
end

function Base.show(io::IO, pair::Sunny.PairCoupling)
    cull_string = pair.isculled ? "(CULLED)," : ""
    bilin_string = iszero(pair.bilin) ? "" : (pair.bilin isa Float64 ? ",J = $(pair.bilin)" : ",J = Exchange Matrix")
    biquad_string = iszero(pair.biquad) ? "" : ",biquad = $(pair.biquad)"
    print(io,"PairCoupling($(cull_string)$(repr(pair.bond))$(bilin_string)$(biquad_string))")
end

function Base.show(io::IO, ::MIME"text/plain", pair::Sunny.PairCoupling)
    cull_string = pair.isculled ? "(CULLED) " : ""
    printstyled(io,"Pair Coupling $(cull_string)on $(repr(pair.bond))\n";bold=true,underline=true)
    #printstyled(io, repr(b); bold=true, color=:underline)

    atol = 1e-12
    digits = 8
    max_denom = 20

    if !iszero(pair.scalar)
      println(io,"Scalar (diagonal pure Heisenberg) = $(Sunny.number_to_math_string(pair.scalar;digits,atol,max_denom))")
    end

    if !iszero(pair.bilin)
      if pair.bilin isa Sunny.Mat3
          strs = Sunny.number_to_math_string.(pair.bilin;digits,atol,max_denom)
          println(Sunny.formatted_matrix(strs; prefix="Bilinear exchange matrix: "))
      else
        println(io,"Heisenberg bilinear (pure diagonal) exchange J = $(Sunny.number_to_math_string(pair.bilin;digits,atol,max_denom))")
      end
    end

    if !iszero(pair.biquad)
      if pair.biquad isa Sunny.Mat5
          strs = Sunny.number_to_math_string.(pair.biquad;digits,atol,max_denom)
          println(Sunny.formatted_matrix(strs; prefix="Biquadratic exchange matrix: "))
      else
        println(io,"Heisenberg biquadratic exchange J = $(Sunny.number_to_math_string(pair.biquad;digits,atol,max_denom))")
      end
    end
end

function Base.show(io::IO, ::MIME"text/plain", ten::Sunny.TensorDecomposition)
  print(io,"General interaction with $(length(ten.data)) SVD terms")
  #for (a,b) in ten.data
  #end
end

