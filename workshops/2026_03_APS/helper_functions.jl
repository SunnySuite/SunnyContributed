function cerium_crystal()
    crystal_full = Crystal(joinpath(@__DIR__, "BaCe2ZnS5.cif"); symprec=0.01)
    crystal = subcrystal(crystal_full, "Ce1")
    positions = crystal.positions
    latvecs = crystal.latvecs
    latvecs_new = [latvecs[:,1] latvecs[:,2] latvecs[:,3]]
    eps = 0.0
    offset = 1/4
    positions_new = [[(p[1]+offset+eps)%1, (p[2]+offset)%1, (p[3]+0.25)%1] for p in positions]

    return Crystal(latvecs_new, positions_new)
end