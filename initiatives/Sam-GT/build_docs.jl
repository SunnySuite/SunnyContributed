using Literate

Literate.markdown("mgcro-investigation/quadratic_casimirs.jl", "docs"; execute = true, documenter = false)

cd("eigenmodes")
Literate.markdown("examples.jl", "../docs"; name ="eigenmode_viewer_examples", execute = true, documenter = false)
Literate.markdown("arnoldi_spin_glass_example.jl", "../docs"; name ="arnoldi_spin_glass_example", execute = true, documenter = false)
cd("..")

cd("realspace")
Literate.markdown("cooperative_chain.jl", "../docs"; execute = true, documenter = false)
cd("..")
