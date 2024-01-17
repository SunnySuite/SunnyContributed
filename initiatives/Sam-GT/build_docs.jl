using Literate

Literate.markdown("mgcro-investigation/quadratic_casimirs.jl", "docs"; execute = true, documenter = false)

cd("eigenmodes")
Literate.markdown("examples.jl", "../docs"; name ="eigenmode_viewer_examples", execute = true, documenter = false)
cd("..")

