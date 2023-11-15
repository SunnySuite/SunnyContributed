using Literate, Git, Dates

draft = false
root = dirname(@__FILE__)
src_dir = joinpath(root, "src") 
build_dir = joinpath(root, "build")

docnames = if length(ARGS) == 0
    [
        "renormalization_tutorial.jl",
        "MgCr2O4-tutorial.jl",
    ]
else
    ARGS
end

# Build the notebooks
map(docnames) do docname
    Literate.markdown(joinpath(src_dir, docname), build_dir; execute=!draft, documenter=false, credit=false)
end

# Sync with github
cd(joinpath(root, ".."))
run(`$(git()) pull`) # Make sure up to date
run(git(["add", build_dir*"/*.md"]))
run(git(["add", build_dir*"/*.png"]))
run(`$(git()) commit -am "Auto-build $(string(Dates.now()))"`)
run(`$(git()) push`)