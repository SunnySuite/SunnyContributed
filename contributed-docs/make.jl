using Literate, Git, Dates

root = dirname(@__FILE__)
contrib_docs_dir = joinpath(root, "src") 
build_dir = joinpath(root, "build")

# If no arguments given, rebuild all notebooks
docnames = if length(ARGS) == 0
    filter(name -> split(name, ".")[end] == "jl", readdir(contrib_docs_dir))
else
    ARGS
end

# Build the notebooks
map(docnames) do docname
    Literate.markdown(joinpath(contrib_docs_dir, docname), build_dir; execute=true, documenter=false, credit=false)
end

# Sync with github
cd("..")
run(`$(git()) pull`) # Make sure up to date
run(git(["add", build_dir*"/*.md"]))
run(git(["add", build_dir*"/*.png"]))
run(`$(git()) commit -am "Auto-build $(string(Dates.now()))"`)
run(`$(git()) push`)