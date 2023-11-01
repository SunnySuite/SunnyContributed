using Literate, Git, Dates

draft = false
root = dirname(@__FILE__)
src_dir = joinpath(root, "src") 
build_dir = joinpath(root, "build")

# If no arguments given, rebuild all notebooks
docnames = if length(ARGS) == 0
    filter(name -> split(name, ".")[end] == "jl", readdir(src_dir))
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