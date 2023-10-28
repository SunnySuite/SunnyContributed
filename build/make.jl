using Literate, Dates, Git

root = dirname(@__FILE__)
example_dir = joinpath(root,"..", "Sunny.jl", "examples", "spinw_ports")
save_dir = joinpath(root, "..", "contributed-doc-assets")

# If no arguments given, rebuild all notebooks
examples = filter(name -> split(name, ".")[end] == "jl", readdir(example_dir))

# Build the notebooks
map(examples) do example
    Literate.markdown(joinpath(example_dir, example), save_dir; documenter=false, execute=true)
end

# Sync with github
cd("..")
run(`$(git()) pull`) # Make sure up to date
run(git(["add", save_dir*"/*.md"]))
run(git(["add", save_dir*"/*.png"]))
run(`$(git()) commit -am "Auto-generated docs $(string(Dates.now()))"`)
run(`$(git()) push`)
