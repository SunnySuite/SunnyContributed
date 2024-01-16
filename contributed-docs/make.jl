using Literate, Git, Dates, Tar, CodecZlib

draft = false
root = dirname(@__FILE__)
src_dir = joinpath(root, "src") 
build_dir = joinpath(root, "build")

docnames = if length(ARGS) == 0
    [
        "kappa_tutorial.jl",
        "MgCr2O4-tutorial.jl",
    ]
else
    ARGS
end

# Build the notebooks
map(docnames) do docname
    Literate.markdown(joinpath(src_dir, docname), build_dir; execute=!draft, documenter=false, credit=false)
end

# Make compressed tarbar of build
tar_gz = open(joinpath(root, "build.tar.gz"), write=true)
tar = GzipCompressorStream(tar_gz)
Tar.create(build_dir, tar)
close(tar)


# Sync with github
cd(joinpath(root, ".."))
run(`$(git()) pull`) # Make sure up to date
run(git(["add", joinpath(root, "build.tar.gz")]))
run(git(["add", build_dir*"/*.md"]))
run(git(["add", build_dir*"/*.png"]))
run(`$(git()) commit -am "Auto-build $(string(Dates.now()))"`)
run(`$(git()) push`)