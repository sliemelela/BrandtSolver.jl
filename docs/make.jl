using BrandtSolver
using Documenter

# Make sure to include BrandtSolver in the load path for doctests
DocMeta.setdocmeta!(BrandtSolver, :DocTestSetup, :(using BrandtSolver); recursive=true)

makedocs(;
    modules=[BrandtSolver],
    authors="Sliem el Ela",
    repo="https://github.com/sliemelela/BrandtSolver.jl",
    sitename="BrandtSolver.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sliemelela.github.io/BrandtSolver.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs = :exports,
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/sliemelela/BrandtSolver.jl",
    devbranch="main",
)