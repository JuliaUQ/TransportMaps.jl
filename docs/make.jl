using Documenter
using DocumenterCitations
using DocumenterVitepress
using Literate
using TransportMaps

# Setup bibliography
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

# Configure Plots to not display interactively
ENV["GKSwstype"] = "100"  # Set GR to non-interactive mode

# Process Literate.jl files
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const OUTPUT_DIR = joinpath(@__DIR__, "src")

# Automatically process all literate files in subdirectories of LITERATE_DIR
if isdir(LITERATE_DIR)
    for subdir in sort(filter(name -> isdir(joinpath(LITERATE_DIR, name)) && !startswith(name, "."), readdir(LITERATE_DIR)))
        srcdir = joinpath(LITERATE_DIR, subdir)
        dest = joinpath(OUTPUT_DIR, subdir)
        mkpath(dest)

        files = sort(
            filter(
                name ->
                !startswith(name, ".") && endswith(name, ".jl"),
                readdir(srcdir)
            )
        )

        for fname in files
            src = joinpath(srcdir, fname)
            Literate.markdown(src, dest; documenter = true, credit = true)
        end
    end
end

makedocs(
    sitename = "TransportMaps.jl",
    authors = "Lukas Fritsch and Jan Grashorn",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/JuliaUQ/TransportMaps.jl",
    ),
    modules = [TransportMaps],
    plugins = [bib],
    repo = Remotes.GitHub("JuliaUQ", "TransportMaps.jl"),
    pages = [
        "Home" => "index.md",
        "Manuals" => [
            "Getting Started" => "Manuals/getting_started.md",
            "Basis Functions" => "Manuals/basis_functions.md",
            "Map Parameterization" => "Manuals/map_parameterization.md",
            "Quadrature Methods" => "Manuals/quadrature_methods.md",
            "Optimization" => "Manuals/optimization.md",
            "Conditional Densities and Samples" => "Manuals/conditional_densities.md",
            "Adaptive Transport Maps" => "Manuals/adaptive_transport_map.md",
        ],
        "Examples" => [
            "Banana: Map from Density" => "Examples/banana_mapfromdensity.md",
            "Banana: Map from Samples" => "Examples/banana_mapfromsamples.md",
            "Banana: Adaptive Transport Map from Samples" => "Examples/banana_adaptive.md",
            "Cubic: Adaptive Transport Map from Density" => "Examples/cubic_adaptive_fromdensity.md",
            "Bayesian Inference: BOD" => "Examples/bod_bayesianinference.md",
        ],
        "API" => [
            "Bases" => "api/bases.md",
            "Rectifiers" => "api/rectifiers.md",
            "Reference and Target Densities" => "api/densities.md",
            "Quadrature" => "api/quadrature.md",
            "Maps" => "api/maps.md",
            "Optimization" => "api/optimization.md",
        ],
        "References" => "references.md",
    ],
    checkdocs = :export,
    doctestfilters = [r"Ptr{0x[0-9a-f]+}"],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaUQ/TransportMaps.jl",
    target = "build",
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
