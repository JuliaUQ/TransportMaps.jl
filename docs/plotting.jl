module DocPlots

using CairoMakie
using TransportMaps

export save_docplot, themed_images

const VARIANTS = ("light", "dark")

function plot_theme(variant)
    variant == "light" && return transportmap_theme()
    variant == "dark" && return transportmap_theme(;
        backgroundcolor = :transparent, foregroundcolor = :white,
    )
    throw(ArgumentError("unknown documentation plot theme: $variant"))
end

function themed_path(path, variant)
    stem, extension = splitext(path)
    return "$stem-$variant$extension"
end

# Only figure construction is repeated. Keep fitting and sampling outside the builder.
function save_docplot(builder, path)
    for variant in VARIANTS
        with_theme(plot_theme(variant)) do
            save(themed_path(path, variant), builder())
        end
    end
    return nothing
end

# Expand image links only for Markdown; the Literate sources and demos stay readable.
function themed_images(source)
    return replace(
        source, r"(?m)^# !\[([^\n]*)\]\(([^\n)]+\.(?:svg|png|gif))\)$" => link -> begin
            parts = match(r"^# !\[([^\n]*)\]\(([^\n)]+)\)$", link)
            alt, path = parts.captures
            alt = replace(alt, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
            startswith(path, ".") || (path = "./$path")
            light = themed_path(path, "light")
            dark = themed_path(path, "dark")
            # Raw HTML keeps the CSS classes intact through Documenter's Markdown pass.
            """
            # ```@raw html
            # <img class="doc-plot-light" src="$light" alt="$alt">
            # <img class="doc-plot-dark" src="$dark" alt="$alt">
            # ```"""
        end
    )
end

end
