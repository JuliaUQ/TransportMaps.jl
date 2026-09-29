# Build Documentation

Run `scripts/build_docs.sh` on Linux/Mac to build docs locally.

Run `scripts/build_docs.ps1` on Windows to build docs locally.

All plotting examples use CairoMakie with
`set_theme!(transportmap_theme())`, provided by the TransportMaps Makie extension.
Figures are at most 600 pixels wide, with a base font size of 18 and regular-weight
axis titles of size 22. Titles are kept where they distinguish panels.
Mathematical labels use `LaTeXStrings` (`L"x_1"`); ordinary text inherits the theme's fonts.

Regenerate the themed q-norm animation with:

```sh
julia --project=docs docs/qnorm_animation.jl
```
