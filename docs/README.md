# Build Documentation

Run `scripts/build_docs.sh` on Linux/Mac to build docs locally.

Run `scripts/build_docs.ps1` on Windows to build docs locally.

All plotting examples use CairoMakie and the theme provided by the TransportMaps
Makie extension. Documentation figures automatically follow VitePress's theme:
the original light ggplot2 style (white figure, gray axis panels, and dark text)
in light mode, and transparent backgrounds with white text in dark mode.
The animation has matching light and dark versions too.


Manuals show the API calls and configuration needed for each topic. Mark incidental
figure assembly, plotting grids, and synthetic-data helpers with `# hide` so they
still execute without expanding the displayed code. Keep plotting recipes visible
in the plotting manual and the complete first workflow in Getting Started.

`docs/plotting.jl` provides `save_docplot`, which renders each figure in both themes.
Fitting and sample generation stay outside its builder, so they run once. Literate's
`#md` and `# hide` markers keep the theme/export plumbing out of the displayed code
and the demo scripts. During Markdown generation, `themed_images` expands each image
link into a light/dark pair. `docs/src/.vitepress/theme/overrides.css` selects the
version that matches the site's theme toggle.

Generate the demo scripts with `julia --project=docs docs/literate_demo.jl`. They use
`set_theme!(transportmap_theme())`, the original light ggplot2 style with white
figure backgrounds and gray axis panels. Use each plot's `color` keyword for data
colors, and `theme(:textcolor)` for reference lines that follow the foreground.

Figures are at most 600 pixels wide, with a base font size of 18 and regular-weight
axis titles of size 22. Titles are kept where they distinguish panels.
Mathematical labels use `LaTeXStrings` (`L"x_1"`); ordinary text inherits the theme's fonts.

Regenerate the themed q-norm animation with:

```sh
julia --project=docs docs/qnorm_animation.jl
```

The animation uses lossless animated PNG (APNG) to preserve antialiased text and
partially transparent grid lines. It renders at 1200 × 1200 pixels and displays
at up to 600 × 600 pixels for sharpness on high-density screens. Each of the ten
frames lasts one second, and the animation loops continuously.
