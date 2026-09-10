using Literate

# Define paths
literate_dir = joinpath(@__DIR__, "literate", "Examples")
demo_dir = joinpath(@__DIR__, "..", "demo")

# Ensure demo directory exists
mkpath(demo_dir)

# Get all .jl files from literate directory
literate_files = filter(f -> endswith(f, ".jl"), readdir(literate_dir))

# Process each literate file
for file in literate_files
    input_file = joinpath(literate_dir, file)

    # Generate output filename - keep the same name for demos
    output_name = file

    println("Processing: $file -> $output_name")

    strip_hide_markers(content) =
        replace(content, r"[ \t]+# hide(?=\r?$)"m => "")

    Literate.script(
        input_file,
        demo_dir;
        name = splitext(output_name)[1],
        execute = false,
        documenter = false,
        credit = true,
        preprocess = strip_hide_markers,
    )

end
