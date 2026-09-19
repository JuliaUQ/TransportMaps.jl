
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'

julia --project=docs/ docs/make.jl
