push!(LOAD_PATH,"../src/")
using Documenter, LinearRegression, StatsModels, DataFrames


makedocs(
    sitename="LinearRegression.jl",
)

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
)
