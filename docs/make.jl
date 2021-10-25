push!(LOAD_PATH,"../src/")
using Documenter, LinearRegression, StatsModels, DataFrames, VegaLite

makedocs(sitename="LinearRegression.jl")

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
    push_preview = false,
    devbranch = "main",
    devurl = "dev",
)
