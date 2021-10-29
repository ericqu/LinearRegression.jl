push!(LOAD_PATH,"../src/")
using Documenter, StatsModels, DataFrames, VegaLite
using Distributions, HypothesisTests
using LinearRegression

makedocs(sitename="LinearRegression.jl")

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
    push_preview = false,
    devbranch = "main",
    devurl = "dev",
)
