push!(LOAD_PATH,"../src/")
using Documenter
using StatsModels, DataFrames, VegaLite
using LinearRegression

makedocs(sitename="LinearRegression.jl", modules = [LinearRegression])

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
    push_preview = false,
    devbranch = "main",
    devurl = "dev",
)
