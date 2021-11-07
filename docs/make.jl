push!(LOAD_PATH,"../src/")
using Documenter
using StatsModels, DataFrames, VegaLite
using LinearRegression

makedocs(sitename="LinearRegression.jl", modules = [LinearRegression] ,
    pages = Any[
        "Home" => "index.md",
        "Tutorial" => "basic_tutorial.md"])

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
    push_preview = false,
    devbranch = "main",
    devurl = "dev",
)
