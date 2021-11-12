push!(LOAD_PATH,"../src/")
using Documenter
using StatsModels, DataFrames, VegaLite
using LinearRegression

makedocs(sitename="LinearRegression.jl", modules = [LinearRegression] ,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => Any[
            "Basic" => "basic_tutorial.md",
            "Multiple regression" => "multi_tutorial.md",
            "Weighted regression" => "weighted_regression_tutorial.md" ]
    ])

deploydocs(
    repo = "github.com/ericqu/LinearRegression.jl.git",
    push_preview = false,
    devbranch = "main",
    devurl = "dev",
)
