using LinearRegression
using Test, DataFrames
using StatsModels
 
include("test_sweep_operator.jl")
include("test_utilities.jl")
include("test_LinearRegression.jl")
include("test_cooksd.jl")
include("test_lessthanfullrank.jl")
include("test_noint.jl")
include("test_heteroscedasticity.jl")
