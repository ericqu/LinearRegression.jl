using LinearRegression
using Test, DataFrames, CSV
using StatsModels

@testset "White's covariance estimators" begin
    ndf = CSV.read("/Users/eric/Downloads/nerlove.csv", DataFrame)
    f = @formula(log(cost) ~ 1 + log(output) + log(labor) + log(fuel) + log(capital))
    lr2 = regress(f, ndf, cov=[:hc0])
    @test isapprox([1.6887096637865764, 0.03203057017940859, 0.2413635423569017, 0.07416986820706634, 0.3181801843194431], lr2.white_stderrors)
    @test isapprox([-2.088282503859893, 22.490828975090466, 1.807817355217008, 5.750542145660426, -0.6910812225036552], lr2.white_t_values)
    @test isapprox([0.03858351163194079, 2.515671235586196e-48, 0.07278163877804528, 5.358556751845478e-8, 0.4906588080040115], lr2.white_p_values)
end 
# lr1 = regress(f, ndf, cov=[:white])
# 
