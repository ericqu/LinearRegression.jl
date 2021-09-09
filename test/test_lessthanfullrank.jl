using StatsBase: isapprox
# include("../src/LinearRegression.jl")

@testset "Less than full rank regression behaviour" begin
    # The package does not remove collinear terms (when there is a less than full rank matrix).
    # This test only checks that the behaviour is similar to other software package (such as R).
    
    data = CSV.read("../test/Car-Training.csv", DataFrame)

    clm = regress(@formula( Price ~ 1 + Year + Mileage ), data)

    @test isapprox([-2.1875899220039356e6, 1096.1497198779816, -0.023836779513595294] , clm.coefs)
    @test isapprox([352441.774385393, 175.28249088168033, 0.009841450356317822] , clm.stderrors)
    @test isapprox([-6.2069541155238275, 6.2536178848457125, -2.4220799425455644], clm.t_values)
    @test isapprox(3, clm.p)
    @test isapprox(1.697508841124793e6, clm.MSE)
    @test true == clm.intercept
    @test isapprox(0.4643179971701117, clm.R2)
    @test isapprox(0.4540164201926138, clm.ADJR2)
    @test isapprox(1302.884814987416, clm.RMSE)
    @test isapprox([1.1199586251463344e-8, 9.016454363548383e-9, 0.017161942174616873], clm.p_values)

end