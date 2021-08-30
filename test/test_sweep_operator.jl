include("../src/sweep_operator.jl")

@testset "Sweep Operator Correctness" begin
    correct_result = [1.1666666666666667 -0.5 -0.0 1.5; -0.5 0.25 -0.0 0.25; 0.0 -0.0 0.16666666666666666 0.3333333333333333; -1.5 -0.25 -0.3333333333333333 3.0833333333333335]
    M  = [  1.   1.   1.   1.
            1.   2.   1.   3.
            1.   3.   1.   3.
            1.   1.  -1.   2.
            1.   2.  -1.   2.
            1.   3.  -1.   1. ]
    M0 = M' * M 
    sweep_op!(M0, 1:3)

    @test isapprox(correct_result, M0)
end