
@testset "from glm" begin 
    fdf = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9], OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, req_stats="all")
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.ADJR2, 0.998808343507198)
    @test isapprox(lm1.AIC, -55.43694668654871) # using the SAS formula rather than the Julia-Statsmodel-GLM
    @test isapprox(lm1.stderrors,  [0.007833679251299831, 0.013534536322659505]) 
    @test isapprox(lm1.t_values, [0.6492114525712505, 64.74442074669666])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.ci_low, [-0.016664066127247305, 0.8387078171615459])
    @test isapprox(lm1.ci_up, [0.02683549469867456, 0.9138636114098853])
    @test isapprox(lm1.t_statistic, 2.7764451051977934)
    @test isapprox(lm1.VIF, [0.,  1.])
end

@testset "from glm regresspredict" begin 
    fdf = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9], OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=[:default, :aic, :vif])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.ADJR2, 0.998808343507198)
    @test isapprox(lm1.AIC, -55.43694668654871) # using the SAS formula rather than the Julia-Statsmodel-GLM
    @test isapprox(lm1.stderrors,  [0.007833679251299831, 0.013534536322659505]) 
    @test isapprox(lm1.t_values, [0.6492114525712505, 64.74442074669666])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.ci_low, [-0.016664066127247305, 0.8387078171615459])
    @test isapprox(lm1.ci_up, [0.02683549469867456, 0.9138636114098853])
    @test isapprox(lm1.t_statistic, 2.7764451051977934)
    @test isapprox(lm1.VIF, [0.,  1.])

    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=["none"])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 

    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=["r2", "P_values"])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])

    df = DataFrame(y=[1., 3., 3., 2., 2., 1.], x1=[1., 2., 3., 1., 2., 3.], x2=[1., 1., 1., -1., -1., -1.])
    lm2 = regress(@formula(y ~ 1 + x1 + x2), df, req_stats=["r2"])
    @test [1.5, 0.25, 0.3333333333333333] == lm2.coefs
    @test 0.22916666666666663 == lm2.R2

end

@testset "weighted regression" begin 
    tw = [
        2.3  7.4  0.058 
        3.0  7.6  0.073 
        2.9  8.2  0.114 
        4.8  9.0  0.144 
        1.3 10.4  0.151 
        3.6 11.7  0.119 
        2.3 11.7  0.119 
        4.6 11.8  0.114 
        3.0 12.4  0.073 
        5.4 12.9  0.035 
        6.4 14.0  0
    ]
    
    df = DataFrame(tw, [:y,:x,:w])
    f = @formula(y ~ x)
    lm = regress(f, df, weights="w")

    @test isapprox([2.328237176867885, 0.08535712911515277], lm.coefs)
    @test isapprox(0.014954934572439349, lm.R2)
    @test isapprox(-0.10817569860600562, lm.ADJR2)
    @test isapprox([2.551864989438224, 0.24492357920520605], lm.stderrors)
    @test isapprox([0.3882424860021164, 0.7364546437428148], lm.p_values)

end

@testset "predictions statistics" begin
    t_carb = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
    t_optden = [0.086, 0.269, 0.446, 0.538, 0.626, 0.782]
    t_leverage = [0.5918367346938774, 0.2816326530612245, 0.1673469387755102, 0.1836734693877552, 0.24897959183673485, 0.5265306122448984]
    t_predicted = [0.09271428571428536, 0.26797142857142836, 0.44322857142857136, 0.5308571428571429, 0.6184857142857143, 0.7937428571428573]
    t_residuals = [-0.006714285714285367, 0.0010285714285716563, 0.002771428571428647, 0.0071428571428571175, 0.007514285714285696, -0.011742857142857277]
    t_stdp = [0.0066535244611503905, 0.00458978457544483, 0.0035380151243903845, 0.003706585424647827, 0.004315515434962329, 0.006275706318490394]
    t_stdi = [0.01091189203370195, 0.009791124677432761, 0.009344386069745653, 0.009409504530540022, 0.009665592246181276, 0.010685714285717253]
    t_stdr = [0.00552545131594831, 0.00733033952495041, 0.007891923021648551, 0.007814168189246363, 0.0074950868260910365, 0.005951093194035971]
    t_student = [-1.2151560714878948, 0.1403170242074994, 0.3511727830880084, 0.9140905301586566, 1.0025615297914663, -1.9732268946192373]
    t_rstudent = [-1.3249515797718379, 0.12181828539462737, 0.3089239861926457, 0.8900235681358146, 1.003419757326401, -10.478921731163984]
    t_lcli = [0.06241801648886679, 0.24078690838638886, 0.4172843964641476, 0.5047321700609886, 0.5916497280049665, 0.7640745580187355]
    t_ucli = [0.12301055493970393, 0.2951559487564679, 0.4691727463929951, 0.5569821156532972, 0.6453217005664621, 0.8234111562669791]
    t_lclp = [0.07424114029181057, 0.2552281436530222, 0.43340546665434193, 0.520566011897882, 0.6065039225799076, 0.7763187030532258]
    t_uclp = [0.11118743113676015, 0.2807147134898345, 0.4530516762028008, 0.5411482738164038, 0.630467505991521, 0.8111670112324888]
    t_press = [-0.016449999999999146, 0.001431818181818499, 0.0033284313725491102, 0.00874999999999997, 0.010005434782608673, -0.02480172413793134]
    t_cooksd = [1.0705381016035724, 0.0038594654616162225, 0.012392684477580572, 0.09400066844914513, 0.16661116000566892, 2.1649894168822432]


    fdf = DataFrame([[0.1,0.3,0.5,0.6,0.7,0.9],[0.086,0.269,0.446,0.538,0.626,0.782]], [:Carb, :OptDen])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf)
    results = predict_in_sample(lm1, fdf, α=0.05, req_stats=["all"])
    @test isapprox(t_leverage, results.leverage)
    @test isapprox(t_predicted, results.predicted) 
    @test isapprox(t_residuals, results.residuals)
    @test isapprox(t_stdp, results.stdp)
    @test isapprox(t_stdi, results.stdi)
    @test isapprox(t_stdr, results.stdr)
    @test isapprox(t_student, results.student)
    @test isapprox(t_rstudent, results.rstudent)
    @test isapprox(t_lcli, results.lcli)
    @test isapprox(t_ucli, results.ucli)
    @test isapprox(t_lclp, results.lclp)
    @test isapprox(t_uclp, results.uclp)
    @test isapprox(t_press, results.press)
    @test isapprox(t_cooksd, results.cooksd)

    results = predict_in_sample(lm1, fdf, α=0.05, req_stats=["none"])
    @test isapprox(t_predicted, results.predicted) 
    @test_throws ArgumentError("column name :leverage not found in the data frame") t_leverage == results.leverage 

    results = predict_out_of_sample(lm1, fdf)
    @test isapprox(t_predicted, results.predicted)
    @test_throws ArgumentError("column name :leverage not found in the data frame") t_leverage == results.leverage 

end

