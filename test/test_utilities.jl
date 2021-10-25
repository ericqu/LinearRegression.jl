include("../src/utilities.jl")

@testset "Plots request massaging" begin
    wanted = "all"
    needed = Set([:fitplot, :residuals_plots, :normal_checks, :cooksd, :leverage, :homoscedasticity])
    @test needed == get_needed_plots(wanted)

    wanted = []
    needed = Set([])
    @test needed == get_needed_plots(wanted)

    wanted = :none
    needed = Set()
    @test needed == get_needed_plots(wanted)

    wanted = Set()
    needed = Set()
    @test needed == get_needed_plots(wanted)

    wanted = ["fitplot", "cooksd", "homoscedasticity"]
    needed = Set([:fitplot, :cooksd, :homoscedasticity])
    @test needed == get_needed_plots(wanted)

    wanted = [:residuals_plots, :normal_checks]
    needed = Set([:residuals_plots, :normal_checks])
    @test needed == get_needed_plots(wanted)

    wanted = :leverage
    needed = Set([:leverage])
    @test needed == get_needed_plots(wanted)

end

@testset "Covariance estimator stats massaging" begin
    wanted = [:white, :white , :bogus , :nw]
    needed = ([:white], [:nw])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = [:hc0, :hc1, :nw]
    needed = ([:hc0, :hc1], [:nw])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = ["White", "HC0" , "nw" , "nothing"]
    needed = ([:white, :hc0], [:nw])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = ["Hc1", "HC0" , "nothing"]
    needed = ([:hc0 , :hc1], [])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = "Nw"
    needed = ([], [:nw])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = :hc0
    needed = ([:hc0], [])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = :hc0
    needed = ([:hc0], [])
    @test needed == get_needed_robust_cov_stats(wanted)
    
    wanted = []
    needed = ([], [])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = :none
    needed = ([], [])
    @test needed == get_needed_robust_cov_stats(wanted)

    wanted = :all
    needed = ([:white, :hc0, :hc1, :hc2, :hc3], [:nw])
    @test needed == get_needed_robust_cov_stats(wanted)

end

@testset "model stats massaging" begin
    wanted = ["default"]
    needed = Set([:coefs, :sse, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    @test needed == get_needed_model_stats(wanted)
    wanted = [:default, :vif]
    needed = Set([:coefs, :sse, :vif, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    @test needed == get_needed_model_stats(wanted)
    wanted = [:default, :diag_ks]
    needed = Set([:coefs, :sse, :diag_ks, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    @test needed == get_needed_model_stats(wanted)
    wanted = [:default, :diag_normality]
    needed = Set([:coefs, :sse, :diag_ks, :diag_ad, :diag_jb, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    @test needed == get_needed_model_stats(wanted)
    wanted = [:default, :diag_heteroskedasticity]
    needed = Set([:coefs, :sse, :diag_white, :diag_bp, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["r2", "rmse"]
    needed = Set([:r2, :rmse, :coefs, :sse, :mse, :sst])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["aic"]
    needed = Set([:coefs, :mse, :sse, :aic])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["Ci"]
    needed = Set([:coefs, :mse, :sse, :ci, :sigma, :stderror, :t_statistic])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["Adjr2"]
    needed = Set([:coefs, :mse, :sse, :r2, :adjr2, :sst])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["stderror"]
    needed = Set([:coefs, :mse, :sse, :stderror, :sigma])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["t_values"]
    needed = Set([:coefs, :mse, :sse, :t_values, :stderror, :sigma])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["p_values"]
    needed = Set([:coefs, :mse, :sse, :p_values, :t_values, :stderror, :sigma])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["ci", "P_values"]
    needed = Set([:coefs, :mse, :sse, :p_values, :t_values, :stderror, :sigma, :ci, :t_statistic])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["none"]
    needed = Set([:coefs, :mse, :sse])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["all"]
    needed = Set([:coefs, :sse, :mse, :sst, :rmse, :aic, :sigma, :t_statistic, :vif, :r2, :adjr2, :stderror, :t_values, :p_values, :ci,
                    :diag_normality, :diag_ks, :diag_ad, :diag_jb, :diag_heteroskedasticity, :diag_white, :diag_bp ])
    @test needed == get_needed_model_stats(wanted)
    wanted = [ ]
    needed = Set([:coefs, :mse, :sse])
    @test needed == get_needed_model_stats(wanted)    
    wanted = [ "bogus"]
    needed = Set([:coefs, :mse, :sse])
    @test needed == get_needed_model_stats(wanted)
    wanted = ["stderror", "Bogus"]
    needed = Set([:coefs, :mse, :sse, :stderror, :sigma])
    @test needed == get_needed_model_stats(wanted)
    wanted = Set([:stderror, :Bogus])
    needed = Set([:coefs, :mse, :sse, :stderror, :sigma])
    @test needed == get_needed_model_stats(wanted)
    wanted = Set()
    needed = Set([:coefs, :mse, :sse])
    @test needed == get_needed_model_stats(wanted)
end


@testset "prediction stats massaging" begin
    wanted = ["none"]
    target_need = Set([:predicted])
    target_present = Set([:predicted])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["all"]
    target_need = Set([:predicted, :residuals, :leverage, :stdp, :stdi, :stdr, :student, :rstudent, :lcli, :ucli, :lclp, :uclp, :press, :cooksd])
    target_present = Set([:predicted, :residuals, :leverage, :stdp, :stdi, :stdr, :student, :rstudent, :lcli, :ucli, :lclp, :uclp, :press, :cooksd])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["CooksD"]
    target_need = Set([:predicted, :residuals, :leverage, :stdp, :stdr, :student, :cooksd])
    target_present = Set([:predicted, :cooksd])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["CooksD", "lcli"]
    target_need = Set([:predicted, :residuals, :leverage, :stdp, :stdr, :student, :cooksd, :stdi, :lcli])
    target_present = Set([:predicted, :cooksd, :lcli])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["CooksD", "Bogus", "lcli"]
    target_need = Set([:predicted, :residuals, :leverage, :stdp, :stdr, :student, :cooksd, :stdi, :lcli])
    target_present = Set([:predicted, :cooksd, :lcli])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["PRESS"]
    target_need = Set([:predicted, :residuals, :leverage, :press])
    target_present = Set([:predicted, :press])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["lclp"]
    target_need = Set([:predicted, :stdp, :leverage, :lclp])
    target_present = Set([:predicted, :lclp])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["uclp"]
    target_need = Set([:predicted, :stdp, :leverage, :uclp])
    target_present = Set([:predicted, :uclp])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["ucli"]
    target_need = Set([:predicted, :stdi, :leverage, :ucli])
    target_present = Set([:predicted, :ucli])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["ucli", "LCLP"]
    target_need = Set([:predicted, :stdi, :leverage, :ucli, :lclp, :stdp])
    target_present = Set([:predicted, :ucli, :lclp])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = ["student"]
    target_need = Set([:predicted, :residuals, :leverage, :stdr, :student])
    target_present = Set([:predicted, :student])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = [:rstudent]
    target_need = Set([:predicted, :residuals, :leverage, :stdr, :student, :rstudent])
    target_present = Set([:predicted, :rstudent])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = [:stdp]
    target_need = Set([:predicted, :leverage, :stdp])
    target_present = Set([:predicted, :stdp])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = [:stdp, :stdi]
    target_need = Set([:predicted, :leverage, :stdp, :stdi])
    target_present = Set([:predicted, :stdp, :stdi])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = [:stdpp, :stdr]
    target_need = Set([:predicted, :leverage, :stdr])
    target_present = Set([:predicted, :stdr])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = []
    target_need = Set([:predicted])
    target_present = Set([:predicted])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

    wanted = Set()
    target_need = Set([:predicted])
    target_present = Set([:predicted])
    need, present = get_prediction_stats(wanted)
    @test target_need == need
    @test target_present == present

end
