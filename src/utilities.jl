get_all_plots_types() = Set([:fit, :residuals, :normal_checks, :cooksd, :leverage, :homoscedasticity])
get_needed_plots(s::String) = return get_needed_plots([s])
get_needed_plots(s::Symbol) = return get_needed_plots([s])
get_needed_plots(s::Vector{String}) = return get_needed_plots(Symbol.(lowercase.(s)))
get_needed_plots(::Vector{Any}) = return get_needed_plots([:none])
get_needed_plots(::Set{Any}) = return get_needed_plots([:none])
get_needed_plots(s::Set{Symbol}) = return get_needed_plots(collect(s))

function get_needed_plots(p::Vector{Symbol})

    needed_plots = Set{Symbol}()
    length(p) == 0 && return needed_plots
    :none in p && return needed_plots

    if :all in p
        return get_all_plots_types()
    end

    :fit in p && push!(needed_plots, :fit)
    :residuals in p && push!(needed_plots, :residuals)
    :normal_checks in p && push!(needed_plots, :normal_checks)
    :cooksd in p && push!(needed_plots, :cooksd)
    :leverage in p && push!(needed_plots, :leverage)
    :homoscedasticity in p && push!(needed_plots, :homoscedasticity)
    return needed_plots

end

"""  
    get_robust_cov_stats()

    Return all robust covariance estimators.
"""
get_all_robust_cov_stats() = Set([:white, :nw, :hc0, :hc1, :hc2, :hc3])
get_needed_robust_cov_stats(s::String) = return get_needed_robust_cov_stats([s])
get_needed_robust_cov_stats(s::Symbol) = return get_needed_robust_cov_stats([s])
get_needed_robust_cov_stats(s::Vector{String}) = return get_needed_robust_cov_stats(Symbol.(lowercase.(s)))
get_needed_robust_cov_stats(::Vector{Any}) = return get_needed_robust_cov_stats([:none])
get_needed_robust_cov_stats(::Set{Any}) = return get_needed_robust_cov_stats(Set([:none])) 
get_needed_robust_cov_stats(s::Set{Symbol}) = return get_needed_robust_cov_stats(collect(s)) 

function get_needed_robust_cov_stats(s::Vector{Symbol})
    
    needed_white = Vector{Symbol}()
    needed_hac = Vector{Symbol}()

    length(s) == 0 && return (needed_white, needed_hac)
    :none in s && return (needed_white, needed_hac)
    if :all in s 
        s = collect(get_all_robust_cov_stats())
    end
    
    :white in s && push!(needed_white, :white)
    :hc0 in s && push!(needed_white, :hc0)
    :hc1 in s && push!(needed_white, :hc1)
    :hc2 in s && push!(needed_white, :hc2)
    :hc3 in s && push!(needed_white, :hc3)

    :nw in s && push!(needed_hac, :nw)

    return (needed_white, needed_hac)

end

"""
    get_all_model_stats()

    Returns all statistics availble for the fitted model.
"""
get_all_model_stats() = Set([:coefs, :sse, :mse, :sst, :rmse, :aic, :sigma, :t_statistic, :vif, :r2, :adjr2, :stderror, :t_values, :p_values, :ci,
                            :diag_normality, :diag_ks, :diag_ad, :diag_jb, :diag_heteroskedasticity, :diag_white, :diag_bp, :press ])

get_needed_model_stats(req_stats::String) = return get_needed_model_stats([req_stats])
get_needed_model_stats(req_stats::Symbol) = return get_needed_model_stats(Set([req_stats]))
get_needed_model_stats(req_stats::Vector{String}) = return get_needed_model_stats(Symbol.(lowercase.(req_stats)))
get_needed_model_stats(::Vector{Any}) = return get_needed_model_stats([:none])
get_needed_model_stats(::Set{Any}) = return get_needed_model_stats(Set([:none]))
get_needed_model_stats(req_stats::Set{Symbol}) = get_needed_model_stats(collect(req_stats))

"""
    get_needed_model_stats(req_stats::Vector{Symbol})

    return the list of needed statistics given the list of statistics about the model the caller wants.
"""
function get_needed_model_stats(req_stats::Vector{Symbol})
    needed = Set([:coefs, :sse, :mse])
    default = Set([:coefs, :sse, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci])
    full = get_all_model_stats()
    unique!(req_stats)

    length(req_stats) == 0 && return needed
    :none in req_stats && return needed
    :all in req_stats && return full

    :default in req_stats && union!(needed, default)

    :sst in req_stats && push!(needed, :sst)
    :press in req_stats && push!(needed, :press)
    :rmse in req_stats && push!(needed, :rmse)
    :aic in req_stats && push!(needed, :aic)
    :sigma in req_stats && push!(needed, :sigma)
    :t_statistic in req_stats && push!(needed, :t_statistic)
    :vif in req_stats && push!(needed, :vif)
    :diag_ks in req_stats && push!(needed, :diag_ks)
    :diag_ad in req_stats && push!(needed, :diag_ad)
    :diag_jb in req_stats && push!(needed, :diag_jb)
    :diag_white in req_stats && push!(needed, :diag_white)
    :diag_bp in req_stats && push!(needed, :diag_bp)

    if :diag_normality in req_stats
        push!(needed, :diag_ks)
        push!(needed, :diag_ad)
        push!(needed, :diag_jb)
    end
    if :diag_heteroskedasticity in req_stats
        push!(needed, :diag_white)
        push!(needed, :diag_bp)
    end
    if :r2 in req_stats
        push!(needed, :sst)
        push!(needed, :r2)
    end
    if :adjr2 in req_stats
        push!(needed, :sst)
        push!(needed, :r2)
        push!(needed, :adjr2)
    end
    if :stderror in req_stats
        push!(needed, :sigma)
        push!(needed, :stderror)
    end
    if :t_values in req_stats
        push!(needed, :sigma)
        push!(needed, :stderror)
        push!(needed, :t_values)
    end
    if :p_values in req_stats
        push!(needed, :sigma)
        push!(needed, :stderror)
        push!(needed, :t_values)
        push!(needed, :p_values)
    end
    if :ci in req_stats
        push!(needed, :sigma)
        push!(needed, :stderror)
        push!(needed, :t_statistic)
        push!(needed, :ci)
    end

    return needed
end

"""
    get_all_prediction_stats()

    get all the available statistics about the values predicted by a fitted model
"""
get_all_prediction_stats() = Set([:predicted, :residuals, :leverage, :stdp, :stdi, :stdr, :student, :rstudent, :lcli, :ucli, :lclp, :uclp, :press, :cooksd])

get_prediction_stats(req_stats::String) = return get_prediction_stats([req_stats]) 
get_prediction_stats(req_stats::Vector{String}) = return get_prediction_stats(Symbol.(lowercase.(req_stats))) 
get_prediction_stats(::Vector{Any}) = return get_prediction_stats([:none])
get_prediction_stats(::Set{Any}) = return get_prediction_stats(Set([:none]))
get_prediction_stats(req_stats::Set{Symbol}) = return get_prediction_stats(collect(req_stats))

"""
    function get_prediction_stats(req_stats::Vector{Symbol})

    return the list of needed statistics and the statistics that need to be presentd given the list of statistics about the predictions the caller wants.
"""
function get_prediction_stats(req_stats::Vector{Symbol})
    needed = Set([:predicted])
    full = get_all_prediction_stats()
    present = Set([:predicted])
    unique!(req_stats)

    length(req_stats) == 0 && return needed, present
    :none in req_stats && return needed, present
    :all in req_stats && return full, full

    :leverage in req_stats && push!(present, :leverage)
    :residuals in req_stats && push!(present, :residuals)
    if :stdp in req_stats
        push!(needed, :leverage)
        push!(present, :stdp)
    end
    if :stdi in req_stats
        push!(needed, :leverage)
        push!(present, :stdi)
    end
    if :stdr in req_stats
        push!(needed, :leverage)
        push!(present, :stdr)
    end
    if :student in req_stats
        push!(needed, :leverage)
        push!(needed, :residuals)
        push!(needed, :stdr)
        push!(present, :student)
    end
    if :rstudent in req_stats
        push!(needed, :leverage)
        push!(needed, :residuals)
        push!(needed, :stdr)
        push!(needed, :student)
        push!(present, :rstudent)
    end
    if :lcli in req_stats
        push!(needed, :leverage)
        push!(needed, :stdi)
        push!(present, :lcli)
    end
    if :ucli in req_stats
        push!(needed, :leverage)
        push!(needed, :stdi)
        push!(present, :ucli)
    end
    if :lclp in req_stats
        push!(needed, :leverage)
        push!(needed, :stdp)
        push!(present, :lclp)
    end
    if :uclp in req_stats
        push!(needed, :leverage)
        push!(needed, :stdp)
        push!(present, :uclp)
    end
    if :press in req_stats
        push!(needed, :residuals)
        push!(needed, :leverage)
        push!(present, :press)
    end
    if :cooksd in req_stats
        push!(needed, :leverage)
        push!(needed, :stdp)
        push!(needed, :stdr)
        push!(needed, :residuals)
        push!(needed, :student)
        push!(present, :cooksd)
    end

    union!(needed, present)

    return needed, present

end

"""
    function encapsulate_string(s)

    (internal) Only used to encapsulate a string into an array.

    used exclusively to handle the function ```StatsBase.coefnames``` which sometime return an array or when there is only one element the element alone. 
"""
function encapsulate_string(s::String)
    return [s]
end

"""
    function encapsulate_string(v)

    (internal) Only used to encapsulate a string into an array.

    used exclusively to handle the function ```StatsBase.coefnames``` which sometime return an array or when there is only one element the element alone. 
"""
function encapsulate_string(v::Vector{String})
    return v
end

import Printf
"""
    macro gprintf(fmt::String)
    
    (internal) used to format with %g

    Taken from message published by user o314 at https://discourse.julialang.org/t/printf-with-variable-format-string/3805/6
"""
macro gprintf(fmt::String)
    :((io::IO, arg) -> Printf.@printf(io, $fmt, arg))
end

"""
    function fmt_pad(s::String, value, pad=0)
    (internal) helper to format and pad string for results display
"""
function fmt_pad(s::String, value, pad=0)
    fmt = @gprintf("%g")
    return rpad(s * sprint(fmt, value), pad)
end

using NamedArrays
"""
    function my_namedarray_print([io::IO = stdout], n::NamedArray)

    (internal) Print the NamedArray without the type annotation (on the first line).
"""
function my_namedarray_print(io::IO, n)
    tmpio = IOBuffer()
    show(tmpio, n)
    println(io, split(String(take!(tmpio)), "\n", limit=2)[2])
end
my_namedarray_print(n::NamedArray) = my_namedarray_print(stdout::IO, n)

"""
    function helper_print_table(io, title, stats::Vector, stats_name::Vector, updformula)

    (Internal) Convenience function to display a table of statistics to the user.
"""
function helper_print_table(io::IO, title, stats::Vector, stats_name::Vector, updformula)
    println(io, "\n$title")
    todelete = [i for (i, v) in enumerate(stats) if isnothing(v)]
    deleteat!(stats, todelete)
    deleteat!(stats_name, todelete)
    m_all_stats = reduce(hcat, stats)
    if m_all_stats isa Vector
        m_all_stats = reshape(m_all_stats, length(m_all_stats), 1)
    end
    na = NamedArray(m_all_stats)
    setnames!(na, encapsulate_string(string.(StatsBase.coefnames(updformula.rhs))), 1)
    setnames!(na, encapsulate_string(string.(stats_name)), 2)
    setdimnames!(na, ("Terms", "Stats"))
    my_namedarray_print(io, na)
end


function present_breusch_pagan_test(X, residuals, α)
    bpt = HypothesisTests.BreuschPaganTest(X, residuals)
    pval = pvalue(bpt)
    alpha_value= round((1 - α)*100, digits=3)
    topresent = string("Breush-Pagan Test (heteroskedasticity of residuals):\n  T*R² statistic: $(round(bpt.lm, sigdigits=6))    degrees of freedom: $(round(bpt.dof, digits=6))    p-value: $(round(pval, digits=6))\n")
    if pval > α
        topresent *= "  with $(alpha_value)% confidence: fail to reject null hyposthesis.\n"
    else 
        topresent *= "  with $(alpha_value)% confidence: reject null hyposthesis.\n"
    end
    return topresent
end

function present_white_test(X, residuals, α)
    bpt = HypothesisTests.WhiteTest(X, residuals)
    pval = pvalue(bpt)
    alpha_value= round((1 - α)*100, digits=3)
    topresent = string("White Test (heteroskedasticity of residuals):\n  T*R² statistic: $(round(bpt.lm, sigdigits=6))    degrees of freedom: $(round(bpt.dof, digits=6))    p-value: $(round(pval, digits=6))\n")
    if pval > α
        topresent *= "  with $(alpha_value)% confidence: fail to reject null hyposthesis.\n"
    else 
        topresent *= "  with $(alpha_value)% confidence: reject null hyposthesis.\n"
    end
    return topresent
end

function present_kolmogorov_smirnov_test(residuals, α)
    fitted_residuals = fit(Normal, residuals)
    kst = HypothesisTests.ApproximateOneSampleKSTest(residuals, fitted_residuals)
    pval = pvalue(kst)
    KS_stat = sqrt(kst.n)*kst.δ
    alpha_value= round((1 - α)*100, digits=3)
    topresent = string("Kolmogorov-Smirnov test (Normality of residuals):\n  KS statistic: $(round(KS_stat, sigdigits=6))    observations: $(kst.n)    p-value: $(round(pval, digits=6))\n")
    if pval > α
        topresent *= "  with $(alpha_value)% confidence: fail to reject null hyposthesis.\n"
    else 
        topresent *= "  with $(alpha_value)% confidence: reject null hyposthesis.\n"
    end
end

function present_anderson_darling_test(residuals, α)
    fitted_residuals = fit(Normal, residuals)
    adt = HypothesisTests.OneSampleADTest(residuals, fitted_residuals)
    pval = pvalue(adt)
    alpha_value= round((1 - α)*100, digits=3)
    topresent = string("Anderson–Darling test (Normality of residuals):\n  A² statistic: $(round(adt.A², digits=6))    observations: $(adt.n)    p-value: $(round(pval, digits=6))\n")
    if pval > α
        topresent *= "  with $(alpha_value)% confidence: fail to reject null hyposthesis.\n"
    else 
        topresent *= "  with $(alpha_value)% confidence: reject null hyposthesis.\n"
    end
end

function present_jarque_bera_test(residuals, α)
    jbt = HypothesisTests.JarqueBeraTest(residuals)
    pval = pvalue(jbt)
    alpha_value= round((1 - α)*100, digits=3)
    topresent = string("Jarque-Bera test (Normality of residuals):\n  JB statistic: $(round(jbt.JB, digits=6))    observations: $(jbt.n)    p-value: $(round(pval, digits=6))\n")
    if pval > α
        topresent *= "  with $(alpha_value)% confidence: fail to reject null hyposthesis.\n"
    else 
        topresent *= "  with $(alpha_value)% confidence: reject null hyposthesis.\n"
    end
end

# function warn_sigma(lm, stat)
#     if length(lm.white_types) + length(lm.hac_types) > 0
#         println(io, "The $(stat) statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
#     end
# end

function warn_sigma(lm, stat)
    warn_sigma(lm.white_types, lm.hac_types , stat)
end

function warn_sigma(white_needed, hac_needed, stat)
    if length(white_needed) > 0 || length(hac_needed) > 0 
        println(io, "The $(stat) statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
    end
end

function real_sqrt(x)
    return @. real(sqrt(complex(x, 0)))
end

isnotintercept(t::AbstractTerm) = t isa InterceptTerm ? false : true
iscontinuousterm(t::AbstractTerm) = t isa ContinuousTerm ? true : false
iscategorical(t::AbstractTerm) = t isa CategoricalTerm ? true : false

function check_cardinality(df::AbstractDataFrame, f, verbose=false)
    cate_terms = [a.sym for a in filter(iscategorical, terms(f.rhs))]
    if length(cate_terms) > 0 
        freqt = freqtable(df, cate_terms...)
        if count(i -> (i == 0), freqt) > 0 
            println("At least one group of categories have no observation. Use frequency tables to identify which one(s).")
            println(my_namedarray_print(freqt))
        elseif verbose == true
            println("No issue identified.")
        end
    end
end