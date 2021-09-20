module LinearRegression

using NamedArrays:length
using LinearAlgebra:length
using Distributions:length
export regress, predict_and_stats

using Base: Tuple, Int64
using StatsBase:eltype, isapprox, length, coefnames, push!
using Distributions
using Printf, NamedArrays
using StatsBase
using StatsModels, DataFrames

include("sweep_operator.jl")
include("utilities.jl")
include("newey_west.jl")

struct linRegRes 
    extended_inverse::Matrix                # Store the extended inverse matrix 
    coefs::Union{Nothing,Vector}            # Store the coefficients of the fitted model
    white_types::Union{Nothing,Vector}      # Store the type of White's covariance estimator(s) used
    hac_types::Union{Nothing,Vector}        # Store the type of White's covariance estimator(s) used
    stderrors::Union{Nothing,Vector}        # Store the standard errors for the fitted model
    white_stderrors::Union{Nothing,Vector}  # Store the standard errors modified for the White's covariance estimator
    hac_stderrors::Union{Nothing,Vector}    # Store the standard errors modified for the Newey-West covariance estimator
    t_values::Union{Nothing,Vector}         # Store the t values for the fitted model
    white_t_values::Union{Nothing,Vector}   # Store the t values modified for the White's covariance estimator
    hac_t_values::Union{Nothing,Vector}     # Store the t values modified for the Newey-West covariance estimator
    p::Int64                                # Store the number of parameters (including the intercept as a parameter)
    MSE::Union{Nothing,Float64}             # Store the Mean squared error for the fitted model
    intercept::Bool                         # Indicate if the model has an intercept
    R2::Union{Nothing,Float64}              # Store the R-squared value for the fitted model
    ADJR2::Union{Nothing,Float64}           # Store the adjusted R-squared value for the fitted model
    RMSE::Union{Nothing,Float64}            # Store the Root mean square error for the fitted model
    AIC::Union{Nothing,Float64}             # Store the Akaike information criterion for the fitted model
    σ̂²::Union{Nothing,Float64}              # Store the σ̂² for the fitted model
    p_values::Union{Nothing,Vector}         # Store the p values for the fitted model
    white_p_values::Union{Nothing,Vector}   # Store the p values modified for the White's covariance estimator
    hac_p_values::Union{Nothing,Vector}     # Store the p values modified for the Newey-West covariance estimator
    ci_up::Union{Nothing,Vector}            # Store the upper values confidence interval of the coefficients 
    ci_low::Union{Nothing,Vector}           # Store the lower values confidence interval of the coefficients 
    white_ci_up::Union{Nothing,Vector}      # Store the upper values confidence interval of the coefficients for White covariance estimators 
    white_ci_low::Union{Nothing,Vector}     # Store the upper values confidence interval of the coefficients for White covariance estimators 
    hac_ci_up::Union{Nothing,Vector}        # Store the upper values confidence interval of the coefficients for Newey-West covariance estimators
    hac_ci_low::Union{Nothing,Vector}       # Store the upper values confidence interval of the coefficients for Newey-West covariance estimators
    observations                            # Store the number of observations used in the model
    t_statistic::Union{Nothing,Float64}     # Store the t statistic
    VIF::Union{Nothing,Vector}              # Store the Varince inflation factor
    modelformula                            # Store the model formula
    dataschema                              # Store the dataschema
    updformula                              # Store the updated model formula (after the dataschema has been applied)
    alpha                                   # Store the alpha used to compute the confidence interval of the coefficients
end

"""
    function Base.show(io::IO, lr::linRegRes) 

    Display information about the fitted model
"""
function Base.show(io::IO, lr::linRegRes) 
    println(io, "Model definition:\t", lr.modelformula)
    println(io, "Used observations:\t", lr.observations)
    println(io, "Model statistics:")
    # Display stats when available
    if !isnothing(lr.R2) && !isnothing(lr.ADJR2)
        @printf(io, "  R²: %g\t\t\tAdjusted R²: %g\n", lr.R2, lr.ADJR2)
    elseif !isnothing(lr.R2) 
        @printf(io, "  R²: %g\n", lr.R2)
    end

    if !isnothing(lr.MSE) && !isnothing(lr.RMSE)
        @printf(io, "  MSE: %g\t\t\tRMSE: %g\n", lr.MSE, lr.RMSE)
    elseif !isnothing(lr.MSE)
        @printf(io, "  MSE: %g\n", lr.MSE)
    end

    if length(lr.white_types) + length(lr.hac_types) == 0
        if !isnothing(lr.σ̂²) && !isnothing(lr.AIC)
            @printf(io, "  σ̂²: %g\t\t\tAIC: %g\n", lr.σ̂², lr.AIC)
        elseif !isnothing(lr.σ̂²)
            @printf(io, "  σ̂²: %g\n", lr.σ̂²)
        elseif !isnothing(lr.AIC)
            @printf(io, "  AIC: %g\n", lr.AIC)
        end
    end
    
    if !isnothing(lr.ci_low) || !isnothing(lr.ci_up)
        @printf(io, "Confidence interval: %g%%\n", (1 - lr.alpha) * 100 )
    end

    if length(lr.white_types) + length(lr.hac_types) == 0
        helper_print_table(io, "Coefficients statistics:", 
            [lr.coefs, lr.stderrors, lr.t_values, lr.p_values, lr.ci_low, lr.ci_up, lr.VIF],
            ["Coefs", "Std err", "t", "Pr(>|t|)", "low ci", "high ci", "VIF"], 
            lr.updformula)
    end

    if length(lr.white_types) > 0
        for (cur_i, cur_type) in enumerate(lr.white_types)
            helper_print_table(io, "White's covariance estimator ($(Base.Unicode.uppercase(string(cur_type)))):", 
                [lr.coefs, lr.white_stderrors[cur_i], lr.white_t_values[cur_i], lr.white_p_values[cur_i], lr.white_ci_low[cur_i], lr.white_ci_up[cur_i] ],
                ["Coefs", "Std err", "t", "Pr(>|t|)", "low ci", "high ci"], 
                lr.updformula)
        end
    end

    if length(lr.hac_types) > 0
        for (cur_i, cur_type) in enumerate(lr.hac_types)
            helper_print_table(io, "Newey-West's covariance estimator:", 
                [lr.coefs, lr.hac_stderrors[cur_i], lr.hac_t_values[cur_i], lr.hac_p_values[cur_i], lr.hac_ci_low[cur_i], lr.hac_ci_up[cur_i] ],
                ["Coefs", "Std err", "t", "Pr(>|t|)", "low ci", "high ci"], 
                lr.updformula)
        end
    end
end

"""
    function getVIF(x, intercept, updf, df, p)

    (internal) Calculates the VIF, Variance Inflation Factor, for a given regression.
    When the has an intercept use the simplified formula. When there is no intercept use the classical formula.
"""
function getVIF(x, intercept, updf, df, p)
    if intercept
        if p == 1
            return [0., 1]
        end
        return vcat(0, diag(inv(cor(@view(x[:, 2:end])))))
    else 
        if p == 1
            return [0]
        end
        vt = terms(updf.rhs)
        results = Vector{Float64}()
        for (i, ct) in enumerate(vt)
            if ct isa InterceptTerm 
                continue
            end
            tf = ct ~ sum(setdiff(vt, [ct]))
            cr = regress(tf, df, req_stats=[:r2])
            push!(results, 1. / (1. - cr.R2))
        end
        return results
    end
end

"""
    function getSST(y, intercept)

    (internal) Calculates "total sum of squares" see link for description.
    https://en.wikipedia.org/wiki/Total_sum_of_squares
    When the mode has no intercept the SST becomes the sum of squares of y
"""
function getSST(y, intercept)
    SST = zero(eltype(y))
    if intercept
        ȳ = mean(y)
        SST = sum(abs2.(y .- ȳ))
    else 
        SST = sum(abs2.(y))
    end

    return SST
end

"""
    lr_predict(xs, coefs, intercept::Bool)

    (internal) Predict the ŷ given the x(s) and the coefficients of the linear regression.
"""
function lr_predict(xs, coefs, intercept::Bool)
    if intercept
        return muladd(@view(xs[:, 2:end]), @view(coefs[2:end]), coefs[1])
    else
        return muladd(xs, coefs, zero(eltype(coefs)))        
    end
end

"""
    function hasintercept(f::StatsModels.FormulaTerm)
    (internal) return a tuple with the first item being true when the formula has an intercept term, the second item being the potentially updated formula.
    If there is no intercept indicated add one.
    If the intercept is specified as absent (y ~ 0 + x) then do not change.
"""
function hasintercept(f::StatsModels.FormulaTerm)
    intercept = true
    if f.rhs isa ConstantTerm{Int64}
        intercept = convert(Bool, f.rhs.n)
        return intercept, f
    elseif f.rhs isa Tuple
        for t in f.rhs
            if t isa ConstantTerm{Int64}
                intercept = convert(Bool, t.n)
                return intercept, f
            end
        end
    end
    f = FormulaTerm(f.lhs, InterceptTerm{true}() + f.rhs)
    return intercept, f
end

"""
    function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=["all"], remove_missing=false)

    Estimate the coefficients of the regression, given a dataset and a formula. 
"""
function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=["all"], remove_missing=false, cov=[:none])
    intercept, f= hasintercept(f)

    (α > 0. && α < 1.) || throw(ArgumentError("α must be between 0 and 1"))

    copieddf = df 
    if remove_missing
        copieddf = copy(df[: , Symbol.(keys(schema(f, df).schema))])
        dropmissing!(copieddf)
    end

    needed_stats = get_needed_model_stats(req_stats)

    dataschema = schema(f, copieddf)
    updatedformula = apply_schema(f, dataschema)

    y, x = modelcols(updatedformula, copieddf)
    n, p = size(x)
    xy = [x y]

    xytxy = xy' * xy

# mandatory stats
    sse = sweep_op_full!(xytxy)[end]
    coefs = xytxy[1:p, end]
    mse = xytxy[p + 1, p + 1] / (n - p)

# optional stats
    total_scalar_stats = Set([:sse, :mse, :sst, :r2, :adjr2, :rmse, :aic, :sigma, :t_statistic, :vif])
    total_vector_stats = Set([:coefs, :stderror, :t_values, :p_values, :ci])

    scalar_stats = Dict{Symbol,Union{Nothing,Float64}}(intersect(total_scalar_stats, needed_stats) .=> nothing)
    vector_stats = Dict{Symbol,Union{Nothing,Vector}}(intersect(total_vector_stats, needed_stats) .=> nothing)

# optional stats
    if :sst in needed_stats
        scalar_stats[:sst] = getSST(y, intercept)
    end
    if :r2 in needed_stats
        scalar_stats[:r2] = 1. - (sse / scalar_stats[:sst])
    end
    if :adjr2 in needed_stats
        scalar_stats[:adjr2] = 1. - ((n - convert(Int64, intercept)) * (1. - scalar_stats[:r2])) / (n - p)
    end
    if :rmse in needed_stats
        scalar_stats[:rmse] = √mse
    end
    if :aic in needed_stats
        scalar_stats[:aic] =  n * log(sse / n) + 2p
    end
    if :sigma in needed_stats
        scalar_stats[:sigma] = sse / (n - p)
    end
    if :t_statistic in needed_stats
        scalar_stats[:t_statistic] = quantile(TDist(n - p), 1 - α / 2)
    end
    if :stderror in needed_stats
        vector_stats[:stderror] = sqrt.(diag(scalar_stats[:sigma] * @view(xytxy[1:end - 1, 1:end - 1])))
    end
    if :t_values in needed_stats
        vector_stats[:t_values] = coefs ./ vector_stats[:stderror]
    end
    if :p_values in needed_stats
        vector_stats[:p_values] = ccdf.(Ref(FDist(1., (n - p))), abs2.(vector_stats[:t_values]))
    end
    if :ci in needed_stats
        vector_stats[:ci] = vector_stats[:stderror] * scalar_stats[:t_statistic]
    end
    if :vif in needed_stats
        vector_stats[:vif] = getVIF(x, intercept, updatedformula, copieddf, p)
    end

# robust estimators stats
    needed_white, needed_hac = get_needed_robust_cov_stats(cov)
    white_types = Vector{Symbol}()
    white_stds = Vector{Vector}()
    white_t_vals = Vector{Vector}()
    white_p_vals = Vector{Vector}()
    white_ci_up = Vector{Vector}()
    white_ci_low = Vector{Vector}()

    if length(needed_white) > 0
        for t in needed_white
            if t in white_types
                continue
            end
            cur_type, cur_std = heteroscedasticity(t, x, y, coefs, intercept, n, p, xytxy)
            push!(white_types, cur_type)
            push!(white_stds, cur_std)

            if !isnothing(get(vector_stats, :t_values, nothing))
                cur_t_vals = coefs ./ cur_std
                push!(white_t_vals, cur_t_vals)
            else 
                white_t_vals = nothing
            end

            if !isnothing(get(vector_stats, :p_values, nothing))
                cur_p_vals = ccdf.(Ref(FDist(1., (n - p))), abs2.(cur_t_vals))
                push!(white_p_vals, cur_p_vals)
            else
                white_p_vals = nothing
            end
            if !isnothing(get(vector_stats, :ci, nothing))
                cur_ci = cur_std * scalar_stats[:t_statistic]
                cur_ci_up = coefs .+ cur_ci
                cur_ci_low = coefs .- cur_ci
                push!(white_ci_up, cur_ci_up)
                push!(white_ci_low, cur_ci_low)
            else 
                white_ci_up = nothing
                white_ci_low = nothing
            end
        end
    end

    hac_types = Vector{Symbol}()
    hac_stds = Vector{Vector}()
    hac_t_vals = Vector{Vector}()
    hac_p_vals = Vector{Vector}()
    hac_ci_up = Vector{Vector}()
    hac_ci_low = Vector{Vector}()

    if length(needed_hac) > 0
        for t in needed_hac
            if t in hac_types
                continue
            end
            cur_type, cur_std = HAC(t, x, y, coefs, intercept, n, p)
            push!(hac_types, cur_type)
            push!(hac_stds, cur_std)

            if !isnothing(get(vector_stats, :t_values, nothing))
                cur_t_vals = coefs ./ cur_std
                push!(hac_t_vals, cur_t_vals)
            else
                hac_t_vals = nothing
            end
            if !isnothing(get(vector_stats, :p_values, nothing))
                cur_p_vals = ccdf.(Ref(FDist(1., (n - p))), abs2.(cur_t_vals))
                push!(hac_p_vals, cur_p_vals)
            else
                hac_p_vals = nothing
            end
            if !isnothing(get(vector_stats, :ci, nothing))
                cur_ci = cur_std * scalar_stats[:t_statistic]
                cur_ci_up = coefs .+ cur_ci
                cur_ci_low = coefs .- cur_ci
                push!(hac_ci_up, cur_ci_up)
                push!(hac_ci_low, cur_ci_low)
            else
                hac_ci_up = nothing
                hac_ci_low = nothing
            end
        end
    end

    sres = linRegRes(xytxy, coefs, 
        white_types, hac_types,
        get(vector_stats, :stderror, nothing), white_stds, hac_stds, 
        get(vector_stats, :t_values, nothing), white_t_vals, hac_t_vals,
        p, mse, intercept, get(scalar_stats, :r2, nothing),
        get(scalar_stats, :adjr2, nothing), get(scalar_stats, :rmse, nothing), get(scalar_stats, :aic, nothing), get(scalar_stats, :sigma, nothing),
        get(vector_stats, :p_values, nothing), white_p_vals, hac_p_vals,
        haskey(vector_stats, :ci) ? coefs .+ vector_stats[:ci] : nothing, 
        haskey(vector_stats, :ci) ? coefs .- vector_stats[:ci] : nothing, 
        white_ci_up, white_ci_low,
        hac_ci_up, hac_ci_low,
        n, get(scalar_stats, :t_statistic, nothing), get(vector_stats, :vif, nothing), f, dataschema, updatedformula, α)
    
    return sres
end

"""
    function HAC(t::Symbol, x, y, coefs, intercept, n, p)

    (Internal) Return the relevant HAC (heteroskedasticity and autocorrelation consistent) estimator.
    In the current version only Newey-West is implemented.
"""
function HAC(t::Symbol, x, y, coefs, intercept, n, p)
    inv_xtx = inv(x' * x)
        e = y - lr_predict(x, coefs, intercept)
    xe = x .* e
    return (t, sqrt.(diag(n * inv_xtx * newey_west(xe) * inv_xtx)))
end

"""
    function heteroscedasticity(t::Symbol, x, y, coefs, intercept, n, p)

    (Internal) Compute the standard errors modified for the White's covariance estimator.
    Currently support HC0, HC1, HC2 and HC3. When :white is passed, select HC3 when the number of observation is below 250 otherwise select HC0.
"""
function heteroscedasticity(t::Symbol, x, y, coefs, intercept, n, p, xytxy)
    inv_xtx = inv(x' * x)
    XX = @view(xytxy[1:end - 1, 1:end - 1])
    e = y - lr_predict(x, coefs, intercept)
    xe = x .* e

    if t == :white && n < 250  
        t = :hc3
    elseif t == :white && n >= 250
        t = :hc0
    end

    if t == :hc0
        xetxe = xe' * xe
        return (:hc0, sqrt.(diag(XX * xetxe * XX)))
    elseif t == :hc1
        scale = (n / (n - p))
        xetxe = xe' * xe
        return (:hc1, sqrt.(diag(XX * xetxe * XX .* scale)))
    elseif t == :hc2
        leverage = diag(x * inv(x'x) * x')
        scale = @.( 1. / (1. - leverage))
        xe = @.(xe .* real(sqrt(Complex(scale))))
        xetxe = xe' * xe
        return (t, sqrt.(diag(inv_xtx * xetxe * inv_xtx)))
    elseif t == :hc3
        leverage = diag(x * inv(x'x) * x')
        scale = @.( 1. / (1. - leverage)^2)
        xe = @.(xe .* real(sqrt(Complex(scale))))
        xetxe = xe' * xe
        return (t, sqrt.(diag(inv_xtx * xetxe * inv_xtx)))
    else
        throw(error("Unknown symbol ($(t)) used as the White's covariance estimator"))
    end

end

"""
    function predict_and_stats(lr::linRegRes, df::DataFrames.DataFrame, α=0.05)

    Using the estimated coefficients from the regression make predictions, and calculate related statistics.
"""
function predict_and_stats(lr::linRegRes, df::DataFrames.DataFrame; α=0.05, req_stats=["none"])

    copieddf = df[: , Symbol.(keys(schema(lr.modelformula, df).schema))]
    dropmissing!(copieddf)
    dataschema = schema(lr.modelformula, copieddf)
    updatedformula = apply_schema(lr.modelformula, dataschema)
    y, x = modelcols(updatedformula, copieddf)
    n, p = size(x)

    needed, present = get_prediction_stats(req_stats)
    needed_stats = Dict{Symbol,Vector}()
    for sym in needed
        needed_stats[sym] = zeros(length(n))
    end
    if :leverage in needed
        needed_stats[:leverage] = diag(x * inv(x'x) * x')
    end
    if :predicted in needed
        needed_stats[:predicted] = lr_predict(x, lr.coefs, lr.intercept)
    end
    if :residuals in needed
        needed_stats[:residuals] = y .- needed_stats[:predicted]
    end
    if :stdp in needed
        if isnothing(lr.σ̂²)
            throw(ArgumentError(":stdp requires that the σ̂² (:sigma) was previously calculated through the regression"))
        end
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The STDP statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:stdp] = sqrt.(needed_stats[:leverage] .* lr.σ̂²)
    end
    if :stdi in needed
        if isnothing(lr.σ̂²)
            throw(ArgumentError(":stdi requires that the σ̂² (:sigma) was previously calculated through the regression"))
        end
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The STDI statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:stdi] = sqrt.((1. .+ needed_stats[:leverage]) .* lr.σ̂²)
    end
    if :stdr in needed
        if isnothing(lr.σ̂²)
            throw(ArgumentError(":stdr requires that the σ̂² (:sigma) was previously calculated through the regression"))
        end
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The STDR statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:stdr] = sqrt.((1. .- needed_stats[:leverage]) .* lr.σ̂²)
    end
    if :student in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The student statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:student] = needed_stats[:residuals] ./ needed_stats[:stdr]
    end
    if :rstudent in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The rstudent statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:rstudent] = needed_stats[:student] .* real.(sqrt.(complex.((n .- p .- 1 ) ./ (n .- p .- needed_stats[:student].^2 ), 0)))
    end
    if :lcli in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The LCLI statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:lcli] = needed_stats[:predicted] .- (lr.t_statistic .* needed_stats[:stdi])
    end
    if :ucli in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The UCLI statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:ucli] = needed_stats[:predicted] .+ (lr.t_statistic .* needed_stats[:stdi])
    end
    if :lclp in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The LCLP statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:lclp] = needed_stats[:predicted] .- (lr.t_statistic .* needed_stats[:stdp])
    end
    if :uclp in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The UCLP statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:uclp] = needed_stats[:predicted] .+ (lr.t_statistic .* needed_stats[:stdp])
    end
    if :press in needed
        needed_stats[:press] = needed_stats[:residuals] ./ (1. .- needed_stats[:leverage])
    end
    if :cooksd in needed
        if length(lr.white_types) + length(lr.hac_types) > 0
            println(io, "The CooksD statistic that relies on Sigma^2 has been requested. At least one robust covariance have been requested indicating that the assumptions needed for Sigma^2 may not be present.")
        end
        needed_stats[:cooksd] = needed_stats[:stdp].^2 ./  needed_stats[:stdr].^2 .* needed_stats[:student].^2 .* (1 / lr.p)
    end

    for sym in present
        copieddf[!, sym] = needed_stats[sym]
end

    return copieddf
end

end # end of module definition
