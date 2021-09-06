module LinearRegression

export regress, predict_and_stats

using Base: Tuple, Int64
using StatsBase:eltype, isapprox, length, coefnames
using Distributions
using Printf, NamedArrays
using StatsBase
using StatsModels, DataFrames

include("sweep_operator.jl")
include("utilities.jl")

struct linRegRes 
    extended_inverse::Matrix                # Store the extended inverse matrix 
    coefs::Union{Nothing,Vector}            # Store the coefficients of the fitted model
    stderrors::Union{Nothing,Vector}        # Store the standard errors for the fitted model
    t_values::Union{Nothing,Vector}         # Store the t values for the fitted model
    p::Int64                                # Store the number of parameters (including the intercept as a parameter)
    MSE::Union{Nothing,Float64}             # Store the Mean squared error for the fitted model
    intercept::Bool                         # Indicate if the model has an intercept
    R2::Union{Nothing,Float64}              # Store the R-squared value for the fitted model
    ADJR2::Union{Nothing,Float64}           # Store the adjusted R-squared value for the fitted model
    RMSE::Union{Nothing,Float64}            # Store the Root mean square error for the fitted model
    AIC::Union{Nothing,Float64}             # Store the Akaike information criterion for the fitted model
    σ̂²::Union{Nothing,Float64}              # Store the σ̂² for the fitted model
    p_values::Union{Nothing,Vector}         # Store the p values for the fitted model
    ci_up::Union{Nothing,Vector}            # Store the upper values confidence interval of the coefficients 
    ci_low::Union{Nothing,Vector}           # Store the lower values confidence interval of the coefficients 
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

    if !isnothing(lr.σ̂²) && !isnothing(lr.AIC)
        @printf(io, "  σ̂²: %g\t\t\tAIC: %g\n", lr.σ̂², lr.AIC)
    elseif !isnothing(lr.σ̂²)
        @printf(io, "  σ̂²: %g\n", lr.σ̂²)
    elseif !isnothing(lr.AIC)
        @printf(io, "  AIC: %g\n", lr.AIC)
    end
    
    if !isnothing(lr.ci_low) || !isnothing(lr.ci_up)
        @printf(io, "Confidence interval: %g%%\n", (1 - lr.alpha)*100 )
    end

    all_stats = [lr.coefs, lr.stderrors, lr.t_values, lr.p_values, lr.ci_low, lr.ci_up, lr.VIF]
    all_stats_name = ["Coefs", "Std err", "t", "Pr(>|t|)", "low ci", "high ci", "VIF"]

    todelete = [i for (i, v) in enumerate(all_stats) if isnothing(v)]
    deleteat!(all_stats, todelete)
    deleteat!(all_stats_name, todelete)

    na = NamedArray(
        reduce(hcat, all_stats),
        (StatsBase.coefnames(lr.updformula.rhs), all_stats_name) , ("Terms", "Stats"))
    
    println(io, "Coefficients statistics:")
    my_namedarray_print(io, na)
end

"""
    function getVIF(x, intercept, p)

    (internal) Calculates the VIF, Variance Inflation Factor, for a given regression.
"""
function getVIF(x, intercept, p)
    if p == 2
        return [0., 1.]
    end
    if intercept
        return vcat(0, diag(inv(cor(@view(x[:, 2:end])))))
    end
    return diag(inv(cor(x)))
end

"""
    function getSST(y, intercept)

    (internal) Calculates "total sum of squares" see link for description.
    TODO: to confirm calculation when there is no intercept.
    https://en.wikipedia.org/wiki/Total_sum_of_squares
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

    (internal) return true when the formula has an intercept term.

"""
function hasintercept(f::StatsModels.FormulaTerm)
    intercept = false
    if f.rhs isa ConstantTerm{Int64}
        intercept = convert(Bool, f.rhs.n)
    elseif f.rhs isa Tuple
        for t in f.rhs
            if t isa ConstantTerm{Int64}
                intercept = convert(Bool, t.n)
                return intercept
            end
        end
    end
    return intercept
end

"""
    function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=["all"], remove_missing=false)

    Estimate the coefficients of the regression, given a dataset and a formula. 
"""
function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=["all"], remove_missing=false)
    intercept = hasintercept(f)
    # intercept || throw(ArgumentError("Only formulas with intercept are supported. Update the forumla to include an intercept."))
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
        vector_stats[:vif] = getVIF(x, intercept, p)
    end

    sres = linRegRes(xytxy, coefs, get(vector_stats, :stderror, nothing), get(vector_stats, :t_values, nothing), p, mse, intercept, get(scalar_stats, :r2, nothing),
                get(scalar_stats, :adjr2, nothing), get(scalar_stats, :rmse, nothing), get(scalar_stats, :aic, nothing), get(scalar_stats, :sigma, nothing),
                get(vector_stats, :p_values, nothing), 
                haskey(vector_stats, :ci) ? coefs .+ vector_stats[:ci] : nothing, 
                haskey(vector_stats, :ci) ? coefs .- vector_stats[:ci] : nothing, 
                n, get(scalar_stats, :t_statistic, nothing), get(vector_stats, :vif, nothing), f, dataschema, updatedformula, α)

    return sres
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
        needed_stats[:stdp] = sqrt.(needed_stats[:leverage] .* lr.σ̂²)
    end
    if :stdi in needed
        if isnothing(lr.σ̂²)
            throw(ArgumentError(":stdi requires that the σ̂² (:sigma) was previously calculated through the regression"))
        end
        needed_stats[:stdi] = sqrt.((1. .+ needed_stats[:leverage]) .* lr.σ̂²)
    end
    if :stdr in needed
        if isnothing(lr.σ̂²)
            throw(ArgumentError(":stdi requires that the σ̂² (:sigma) was previously calculated through the regression"))
        end
        needed_stats[:stdr] = sqrt.((1. .- needed_stats[:leverage]) .* lr.σ̂²)
    end
    if :student in needed
        needed_stats[:student] = needed_stats[:residuals] ./ needed_stats[:stdr]
    end
    if :rstudent in needed
        needed_stats[:rstudent] = needed_stats[:student] .* real.(sqrt.(complex.((n .- p .- 1 ) ./ (n .- p .- needed_stats[:student].^2 ), 0)))
    end
    if :lcli in needed
        needed_stats[:lcli] = needed_stats[:predicted] .- (lr.t_statistic .* needed_stats[:stdi])
    end
    if :ucli in needed
        needed_stats[:ucli] = needed_stats[:predicted] .+ (lr.t_statistic .* needed_stats[:stdi])
    end
    if :lclp in needed
        needed_stats[:lclp] = needed_stats[:predicted] .- (lr.t_statistic .* needed_stats[:stdp])
    end
    if :uclp in needed
        needed_stats[:uclp] = needed_stats[:predicted] .+ (lr.t_statistic .* needed_stats[:stdp])
    end
    if :uclp in needed
        needed_stats[:uclp] = needed_stats[:predicted] .+ (lr.t_statistic .* needed_stats[:stdp])
    end
    if :press in needed
        needed_stats[:press] = needed_stats[:residuals] ./ (1. .- needed_stats[:leverage])
    end
    if :cooksd in needed
        needed_stats[:cooksd] = needed_stats[:stdp].^2 ./  needed_stats[:stdr].^2 .* needed_stats[:student].^2 .* (1 / lr.p)
    end

    for sym in present
        copieddf[!, sym] = needed_stats[sym]
end

    return copieddf
end



end # end of module definition