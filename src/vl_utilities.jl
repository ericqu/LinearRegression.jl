using VegaLite
using StatsBase, LinearAlgebra , Distributions # for the density/histogram plot

isnotintercept(t::AbstractTerm) = t isa InterceptTerm ? false : true
iscontinuousterm(t::AbstractTerm) = t isa ContinuousTerm ? true : false

function fitplot(results, lm, plot_args)

    lhs = terms(lm.updformula.lhs)
    rhs_noint = filter(isnotintercept, terms(lm.updformula.rhs))

    length(rhs_noint) == 0 && return nothing
    length(rhs_noint) > 1 && begin
        println("LinearRegression: Fitplot was requested but not appropriate for regression with multiple independent variables")
        return nothing
    end
    length(lhs) == 0 && return nothing

    actx = first(rhs_noint).sym
    acty = first(lhs).sym
    acttitle = "Fit Plot: " * string(actx) * " vs " * string(acty)

    fp = select(results, [actx, acty, :ucli, :lcli, :uclp, :lclp, :predicted]) |> @vlplot(
        layer = [
            {   mark = {:errorband, color = "lightgrey" },
                y = { field = :ucli, type = "quantitative", title = :y} ,
                y2 = { field = :lcli, type = "quantitative", }, 
                x = actx, 
            },
            {   mark = {:errorband, color = "darkgrey" },
                y = { field = :uclp, type = "quantitative", title = :y}, 
                y2 = { field = :lclp, type = "quantitative", }, 
                x = actx, 
            },
            {
                mark = { :line, color = "darkorange", },
                x = actx,
                y = :predicted
            },
            {
                :point, 
                x = { actx,  axis = {grid = false}, scale = {zero = false, padding = 5}},
                y = { acty, axis = {grid = false}, scale = {zero = false, padding = 5}},
                title = acttitle, width = plot_args["plot_width"], height = plot_args["plot_width"]
    } ,
        ]
    )
    return fp
end

function multiple_effect_plots(results, lm, plot_args)

end

function simple_residuals_plot(results, dep_var=nothing, show_density=true; plot_width=400, loess_bandwidth::Union{Nothing,Float64}=0.99)
    if isnothing(dep_var)
        dep_var = :predicted
    end
    
    loess_p = @vlplot()
    if !isnothing(loess_bandwidth)
        loess_p = @vlplot(
            transform = [ { loess = :residuals, on = dep_var, bandwidth = loess_bandwidth } ],
            mark = {:line, color = "firebrick"},
            x = dep_var,
            y = :residuals
            )
    end

    title = "Residuals plot: $(string(dep_var))"
    sresults = select(results, [:residuals, dep_var])

    p = sresults |> 
    @vlplot(title = title, width = plot_width, height = plot_width,
    x = {type = "quantitative", axis = {grid = false}, scale = {zero = false, padding = 5}},
    y = {type = "quantitative", axis = {grid = false}}) +
    @vlplot(:point, dep_var, :residuals) +
    loess_p +
    @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = 0})

    if show_density == false
        return p
    end

    mp = sresults |> @vlplot(
        width = 100, height = plot_width,
        mark = {:area, orient = "horizontal"},
        transform = [{density = :residuals, bandwidth = 0.4}],
        x = {"density:q", title = nothing, axis = nothing},
        y = {"value:q", title = nothing, axis = nothing } )
        
    tp = @vlplot(bounds = :flush, spacing = 5, config = {view = {stroke = :transparent}}) + [p mp]

    return tp
end

function residuals_plots(results, lm, plot_args)
    rhs_noint = filter(isnotintercept, terms(lm.updformula.rhs))
    plots = Vector{VegaLite.VLSpec}()
    length(rhs_noint) == 0 && return nothing

    plot_width = get(plot_args, "plot_width", nothing)
    loess_bw = get(plot_args, "loess_bw", nothing)

    # main residual plot 
    push!(plots, simple_residuals_plot(results, plot_width=plot_width, loess_bandwidth=loess_bw))

    # additional plot per dependent variable
    for c_dependent_var in rhs_noint
        c_sym = c_dependent_var.sym
        show_density = iscontinuousterm(c_dependent_var)
        push!(plots, simple_residuals_plot(results, c_sym, show_density, plot_width=plot_width, loess_bandwidth=loess_bw))
    end

    return plots
end

function qqplot(results, fitted_residuals, plot_width)
    n = length(results.residuals)
    grid = [1 / (n - 1):1 / (n - 1):1 - (1 / (n - 1));]
    qu_theo = quantile.(fitted_residuals, grid)
    qu_empi = quantile(results.residuals, grid)
    qqdf = DataFrame(x=qu_theo, y=qu_empi)
    qqline = [first(qu_theo), last(qu_theo)]
    qqldf = DataFrame(x=qqline, y=qqline)

    qqplot = qqdf |> @vlplot() + @vlplot(
            title = "Residuals QQ-Plot", 
            width = plot_width, height = plot_width, 
            :point,
            x = {:x, title = "Theoritical quantiles", axis = {grid = false}, scale = {zero = false, padding = 5}}, 
            y = {:y, title = "Empirical quantiles", axis = {grid = false}, scale = {zero = false, padding = 5} }
        ) + @vlplot(
            {:line, color = "darkgrey"}, 
            data = qqldf, x = :x, y = :y )

    return qqplot
end

function default_range(dist::Distribution, alpha=0.0001)
    minval = isfinite(minimum(dist)) ? minimum(dist) : quantile(dist, alpha)
    maxval = isfinite(maximum(dist)) ? maximum(dist) : quantile(dist, 1 - alpha)
    minval, maxval
end

rice_rule(obs) = round(Int, 2. * obs^(1 / 3))

function histogram_density(results, fitted_residuals, plot_width)
    # data for the density curve
    frmin, frmax = default_range(fitted_residuals)
    rangetest = (frmax - frmin) / plot_width
    qpdf = [pdf(fitted_residuals, x) for x in frmin:rangetest:frmax]
    xs = [x for x in frmin:rangetest:frmax]
    tdf = DataFrame(x=xs, y=qpdf)

    # data for the histogram
    hhh = fit(Histogram, results.residuals)
    nhh = normalize(hhh)
    all_edges = collect(first(nhh.edges))
    bin_starts = [x for x in all_edges[1:end - 1]]
    bin_ends = [x for x in all_edges[2:end]]
    counts = nhh.weights

    hdf = DataFrame(bs=bin_starts, be=bin_ends, y=counts)
    step_size = bin_starts[2] - bin_starts[1]

    hdplot = hdf |> @vlplot(width = plot_width, height = plot_width, title = "Residuals Histogram and PDF") +
    @vlplot(
        :bar, 
        x = {:bs, title = "residuals", bin = {binned = true, step = step_size}, axis = {grid = false}},
        x2 = :be , 
        y = {:y, stack = "zero",  axis = {grid = false} }
    ) +
    @vlplot(
        data = tdf,
        {:line, color = "darkorange"}, 
        x = {"x:q", scale = {zero = false}, axis = {grid = false}},
    y = {"y:q", scale = {zero = false}, axis = {grid = false}}, 
    ) 

    return hdplot
end

function normality_plots(results, lm, plot_args)
    plots = Vector{VegaLite.VLSpec}()
    plot_width = get(plot_args, "plot_width", nothing)
    fitted_residuals = fit(Normal, results.residuals)

    push!(plots, qqplot(results, fitted_residuals, plot_width))
    push!(plots, histogram_density(results, fitted_residuals, plot_width))
    
    return plots

end

function cooksd_plot(results, lm, plot_args)
    plot_width = get(plot_args, "plot_width", nothing)
    plot_height = plot_width / 2

    sdf = select(results, [:cooksd])
    sdf.Observations = rownumber.(eachrow(sdf))

    threshold_cooksd = 4 / lm.observations
    p = sdf |> 
        @vlplot(title = "Cook's Distance", width = plot_width, height = plot_height) +
        @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = threshold_cooksd}) +
        @vlplot(
            mark = {:rule, color = :steelblue},
            x = {:Observations, type = :quantitative, axis = {grid = false}},
            y = {datum = 0}, y2 = :cooksd
            ) 
    return p 
end

function scalelocation_plot(results, lm, plot_args)
    plot_width = get(plot_args, "plot_width", nothing)
    sdf = select(results, [:predicted, :student])
    sdf.sqrtstudent = sqrt.(abs.(sdf.student))

    p = sdf |> 
        @vlplot() +
        @vlplot(
                title = "Scale and location plot" , 
                width = plot_width , height = plot_width,
                :point, 
                x = {:predicted,  scale = {zero = false}, axis = { grid = false} },
                y = {:sqrtstudent, title = "√student", scale = {zero = false}, axis = { grid = false} }
            ) + @vlplot(
                transform = [ { loess = :sqrtstudent, on = :predicted, bandwidth = 0.6 } ],
                mark = {:line, color = "firebrick"},
                x = :predicted, y = :sqrtstudent
            )
    return p 
end

function leverage_plot(results, lm, plot_args)
    threshold_leverage = 2 * lm.p / lm.observations
    plot_width = plot_args["plot_width"]
    p = select(results, [:leverage, :rstudent]) |> 
        @vlplot(title = "Leverage vs Rstudent", width = plot_width, height = plot_width,
            x = {axis = {grid = false}}, y = {axis = {grid = false}}   ) +
        @vlplot(:point, :leverage, :rstudent) +
        @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = -2}) +
        @vlplot(mark = {:rule, color = :darkgrey}, x = {datum = threshold_leverage}) +
        @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = 2}) 
    return p
end

