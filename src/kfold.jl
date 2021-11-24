"""
    function kfold(f, df, k, r = 1, shuffle=true; kwargs...)

    Provide a simple `k` fold cross validation, repeated `r` time(s).
    `kwargs` arguments are passed to the `regress` function call.
    This feature overlap in part with the PRESS statistics available from `predict_in_sample`.
    
"""
function kfold(f, df, k, r = 1, shuffle=true; kwargs...)
    totalrows = nrow(df)
    gindexes = Vector(undef, k)
    resvec = Vector(undef, k*r)
    sdf = @view df[!, :]

    for cr in 1:r
        if shuffle
            sdf = @view df[shuffle(axes(df, 1)), :]
        end
        for ck in 1:k 
            gindexes[ck] = ck:k:totalrows
        end
        for ck in 1:k
            training_range = reduce(vcat, map(x->gindexes[x], filter(x->x!=ck, 1:k)))
            training = @view sdf[training_range, :]
            testing = @view sdf[gindexes[ck], :]

            lr = regress(f, training; kwargs...)
            cres = predict_in_sample(lr, testing, req_stats=[:residuals])
            t_mse= mean(cres.residuals .^2)
            resvec[ (k * (cr -1)) + ck] = (R2 = lr.R2, ADJR2 = lr.ADJR2, TRAIN_MSE = lr.MSE,
                        TRAIN_RMSE = lr.RMSE, 
                        TEST_MSE= t_mse , TEST_RMSE= √t_mse )
        end
    end
    return DataFrame(resvec) 
end
