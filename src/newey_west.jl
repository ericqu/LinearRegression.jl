# Newey-West covariance estimator
# from https://github.com/mcreel/Econometrics/blob/508aee681ca42ff1f361fd48cd64de6565ece221/src/NP/NeweyWest.jl
# under MIT licence https://github.com/mcreel/Econometrics/blob/508aee681ca42ff1f361fd48cd64de6565ece221/LICENSE
# and adapted 

"""
    function newey_west(Z,nlags=0)

Returns the Newey-West estimator of the asymptotic variance matrix
INPUTS: Z, a nxk matrix with rows the vector zt'
        nlags, the number of lags
OUTPUTS: omegahat, the Newey-West estimator of the covariance matrix
"""
function newey_west(Z,nlags=0)
    n,k = size(Z)
    # de-mean the variables
    Z = Z .- mean(Z,dims=1)
    omegahat = Z'*Z/n       # sample variance
    # automatic lags?
    if nlags == 0
        nlags = max(1, round(Int, n^0.25))
    end    
    # sample autocovariances
    for i = 1:nlags
       Zlag = @view(Z[1:n-i,:])
       ZZ = @view(Z[i+1:n,:])
       gamma = (ZZ'*Zlag)/n
       weight = 1.0 - (i/(nlags+1.0))
       omegahat += weight*(gamma + gamma')
    end    
    return omegahat
end