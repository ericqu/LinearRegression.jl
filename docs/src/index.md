# LinearRegression.jl Documentation
```@contents
```
LinearRegression.jl implements linear regression using the least-squares algorithm (relying on the sweep operator). This package is in the alpha stage. Hence it is likely that some bugs exist. Furthermore, the API might change in future versions.

The usage aims to be straightforward, a call to ```regress``` to build a linear regression model, and a call to ```predict_in_sample``` to predict data using the built linear regression model.

The regress call will compute some statistics about the fitted model in addition to the coefficients. 

#### Number of observations and variables
The number of observations ``n`` used to fit the model.

The number of independent variables ``p`` used in the model.

#### Total Sum of Squares
The Total Sum of Squares (SST) is calculated but not presented to the user. In case of model with intercept the SST is computed with the following:
```math
\mathrm{SST}=\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^2
```
And when there is no intercept with the following: 
```math
\mathrm{SST}=\sum_{i=1}^{n} y_{i}^2
```

#### Error Sum of Squares
The Error Sum of Squares (or SSE) also known as Residual Sum of Square (RSS). This package uses the sweep operator (Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.) to compute the SSE.

#### Mean Squared Error
The Mean Squared Error (MSE) is calculated as 
```math
\mathrm{MSE} = \displaystyle{\frac{{\mathrm{SSE}}}{{n - p}}}
```

The Root Mean Squared Error (RMSE) is calculated as 
```math
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}
```

The MSE is the estimator of σ̂² unless at least one robust covariance estimator is requested.

#### R² and Adjusted R² 
The R² (R2 or R-squared) see (https://en.wikipedia.org/wiki/Coefficient_of_determination) is calculated with the following formula: 
```math
\mathrm{R}^2 = 1 - \displaystyle{\frac{{\mathrm{SSE}}}{{\mathrm{SST}}}}
```
The Adjusted R² (ADJR2) is computed with the following formulas:

when it is a model with an intercept:
```math
\mathrm{ADJR}^2 = 1 - \displaystyle \frac{(n-1)(1-\mathrm{R}^2)}{n-p}
```
And when there is no intercept:
```math
\mathrm{ADJR}^2 = 1 - \displaystyle \frac{(n)(1-\mathrm{R}^2)}{n-p}
```

#### Akaike information criterion
The Akaike information criterion is calculated with the Linear Regression specific formula:
```math
\mathrm{AIC} = \displaystyle n \ln \left( \frac{\mathrm{SSE}}{n} \right) + 2p
```

#### t-statistic and confidence interval 
The t-statistic is computed by using the inverse cumulative t-distribution (with ```quantile()```) with parameter (``n - p``) at ``1 - \frac{α}{2}``. 

The standard errors of the coefficients are calculated by multiplying the Sigma (estimated by the MSE) with the pseudo inverse matrix (resulting from the sweep operator), out of which the square root of the diagonal elements are extracted.

The t-values are calculated as the coefficients divided by their standard deviation.

The upper bound of the confidence interval for each coefficient is calculated as the coeffiecent + coefficient's standard error * t_statistic.

The lower bound of the confidence interval for each coefficient is calculated as the coeffiecent - coefficient's standard error * t_statistic.

#### p-values

The p-values are computed using the F Distribution, the degree of freedom for each coefficent.

#### Variance inflation factor
Variance inflation factor (VIF) is calculated by taking the  diagonal elements of the inverse of the correlation matrix formed by the independent variables.

### Robust covariance estimators

#### Heteroscedasticity estimators
The user can select estimators from these list. If the user select "White" as an estimator then HC3 will be selected for a small size (n < 250) otherwise HC0 will be selected.
##### HC0
The following estimators can be calculated.
Having InvMat the pseudo inverse resulting from the sweep operator. And having ``xe`` being the matrix of the independent variables times the residuals. Then HC0 is calculated as:
```math
\textup{HC0} = \sqrt{diag(\textup{InvMat } \textup{xe}' \textup{xe} \textup{ InvMat})}
```

##### HC1
Having n being the number of observations and p the number of variables. Then HC1 is calculated as:
```math
\textup{HC1} = \sqrt{diag(\textup{InvMat } \textup{xe}' \textup{xe} \textup{ InvMat } \frac{n}{n-p})}
```

##### HC2
The leverage or hat matrix is calculated as:
```math
\textup{H} = \textup{X} (\textup{X'X})^{-1}\textup{X'}
```

``xe`` is scaled by ``\frac{1}{1 - H}`` then 
```math
\textup{HC2} = \sqrt{diag(\textup{InvMat } \textup{xe}' \textup{xe} \textup{ InvMat } )}
```

##### HC3
``xe`` is scaled by ``\frac{1}{{\left( 1 - H \right)^2}}`` then 
```math
\textup{HC3} = \sqrt{diag(\textup{InvMat } \textup{xe}' \textup{xe} \textup{ InvMat } )}
```


#### Heteroskedasticity and autocorrelation consistent estimator

Newey-West estimator calculation is not documented yet.
See [reference implementation](https://github.com/mcreel/Econometrics/blob/508aee681ca42ff1f361fd48cd64de6565ece221/src/NP/NeweyWest.jl) [current implementation](https://github.com/ericqu/LinearRegression.jl/blob/docu/src/newey_west.jl) for details.


#### Weighted regression

This version is the initial implementation of a weighted regression using analytical weights.
Here is a minimal example illustrating its usage.
```julia 
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
] # data from https://blogs.sas.com/content/iml/2016/10/05/weighted-regression.html

df = DataFrame(tw, [:y,:x,:w])
f = @formula(y ~ x)
lm, ps= regress(f, df, "fit", weights="w")
```
Which gives the following output:
```
Model definition:      y ~ 1 + x
Used observations:      3
Weighted regression
Model statistics:
  R²: 0.96                      Adjusted R²: 0.92
  MSE: 0.48                     RMSE: 0.69282
  σ̂²: 0.48
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │     Coefs    Std err          t   Pr(>|t|)     low ci    high ci
──────────────┼─────────────────────────────────────────────────────────────────
(Intercept)   │      -0.2    0.69282  -0.288675   0.821088   -9.00312    8.60312
x             │      1.44   0.293939    4.89898   0.128188   -2.29485    5.17485
```


## Functions
```@docs
    function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame, req_plots; α::Float64=0.05, req_stats=["default"],
    weights::Union{Nothing,String}=nothing, remove_missing=false, cov=[:none], contrasts=nothing, 
    plot_args=Dict("plot_width" => 400, "loess_bw" => 0.6, "residuals_with_density" => false))

function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=["default"], weights::Union{Nothing,String}=nothing, remove_missing=false, cov=[:none], contrasts=nothing)

function predict_in_sample(lr::linRegRes, df::DataFrames.DataFrame; α=0.05, req_stats=["none"], dropmissingvalues=true)

function predict_out_of_sample(lr::linRegRes, df::DataFrames.DataFrame; α=0.05, req_stats=["none"], dropmissingvalues=true)

```

## Index

```@index
```
