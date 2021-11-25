# LinearRegression.jl Documentation
LinearRegression.jl implements linear regression using the least-squares algorithm (relying on the sweep operator). This package is in the alpha stage. Hence it is likely that some bugs exist. Furthermore, the API might change in future versions.

The usage aims to be straightforward, a call to ```regress``` to build a linear regression model, and a call to ```predict_in_sample``` to predict data using the built linear regression model.
When predicting on data not present during the regression, use the ```predict_out_of_sample``` function as this does not require a response value (consequently, statistics that need a response, as the residuals, are not available.)

The regress call will compute some statistics about the fitted model in addition to the coefficients. The statistics computed depend on the value of the ```req_stats``` argument. 
The prediction functions compute predicted values together with some statistics. Like for the regress calls, the statistics computed depend on the value of the ```req_stats``` argument.

When some analytical positive weights are used, a weighted regression is performed.

### Statistics related to the regression (the fitting)
Fitting the model generates some statistics dependent on the `req_stats` argument of the `regress` function.
- ``n``, ``p``, `"coefs"` and `"see"` are always computed
- `"mse"`, `"sst"`, `"rmse"`, `"aic"`,  `"sigma"`, `"t_statistic"`, `"vif"`, `"r2"`, `"adjr2"`, `"stderror"`, `"t_values"`, `"p_values"`, `"ci"`,  are computed upon request.
  - some diagnostics can be requested as well. Here is the full list as Symbols `[:diag_normality, :diag_ks, :diag_ad, :diag_jb, :diag_heteroskedasticity, :diag_white, :diag_bp ]`, `"diag_normality"` is a shortcut for `[:diag_ks, :diag_ad, :diag_jb]` and `:diag_heteroskedasticity` is a shortcut for `[:diag_white, :diag_bp]`. 
- "default", includes the mandatory stats, and some of the optional statistics here as Symbols: `[:coefs, :sse, :mse, :sst, :rmse, :sigma, :t_statistic, :r2, :adjr2, :stderror, :t_values, :p_values, :ci]`
- `"all"` includes all availble statistics
- `"none"` include only the mandatory statistics

The meaning for these statistics is given below.

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

#### t\_statistic and confidence interval 
The t\_statistic is computed by using the inverse cumulative t_distribution (with ```quantile()```) with parameter (``n - p``) at ``1 - \frac{α}{2}``. 

The standard errors of the coefficients are calculated by multiplying the Sigma (estimated by the MSE) with the pseudo inverse matrix (resulting from the sweep operator), out of which the square root of the diagonal elements are extracted.

The t-values are calculated as the coefficients divided by their standard deviation.

The upper bound of the confidence interval for each coefficient is calculated as the coeffiecent + coefficient's standard error * t\_statistic.

The lower bound of the confidence interval for each coefficient is calculated as the coeffiecent - coefficient's standard error * t\_statistic.

#### p-values
The p-values are computed using the F Distribution, the degree of freedom for each coefficent.

#### Variance inflation factor
Variance inflation factor (VIF) is calculated by taking the diagonal elements of the inverse of the correlation matrix formed by the independent variables.

#### PRESS predicted residual error sum of squares
The predicted residual error sum of squares is calculated by taking the sum of squares from the `PRESS` (see below the statistics related to predictions) of each observations. 

### Robust covariance estimators
Robust Covariance estimator can be requested through the ```cov``` argument of the ```regress``` function.
The options are (as Symbols):
- `:white`: Heteroscedasticity 
- `:hc0`: Heteroscedasticity
- `:hc1`: Heteroscedasticity)
- `:hc2`: Heteroscedasticity)
- `:hc3`: Heteroscedasticity)
- `:nw`: HAC (Heteroskedasticity and Autocorrelation Consistent estimator)

#### Heteroscedasticity estimators
The user can select estimators from above list. If the user select `:white` as an estimator then HC3 will be selected for a small size (n <= 250) otherwise HC0 will be selected. (see "Using Heteroscedasticity Consitent Standard Errors in the Linear Regression Model" J. Scott Long and Laurie H. Ervin (1998-2000)).
If another estimator is requested it is provided. A list of estimator can be requested as in for instance `cov=[:hc2, hc3]`.
Comprehensive descriptions of the estimators and their applications shoudl in found in a text book, here only a brief description of the implementation is provided. 

##### HC0
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

#### Heteroskedasticity and autocorrelation consistent estimator (HAC)

Newey-West estimator calculation is not documented yet.
See [reference implementation](https://github.com/mcreel/Econometrics/blob/508aee681ca42ff1f361fd48cd64de6565ece221/src/NP/NeweyWest.jl) [current implementation](https://github.com/ericqu/LinearRegression.jl/blob/docu/src/newey_west.jl) for details.

### Statistics related to the prediction
Predicting values using independent variables and a model will generate predicted values and some additional statistics dependent on the value of the `req_stats` argument of the `predict*` functions.
Here is a list of the available statistics:
[:predicted, :residuals, :leverage, :stdp, :stdi, :stdr, :student, :rstudent, :lcli, :ucli, :lclp, :uclp, :press, :cooksd]

#### Predicted
The predicted value is the sum of the dependant variable(s) multiplied by the coefficients from the regression and the intercept (if the model has one). The predicted value is also known as the Y-hat.

#### Residuals
The residuals are here defined as the known responses variables minus the predicted values.

#### Leverage
The leverage for the i-th independent observation x_i when it is not a weighted regression is calculated as:
```math
\mathrm{h_i} = \mathrm{x_i' (X' X)^{-1} x_i}
```
And as per below when it is a weighted regression with a vector of weights ``W`` with the i-th weight being ``w_i`` then the i-th leverage is calculated as such:
```math
\mathrm{h_i} = \mathrm{w_i \cdot  x_i' (X' W X)^{-1} x_i}
```
#### STDP
STDP is the standard error of the mean predicted value, and is calculated as 
```math
\textup{STDP} = \sqrt{\hat{\sigma}^2 h_i }
```
and for a weighted regression as:
```math
\textup{STDP} = \sqrt{\hat{\sigma}^2 h_i  / w_i}
```
#### STDI
STDI is the standard error of the individual predicted value, and is calculated as 
```math
\textup{STDI} = \sqrt{\hat{\sigma}^2 (1 + h_i)}
```
and for a weighted regression as:
```math
\textup{STDI} = \sqrt{\hat{\sigma}^2 (1 + h_i) / w_i}
```
#### STDR
STDR is the standard error of the residual, and is calculated as 
```math
\textup{STDR} = \sqrt{\hat{\sigma}^2 (1 - h_i) }
```
and for a weighted regression as:
```math
\textup{STDR} = \sqrt{\hat{\sigma}^2 (1 - h_i) / w_i}
```
#### Student
Student represents the standardized residuals, and is calculated by using the residuals over the standard error of the residuals.
#### RStudent
RStudent is the studentized residuals calculated as
```math
\textup{RSTUDENT} = \sqrt{ \frac{n - p - 1}{n - p - \textup{student}^2}} 
```
#### LCLI
LCLI is the lower bound of the prediction interval and is calculated as:
```math
\textup{LCLI} = \mathrm{predicted} - ( \mathrm{t\_statistic} \cdot \mathrm{STDI} )
```
#### UCLI
UCLI is the upper bound of the prediction interval and is calculated as:
```math
\textup{UCLI} = \mathrm{predicted} + ( \mathrm{t\_statistic} \cdot \mathrm{STDI} )
```
#### LCLP
LCLP is the lower bound of the predicted mean confidence interval and is calculated as:
```math
\textup{LCLP} = \mathrm{predicted} - ( \mathrm{t\_statistic} \cdot \mathrm{STDP} )
```
#### UCLP
UCLP is the upper bound of the predicted mean confidence interval and is calculated as:
```math
\textup{UCLI} = \mathrm{predicted} + ( \mathrm{t\_statistic} \cdot \mathrm{STDP} )
```
#### COOKSD
COOKSD is the Cook's Distance for each predicted value, and is calculated as 
```math
\textup{COOKSD} = \frac{1}{p} \frac{\textup{STDP}^2}{\textup{STDR}^2 \cdot \textup{student}^2}
```
#### PRESS 
PRESS is the predicted residual error sum of squares and is calculated as 
```math
\textup{PRESS} = \frac{\textup{residuals}}{1 - \textup{leverage}}
```
#### Type 1 SS
Type 1 Sum of squares, are calculated as a by-product of the sweep operator.

#### Type 2 SS
Type 2 Sum of squares, are calculated using the pseudo-inverse matrix. The Type 2 SS of the ith independent variable is the square of the coefficient of the independent variable divided by the ith element of the diagonal from the pseudo-inverse matrix.

#### Pcorr 1 and 2 
`pcorr1` and `pcorr2` are the squared partial correlation coefficient calculated as:
```math
\textup{pcorr1} = \frac{\textup{Type 1 SS}}{\textup{Type 1 SS}+ \textup{SSE}}
```

```math
\textup{pcorr2} = \frac{\textup{Type 2 SS}}{\textup{Type 2 SS}+ \textup{SSE}}
```
When there is an intercept in the model the `pcorr1` and `pcorr2` are considered `missing` for the intercept.

#### Scorr 1 and 2 
`scorr1` and `scorr2` are the squared semi-partial correlation coefficient calculated as:
```math
\textup{scorr1} = \frac{\textup{Type 1 SS}}{\textup{SST}}
```

```math
\textup{scorr2} = \frac{\textup{Type 2 SS}}{\textup{SST}}
```
When there is an intercept in the model the `scorr1` and `scorr2` are considered `missing` for the intercept.

### General remarks
For all options and parameters they can be passed as a `Vector{String}` or a `Vector{Symbol}` or alternatively if only options is needed as a single `String` or `Symbol`. For instance `"all"`, `:all` or `["R2", "VIF"]` or `[:r2, :vif]`. 

## Functions
```@docs
regress(f::StatsModels.FormulaTerm, df::AbstractDataFrame, req_plots; α::Float64=0.05, req_stats=["default"], weights::Union{Nothing,String}=nothing, remove_missing=false, cov=[:none], contrasts=nothing, plot_args=Dict("plot_width" => 400, "loess_bw" => 0.6, "residuals_with_density" => false))

regress(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame; α::Float64=0.05, req_stats=["default"], weights::Union{Nothing,String}=nothing, remove_missing=false, cov=[:none], contrasts=nothing)

predict_out_of_sample(lr::linRegRes, df::AbstractDataFrame; α=0.05, req_stats=["none"], dropmissingvalues=true)

predict_in_sample(lr::linRegRes, df::AbstractDataFrame; α=0.05, req_stats=["none"], dropmissingvalues=true)

kfold(f, df, k, r = 1, shuffle=true; kwargs...)
```

## Index

```@index
```

## Content

```@contents
```
