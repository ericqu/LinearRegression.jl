# LinearRegression.jl

LinearRegression.jl implements linear regression using the least-squares algorithm (relying on the sweep operator). This package is in the alpha stage. Hence it is likely that some bugs exist. Furthermore, the API might change in future versions. User's or prospective users' feedback is welcome.

# Installation
Enter the Pkg REPL by pressing ] from the Julia REPL. Then install the package with: ``` pkg> add https://github.com/ericqu/LinearRegression.jl.git ```. 
To uninstall use ```  pkg> rm LinearRegression```

# Usage

The following is a simple usage:

```julia 
using LinearRegression, DataFrames, StatsModels

x = [0.68, 0.631, 0.348, 0.413, 0.698, 0.368, 0.571, 0.433, 0.252, 0.387, 0.409, 0.456, 0.375, 0.495, 0.55, 0.576, 0.265, 0.299, 0.612, 0.631]
y = [15.72, 14.86, 6.14, 8.21, 17.07, 9.07, 14.68, 10.37, 5.18, 9.36, 7.61, 10.43, 8.93, 10.33, 14.46, 12.39, 4.06, 4.67, 13.73, 14.75]

df = DataFrame(y=y, x=x)

lr = regress(@formula(y ~ 1 + x), df)

```

which outputs the following information:
```
Model definition:       y ~ 1 + x
Used observations:      20
Model statistics:
  R²: 0.938467                  Adjusted R²: 0.935049
  MSE: 1.01417                  RMSE: 1.00706
  σ̂²: 1.01417                   AIC: 2.17421
Confidence interval: 95%
Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -2.44811     0.819131     -2.98867     0.007877     -4.16904    -0.727184          0.0
x             │     27.6201      1.66699      16.5688  2.41337e-12      24.1179      31.1223          1.0
```

# Contrasts with Julia Stats GLM package
First, the GLM package provides more than linear regression with Ordinary Least-Squares through the Generalized Linear Model with Maximum Likelihood Estimation.

LinearRegression package only supports models with an intercept, GLM supports models with and without intercept.

LinearRegression does not support analytical weights (however, it is under consideration); GLM supports frequency weights.

Both LinearRegression and GLM rely on StatsModels.jl for the model's description (@formula); hence it is easy to move between the two packages. Similarly, categorical variables are defined in the same way facilitating moving from one to the other when needed.

LinearRegression relies on the Sweep operator to estimate the coefficients, and GLM depends on Cholesky and QR factorizations.

The Akaike information criterion (AIC) is calculated with the formula relevant only for Linear Regression hence enabling comparison between linear regressions (AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors). On the other hand, the AIC calculated with GLM is more general (based on log-likelihood), enabling comparison between a broader range of models.

# List of Statistics 
## List of Statistics calculated about the linear regression model:
- AIC: Akaike information criterion with the formula AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors.
- P values for each predictor.
- SSE Sum of Squared Errors as the output from the sweep operator.
- SST as the Total Sum of Squares as the sum over all squared differences between the observations and their overall mean.
- R² as 1 - SSE/SST.
- Adjusted R².
- σ̂² (sigma) Estimate of the error variance.
- Variance Inflation Factor.
- CI the confidence interval based the \alpha default value of 0.05 giving the 95% confidence interval.
- The t-statistic.
- The mean squared error.
- The root of the mean squared error.
- The standard errors.
- The t values.

## List of Statistics about the predicted values:
- The predicted values
- The residuals values (as the actual values minus the predicted ones)
- The Leverage or the i-th diagonal element of the projection matrix.
- STDI is the standard error of the individual predicted value.
- STDP is the standard error of the mean predicted value
- STDR is the standard error of the residual
- Student as the studentized residuals also knows as the Standardized residuals or internally studentized residuals.
- Rstudent is the studentized residual with the current observation deleted.
- LCLI is the lower bound of the confidence interval for the individual prediction.
- UCLI is the upper bound of the confidence interval for the individual prediction.
- LCLP is the lower bound of the confidence interval for the expected (mean) value.
- UCLP is the upper bound of the confidence interval for the expected (mean) value.
- Cook's Distance
- PRESS as the sum of squares of predicted residual errors

# Questions and Feedback
Please post your questions, feedabck or issues in the Issues tabs. As much as possible, please provide relevant contextual information.

# Credits and additional information
- Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.
- Gordon, R. A. (2015). Regression Analysis for the Social Sciences. New York and London: Routledge.
- https://blogs.sas.com/content/iml/2021/07/14/performance-ls-regression.html
- https://github.com/joshday/SweepOperator.jl
- http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/12-sweep/sweep.html


# Example

The following is a short example illustrating some statistics about the predicted data.
First we simulate some data with a polynomial function.

```julia 
using LinearRegression, DataFrames, StatsModels
using Distributions # for the data generation with Normal()
using VegaLite # for plotting

# Data simulation
f(x) = @. (x^3 + 2.2345x - 1.2345 + rand(Normal(0, 3)))
xs = [x for x in -2:0.1:8]
ys = f(xs)
vdf = DataFrame(y=ys, x=xs)

```
Then we can make the first model and look at the results:

```julia 
lr = regress(@formula(y ~ 1 + x ), vdf)
```
```
Model definition:  y ~ 1 + x
Used observations:      101
Model statistics:
  R²: 0.770534                  Adjusted R²: 0.768216
  MSE: 5169.6                   RMSE: 71.8999
  σ̂²: 5169.6                    AIC: 865.586
Confidence interval: 95%
Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -24.8724      10.2654     -2.42292    0.0172133     -45.2412     -4.50351          0.0
x             │     44.7417      2.45391      18.2329  2.06014e-33      39.8727      49.6108          1.0
```
Which is pretty good, so let's further review some diagnostic plots.

```julia
select(results, [:predicted, :y]) |> @vlplot(
    :point, 
    x = { :predicted,  axis = {grid = false}},
    y = { :y, axis = {grid = false}},
    title = "Predicted vs actual", width = 400, height = 400
)
```
![Predicted vs actual model 1](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_001.svg "Predicted vs actual model 1")


```julia
select(results, [:predicted, :residuals]) |> 
        @vlplot(title = "Predicted vs residuals", width = 400, height = 400, x = {axis = {grid = false}}, y = {axis = {grid = false}}   ) +
        @vlplot(:point, :predicted, :residuals) +
        @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = 0})
```

![Predicted vs residuals model 1](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_002.svg "Predicted vs residuals model 1")

Both plots indicating the potential presence a polynomial component.
Hence one might try to add one by doing the following:

```julia 
lr = regress(@formula(y ~ 1 + x + x^2 ), vdf)
```
Giving:
```
Model definition:  y ~ 1 + x + :(x ^ 2)
Used observations:      101
Model statistics:
  R²: 0.982048                  Adjusted R²: 0.981681
  MSE: 408.574                  RMSE: 20.2132
  σ̂²: 408.574                   AIC: 610.235
Confidence interval: 95%
Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     -20.377      2.88895     -7.05342  2.49191e-10       -26.11      -14.644          0.0
x             │    -9.20267      1.73096     -5.31652   6.65925e-7     -12.6377     -5.76764      6.29568
x ^ 2         │     8.99074     0.264591      33.9798  5.00484e-56      8.46566      9.51581      6.29568
```
Which aside from the slightly high VIF does not indicate anything wrong. 
Lets look at the updated "predicted vs residuals" plots (same code used):

![Predicted vs residuals model 2](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_003.svg "Predicted vs residuals model 2")

Which still show a strong pattern. 
To shorten the analysis we can add the cubic component in the model:

```julia 
lr = regress(@formula(y ~ 1 + x + x^3 ), vdf)
```
Giving:
```
Model definition:  y ~ 1 + x + :(x ^ 3)
Used observations:      101
Model statistics:
  R²: 0.999623                  Adjusted R²: 0.999615
  MSE: 8.58223                  RMSE: 2.92954
  σ̂²: 8.58223                   AIC: 220.074
Confidence interval: 95%
Coefficients statistics:
Terms ╲ Stats │        Coefs       Std err             t      Pr(>|t|)        low ci       high ci           VIF
──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     -1.47862       0.42911      -3.44578   0.000839903      -2.33018     -0.627064           0.0
x             │      2.44422      0.200118       12.2139   2.11167e-21       2.04709       2.84134       4.00602
x ^ 3         │     0.999989    0.00409832       243.999  2.99597e-138      0.991856       1.00812       4.00602
```
And the following Predicted vs Residuals plots:

![Predicted vs residuals model 3](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_004.svg "Predicted vs residuals model 3")

Which this time show residuals scatters without an obvious pattern, potentially showing the Normal error component of the model.

Without surprise, the "Predicted vs actual" plot now shows a linear relationship. 
![Predicted vs actual](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_005.svg "Predicted vs actual, model 3")

To look to outliers of interest additional plots can be used such as: Leverage vs Rstudent (or studentized residual with the current observation deleted), as well as the Cook's Distance.

```julia
threshold_leverage = 2 * lr.p / lr.observations
select(results, [:leverage, :rstudent]) |> 
    @vlplot(title = "Leverage vs Rstudent", width = 400, height = 400,
        x = {axis = {grid= false}}, y = {axis = {grid= false}}   ) +
    @vlplot(:point, :leverage, :rstudent) +
    @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = -2}) +
    @vlplot(mark = {:rule, color = :darkgrey}, x = {datum = threshold_leverage}) +
    @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = 2}) 

threshold_cooksd = 4 / lr.observations
select(results, [:x, :cooksd]) |> 
    @vlplot(title = "x vs Cook's Distance", width = 400, height = 200) +
    @vlplot(
        mark={:rule, color = :steelblue},
        enc =
        {
            x = {:x, type = :quantitative, axis = {grid = false}},
            y = {datum = 0} ,
            y2 = :cooksd
        }
        ) +
    @vlplot(mark = {:rule, color = :darkgrey}, y = {datum = threshold_cooksd})  
```

![Leverage vs RStudent](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_006.svg "Leverage vs RStudent")

![x vs Cook's distance](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_007.svg "x vs Cook's distance")

