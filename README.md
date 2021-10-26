[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ericqu.github.io/LinearRegression.jl/dev/)
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
  σ̂²: 1.01417
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci
──────────────┼─────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -2.44811     0.819131     -2.98867     0.007877     -4.16904    -0.727184
x             │     27.6201      1.66699      16.5688  2.41337e-12      24.1179      31.1223
```

# Contrasts with Julia Stats GLM package
First, the GLM package provides more than linear regression with Ordinary Least-Squares through the Generalized Linear Model with Maximum Likelihood Estimation.

LinearRegression now accept model without intercept. Like models made with GLM the intercept is implicit, and to enable the no intercept the user must specify it in the formula (for instance ```y  ~ 0 + x```).

LinearRegression now supports analytical weights; GLM supports frequency weights.

Both LinearRegression and GLM rely on StatsModels.jl for the model's description (@formula); hence it is easy to move between the two packages. Similarly, contrasts and categorical variables are defined in the same way facilitating moving from one to the other when needed.

LinearRegression relies on the Sweep operator to estimate the coefficients, and GLM depends on Cholesky and QR factorizations.

The Akaike information criterion (AIC) is calculated with the formula relevant only for Linear Regression hence enabling comparison between linear regressions (AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors). On the other hand, the AIC calculated with GLM is more general (based on log-likelihood), enabling comparison between a broader range of models.

LinearRegression package provides access to some robust covariance estimators (for Heteroscedasticity: White, HC0, HC1, HC2 and HC3 and for HAC: Newey-West)

# List of Statistics 
## List of Statistics calculated about the linear regression model:
- AIC: Akaike information criterion with the formula AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors.
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
- The standard errors and their equivalent with a Heteroscedasticity or HAC covariance estimator
- The t values and their equivalent with a Heteroscedasticity or HAC covariance estimator
- P values for each predictor and their equivalent with a Heteroscedasticity or HAC covariance estimator

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
- https://github.com/mcreel/Econometrics for the Newey-West implementation

# Examples

The following is a short example illustrating some statistics about the predicted data.
First, a simulation of some data with a polynomial function.

```julia 
using LinearRegression, DataFrames, StatsModels
using Distributions # for the data generation with Normal() and Uniform()
using VegaLite

# Data simulation
f(x) = @. (x^3 + 2.2345x - 1.2345 + rand(Normal(0, 20)))
xs = [x for x in -2:0.1:8]
ys = f(xs)
vdf = DataFrame(y=ys, x=xs)
```
Then we can make the first model and look at the results:

```julia 
lr, ps = regress(@formula(y ~ 1 + x^3 ), vdf, "all", 
    req_stats=["default", "vif", "AIC"], 
    plot_args=Dict("plot_width" => 200 ))
lr
```
```
Model definition:       y ~ 1 + x
Used observations:      101
Model statistics:
  R²: 0.750957                  Adjusted R²: 0.748441
  MSE: 5693.68                  RMSE: 75.4565
  σ̂²: 5693.68                   AIC: 875.338
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -24.5318      10.7732     -2.27711    0.0249316     -45.9082     -3.15535          0.0
x             │     44.4953      2.57529      17.2778  1.20063e-31      39.3854      49.6052          1.0
```
This is pretty good, so let's further review some diagnostic plots.

```julia
[[ps["fit"] ps["residuals"]]
    [ps["histogram density"] ps["qq plot"]]]
```
![Overview Plots](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_072_01.svg "Overview Plots")

Please note that for the fit plot, the orange line shows the regression line, in dark grey the confidence interval for the mean, and in light grey the interval for the individuals predictions.

Plots are indicating the potential presence of a polynomial component. Hence one might try to add one by doing the following:

```julia 
lr, ps = regress(@formula(y ~ 1 + x^3 ), vdf, "all", 
    req_stats=["default", "vif", "AIC"], 
    plot_args=Dict("plot_width" => 200 ))
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      101
Model statistics:
  R²: 0.979585                  Adjusted R²: 0.979379
  MSE: 466.724                  RMSE: 21.6038
  σ̂²: 466.724                   AIC: 622.699
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     1.23626      2.65774     0.465157     0.642841     -4.03726      6.50979          0.0
x ^ 3         │     1.04075    0.0151001      68.9236  1.77641e-85      1.01079      1.07071          1.0
```
![Overview Plots](https://github.com/ericqu/LinearRegression.jl/raw/main/assets/asset_exe_072_02.svg "Overview Plots")

Further, in addition to the diagnostic plots helping confirm if the residuals are normally distributed, a few tests can be requested:

```julia
# Data simulation
f(x) = @. (x^3 + 2.2345x - 1.2345 + rand(Uniform(0, 20)))
xs = [x for x in -2:0.001:8]
ys = f(xs)
vdf = DataFrame(y=ys, x=xs)

lr = regress(@formula(y ~ 1 + x^3 ), vdf, 
    req_stats=["default", "vif", "AIC", "diag_normality"])
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      10001
Model statistics:
  R²: 0.997951                  Adjusted R²: 0.997951
  MSE: 43.4392                  RMSE: 6.59084
  σ̂²: 43.4392                   AIC: 37719.4
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     11.3151    0.0815719      138.714          0.0      11.1552       11.475          0.0
x ^ 3         │     1.03984  0.000471181      2206.87          0.0      1.03892      1.04076          1.0

Diagnostic Tests:

Kolmogorov-Smirnov test (Normality of residuals):
  KS statistic: 3.47709    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
Anderson–Darling test (Normality of residuals):
  A² statistic: 24.924901    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
Jarque-Bera test (Normality of residuals):
  JB statistic: 241.764504    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
```

Here is how to request the robust covariance estimators:

```julia 
lr = regress(@formula(y ~ 1 + x^3 ), vdf, cov=["white", "nw"])
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      101
Model statistics:
  R²: 0.979585                  Adjusted R²: 0.979379
  MSE: 466.724                  RMSE: 21.6038
Confidence interval: 95%

White's covariance estimator (HC3):
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci
──────────────┼─────────────────────────────────────────────────────────────────────────────
(Intercept)   │     1.23626      2.66559     0.463785      0.64382     -4.05285      6.52538
x ^ 3         │     1.04075    0.0145322      71.6169  4.30034e-87      1.01192      1.06959

Newey-West's covariance estimator:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci
──────────────┼─────────────────────────────────────────────────────────────────────────────
(Intercept)   │     1.23626       2.4218     0.510472     0.610857     -3.56912      6.04165
x ^ 3         │     1.04075    0.0129463      80.3897  5.60424e-92      1.01506      1.06644
```


## Notable changes since last version
- Addition of weighted regression.
- The function to predict values have been renamed. Now there is a function to predict in-sample values and one to predict out-of-sample values.
- It is possible to generate, through Vega-lite, some plot about the regression being studied.
- "contrasts" following the same syntax as GLM can be passed as a parameter to the ```regress``` function
- It is possible to request some test about normality of the residuals and heteroscedasticity (this relies on the HypothesisTests package)

