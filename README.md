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
Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)       low ci      high ci          VIF
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -2.44811     0.819131     -2.98867     0.007877     -4.16904    -0.727184          0.0
x             │     27.6201      1.66699      16.5688  2.41337e-12      24.1179      31.1223          1.0
```

# Contrasts with Julia Stats GLM package
First, the GLM package provides more than linear regression with Ordinary Least-Squares through the Generalized Linear Model with Maximum Likelihood Estimation.

LinearRegression package only supports models with an intercept, GLM supports models with and without intercept.

LinearRegression does not support analytical weights (however it is under consideration), GLM supports frequency weights. 

Both LinearRegression and GLM rely on StatsModels.jl for the model's description (@formula); hence it is easy to move between the two packages. Similarly, categorical variables are defined in the same way facilitating moving from one to the other when needed.

LinearRegression relies on the Sweep operator to estimate the coefficients, and GLM depends on Cholesky and QR factorizations.

The Akaike information criterion (AIC) is calculated with the formula relevant only for Linear Regression hence enabling comparison between linear regressions (AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p the number of predictors). The AIC calculated with GLM is more general (based on log likelihood) enabling comparison between a wider range of models.

# Questions and Feedback
Please post your questions, feedabck or issues in the Issues tabs. As much as possible, please provide relevant contextual information.

# Credits and additional information
- Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.
- Gordon, R. A. (2015). Regression Analysis for the Social Sciences. New York and London: Routledge.
- https://blogs.sas.com/content/iml/2021/07/14/performance-ls-regression.html
- https://github.com/joshday/SweepOperator.jl
- http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/12-sweep/sweep.html
