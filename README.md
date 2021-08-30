# LinearRegression.jl

LinearRegression.jl implements linear regression using the ordinary least-squares algorithm (relying on the sweep operator). 
This package is in alpha stage, hence it is likely that some bugs exists. Furthermore the API might changes in future versions. And users or prospective users 's feedback is welcome.

# Installation
Enter the Pkg REPL by pressing ] from the Julia REPL. Then install the package with: ``` pkg> add https://github.com/ericqu/LinearRegression.jl.git ```. 
To uninstall use ```  pkg> rm LinearRegression```

# Usage

The following is a simple usage:

```julia 
using LinearRegression, DataFrames, StatsModels

# x specific gravity
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
First, the GLM package does not focus on linear regression with OLS (Ordinary Least-Squares) but also provide Maximum Likelihood Estimation.

LinearRegression only support model with an intercept, GLM support model and with and without intercept.

LinearRegression does not support analytical weights (but plans to do so), GLM supports frequency weights. 

Both LinearRegression and GLM rely on StatsModels.jl for the model's description (formula) hence it is easy to move between the two packages. And categorical variables will be defined in the same way.

LinearRegression relies on the Sweep operator to estimate the coefficients and GLM relies on Cholesky and QR factorizations.

# Questions and Feedback
Please post your questions, feedabck or issues in the Issues tabs. As much as possible, please provide relevant contextual information.