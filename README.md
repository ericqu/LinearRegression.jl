# LinearRegression.jl

LinearRegression.jl implements linear regression using the ordinary least-squares algorithm (relying on the sweep operator). 
This package is in alpha stage, hence it is likely that some bugs exists. Furthermore the API might changes in future versions. And users or prospective users 's feedback is welcome.

# Installation
Enter the Pkg REPL by pressing ] from the Julia REPL. Then install the package with: ``` pkg> add https://github.com/ericqu/ ```. 
To uninstall use ```  pkg> rm LinearRegression```

# Usage

# Contrasts with Julia Stats GLM package
First, the GLM package does not focus on linear regression with OLS (Ordinary Least-Squares) but also provide Maximum Likelihood Estimation.

LinearRegression only support model with an intercept, GLM support model and with and without intercept.

LinearRegression does not support analytical weights (but plans to do so), GLM supports frequency weights. 

Both LinearRegression and GLM rely on StatsModels.jl for the model's description (formula) hence it is easy to move between the two packages. And categorical variables will be defined in the same way.

LinearRegression relies on the Sweep operator to estimate the coefficients and GLM relies on Cholesky and QR factorizations.

# Questions and Feedback
Please post your questions, feedabck or issues in the Issues tabs. As much as possible, please provide relevant contextual information.