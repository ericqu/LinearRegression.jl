var documenterSearchIndex = {"docs":
[{"location":"#LinearRegression.jl-Documentation","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"LinearRegression.jl implements linear regression using the least-squares algorithm (relying on the sweep operator). This package is in the alpha stage. Hence it is likely that some bugs exist. Furthermore, the API might change in future versions.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The usage aims to be straightforward, a call to regress to build a linear regression model, and a call to predict_and_stats to predict data using the built linear regression model.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The regress call will compute some statistics about the fitted model in addition to the coefficients. ","category":"page"},{"location":"#Number-of-observations-and-variables","page":"LinearRegression.jl Documentation","title":"Number of observations and variables","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The number of observations n used to fit the model.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The number of independent variables p used in the model.","category":"page"},{"location":"#Total-Sum-of-Squares","page":"LinearRegression.jl Documentation","title":"Total Sum of Squares","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Total Sum of Squares (SST) is calculated but not presented to the user. In case of model with intercept the SST is computed with the following:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmSST=sum_i=1^nleft(y_i-baryright)^2","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"And when there is no intercept with the following: ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmSST=sum_i=1^n y_i^2","category":"page"},{"location":"#Error-Sum-of-Squares","page":"LinearRegression.jl Documentation","title":"Error Sum of Squares","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Error Sum of Squares (or SSE) also known as Residual Sum of Square (RSS). This package uses the sweep operator (Goodnight, J. (1979). \"A Tutorial on the SWEEP Operator.\" The American Statistician.) to compute the SSE.","category":"page"},{"location":"#Mean-Squared-Error","page":"LinearRegression.jl Documentation","title":"Mean Squared Error","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Mean Squared Error (MSE) is calculated as ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmMSE = displaystylefracmathrmSSEn - p","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Root Mean Squared Error (RMSE) is calculated as ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmRMSE = sqrtmathrmMSE","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The MSE is the estimator of σ̂² unless at least one robust covariance estimator is requested.","category":"page"},{"location":"#R-and-Adjusted-R","page":"LinearRegression.jl Documentation","title":"R² and Adjusted R²","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The R² (R2 or R-squared) see (https://en.wikipedia.org/wiki/Coefficientofdetermination) is calculated with the following formula: ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmR^2 = 1 - displaystylefracmathrmSSEmathrmSST","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Adjusted R² (ADJR2) is computed with the following formulas:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"when it is a model with an intercept:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmADJR^2 = 1 - displaystyle frac(n-1)(1-mathrmR^2)n-p","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"And when there is no intercept:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmADJR^2 = 1 - displaystyle frac(n)(1-mathrmR^2)n-p","category":"page"},{"location":"#Akaike-information-criterion","page":"LinearRegression.jl Documentation","title":"Akaike information criterion","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The Akaike information criterion is calculated with the Linear Regression specific formula:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"mathrmAIC = displaystyle n ln left( fracmathrmSSEn right) + 2p","category":"page"},{"location":"#t-statistic-and-confidence-interval","page":"LinearRegression.jl Documentation","title":"t-statistic and confidence interval","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The t-statistic is computed by using the inverse cumulative t-distribution (with quantile()) with parameter (n - p) at 1 - fracα2. ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The standard errors of the coefficients are calculated by multiplying the Sigma (estimated by the MSE) with the pseudo inverse matrix (resulting from the sweep operator), out of which the square root of the diagonal elements are extracted.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The t-values are calculated as the coefficients divided by their standard deviation.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The upper bound of the confidence interval for each coefficient is calculated as the coeffiecent + coefficient's standard error * t_statistic.","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The lower bound of the confidence interval for each coefficient is calculated as the coeffiecent - coefficient's standard error * t_statistic.","category":"page"},{"location":"#p-values","page":"LinearRegression.jl Documentation","title":"p-values","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The p-values are computed using the F Distribution, the degree of freedom for each coefficent.","category":"page"},{"location":"#Variance-inflation-factor","page":"LinearRegression.jl Documentation","title":"Variance inflation factor","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"Variance inflation factor (VIF) is calculated by taking the  diagonal elements of the inverse of the correlation matrix formed by the independent variables.","category":"page"},{"location":"#Robust-covariance-estimators","page":"LinearRegression.jl Documentation","title":"Robust covariance estimators","text":"","category":"section"},{"location":"#Heteroscedasticity-estimators","page":"LinearRegression.jl Documentation","title":"Heteroscedasticity estimators","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The user can select estimators from these list. If the user select \"White\" as an estimator then HC3 will be selected for a small size (n < 250) otherwise HC0 will be selected.","category":"page"},{"location":"#HC0","page":"LinearRegression.jl Documentation","title":"HC0","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The following estimators can be calculated. Having InvMat the pseudo inverse resulting from the sweep operator. And having xe being the matrix of the independent variables times the residuals. Then HC0 is calculated as:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"textupHC0 = sqrtdiag(textupInvMat  textupxe textupxe textup InvMat)","category":"page"},{"location":"#HC1","page":"LinearRegression.jl Documentation","title":"HC1","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"Having n being the number of observations and p the number of variables. Then HC1 is calculated as:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"textupHC1 = sqrtdiag(textupInvMat  textupxe textupxe textup InvMat  fracnn-p)","category":"page"},{"location":"#HC2","page":"LinearRegression.jl Documentation","title":"HC2","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"The leverage or hat matrix is calculated as:","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"textupH = textupX (textupXX)^-1textupX","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"xe is scaled by frac11 - H then ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"textupHC2 = sqrtdiag(textupInvMat  textupxe textupxe textup InvMat  )","category":"page"},{"location":"#HC3","page":"LinearRegression.jl Documentation","title":"HC3","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"xe is scaled by frac1left( 1 - H right)^2 then ","category":"page"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"textupHC3 = sqrtdiag(textupInvMat  textupxe textupxe textup InvMat  )","category":"page"},{"location":"#Heteroskedasticity-and-autocorrelation-consistent-estimator","page":"LinearRegression.jl Documentation","title":"Heteroskedasticity and autocorrelation consistent estimator","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"Newey-West estimator calculation is not documented yet. See reference implementation current implementation for details.","category":"page"},{"location":"#Functions","page":"LinearRegression.jl Documentation","title":"Functions","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=[\"all\"], remove_missing=false, cov=[:none])\npredict_and_stats(lr::linRegRes, df::DataFrames.DataFrame; α=0.05, req_stats=[\"none\"])","category":"page"},{"location":"#LinearRegression.regress-Tuple{FormulaTerm, DataFrame}","page":"LinearRegression.jl Documentation","title":"LinearRegression.regress","text":"function regress(f::StatsModels.FormulaTerm, df::DataFrames.DataFrame; α::Float64=0.05, req_stats=[\"all\"], remove_missing=false, cov=[:none])\n\nEstimate the coefficients of the regression, given a dataset and a formula. \n\nThe formula details are provided in the StatsModels package and the behaviour aims to be similar as what the Julia GLM package provides.\nThe data shall be provided as a DataFrame without missing data.\nIf remove_missing is set to true a copy of the dataframe will be made and the row with missing data will be removed.\n\n\n\n\n\n","category":"method"},{"location":"#LinearRegression.predict_and_stats-Tuple{linRegRes, DataFrame}","page":"LinearRegression.jl Documentation","title":"LinearRegression.predict_and_stats","text":"function predict_and_stats(lr::linRegRes, df::DataFrames.DataFrame, α=0.05)\n\nUsing the estimated coefficients from the regression make predictions, and calculate related statistics.\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"LinearRegression.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"LinearRegression.jl Documentation","title":"LinearRegression.jl Documentation","text":"","category":"page"}]
}
