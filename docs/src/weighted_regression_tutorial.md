## Tutorial weighted regression

This tutorial gives a brief introduction to simple weighted regression using analytical weights. The tutorial makes use of the short dataset available on this [sas blog post](https://blogs.sas.com/content/iml/2016/10/05/weighted-regression.html).

### First, creating the dataset.

We create the dataset with the help of the `DataFrames.jl` package.

```@example weightedregression
using DataFrames
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
    ]
    df = DataFrame(tw, [:y,:x,:w])
```

### Second, make a basic analysis

We make a simple linear regression.

```@example weightedregression
using LinearRegression, StatsModels
using VegaLite

f = @formula(y ~ x)
lms, pss = regress(f, df, "fit")
lms
```

And then the weighted regression version:

```@example weightedregression
lmw, psw = regress(f, df, "fit", weights="w")
lmw
```
The output of the model indicates that this is a weighted regression.
We also note that the number of observations is 10 instead of 11 for the simple regression. This is because the last observation weights 0, and as the package only uses positive weights, it is not used to fit the regression model.

For comparison, we fit the simple regression with only the first 10 observations.


```@example weightedregression
df = first(df, 10)
lms, pss = regress(f, df, "fit")
lms
```
We can now realise that the coefficients are indeed differents with the weighted regression.

We can then contrast the fit plot from both regressions.
```@example weightedregression
[pss["fit"] psw["fit"]]
```

We note that the regression line is indeed "flatter" in the weighted regression case.
We also note that the prediction interval is presented differently (using error bars), and it shows a different shape, reflecting the weights' importance.

