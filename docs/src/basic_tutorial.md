## Tutorial Linear Regression Basics

This tutorial details a simple regerssion analysis based on the Formaldehyde dataset.

### First, creating the dataset

This is done relying on the `DataFrames.jl` package.

```@example basic1
using DataFrames
df = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9], OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
```

### Second, the model is defined
We want OptDen as the dependent variable (the response) and Carb as the independent variable (the predictor). Our model will have an intercept; however, the package will implicitly add the intercept to the model. We define the model as `Optden ~ Carb`; the variable's names need to be column names from the DataFrame, which is the second argument to the `regress` function. The `lm` object will then present essential information from the regression.

```@example basic1
using LinearRegression
using StatsModels # this is requested to use the @formula

lm = regress(@formula(OptDen ~ Carb), df)
```

### Third, some illustration about the model is created
Here we will only look at the fit-plot. To obtain it, we only need to add a third argument to the `regress` function. Namely, the name of the plot requested ("fit"). When at least one plot is requested, the `regress` function will return a pair of objects: the information about the regression (as before), and an object (`Dict`) to access the requested plot(s).

```@example basic1
using VegaLite # this is the package use for plotting

lm, ps = regress(@formula(OptDen ~ Carb), df, "fit")
ps["fit"]
```

The response is plotted on the y-axis, and the predictor is plotted on the x-axis. The dark orange line represents the regression equation. The dark grey band represents the confidence interval given the α (which defaults to 0.05 and gives a 95% confidence interval). The light grey band represents the individual prediction interval. Finally, the blue circles represent the actual observations from the dataset.

#### Fourth, generate the predictions from the model
Here we get the predicted values from the model using the same Dataframe.
```@example basic1
results = predict_in_sample(lm, df)
```

#### Fifth, generate the others statistics about the model
In order to get all the statistics, one can use the "all" keyword as an argument of the `req_stats` argument.

```@example basic1
results = predict_in_sample(lm, df, req_stats="all")
```

#### Sixth, generate prediction for new data
We first create a new DataFrame that needs to use the same column names used in the model. In our case, there is only one column: "Carb".

```@example basic1
ndf = DataFrame(Carb= [0.11, 0.22, 0.55, 0.77])
predictions = predict_out_of_sample(lm, ndf)
```

