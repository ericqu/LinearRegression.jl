## Tutorial multiple linear regression with categorical variables

This tutorial details a multiple regression analysis based on the "carseat" dataset (Information about car seat sales in 400 stores). This tutorial follows roughly the same steps done the datasets in the "An Introduction to Statistical Learning" book (https://www.statlearning.com/), from pages 119, 120, and 124.

### First, creating the dataset.

We create the dataset with the help of the `DataFrames.jl` and `Download` packages.

```@example multi1
using Downloads, DataFrames, CSV
df = DataFrame(CSV.File(Downloads.download("https://raw.githubusercontent.com/Kulbear/ISLR-Python/master/data/Carseats.csv")))
describe(df)
```

### Second, basic analysis.
We make a model with all variables and a couple of interactions.

```@example multi1
using LinearRegression, StatsModels
lm = regress(@formula(Sales ~ CompPrice + Income + Advertising + Population + Price + 
            ShelveLoc + Age + Education + Urban + US + Income & Advertising + Price & Age), df)
```

To have better explainability, we choose to set the base for the Shelve Location (ShelveLoc) as "Medium" so that the results highlight what happens when it is "Bad" or "Good". Furthermore, to form an idea about how collinear the predictors are, we request the Variance inflation factor (VIF).

```@example multi1
lm = regress(@formula(Sales ~ CompPrice + Income + Advertising + Population + Price + 
            ShelveLoc + Age + Education + Urban + US + Income & Advertising + Price & Age), df, 
            req_stats=["default", "vif"],
            contrasts= Dict(:ShelveLoc => DummyCoding(base="Medium"), :Urban => DummyCoding(base="No"), :US => DummyCoding(base="No") ))
```

Now let's assume we want our response to be Sales and the predictors to be Price, Urban, and US:

```@example multi1
lm = regress(@formula(Sales ~ Price + Urban + US), df,
            contrasts= Dict(:ShelveLoc => DummyCoding(base="Medium"), :Urban => DummyCoding(base="No"), :US => DummyCoding(base="No") ))
```

Indeed, we note that "Urban:Yes" appears to have a low significance. Hence we could decide to make our model without this predictor:
```@example multi1
lm = regress(@formula(Sales ~ Price + US), df,
            contrasts= Dict(:ShelveLoc => DummyCoding(base="Medium"), :Urban => DummyCoding(base="No"), :US => DummyCoding(base="No") ))
```

To identify potential outliers and high leverage variables, we choose to plot the Cook's Distance and the leverage plot.

```@example multi1
using VegaLite
lm, ps = regress(@formula(Sales ~ Price + US), df, "all", 
            req_stats=["default", "vif"],
            contrasts= Dict(:ShelveLoc => DummyCoding(base="Medium"), :Urban => DummyCoding(base="No"), :US => DummyCoding(base="No") )) 
p = [ps["leverage"]
    ps["cooksd"]]
```

Alternatively, we can also use the predicted values and their statistics to create a new data frame with the entries of interest (here, we show only the first three entries).

```@example multi1
results = predict_in_sample(lm, df, req_stats="all")

threshold_cooksd = 4 / lm.observations
potential_outliers = results[ results.cooksd .> threshold_cooksd , :]
potential_outliers[1:3, 1:3]
```

```@example multi1
threshold_leverage = 2 * lm.p / lm.observations
potential_highleverage = results[ abs.(results.leverage) .> threshold_leverage , : ]
potential_highleverage[1:3, 1:3]
```
