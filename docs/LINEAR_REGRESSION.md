# Mathematics Behind Simple Linear Regression

Simple linear regression is a method used to find a linear relationship between a dependent variable (y) and an independent variable (x).

## Equation of a Line

The equation of a line in slope-intercept form is given by:

`y = m * x + c`

Where:
- `y` is the dependent variable (what we're trying to predict),
- `x` is the independent variable (the input),
- `m` is the slope of the line,
- `c` is the y-intercept.

## Calculating the Slope and Y-intercept

For simple linear regression, the slope (m) and y-intercept (c) can be calculated using the following formulas:

Slope `m` is:
```
m = (n * sum of (x*y) - sum of x * sum of y) / (n * sum of (x^2) - (sum of x)^2)
```

Y-intercept `c` is:
```
c = (sum of y - m * sum of x) / n
```

Where:
- `n` is the number of observations
- `sum of x` represents the total of all x values
- `sum of y` represents the total of all y values
- `sum of (x*y)` is the sum of the product of each pair of x and y values
- `sum of (x^2)` is the sum of the squared x values

## Mean Squared Error (MSE)

Once we have our regression line, we need a way to measure how well it fits the data. One common method is to use the Mean Squared Error (MSE). The formula for MSE is:

```
MSE = 1/n * sum of ((actual y value - predicted y value from our regression line)^2)
```

Where:
- Each `actual y value` corresponds to the observed y values from the data.
- Each `predicted y value from our regression line` corresponds to the y value predicted by our regression equation for the given x value.
- `n` is the number of observations.
