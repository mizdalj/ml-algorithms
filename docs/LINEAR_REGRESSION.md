# Mathematics Behind Simple Linear Regression

Simple linear regression is a method used to find a linear relationship between a dependent variable \(y\) and an independent variable \(x\).

## Equation of a Line

The equation of a line in slope-intercept form is:

\[ y = mx + c \]

Where:
- \( y \) is the dependent variable (what we're trying to predict),
- \( x \) is the independent variable (the input),
- \( m \) is the slope of the line,
- \( c \) is the y-intercept.

## Calculating the Slope and Y-intercept

For simple linear regression, the formulas to calculate the slope \( m \) and y-intercept \( c \) are:

\[ m = \frac{n(\sum xy) - (\sum x)(\sum y)}{n(\sum x^2) - (\sum x)^2} \]
\[ c = \frac{\sum y - m(\sum x)}{n} \]

Where:
- \( n \) is the number of observations
- \( \sum x \) is the sum of the x values
- \( \sum y \) is the sum of the y values
- \( \sum xy \) is the sum of the product of each pair of x and y values
- \( \sum x^2 \) is the sum of the squared x values

## Mean Squared Error (MSE)

Once we have our regression line, we need a way to measure how well it fits the data. One common method is to use the Mean Squared Error (MSE). The formula for the MSE is:

\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

Where:
- \( y_i \) is the actual value
- \( \hat{y}_i \) is the predicted value from our regression line
- \( n \) is the number of observations
