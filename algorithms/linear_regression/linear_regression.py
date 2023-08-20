import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import mean_squared_error as SKMeanSquaredError

from algorithms.linear_regression.my_linear_regression import simple_linear_regression, predict, mean_squared_error, \
    my_train_test_split


def linear_regression_analysis(data_file, implementation='scikit-learn'):
    # 1. Data Preparation
    data = pd.read_csv(data_file)
    filtered_data = data[(data["LocTypeName"] == "Country/Area") & (data["Time"] < 2022)]

    # User's choice of country
    countries = sorted(filtered_data['Location'].unique())
    print("Please choose a country from the list:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")
    choice = int(input("Enter the number corresponding to your choice: "))
    chosen_country = countries[choice - 1]
    country_data = filtered_data[filtered_data['Location'] == chosen_country]

    X = country_data[["Time"]]
    y = country_data["PopTotal"]

    # 2. Model Creation and Training

    if implementation == 'scikit-learn':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        model = SKLinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = SKMeanSquaredError(y_test, y_pred)
    else:
        X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.5, random_seed=42)
        m, c = simple_linear_regression(X_train["Time"].values, y_train.values)
        y_pred = predict(X_test["Time"].values, m, c)
        mse = mean_squared_error(y_test.values, y_pred)

    # 3. Evaluation
    print(f"Country: {chosen_country}")
    print(f"Mean Squared Error: {mse}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.title(f'{chosen_country} - Time vs PopTotal')
    plt.xlabel('Time')
    plt.ylabel('PopTotal')
    plt.legend()
    plt.grid(True)
    plt.show()
