import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def linear_regression_analysis(data_file):
    # 1. Data Preparation
    data = pd.read_csv(data_file)
    filtered_data = data[(data["LocTypeName"] == "Country/Area") & (data["Time"] < 2022)]

    # Get all unique countries and sort them alphabetically
    countries = sorted(filtered_data['Location'].unique())

    # Ask the user to choose a country
    print("Please choose a country from the list:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    choice = int(input("Enter the number corresponding to your choice: "))

    # Ensure the choice is valid
    while choice < 1 or choice > len(countries):
        print("Invalid choice. Please try again.")
        choice = int(input("Enter the number corresponding to your choice: "))

    # Set the chosen country
    chosen_country = countries[choice - 1]

    country_data = filtered_data[filtered_data['Location'] == chosen_country]

    X = country_data[["Time"]]
    y = country_data["PopTotal"]

    # 2. Model Creation and Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Country: {chosen_country}")
    print(f"Mean Squared Error: {mse}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))

    # Scatter plot of the data
    plt.scatter(X, y, color='blue', label='Actual Data')

    # Plotting the regression line
    plt.plot(X, model.predict(X), color='red', label='Regression Line')

    plt.title(f'{chosen_country} - Time vs PopTotal')
    plt.xlabel('Time')
    plt.ylabel('PopTotal')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function from main
if __name__ == '__main__':
    linear_regression_analysis("your_data_file.csv")
