from algorithms.linear_regression import linear_regression as lr

if __name__ == '__main__':
    implementation_choice = input("Which implementation would you like to use? (scikit-learn/custom): ").lower()
    lr.linear_regression_analysis("./datasets/WPP2022_TotalPopulationBySex.csv", implementation_choice)
