from algorithms.linear_regression import linear_regression as lr
from algorithms.logistic_regression import logistic_regression as lor


def main_menu():
    print("Welcome to the ML Algorithms Showcase")
    print("======================================")
    print("1. Logistic Regression: Spam Classification")
    print("2. Linear Regression Analysis")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")
    return choice


def logistic_regression_menu():
    print("\nLogistic Regression for Spam Classification")
    implementation_choice = input("Which implementation would you like to use? (scikit-learn/custom): ").lower()
    if implementation_choice in ['scikit-learn', 'custom']:
        lor.spam_classification("./datasets/spam_or_not_spam.csv", implementation_choice)
    else:
        print("Invalid choice. Going back to main menu.")


def linear_regression_menu():
    implementation_choice = input(
        "\nLinear Regression Analysis\nWhich implementation would you like to use? (scikit-learn/custom): ").lower()
    if implementation_choice in ['scikit-learn', 'custom']:
        lr.linear_regression_analysis("./datasets/WPP2022_TotalPopulationBySex.csv", implementation_choice)
    else:
        print("Invalid choice. Going back to main menu.")


if __name__ == '__main__':
    while True:
        choice = main_menu()
        if choice == '1':
            logistic_regression_menu()
        elif choice == '2':
            linear_regression_menu()
        elif choice == '3':
            print("Thank you for using the ML Algorithms Showcase. Goodbye!")
            break
        else:
            print("Invalid choice. Please choose again.")
