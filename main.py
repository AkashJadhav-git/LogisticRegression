import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():

    # Read in the advertising.csv file and set it to a data frame called ad_data.
    ad_data = pd.read_csv('advertising.csv')

    # Print the information related to data.
    print(ad_data.info())

    # Split the data into training set and testing set using train_test_split.
    X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
    y = ad_data['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train and fit a logistic regression model on the training set.
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)

    # predict values for the testing data.
    prediction = logmodel.predict(X_test)

    # Create a classification report for the model.
    print(classification_report(y_test, prediction))


if __name__ == '__main__':
    main()
