# Ensemble classifier to predict Customer Attrition
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold                                                  # KFold Cross Validation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier                                               # Ensemble Max Voting
from sklearn.externals import joblib                                                        # Joblib for Persistence Model
from matplotlib import pyplot as plt
import sys
import os

classifier_name='churn-classifier.pkl'
feature_vector_max_length = 7                                                               # Maximum length of the Input Feature Vector
max_val_of_each_feature = 5                                                                 # Maximum value of each feature
churn_label_dictionary={1: 'Churn', 0:'No Churn'}                                           # Dictionary of Class Labels (Churn, No Churn)
churn_df = pd.read_csv('Bank-customers-transformed-dataset.csv')                            # Read the Customer Churn Dataset
'''
CROSS VALIDATION  -  Cross validation attempts to avoid overfitting (training on and predicting the same datapoint)
while still producing a prediction for each observation dataset.
'''
def cross_validation(X, y):                                                                 # Pass the algorithm name as the 3rd paramter
    # Construct a kfolds object
    kf = KFold(n_splits=5, random_state=None, shuffle=True)                                 # Construct 5 folds for cross validation
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]                                     # Partition the Features into Train Set and Test Set
        y_train = y[train_index]                                                            # Labels for Training
        # Initialize a classifier with key word arguments
        clf1 = SVC()
        clf2 = LogisticRegression(random_state=1)
        clf3 = KNeighborsClassifier()

        clf = VotingClassifier(estimators=[('svm', clf1), ('lr', clf2), ('knn', clf3)])     # Voting classifier using the Ensemble method
        clf.fit(X_train, y_train)                                                           # Build the Classifier
        y_pred[test_index] = clf.predict(X_test)                                            # Predict the Target labels (Y_Pred) using the Feature Test Set (X)

    joblib.dump(clf, classifier_name)                                                       # Persistence storage of the classifier
    return y_pred                                                                           # Return the Predicted y values


# Measure the accuracy. Compare the Ground-Truth Labels with the Predicted Labels
def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)                                                        # Return the accuracy

def input_validation(input_feature_vector):                                                 # Validate the Input Feature Vector
    features_with_high_value = np.where(input_feature_vector > max_val_of_each_feature)     # Check whether a feature exceeds the Maximum value specified
    # print(features_with_high_value)
    if len(features_with_high_value[0]) != 0:                                               # If features have exceeded the maximum value limit
        print('Element(s) exceeds the value of ', max_val_of_each_feature)
        sys.exit(0)

    print(input_feature_vector[0])
    if len(input_feature_vector[0]) != feature_vector_max_length:                           # If the length of the input feature exceeds the maximum length
        print('Correct number of features have not been entered')
        sys.exit(0)
    return input_feature_vector


def input_data(df):
    np.random.seed(0)                                                                       # Stop the random generation of numbers
    col_names = df.columns.tolist()
    print("Column names:")
    print(col_names)

    # Isolate target data
    y = df['Churn'].values

    # These columns are not required
    to_drop = ['CustID','Churn']
    feature_space = df.drop(to_drop, axis=1)                                              # Drop these 2 columns from the features table
    # features = feature_space.columns
    X = feature_space.as_matrix().astype(np.float)                                        # Convert the features to a feature matrix
    print("Feature space holds %d observations and %d features" % X.shape)
    print("Unique target labels:", np.unique(y))

    return X, y


def visualization(churn_df):                                                                # Visualize  the dataset
    churn_df.plot.hexbin(x='CustID', y='Numberofcheckswritten', gridsize=10)                # Draw a hexagonal bin plot
    plt.xlabel('Customer ID')
    plt.ylabel('Number of checks written')
    plt.show()


def predict_churn(input_feature_vector):                                                    # Predict Churn For the Feature Vector
    clf = joblib.load(classifier_name)                                                      # Load the pickled classifier
    # Reshape your data either using X.reshape(-1, 1)
    # if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
    # input_feature_vector = np.array([2, 3, 0, 4, 1, 1, 0])
    input_feature_vector = input_feature_vector.reshape(1, -1)
    input_feature_vector = input_validation(input_feature_vector)                           # Validate the Input Feature Vector
    y_pred = clf.predict(input_feature_vector)                                              # Predict the Class Label for the Input Feature Vector

    print('\nThe predicted label is '+str(y_pred[0])+ '  :  '+churn_label_dictionary[y_pred[0]])


def main():
    print("\nEnsemble Classifier:\n")
    print("Input feature vector for the Bank Customer: ")
    input_feature_vector = np.array([3, 3, 3, 3, 0, 4, 3])                                  # Input feature vector for the Bank Customer
    if os.path.isfile(classifier_name):                                                     # If the classifier (model) is already built
        predict_churn(input_feature_vector)                                                 # Predict churn for the input customer
        visualization(churn_df)                                                             # Visualize the dataset
    else:                                                                                   # If the classifier (model) is not built
        X, y = input_data(churn_df)                                                         # Take the dataset as the input
        print("\nAccuracy is:  %.3f\n" % accuracy(y, cross_validation(X, y)))               # Cross validate, build mdel and compute accuracy


if __name__ == "__main__":
    main()                                                                                  # Call the main function (Entry point)