# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-21 12:38:10
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-21 13:08:34


import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_csv_no_header(file_path: str) -> tuple:
    """
    Read a CSV file without header and separate attributes and classes.

    Args:
    - file_path: The path to the CSV file.

    Returns:
    - A tuple containing attributes DataFrame and classes Series.
    """
    data = pd.read_csv(file_path, header=None)
    attributes = data.iloc[:, :-1]  # All columns except the last one
    classes = data.iloc[:, -1]       # Last column
    return attributes, classes

# Example usage:
file_path = 'data.csv'
attributes, classes = read_csv_no_header(file_path)

def preprocess_data(attributes: pd.DataFrame, classes: pd.Series) -> tuple:
    """
    Preprocess the data by splitting it into training and testing sets and scaling the attributes.

    Args:
    - attributes: DataFrame containing the attributes.
    - classes: Series containing the classes.

    Returns:
    - A tuple containing scaled attributes and split data (X_train, X_test, y_train, y_test).
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.2, random_state=42)
    
    # Standardize the attributes
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Preprocess the data
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(attributes, classes)

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> float:
    """
    Train and evaluate a logistic regression model.

    Args:
    - X_train: DataFrame containing the scaled attributes for training.
    - X_test: DataFrame containing the scaled attributes for testing.
    - y_train: Series containing the classes for training.
    - y_test: Series containing the classes for testing.

    Returns:
    - The accuracy of the trained model on the testing data.
    """
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Train and evaluate the model
accuracy = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
print("Accuracy:", accuracy)
