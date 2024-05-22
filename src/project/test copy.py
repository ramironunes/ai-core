# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-21 17:49:33
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-21 18:48:59


import os
import pandas as pd

from abc import ABC, abstractmethod
from collections import Counter
from openpyxl import Workbook

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """
    Abstract base class for different machine learning models.
    """

    def __init__(self):
        self.model = None
        self.params = {}

    @abstractmethod
    def define_model(self):
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        # Dynamically determine the number of splits for cross-validation
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        cv_splits = max(2, min(5, min_class_count))  # Ensure cv is at least 2

        if self.params:
            grid_search = GridSearchCV(self.model, self.params, cv=cv_splits, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


class LogisticRegressionModel(BaseModel):
    def define_model(self):
        self.model = LogisticRegression(max_iter=2000, solver='saga')  # Increased max_iter


class KNNModel(BaseModel):
    def define_model(self):
        self.model = KNeighborsClassifier()
        self.params = {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }


class PerceptronModel(BaseModel):
    def define_model(self):
        self.model = Perceptron(max_iter=1000)


class SVMModel(BaseModel):
    def define_model(self):
        self.model = SVC()
        self.params = {
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }


class DecisionTreeModel(BaseModel):
    def define_model(self):
        self.model = DecisionTreeClassifier()


class MLPModel(BaseModel):
    def define_model(self):
        self.model = MLPClassifier(max_iter=1000)  # Increased max_iter
        self.params = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001]
        }


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


def preprocess_data(attributes: pd.DataFrame, classes: pd.Series) -> tuple:
    """
    Preprocess the data by splitting it into training and testing sets 
    and scaling the attributes.

    Args:
    - attributes: DataFrame containing the attributes.
    - classes: Series containing the classes.

    Returns:
    - A tuple containing scaled attributes and split data (X_train, X_test, y_train, y_test).
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        attributes, classes, test_size=0.2, random_state=42)
    
    # Standardize the attributes
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_all_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                  y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Train and evaluate multiple models and return their accuracies.

    Args:
    - X_train: DataFrame containing the scaled attributes for training.
    - X_test: DataFrame containing the scaled attributes for testing.
    - y_train: Series containing the classes for training.
    - y_test: Series containing the classes for testing.

    Returns:
    - A dictionary containing the accuracy of each model.
    """
    models = [
        LogisticRegressionModel(),
        KNNModel(),
        PerceptronModel(),
        SVMModel(),
        DecisionTreeModel(),
        MLPModel()
    ]

    accuracies = {}
    for model in models:
        model.define_model()
        model.train(X_train, y_train)
        accuracy = model.evaluate(X_test, y_test)
        accuracies[model.__class__.__name__] = accuracy
    
    return accuracies


def process_all_datasets(directory_path: str) -> pd.DataFrame:
    """
    Process all CSV files in the given directory, train and evaluate models,
    and print the accuracy for each dataset. Also, save the results to a DataFrame.

    Args:
    - directory_path: The path to the directory containing the CSV files.

    Returns:
    - A DataFrame containing the results for all datasets.
    """
    results = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")
            
            # Read and preprocess the data
            attributes, classes = read_csv_no_header(file_path)
            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(attributes, classes)
            
            # Train and evaluate the models
            accuracies = train_and_evaluate_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
            accuracies['filename'] = filename
            results.append(accuracies)
            print(f"Accuracies for {filename}: {accuracies}")
    
    return pd.DataFrame(results)


def save_results_to_excel(results: pd.DataFrame, output_path: str) -> None:
    """
    Save the results DataFrame to an Excel file.

    Args:
    - results: DataFrame containing the results.
    - output_path: The path where the Excel file will be saved.
    """
    results.to_excel(output_path, index=False)


# Directory containing the CSV files
directory_path = 'src/project/resources/dataset'

# Process all datasets and save the results
results_df = process_all_datasets(directory_path)
output_path = 'model_accuracies.xlsx'
save_results_to_excel(results_df, output_path)
print(f"Results saved to {output_path}")
