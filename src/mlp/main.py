# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-04-02 07:13:47
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-04-02 19:23:29


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd


# Load the data from the Excel file
data = pd.read_excel('dataset/developers.xlsx')

# Suppose the spreadsheet has columns 'Name', 'NumLanguages', and 'IsFullStack'
# 'NumLanguages' is the feature (X) and 'IsFullStack' is the label (Y)

# Extracting the feature (number of programming languages known)
X = data['NumLanguages'].values.reshape(-1, 1)

# Creating the label (IsFullStack), determining if someone is a
# full stack developer based on the number of languages known
data['IsFullStack'] = data['NumLanguages'].apply(lambda x: 1 if x > 2 else 0)
Y = data['IsFullStack'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42,
)

# Create the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=1)

# Train the model with the training data
mlp.fit(X_train, Y_train)

# Predict the output with the testing data
predictions = mlp.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Debugging: Print the names of programmers and their full stack status
for index, row in data.iterrows():
    print(
        f"Programmer: {row['Name']}, Full Stack: " + \
        f"{'Yes' if row['IsFullStack'] == 1 else 'No'}"
    )
    