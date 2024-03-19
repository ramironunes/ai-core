# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-03-19 13:57:09
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-03-19 14:11:30


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


# Load the data from the .xlsx file
df: pd.DataFrame = pd.read_excel('dataset/people.xlsx')

# Prepare the data
X: object = df[['age', 'employment', 'salary']]  # Features
y: object = df['has_money']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Make predictions
df['predicted_has_money_dt'] = classifier.predict(X)

print("-"*50)
print("* Decision Tree")

# Display the results
for index, row in df.iterrows():
    username: str = row['name']
    money: str = 'has' if row['predicted_has_money_dt'] == 1 else 'does not have'

    print(f" --> {username} {money} money")

print("-"*50)
