# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-03-19 14:10:25
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-03-19 14:16:22


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd


# Assuming 'people.xlsx' is in the correct directory and has the necessary columns
df: pd.DataFrame = pd.read_excel('dataset/people.xlsx')

# Preparing the data
X: object = df[['age', 'employment', 'salary']]  # Features
y: object = df['has_money']  # Target

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the kNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Making predictions with kNN
df['predicted_has_money_knn'] = knn.predict(X)

# Displaying the results
print("-"*50)
print("* k-Nearest Neighbors (kNN)")

for index, row in df.iterrows():
    username: str = row['name']
    money: str = 'has' if row['predicted_has_money_knn'] == 1 else 'does not have'

    print(f"{row['name']} {money} money")

print("-"*50)
