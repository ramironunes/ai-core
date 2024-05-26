# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:18:33
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:18:50


import pandas as pd

from abc import ABC, abstractmethod
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


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
