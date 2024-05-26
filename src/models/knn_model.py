# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:20:04
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:24:11


from src.abstracts.base_model import BaseModel

from sklearn.neighbors import KNeighborsClassifier


class KNNModel(BaseModel):
    def define_model(self):
        self.model = KNeighborsClassifier()
        self.params = {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

