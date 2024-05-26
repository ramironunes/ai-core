# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:21:07
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:25:20


from src.abstracts.base_model import BaseModel

from sklearn.neural_network import MLPClassifier


class MLPModel(BaseModel):
    def define_model(self):
        self.model = MLPClassifier(max_iter=1000)
        self.params = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001]
        }
