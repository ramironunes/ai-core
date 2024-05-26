# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:20:46
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:24:37


from src.abstracts.base_model import BaseModel

from sklearn.svm import SVC


class SVMModel(BaseModel):
    def define_model(self):
        self.model = SVC()
        self.params = {
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }
