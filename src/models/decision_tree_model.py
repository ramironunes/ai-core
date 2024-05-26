# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:24:59
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:25:04


from src.abstracts.base_model import BaseModel

from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel(BaseModel):
    def define_model(self):
        self.model = DecisionTreeClassifier()
