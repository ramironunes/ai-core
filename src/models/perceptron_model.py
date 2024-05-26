# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:20:28
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:24:25


from src.abstracts.base_model import BaseModel

from sklearn.linear_model import Perceptron


class PerceptronModel(BaseModel):
    def define_model(self):
        self.model = Perceptron(max_iter=1000)

