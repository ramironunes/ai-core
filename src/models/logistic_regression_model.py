# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:19:39
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:24:01


from src.abstracts.base_model import BaseModel

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseModel):
    def define_model(self):
        self.model = LogisticRegression(max_iter=2000, solver='saga')

