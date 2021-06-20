# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from .PQKNNBase import PQKNNBase

class PQKNNClasifier(BaseEstimator, ClassifierMixin, PQKNNBase):

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights=None):
        # TODO: support sample weights
        self._compress(X, y)
        return self

    def predict(self, X: np.ndarray):
        return self._predict(X)
