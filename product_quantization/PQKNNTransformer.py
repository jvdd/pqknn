# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from .PQKNNBase import PQKNNBase


class PQKNNTransformer(BaseEstimator, TransformerMixin, PQKNNBase):

    def fit(self, X: np.ndarray, y=None, sample_weights=None):
        # TODO: support sample weights
        self._compress(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the given samples.

        Parameters
        ----------
        X: np.ndarray
            The samples, a 2D array where each row represents a sample.

        Returns
        -------
        np.ndarray
            The transformed samples.

        """
        X = np.atleast_2d(X)
        assert X.ndim == 2, "The dimensionality of the data should be 2"
        return self._encode(X)