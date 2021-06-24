# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

import numpy as np
import multiprocessing

from sklearn.base import BaseEstimator, RegressorMixin

from .PQKNNBase import PQKNNBase
from .util import regress_label


class PQKNNRegressor(BaseEstimator, RegressorMixin, PQKNNBase):

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights=None):
        # TODO: support sample weights
        self._compress(X, y)
        return self

    def _predict_single_sample(self, sample: np.ndarray):
        # Calculate (approximate) distance to stored data
        distance_sums = self._get_distances_to_compressed_data(sample)
        # Regress label among k nearest neighbors
        return regress_label(distance_sums, self.train_labels, self.n_neighbors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label of the given test samples.

        Parameters
        ----------
        X: np.ndarray
            The samples, a 2D array where each row represents a sample.

        Returns
        -------
        np.ndarray
            The predicted labels, a 1D array.

        """
        X = np.atleast_2d(X)
        assert X.ndim == 2, "The dimensionality of the data should be 2"
        if len(X) > 2000:  
            # Fuzzy rule to decide whether or not multiple threads should be spawned
            with multiprocessing.Pool(self.n_jobs) as pool:
                preds = pool.map(self._predict_single_sample, X)
        else:
            preds = [self._predict_single_sample(row) for row in X]
        return np.array(preds)
