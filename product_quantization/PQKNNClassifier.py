# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

import numpy as np
import multiprocessing

from sklearn.base import BaseEstimator, ClassifierMixin

from .PQKNNBase import PQKNNBase
from .util import classify_label, select_proba_labels


class PQKNNClasifier(BaseEstimator, ClassifierMixin, PQKNNBase):

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights=None):
        # TODO: support sample weights
        self._compress(X, y)
        self.unique_labels = sorted(list(set(y)))
        return self


    def _predict_single_sample(self, sample: np.ndarray):
        # Calculate (approximate) distance to stored data
        distance_sums = self._get_distances_to_compressed_data(sample)
        # Select label among k nearest neighbors
        return classify_label(distance_sums, self.train_labels, self.n_neighbors)

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


    def _predict_proba_single_sample(self, sample: np.ndarray) -> np.ndarray:
        distance_sums = self._get_distances_to_compressed_data(sample)
        label_proba_dict = select_proba_labels(distance_sums, self.train_labels, self.n_neighbors)
        probas = np.zeros(shape=(len(self.unique_labels)))
        for idx, label in enumerate(self.unique_labels):
            if label in label_proba_dict:
                probas[idx] = label_proba_dict[label]
        return probas

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probabilities for the label of the given test samples.

        Parameters
        ----------
        X: np.ndarray
            The samples, a 2D array where each row represents a sample.

        Returns
        -------
        np.ndarray
            The predicted probabilities for the labels, a 2D array.

        """
        X = np.atleast_2d(X)
        assert X.ndim == 2, "The dimensionality of the data should be 2"
        if len(X) > 2000:  
            # Fuzzy rule to decide whether or not multiple threads should be spawned
            with multiprocessing.Pool(self.n_jobs) as pool:
                preds = pool.map(self._predict_proba_single_sample, X)
        else:
            preds = [self._predict_proba_single_sample(row) for row in X]
        return np.array(preds)
