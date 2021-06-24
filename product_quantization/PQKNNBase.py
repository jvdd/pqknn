# -*- coding: utf-8 -*-
__author__ = "Jeroen Van Der Donckt"

import multiprocessing
from typing import Callable

import numpy as np
from numpy.ma.core import common_fill_value
from scipy.cluster.vq import kmeans2, vq

from .util import log_nb_clusters_to_np_int_type, euclidean_dist
from typing import Optional, Union


class PQKNNBase:
    """Base class for product quantization k-nearest neighbors estimators."""

    def __init__(
        self,
        n: int,
        c: int,  
        n_neighbors: Optional[int] = 5,
        metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_dist,
        random_state: Optional[Union[int, np.random.RandomState]] = None,   
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        """Construct a new instance of PQKNN.

        Parameters
        ----------
        n : int
            The number of subvectors.
        c : int
            Determines the amount of clusters for KMeans, i.e., k = 2**c.
        n_neighbors : int, default = 5
            The number of neighborsto use by default for pq k-neighbors queries.
        metric : Callable[[np.ndarray, np.ndarray], np.ndarray], default = euclidean_dist
            The distance metric to use for querying.
        random_state : Union[int, np.random.RandomState], default = None
            Determines random number generation for centroid initialization. Use an int
            to make the randomness deterministic.
        n_jobs : int, default = None
            The number of processes to use for computation.
        kwargs: dict
            Keyword arguments for the kmeans algorithm. See more 
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans
        """
        # Product Quantization parameters
        self.n = n
        self.k = 2 ** c
        # K-NN parameters
        self.n_neighbors = n_neighbors
        self.metric = metric
        # KMeans parameters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        # Precompute the int type (nb. bytes)
        self.int_type = log_nb_clusters_to_np_int_type(c)
        # Dict storing each KMeans centroids at the partition_idx, filled in compress()
        self.subvector_centroids = {}

        # The field below are initialized in the methods of ProductQuantizationKNN
        # self.partition_size = -1 # initialized in compress()
        # self.train_labels = [] # initialized in compress() if train_labels given
        # self.compressed_data = [[]] # initialized in compress() if train_labels given

    def _get_data_partition(self, data, partition_idx):
        # TODO: make sure these are views
        partition_start = partition_idx * self.partition_size
        partition_end = (partition_idx + 1) * self.partition_size
        if data.ndim == 1:
            return data[partition_start:partition_end]
        elif data.ndim == 2:
            return data[:, partition_start:partition_end]
        raise IndexError(f'Given data has the wrong dimensionality {data.ndim}')

    def _compress_partition(self, partition_idx: int, train_data_partition, return_compressed_partition: bool):
        # TODO: sample weights
        partition_centroids, _ = kmeans2(train_data_partition, self.k, iter=20, minit="points")
        compressed_data_partition = None
        if return_compressed_partition:
             compressed_data_partition, _ = vq(train_data_partition, partition_centroids)
        return partition_idx, partition_centroids, compressed_data_partition

    def _compress(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """Compress the given training data via the product quantization method.

        :param train_data: the training examples, a 2D array where each row represents a training sample.
        :param train_labels: the labels for the training data (a 1D array).
        """
        save_compressed = False
        if train_labels is not None:
            save_compressed = True
            nb_samples = len(train_data)
            assert nb_samples == len(
                train_labels
            ), "The number of train samples do not match the length of the labels"
            self.train_labels = train_labels
            self.compressed_data = np.empty(shape=(nb_samples, self.n), dtype=self.int_type)

        d = len(train_data[0])
        self.partition_size = d // self.n

        with multiprocessing.Pool() as pool:
            params = [
                (partition_idx, self._get_data_partition(train_data, partition_idx), save_compressed)
                for partition_idx in range(self.n)
            ]
            kms = pool.starmap(self._compress_partition, params)
            for (partition_idx, partition_centroids, compressed_data_partition) in kms:
                self.subvector_centroids[partition_idx] = partition_centroids
                if save_compressed:
                    self.compressed_data[:, partition_idx] = compressed_data_partition


    # def _compress_partition(self, partition_idx: int, train_data_partition):
    #     # TODO: sample weights
    #     partition_centroids, _ = kmeans2(train_data_partition, self.k, iter=20, minit="points")
    #     compressed_data_partition, _ = vq(train_data_partition, partition_centroids)
    #     return partition_idx, compressed_data_partition, partition_centroids

    # def _compress(self, train_data: np.ndarray, train_labels: np.ndarray):
    #     """Compress the given training data via the product quantization method.

    #     :param train_data: the training examples, a 2D array where each row represents a training sample.
    #     :param train_labels: the labels for the training data (a 1D array).
    #     """
    #     # TODO hier de y niet bij fitten?
    #     nb_samples = len(train_data)
    #     assert nb_samples == len(
    #         train_labels
    #     ), "The number of train samples do not match the length of the labels"
    #     self.train_labels = train_labels
    #     self.compressed_data = np.empty(shape=(nb_samples, self.n), dtype=self.int_type)

    #     d = len(train_data[0])
    #     self.partition_size = d // self.n

    #     with multiprocessing.Pool() as pool:
    #         params = [
    #             (partition_idx, self._get_data_partition(train_data, partition_idx))
    #             for partition_idx in range(self.n)
    #         ]
    #         kms = pool.starmap(self._compress_partition, params)
    #         for (partition_idx, compressed_data_partition, partition_centroids) in kms:
    #             self.compressed_data[:, partition_idx] = compressed_data_partition
    #             self.subvector_centroids[partition_idx] = partition_centroids

    def _encode(self, samples):
        samples = np.atleast_2d(samples)
        codes = np.empty((len(samples), self.n), dtype=self.int_type)
        for partition_idx in range(self.n):
            partition_data = self._get_data_partition(samples, partition_idx)
            codes[:, partition_idx], _ = vq(partition_data, self.subvector_centroids[partition_idx])
        return codes


    def _get_distances_to_compressed_data(self, sample: np.ndarray):
        assert hasattr(self, "compressed_data") and hasattr(
            self, "train_labels"
        ), "There is no stored compressed data => PQKNN can not do a k-NN search"
        # Create the distance table, containing the distances of the sample to the 
        # centroids of each partition
        distances = np.empty(shape=(self.k, self.n), dtype=np.float32)
        for partition_idx in range(self.n):
            sample_partition = self._get_data_partition(sample, partition_idx)
            centroids_partition = self.subvector_centroids[partition_idx]
            distances[:, partition_idx] = self.metric(sample_partition, centroids_partition)

        # Calculate (approximate) distance to stored data
        distance_sums = np.zeros(shape=len(self.compressed_data))
        for partition_idx in range(self.n):
            distance_sums += distances[:, partition_idx][
                self.compressed_data[:, partition_idx]
            ]
        return distance_sums       