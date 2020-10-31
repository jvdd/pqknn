# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

import numpy as np


def log_nb_clusters_to_np_int_type(log_nb_clusters: int) -> type:
    # Considering that the cluster indices start from zero
    if log_nb_clusters <= 8:
        return np.uint8
    elif log_nb_clusters <= 16:
        return np.uint16
    elif log_nb_clusters <= 32:
        return np.uint32
    else:
        return np.uint64


def squared_euclidean_dist(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sum(np.square(vec2 - vec1), axis=-1)
