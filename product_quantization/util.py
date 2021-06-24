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


def euclidean_dist(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sum(np.square(vec2 - vec1), axis=-1)


def classify_label(distance_sums, train_labels, n_neighbors):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    indices = np.argpartition(distance_sums, n_neighbors)
    labels = train_labels[indices][:n_neighbors]
    unique_labels, counts = np.unique(labels, return_counts=True)
    # 1) If there is only 1 label (among the nearest neighbors)
    if len(unique_labels) == 1:
        return unique_labels[0]
    # 2) Else -> there are 2 or more labels (among the nearest neighbors)
    sorted_idxs = np.argsort(counts)[::-1]  # Get idxs from max to min nb of counts
    unique_labels = unique_labels[sorted_idxs]  # Sort labels in descending order of their frequency
    counts = counts[sorted_idxs]  # Sorted counts in descending order of their value
    # 2.1) If there is no tie
    if counts[0] != counts[1]:
        return unique_labels[0]
    # 2.2) If there is an tie
    max_count = counts[0]
    idx = 0
    min_distance = float("inf")
    selected_label = None
    # Select label with minimal summed distance amongst the labels with frequency == max_count
    while idx < len(unique_labels) and counts[idx] == max_count:
        label = unique_labels[idx]
        label_indices = np.where(labels == label)
        summed_distance = np.sum(distance_sums[indices[label_indices]])
        if summed_distance < min_distance:
            selected_label = label
            min_distance = summed_distance
        idx += 1
    return selected_label

def select_proba_labels(distance_sums, train_labels, n_neighbors):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    indices = np.argpartition(distance_sums, n_neighbors)
    labels = train_labels[indices][:n_neighbors]
    unique_labels, counts = np.unique(labels, return_counts=True)

    label_weights = {}
    for idx in range(len(unique_labels)):
        label = unique_labels[idx]
        label_indices = np.where(labels == label)
        label_weights[label] = 1 / np.mean(distance_sums[indices[label_indices]])
    
    proba_labels = {label: count * label_weights[label] for label, count in zip(unique_labels, counts)}
    value_sum = np.sum(list(proba_labels.values()))
    return {label: val / value_sum for label, val in proba_labels.items()}

    
    # Return the probabilities just based on the count
    # return {label: count / n_neighbors for label, count in zip(unique_labels, counts)}


def regress_label(distance_sums, train_labels, n_neighbors):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    indices = np.argpartition(distance_sums, n_neighbors)
    labels = train_labels[indices][:n_neighbors]

    return np.average(labels, weights=1/distance_sums[indices[:n_neighbors]])

    # Return just the mean
    # return np.mean(labels)