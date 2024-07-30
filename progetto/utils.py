# coding:utf-8
"""utils.py
Include distance calculation for evaluation metrics
"""
import sys
import os
import glob
import math
import sklearn
import numpy as np
from scipy import stats, integrate
from scipy.integrate import quad


# Calculate overlap between the two PDF
def overlap_area(A, B):
    # Print the shapes of the datasets for debugging
    #print(f"Shape of A: {A.shape}, Shape of B: {B.shape}")

    # Check if datasets have fewer than 2 elements
    if len(A) < 2 or len(B) < 2:
        raise ValueError("Both input datasets must have at least two elements.")

    # Add small noise to avoid singular covariance matrix
    A = A + np.random.normal(0, 1e-8, A.shape)
    B = B + np.random.normal(0, 1e-8, B.shape)

    # Remove duplicates to avoid singular covariance matrix
    A = np.unique(A)
    B = np.unique(B)

    # Create Gaussian KDEs for the datasets A and B
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    # Define the range for integration based on the data in A and B
    range_min = min(np.min(A), np.min(B))
    range_max = max(np.max(A), np.max(B))

    # Define the integrand as the minimum of the two PDFs
    integrand = lambda x: np.minimum(pdf_A(x), pdf_B(x))

    # Perform the integration to find the overlapping area
    overlap_area, _ = quad(integrand, range_min, range_max)

    return overlap_area


# Calculate KL distance between the two PDF
def kl_dist(A, B, num_sample=1000):
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def c_dist(A, B, mode='None', normalize=0):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == 'None':
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == 'EMD':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == 'KL':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist
