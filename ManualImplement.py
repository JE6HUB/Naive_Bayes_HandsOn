from scipy.sparse import csr_matrix
import math
import numpy as np

class SparseMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = {}
        self.classes_ = None

    def fit(self, X: csr_matrix, y):
        n_docs, n_features = X.shape
        self.classes_ = np.unique(y)
        class_count = np.zeros(len(self.classes_))
        feature_count = np.zeros((len(self.classes_), n_features))

        for i, c in enumerate(self.classes_):
            idx = np.where(y == c)[0]
            X_c = X[idx]
            class_count[i] = X_c.shape[0]
            feature_count[i, :] = X_c.sum(axis=0)

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.class_log_prior_ = np.log(class_count / class_count.sum())
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc[:, None])

    def predict(self, X: csr_matrix):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]