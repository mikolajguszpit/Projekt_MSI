import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state, safe_indexing
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object

class ADASYN(BaseOverSampler):

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 n_neighbors=5,
                 n_jobs=1):

        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors, additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

    def _fit_resample(self, X, y):
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.nn_.fit(X)
            _, nn_index = self.nn_.kneighbors(X_class)
            ratio_nn = (np.sum(y[nn_index[:, 1:]] != class_sample, axis=1) /
                        (self.nn_.n_neighbors - 1))
            if not np.sum(ratio_nn):
                raise RuntimeError('Not any neighbours belong to the majority'
                                   ' class. This case will induce a NaN case'
                                   ' with a division by zero. ADASYN is not'
                                   ' suited for this specific dataset.'
                                   ' Use SMOTE instead.')
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            if not np.sum(n_samples_generate):
                raise ValueError("No samples will be generated with the"
                                 " provided ratio settings.")

            self.nn_.fit(X_class)
            _, nn_index = self.nn_.kneighbors(X_class)

            if sparse.issparse(X):
                row_indices, col_indices, samples = [], [], []
                n_samples_generated = 0
                for x_i, x_i_nn, num_sample_i in zip(X_class, nn_index,
                                                     n_samples_generate):
                    if num_sample_i == 0:
                        continue
                    nn_zs = random_state.randint(
                        1, high=self.nn_.n_neighbors, size=num_sample_i)
                    steps = random_state.uniform(size=len(nn_zs))
                    if x_i.nnz:
                        for step, nn_z in zip(steps, nn_zs):
                            sample = (x_i + step *
                                      (X_class[x_i_nn[nn_z], :] - x_i))
                            row_indices += (
                                    [n_samples_generated] * len(sample.indices))
                            col_indices += sample.indices.tolist()
                            samples += sample.data.tolist()
                            n_samples_generated += 1
                X_new = (sparse.csr_matrix(
                    (samples, (row_indices, col_indices)),
                    [np.sum(n_samples_generate), X.shape[1]], dtype=X.dtype))
                y_new = np.array([class_sample] * np.sum(n_samples_generate),
                                 dtype=y.dtype)
            else:
                x_class_gen = []
                for x_i, x_i_nn, num_sample_i in zip(X_class, nn_index,
                                                     n_samples_generate):
                    if num_sample_i == 0:
                        continue
                    nn_zs = random_state.randint(
                        1, high=self.nn_.n_neighbors, size=num_sample_i)
                    steps = random_state.uniform(size=len(nn_zs))
                    x_class_gen.append([
                        x_i + step * (X_class[x_i_nn[nn_z], :] - x_i)
                        for step, nn_z in zip(steps, nn_zs)
                    ])

                X_new = np.concatenate(x_class_gen).astype(X.dtype)
                y_new = np.array([class_sample] * np.sum(n_samples_generate),
                                 dtype=y.dtype)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled