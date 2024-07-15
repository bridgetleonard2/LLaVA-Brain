import os
import numpy as np
from classes import visual_featuresCLASS
from scipy.signal import resample
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state, check_is_fitted, check_array
from himalaya.backend import set_backend
from sklearn.base import BaseEstimator, TransformerMixin
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn import set_config


def resample_to_acq(feature_data, fmri_data_shape):
    dimensions = fmri_data_shape[0]
    data_transposed = feature_data.T
    data_resampled = np.empty((data_transposed.shape[0], dimensions))

    for i in range(data_transposed.shape[0]):
        data_resampled[i, :] = resample(data_transposed[i, :],
                                        dimensions, window=('kaiser', 14))

    print("Shape after resampling:", data_resampled.T.shape)
    return data_resampled.T


def remove_nan(data):
    mask = ~np.isnan(data)

    # Apply the mask and then flatten
    # This will keep only the non-NaN values
    data_reshaped = data[mask].reshape(data.shape[0], -1)

    print("fMRI shape:", data_reshaped.shape)
    return data_reshaped


def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    # With permutations, we are sure that all runs are used as validation runs.
    # However here for n_runs_out > 1, a run can be chosen twice as validation
    # in the same split.
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val


class Delayer(BaseEstimator, TransformerMixin):
    """Scikit-learn Transformer to add delays to features.

    This assumes that the samples are ordered in time.
    Adding a delay of 0 corresponds to leaving the features unchanged.
    Adding a delay of 1 corresponds to using features from the previous sample.

    Adding multiple delays can be used to take into account the slow
    hemodynamic response, with for example `delays=[1, 2, 3, 4]`.

    Parameters
    ----------
    delays : array-like or None
        Indices of the delays applied to each feature. If multiple values are
        given, each feature is duplicated for each delay.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during the fit.

    Example
    -------
    >>> from sklearn.pipeline import make_pipeline
    >>> from voxelwise_tutorials.delayer import Delayer
    >>> from himalaya.kernel_ridge import KernelRidgeCV
    >>> pipeline = make_pipeline(Delayer(delays=[1, 2, 3, 4]), KernelRidgeCV())
    """

    def __init__(self, delays=None):
        self.delays = delays

    def fit(self, X, y=None):
        """Fit the delayer.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values. Ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        X = self._validate_data(X, dtype='numeric')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform the input data X, copying features with different delays.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X, copy=True)

        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        if self.delays is None:
            return X

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)),
                             dtype=X.dtype)
        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[:-abs(delay), beg:end] = X[abs(delay):]

        return X_delayed

    def reshape_by_delays(self, Xt, axis=1):
        """Reshape an array, splitting and stacking across delays.

        Parameters
        ----------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed array.
        axis : int, default=1
            Axis to split.

        Returns
        -------
        Xt_split :array of shape (n_delays, n_samples, n_features)
            Reshaped array, splitting across delays.
        """
        delays = self.delays or [0]  # deals with None
        return np.stack(np.split(Xt, len(delays), axis=axis))


def remove_run(arrays, index_to_remove):
    # Return a new list with the specified run removed
    return [array for idx, array in enumerate(arrays)
            if idx != index_to_remove]


def set_params(feature_arrays):
    run_onsets = []
    current_index = 0
    for arr in feature_arrays:
        next_index = current_index + arr.shape[0]
        run_onsets.append(current_index)
        current_index = next_index

    print(run_onsets)
    n_samples_train = len(feature_arrays)
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = Delayer(delays=[1, 2, 3, 4])
    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)
    alphas = np.logspace(1, 20, 20)
    return cv, scaler, delayer, backend, alphas


def safe_correlation(x, y):
    """Calculate the Pearson correlation coefficient safely."""
    # Mean centering
    x_mean = x - np.mean(x)
    y_mean = y - np.mean(y)

    # Numerator: sum of the product of mean-centered variables
    numerator = np.sum(x_mean * y_mean)

    # Denominator: sqrt of product of sums of squared mean-centered variables
    denominator = np.sqrt(np.sum(x_mean**2) * np.sum(y_mean**2))

    # Safe division
    if denominator == 0:
        # Return NaN or another value to indicate undefined correlation
        return np.nan
    else:
        return numerator / denominator


def calc_correlation(predicted_fMRI, real_fMRI):
    # Calculate correlations for each voxel
    correlation_coefficients = [safe_correlation(predicted_fMRI[:, i],
                                                 real_fMRI[:, i]) for i in
                                range(predicted_fMRI.shape[1])]
    correlation_coefficients = np.array(correlation_coefficients)

    # Check for NaNs in the result
    nans_in_correlations = np.isnan(correlation_coefficients).any()
    print(f"NaNs in correlation coefficients: {nans_in_correlations}")

    return correlation_coefficients

class EncodingModels:
    def __init__(self, stim_dir, fmri_dir, model_handler):
        """Initialize the EncodingModels class.
        stim_dir (str): Path to the directory containing the stimuli.
        fmri_dir (str): Path to the directory containing the fMRI data.

        len(stim_dir) == len(fmri_dir)
        Order of files in stim_dir and fmri_dir must match.

        Naming convention for files:
        stim_dir: 'stim_01', 'stim_02', ...
        fmri_dir: 'fmri_01', 'fmri_02', ..."""
        self.stim_dir = stim_dir
        self.fmri_dir = fmri_dir
        self.model_handler = model_handler

        # Prep files
        self.stim_files = os.listdir(stim_dir)
        self.fmri_files = os.listdir(fmri_dir)
        if len(self.stim_files) != len(self.fmri_files):
            raise ValueError("Length of stim_dir and fmri_dir must be equal.")
        self.stim_files.sort()
        self.fmri_files.sort()
        # check naming convention
        for stim, fmri in zip(self.stim_files, self.fmri_files):
            if stim.split('_')[1] != fmri.split('_')[1]:
                raise ValueError(
                    "Naming convention mismatch between stim_dir and fmri_dir."
                    )

    def load_fmri(self):
        self.fmri_arrays = []
        for fmri_file in self.fmri_files:
            fmri_path = os.path.join(self.fmri_dir, fmri_file)
            fmri_data = np.load(fmri_path)
            fmri_data_clean = remove_nan(fmri_data)
            new_feat_arrays.append(fmri_data_clean)
        self.fmri_shape = fmri_data.shape

    def load_features(self):
        self.feature_arrays = []
        for stim_file in self.stim_files:
            stim_path = os.path.join(self.stim_dir, stim_file)
            visual_features = visual_featuresCLASS.VisualFeatures(
                stim_path, self.model_handler)
            visual_features.load_image()
            stim_features = visual_features.get_features()

            # Only resample features if dimensions don't match fmri
            if stim_features.shape[0] != self.fmri_shape[0]:
                stim_features_resampled = resample_to_acq(stim_features,
                                                          self.fmri_shape)
            else:
                stim_features_resampled = stim_features
            self.feature_arrays.append(stim_features_resampled)

    def evaluate(self):
        self.correlations = []

        for i in range(len(self.feature_arrays)):
            print("leaving out run", i)
            new_feat_arrays = remove_run(self.feature_arrays, i)
            X_train = np.vstack(new_feat_arrays)
            Y_train = np.vstack(remove_run(self.fmri_arrays, i))

            print("X_train shape", X_train.shape)

            cv, scaler, delayer, backend, alphas = set_params(new_feat_arrays)
            X_train = X_train.astype(np.float32)

            ridge_cv = RidgeCV(
                alphas=alphas,
                cv=cv,
                solver_params=dict(
                    n_targets_batch=500, n_alphas_batch=5,
                    n_targets_batch_refit=100))

            pipeline = make_pipeline(
                scaler,
                delayer,
                ridge_cv,
            )

            set_config(display='diagram')  # requires scikit-learn 0.23
            pipeline

            _ = pipeline.fit(X_train, Y_train)

            coef = pipeline[-1].coef_
            coef = backend.to_numpy(coef)
            print("(n_delays * n_features, n_voxels) =", coef.shape)

            # Regularize coefficients
            coef /= np.linalg.norm(coef, axis=0)[None]

            # split the ridge coefficients per delays
            delayer = pipeline.named_steps['delayer']
            coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
            print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)
            del coef

            # average over delays
            average_coef = np.mean(coef_per_delay, axis=0)
            print("(n_features, n_voxels) =", average_coef.shape)
            del coef_per_delay

            # Test the model
            X_test = self.feature_arrays[i]
            Y_test = self.fmri_arrays[i]

            # Predict
            Y_pred = np.dot(X_test, average_coef)

            test_correlations = calc_correlation(Y_pred, Y_test)

            print("Max correlation:", np.nanmax(test_correlations))

            self.correlations.append(test_correlations)
