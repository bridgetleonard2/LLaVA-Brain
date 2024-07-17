import os
import numpy as np
from classes import visual_featuresCLASS
from sklearn import set_config

import utils


class EncodingModels:
    def __init__(self, train_stim_dir, train_fmri_dir, model_handler,
                 test_stim_dir=None, test_fmri_dir=None, features_dir=None):
        """Initialize the EncodingModels class.
        train_stim_dir (str): Path to the directory with training stimuli.
        train_fmri_dir (str): Path to the directory with training fMRI data.

        len(train_stim_dir) == len(train_fmri_dir)
        Order of files in train_fmri_dir and train_fmri_dir must match.

        Naming convention for files:
        train_stim_dir: 'stim_01', 'stim_02', ...
        train_fmri_dir: 'fmri_01', 'fmri_02', ...

        Same rules apply to test data if provided."""
        self.train_stim_dir = train_stim_dir
        self.train_fmri_dir = train_fmri_dir
        self.model_handler = model_handler
        self.test_stim_dir = test_stim_dir
        self.test_fmri_dir = test_fmri_dir
        self.features_dir = features_dir

        # Prep files
        self.train_stim_files = os.listdir(train_stim_dir)
        self.train_fmri_files = os.listdir(train_fmri_dir)

        # Check rules
        if len(self.train_stim_files) != len(self.train_fmri_files):
            raise ValueError("Length of stim_dir and fmri_dir must be equal.")
        self.train_stim_files.sort()
        self.train_fmri_files.sort()
        # check naming convention
        for stim, fmri in zip(self.train_stim_files, self.train_fmri_files):
            stim_title = stim.split('_')[1].split('.')[0]
            fmri_title = fmri.split('_')[1].split('.')[0]
            if stim_title != fmri_title:
                raise ValueError(
                    "Naming convention mismatch between stim_dir and fmri_dir."
                    )

        # Can have just test_stim
        if test_stim_dir:
            # Prep files
            self.test_stim_files = os.listdir(test_stim_dir)
            if test_fmri_dir:
                # Prep files
                self.test_fmri_files = os.listdir(test_fmri_dir)

                # Check rules
                if len(self.test_stim_files) != len(self.test_fmri_files):
                    raise ValueError(
                        "Length of stim_dir and fmri_dir must be equal.")
                self.test_stim_files.sort()
                self.test_fmri_files.sort()
                # check naming convention
                for stim, fmri in zip(self.test_stim_files,
                                      self.test_fmri_files):
                    stim_title = stim.split('_')[1].split('.')[0]
                    fmri_title = fmri.split('_')[1].split('.')[0]
                    if stim_title != fmri_title:
                        raise ValueError(
                            "Naming convention mismatch between "
                            "stim_dir and fmri_dir."
                            )

    def load_fmri(self):
        self.train_fmri_arrays = []
        for fmri_file in self.train_fmri_files:
            fmri_path = os.path.join(self.train_fmri_dir, fmri_file)
            fmri_data = np.load(fmri_path)
            fmri_data_clean = utils.remove_nan(fmri_data)
            self.train_fmri_arrays.append(fmri_data_clean)

        # Only load the test data if test stim provided
        if self.test_stim_dir and self.test_fmri_dir:
            self.test_fmri_arrays = []
            for fmri_file in self.test_fmri_files:
                fmri_path = os.path.join(self.test_fmri_dir, fmri_file)
                fmri_data = np.load(fmri_path)
                fmri_data_clean = utils.remove_nan(fmri_data)
                self.test_fmri_arrays.append(fmri_data_clean)

    def load_features(self):
        self.train_feature_arrays = []
        for i, stim_file in enumerate(self.train_stim_files):
            try:
                # load features if they exist
                feat_file = stim_file.split('.')[0] + '_features.npy'
                feat_path = os.path.join(self.features_dir, feat_file)
                stim_features = np.load(feat_path, allow_pickle=True)
            except FileNotFoundError:
                stim_path = os.path.join(self.train_stim_dir, stim_file)
                visual_features = visual_featuresCLASS.VisualFeatures(
                    stim_path, self.model_handler)
                visual_features.load_image()
                visual_features.get_features()
                stim_features = visual_features.visualFeatures

                np.save(feat_path, stim_features)

            # Only resample features if dimensions don't match fmri
            fmri_shape = self.train_fmri_arrays[i].shape
            print("features shape", stim_features.shape[0])
            print("fmri shape", fmri_shape[0])
            if stim_features.shape[0] != fmri_shape[0]:
                stim_features_resampled = utils.resample_to_acq(
                    stim_features, fmri_shape)
            else:
                stim_features_resampled = stim_features
            self.train_feature_arrays.append(stim_features_resampled)

        if self.test_stim_dir:
            self.test_feature_arrays = []
            for i, stim_file in enumerate(self.test_stim_files):
                try:
                    # load features if they exist
                    feat_file = stim_file.split('.')[0] + '_features.npy'
                    feat_path = os.path.join(self.features_dir, feat_file)
                    stim_features = np.load(feat_path, allow_pickle=True)
                except FileNotFoundError:
                    stim_path = os.path.join(self.test_stim_dir, stim_file)
                    visual_features = visual_featuresCLASS.VisualFeatures(
                        stim_path, self.model_handler)
                    visual_features.load_image()
                    stim_features = visual_features.get_features()
                    np.save(feat_path, stim_features)

                # Only resample features if dimensions don't match fmri
                fmri_shape = self.test_fmri_arrays[i].shape
                if stim_features.shape[0] != fmri_shape[0]:
                    stim_features_resampled = utils.resample_to_acq(
                        stim_features, fmri_shape)
                else:
                    stim_features_resampled = stim_features
                self.test_feature_arrays.append(stim_features_resampled)

    def evaluate(self):
        """Evaluate the encoding models using leave-one-run-out
        cross-validation."""
        self.correlations = []

        for i in range(len(self.train_feature_arrays)):
            print("leaving out run", i)
            new_feat_arrays = utils.remove_run(self.train_feature_arrays, i)
            X_train = np.vstack(new_feat_arrays)
            Y_train = np.vstack(utils.remove_run(self.train_fmri_arrays, i))

            print("X_train shape", X_train.shape)

            pipeline, backend = utils.set_pipeline(new_feat_arrays)

            set_config(display='diagram')  # requires scikit-learn 0.23
            pipeline

            X_train = X_train.astype(np.float32)

            # if X_train > 2 dimensions, flatten all but the first
            if len(X_train.shape) > 2:
                X_train = np.reshape(X_train, (X_train.shape[0], -1))
                print("X_train reshaped", X_train.shape)

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
            X_test = self.train_feature_arrays[i]
            Y_test = self.train_fmri_arrays[i]

            # Predict
            Y_pred = np.dot(X_test, average_coef)

            test_correlations = utils.calc_correlation(Y_pred, Y_test)

            print("Max correlation:", np.nanmax(test_correlations))

            self.correlations.append(test_correlations)

    def build(self, alignment=False):
        """Build the encoding model."""
        if alignment:
            # replace with actual later***
            align_matrix = np.random.randn(768,
                                           81111)
            transformed_features = []
            for i in range(len(self.train_feature_arrays)):
                transformed_features.append(np.dot(
                    self.train_feature_arrays[i],
                    align_matrix))
            self.train_feature_arrays = transformed_features

        print("Building encoding model using all training data")
        X_train = np.vstack(self.train_feature_arrays)
        Y_train = np.vstack(self.train_fmri_arrays)

        pipeline, backend = utils.set_pipeline(self.train_feature_arrays)

        set_config(display='diagram')  # requires scikit-learn 0.23
        pipeline

        X_train = X_train.astype(np.float32)
        _ = pipeline.fit(X_train, Y_train)

        coef = pipeline[-1].coef_
        coef = backend.to_numpy(coef)
        print("(n_delays * n_features, n_voxels) =", coef.shape)
        # Get encoding model from coefficients
        # Regularize coefficients
        coef /= np.linalg.norm(coef, axis=0)[None]

        delayer = pipeline.named_steps['delayer']
        coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
        print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)

        average_coef = np.mean(coef_per_delay, axis=0)
        print("(n_features, n_voxels) =", average_coef.shape)

        self.encoding_model = average_coef

    def predict(self):
        """Predict fMRI data using the encoding model."""
        self.predictions = []
        for i in range(len(self.test_feature_arrays)):
            X_test = self.test_feature_arrays[i]
            Y_pred = np.dot(X_test, self.encoding_model)
            self.predictions.append(Y_pred)

    def correlate(self):
        """Calculate the correlation between predicted and actual fMRI data."""
        self.correlations = []
        for i in range(len(self.predictions)):
            test_correlations = utils.calc_correlation(
                self.predictions[i], self.test_fmri_arrays[i])
            self.correlations.append(test_correlations)
            # Take the mean of the correlations
            self.mean_correlations = np.nanmean(np.stack((test_correlations)),
                                                axis=0)
            print("Max correlation:", np.nanmax(self.mean_correlations))

    def encoding_pipeline(self):
        """The encoding pipeline depends on the kind of data provided."""
        # Loading features and fmri data will always occur before this
        if self.test_stim_dir:
            # In this case we build the full training model
            # and use it to predict data from test_stim_files
            print("Building encoding model and running predictions")
            self.build()
            self.predict()
            if self.test_fmri_dir:
                # In this case we add on to the last step and
                # calculate correlations between predicted and actual data
                print("Calculating correlations")
                self.correlate()
        else:
            # In this case we evaluate the model using leave-one-run-out
            # cross-validation
            print("Evaluating encoding model")
            self.evaluate()
