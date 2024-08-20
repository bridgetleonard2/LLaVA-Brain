import os
import numpy as np
from classes import visual_featuresCLASS
from classes import language_featuresCLASS
from sklearn import set_config
from datasets import load_dataset  # type: ignore
from scipy.stats import pearsonr

import utils

flatten = False


class EncodingModels:
    def __init__(self, model_handler, train_stim_dir, train_fmri_dir,
                 train_stim_type, test_stim_dir=None, test_fmri_dir=None,
                 test_stim_type=None, features_dir=None):
        """Initialize the EncodingModels class.
        model_handler (ModelHandler): ModelHandler object with the model.

        train_stim_dir (str): Path to the directory with training stimuli.
        train_fmri_dir (str): Path to the directory with training fMRI data.

        train_stim_type (str): Type of training stimuli. Can be "visual" or
        "language".

        test_stim_dir (str): Path to the directory with test stimuli.
        Default is None.

        test_fmri_dir (str): Path to the directory with test fMRI data. Default
        is None.

        test_stim_type (str): Type of test stimuli. Can be "visual" or
        "language". Default is None.

        features_dir (str): Path to the directory with features.
        Default is None.

        len(train_stim_dir) == len(train_fmri_dir)
        Order of files in train_fmri_dir and train_fmri_dir must match.

        Naming convention for files:
        Stim and fmri have same name but different types (name.npy, name.hdf5)

        Same rules apply to test data if provided."""
        self.model_handler = model_handler
        self.train_stim_dir = train_stim_dir
        self.train_fmri_dir = train_fmri_dir
        self.train_stim_type = train_stim_type

        self.test_stim_dir = test_stim_dir
        self.test_fmri_dir = test_fmri_dir
        self.test_stim_type = test_stim_type

        self.features_dir = features_dir

        # Prep files
        # files_only = [f for f in os.listdir(directory_path) if
        # os.path.isfile(os.path.join(directory_path, f))]
        self.train_stim_files = [f for f in os.listdir(train_stim_dir) if
                                 os.path.isfile(os.path.join(train_stim_dir,
                                                             f))]
        self.train_fmri_files = [f for f in os.listdir(train_fmri_dir) if
                                 os.path.isfile(os.path.join(train_fmri_dir,
                                                             f))]

        # Check rules
        if len(self.train_stim_files) != len(self.train_fmri_files):
            raise ValueError("Length of stim_dir and fmri_dir must be equal.")
        self.train_stim_files.sort()
        self.train_fmri_files.sort()
        # check naming convention
        for stim, fmri in zip(self.train_stim_files, self.train_fmri_files):
            stim_title = stim.split('.')[0]
            fmri_title = fmri.split('.')[0]
            if stim_title != fmri_title:
                raise ValueError(
                    "Naming convention mismatch between stim_dir and fmri_dir."
                    )

        # Can have just test_stim
        if test_stim_dir:
            # Prep files
            self.test_stim_files = [f for f in os.listdir(test_stim_dir) if
                                    os.path.isfile(os.path.join(test_stim_dir,
                                                                f))]
            if test_fmri_dir:
                # Prep files
                self.test_fmri_files = [f for f in os.listdir(test_fmri_dir) if
                                        os.path.isfile(os.path.join(
                                            test_fmri_dir, f))]

                # Check rules
                if len(self.test_stim_files) != len(self.test_fmri_files):
                    raise ValueError(
                        "Length of stim_dir and fmri_dir must be equal.")
                self.test_stim_files.sort()
                self.test_fmri_files.sort()
                # check naming convention
                for stim, fmri in zip(self.test_stim_files,
                                      self.test_fmri_files):
                    stim_title = stim.split('.')[0]
                    fmri_title = fmri.split('.')[0]
                    if stim_title != fmri_title:
                        raise ValueError(
                            "Naming convention mismatch between "
                            "stim_dir and fmri_dir."
                            )

    def load_fmri(self):
        """Load the fMRI data.

        Turns nan values to 0 and reshapes the data to 2D."""
        self.train_fmri_arrays = []
        for fmri_file in self.train_fmri_files:
            fmri_path = os.path.join(self.train_fmri_dir, fmri_file)
            fmri_data = np.load(fmri_path)
            fmri_data_clean = np.nan_to_num(fmri_data)
            fmri_data_clean = fmri_data_clean.reshape(
                fmri_data_clean.shape[0], -1)
            print('fmri_data_clean shape:', fmri_data_clean.shape)
            self.train_fmri_arrays.append(fmri_data_clean)

        # Only load the test data if test stim provided
        if self.test_stim_dir and self.test_fmri_dir:
            self.test_fmri_arrays = []
            for fmri_file in self.test_fmri_files:
                fmri_path = os.path.join(self.test_fmri_dir, fmri_file)
                fmri_data = np.load(fmri_path)
                fmri_data_clean = np.nan_to_num(fmri_data)
                fmri_data_clean = fmri_data_clean.reshape(
                    fmri_data_clean.shape[0], -1)
                print('fmri_data_clean shape:', fmri_data_clean.shape)
                self.test_fmri_arrays.append(fmri_data_clean)

    def load_features(self, n=30):
        """Load the features.

        If features don't exist, they are created and saved.

        n (int): specifies the number of items to average features for.
        Default is 30 to mimic 30 frames per second."""
        self.train_feature_arrays = []
        for i, stim_file in enumerate(self.train_stim_files):
            try:
                # load features if they exist
                feat_file = stim_file.split('.')[0] + '_features.npy'
                feat_path = os.path.join(self.features_dir, feat_file)
                train_stim_features = np.load(feat_path, allow_pickle=True)
            except FileNotFoundError:
                stim_path = os.path.join(self.train_stim_dir, stim_file)
                if self.train_stim_type == "visual":
                    visual_features = visual_featuresCLASS.VisualFeatures(
                        stim_path, self.model_handler)
                    visual_features.load_image()
                    visual_features.get_features(n=n)
                    train_stim_features = visual_features.visualFeatures
                elif self.train_stim_type == "language":
                    language_features = (
                        language_featuresCLASS.LanguageFeatures(
                            stim_path, self.model_handler))
                    language_features.load_text()
                    language_features.get_features()
                    train_stim_features = (
                        language_features.languageFeatures
                    )

                np.save(feat_path, train_stim_features)

            # Only resample features if dimensions don't match fmri
            train_fmri_shape = self.train_fmri_arrays[i].shape
            print("features shape", train_stim_features.shape)
            print("fmri shape", train_fmri_shape)

            # if features are >2 dimensions, average across the
            # second dimension
            if len(train_stim_features.shape) > 2:
                if flatten:
                    train_stim_features = train_stim_features.reshape(
                        train_stim_features.shape[0], -1)
                else:
                    train_stim_features = np.mean(train_stim_features, axis=1)
                print("new features shape", train_stim_features.shape)

            if train_stim_features.shape[0] != train_fmri_shape[0]:
                train_stim_features_resampled = utils.resample_to_acq(
                    train_stim_features, train_fmri_shape)
            else:
                train_stim_features_resampled = train_stim_features
            self.train_feature_arrays.append(train_stim_features_resampled)

        if self.test_stim_dir:
            self.test_feature_arrays = []
            for i, stim_file in enumerate(self.test_stim_files):
                try:
                    # load features if they exist
                    feat_file = stim_file.split('.')[0] + '_features.npy'
                    feat_path = os.path.join(self.features_dir, feat_file)
                    test_stim_features = np.load(feat_path, allow_pickle=True)
                except FileNotFoundError:
                    stim_path = os.path.join(self.test_stim_dir, stim_file)
                    if self.test_stim_type == "visual":
                        visual_features = visual_featuresCLASS.VisualFeatures(
                            stim_path, self.model_handler)
                        visual_features.load_image()
                        visual_features.get_features(n=n)
                        test_stim_features = visual_features.visualFeatures
                    elif self.test_stim_type == "language":
                        language_features = (
                            language_featuresCLASS.LanguageFeatures(
                                stim_path, self.model_handler))
                        language_features.load_text()
                        language_features.get_features()
                        test_stim_features = (
                            language_features.languageFeatures
                        )
                    print("saving features", test_stim_features.shape)

                    np.save(feat_path, test_stim_features)

                # Only resample features if dimensions don't match fmri
                if self.test_fmri_dir:
                    test_fmri_shape = self.test_fmri_arrays[i].shape
                    print("features shape", test_stim_features.shape)
                    print("fmri shape", test_fmri_shape)

                    # if features are >2 dimensions, average across the second
                    # dimension
                    if len(test_stim_features.shape) > 2:
                        if flatten:
                            test_stim_features = test_stim_features.reshape(
                                test_stim_features.shape[0], -1)
                        else:
                            test_stim_features = np.mean(test_stim_features,
                                                         axis=1)
                        print("new features shape", test_stim_features.shape)

                    fmri_shape = self.test_fmri_arrays[i].shape
                    if test_stim_features.shape[0] != fmri_shape[0]:
                        test_stim_features_resampled = utils.resample_to_acq(
                            test_stim_features, fmri_shape)
                    else:
                        test_stim_features_resampled = test_stim_features
                else:
                    test_stim_features_resampled = test_stim_features
                self.test_feature_arrays.append(test_stim_features_resampled)

    def evaluate(self):
        """Evaluate the encoding models using leave-one-run-out
        cross-validation.

        If only test and train data is given, the model is built and
        evaluated with leave-one-run-out cross-validation."""
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

            # if X_train > 2 dimensions, average across the second dimension
            if len(X_train.shape) > 2:
                if flatten:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                else:
                    X_train = np.mean(X_train, axis=1)
                print("X_train shape", X_train.shape)

            _ = pipeline.fit(X_train, Y_train)

            coef = pipeline[-1].get_primal_coef()
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

            # check if any coef are zero or nan
            num_zeroes = np.count_nonzero(average_coef == 0)
            print("coef zeros:", num_zeroes)
            num_nan = np.count_nonzero(np.isnan(average_coef))
            print("coef nans:", num_nan)

            # Test the model
            X_test = self.train_feature_arrays[i]
            Y_test = self.train_fmri_arrays[i]

            # if X_test > 2 dimensions, average across the second dimension
            if len(X_test.shape) > 2:
                if flatten:
                    X_test = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_test = np.mean(X_test, axis=1)
                print("X_test shape", X_test.shape)
            print("Y_test shape:", Y_test.shape)

            # Predict
            Y_pred = np.matmul(X_test, average_coef)
            print("Y_pred shape:", Y_pred.shape)

            # Calculate correlation
            test_correlations = utils.calc_correlation(Y_pred, Y_test)
            print("Test correlations calculated")

            print("Max correlation:", np.nanmax(test_correlations))

            self.correlations.append(test_correlations)

    def alignment(self):
        """Align the features and fMRI data.
        If train_stim_type != test_stim_type, alignment is needed.

        This accounts for the linear transformation between the
        two types of data. This allows us to project one type of
        data into the other's space."""
        # try loading alignment
        im_to_cap_filename = (f"{self.model_handler.layer_name}"
                              "_im_to_cap_alignment.npy")
        cap_to_im_filename = (f"{self.model_handler.layer_name}"
                              "_cap_to_im_alignment.npy")
        im_to_cap_path = os.path.join(self.features_dir, im_to_cap_filename)
        cap_to_im_path = os.path.join(self.features_dir, cap_to_im_filename)
        try:
            self.coef_image_to_caption = np.load(im_to_cap_path)
            self.coef_caption_to_image = np.load(cap_to_im_path)
        except FileNotFoundError:
            def preprocess_image(image):
                # Convert the image to RGB (if not already in RGB)
                image = image.convert('RGB')
                # Resize the image to a fixed size, e.g., 224x224
                image = image.resize((224, 224))
                # Convert the image to a numpy array
                image_array = np.array(image)
                return image_array

            # if alignment doesn't exist, create one
            alignment_data = load_dataset("nlphuji/flickr30k", split='test',
                                          streaming=True)
            # Take 1000 samples
            shuffled_data = alignment_data.shuffle(seed=42)
            alignment_data = shuffled_data.take(1000)

            images = [preprocess_image(item['image']) for
                      item in alignment_data]
            captions = [" ".join(item['caption']) for item in alignment_data]

            images_array = np.array(images)
            captions_array = np.array(captions)

            print("images shape", images_array.shape)
            print("captions shape", captions_array.shape)

            stim_path = ""

            visual_features = visual_featuresCLASS.VisualFeatures(
                    stim_path, self.model_handler)
            visual_features.stim_data = images_array
            visual_features.get_features(batch_size=20, n=1)
            image_features = visual_features.visualFeatures

            language_features = (
                    language_featuresCLASS.LanguageFeatures(
                        stim_path, self.model_handler))

            language_features.stim_data = captions_array
            language_features.get_features(batch_size=10, alignment=True)
            caption_features = language_features.languageFeatures

            # Data should be 2d of shape (n_images/n, num_features)
            # if data is above 2d, average 2nd+ dimensions
            if caption_features.ndim > 2:
                if flatten:
                    caption_features = caption_features.reshape(
                        caption_features.shape[0], -1)
                    image_features = image_features.reshape(
                        image_features.shape[0], -1)
                else:
                    caption_features = np.mean(caption_features, axis=1)
                    image_features = np.mean(image_features, axis=1)

            print("image features shape", image_features.shape)
            print("caption features shape", caption_features.shape)

            pipeline, backend = utils.set_pipeline("", cv=5)

            set_config(display='diagram')  # requires scikit-learn 0.23
            pipeline

            _ = pipeline.fit(image_features, caption_features)
            coef_im_cap = backend.to_numpy(pipeline[-1].get_primal_coef())

            print("coef_image_to_caption shape",
                  coef_im_cap.shape)

            # Check if zeroes in coef_images_to_captions
            num_zeroes_im_to_cap = np.count_nonzero(
                coef_im_cap == 0)
            print("image to caption zeros:", num_zeroes_im_to_cap)

            epsilon = 1e-10

            if num_zeroes_im_to_cap > 0:
                coef_im_cap = coef_im_cap.astype(
                    float)
                coef_im_cap[
                    coef_im_cap == 0] = epsilon

            print("coef_image_to_caption shape",
                  coef_im_cap.shape)

            # Regularize coefficients
            coef_im_cap /= np.linalg.norm(coef_im_cap, axis=0)[None]

            # split the ridge coefficients per delays
            delayer = pipeline.named_steps['delayer']
            coef_im_cap_per_delay = delayer.reshape_by_delays(coef_im_cap,
                                                              axis=0)
            print("(n_delays, n_features, n_voxels) =",
                  coef_im_cap_per_delay.shape)
            del coef_im_cap

            # average over delays
            average_im_cap_coef = np.mean(coef_im_cap_per_delay, axis=0)
            print("(n_features, n_voxels) =", average_im_cap_coef.shape)
            del coef_im_cap_per_delay

            self.coef_image_to_caption = average_im_cap_coef

            print("coef_image_to_caption shape",
                  self.coef_image_to_caption.shape)

            # save alignment
            np.save(im_to_cap_path, self.coef_image_to_caption)

            _ = pipeline.fit(caption_features, image_features)
            coef_cap_im = backend.to_numpy(pipeline[-1].get_primal_coef())

            # Check if zeroes in coef_captions_to_images
            num_zeroes_cap_to_im = np.count_nonzero(
                coef_cap_im == 0)
            print("caption to image zeros:", num_zeroes_cap_to_im)

            if num_zeroes_cap_to_im > 0:
                coef_cap_im = coef_cap_im.astype(
                    float)
                coef_cap_im[
                    coef_cap_im == 0] = epsilon

            # Regularize coefficients
            coef_cap_im /= np.linalg.norm(coef_cap_im, axis=0)[None]

            # split the ridge coefficients per delays
            delayer = pipeline.named_steps['delayer']
            coef_cap_im_per_delay = delayer.reshape_by_delays(coef_cap_im,
                                                              axis=0)
            print("(n_delays, n_features, n_voxels) =",
                  coef_cap_im_per_delay.shape)
            del coef_cap_im

            # average over delays
            average_cap_im_coef = np.mean(coef_cap_im_per_delay, axis=0)
            print("(n_features, n_voxels) =", average_cap_im_coef.shape)
            del coef_cap_im_per_delay

            self.coef_caption_to_image = average_cap_im_coef

            print("coef_caption_to_image shape",
                  self.coef_caption_to_image.shape)

            # save alignment
            np.save(cap_to_im_path, self.coef_caption_to_image)

    def build(self, cv=None):
        """Build the encoding model.

        cv (int): Number of cross-validation folds. Default is None.
        If None, the cv is built on when each data set starts and ends
        (i.e., movie data start/ends)."""

        print("Building encoding model using all training data")
        X_train = np.vstack(self.train_feature_arrays)
        Y_train = np.vstack(self.train_fmri_arrays)

        # Save x_train and y_train
        np.save('output/clip/model/x_train.npy',
                X_train)
        np.save('output/clip/model/y_train.npy',
                Y_train)

        self.pipeline, backend = utils.set_pipeline(self.train_feature_arrays,
                                                    cv=cv)

        set_config(display='diagram')  # requires scikit-learn 0.23
        self.pipeline

        X_train = X_train.astype(np.float32)

        # if X_train > 2 dimensions, average across the second dimension
        if len(X_train.shape) > 2:
            if flatten:
                X_train = X_train.reshape(X_train.shape[0], -1)
            else:
                X_train = np.mean(X_train, axis=1)
            print("X_train shape", X_train.shape)

        _ = self.pipeline.fit(X_train, Y_train)

        coef = self.pipeline[-1].get_primal_coef()
        coef = backend.to_numpy(coef)
        print("(n_delays * n_features, n_voxels) =", coef.shape)
        # Get encoding model from coefficients
        # Regularize coefficients
        self.coef /= np.linalg.norm(coef, axis=0)[None]

        delayer = self.pipeline.named_steps['delayer']
        coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
        print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)

        average_coef = np.mean(coef_per_delay, axis=0)
        print("(n_features, n_voxels) =", average_coef.shape)

        self.encoding_model = average_coef
        np.save('output/clip/model/encoding_model.npy',
                self.encoding_model)

    def predict(self, alignment=False):
        """Predict fMRI data using the encoding model.

        If test_stim_data is given but not test_fmri_data, the model
        will predict the fMRI data."""
        if alignment:
            self.alignment()
            if self.test_stim_type == "visual":
                self.test_feature_arrays = [
                    np.dot(X, self.coef_image_to_caption.T) for X
                    in self.test_feature_arrays]
            elif self.test_stim_type == "language":
                self.test_feature_arrays = [
                    np.dot(X, self.coef_caption_to_image.T) for X
                    in self.test_feature_arrays]

        self.pipeline_predictions = []
        self.coef_predictions = []
        for i in range(len(self.test_feature_arrays)):
            X_test = self.test_feature_arrays[i]

            # if X_test > 2 dimensions, average across the second dimension
            if len(X_test.shape) > 2:
                if flatten:
                    X_test = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_test = np.mean(X_test, axis=1)
                print("X_test shape", X_test.shape)

            Y_pred_pipeline = self.pipeline.predict(X_test)
            print("Encoding model shape:", self.encoding_model.shape)
            print("X_test shape:", X_test.shape)
            X_test_scaled = (
                self.pipeline.named_steps['standardscaler'].transform(X_test))
            delayer = self.pipeline.named_steps['delayer']
            X_test_delayed = delayer.transform(
                X_test_scaled)
            Y_pred_delay = np.matmul(X_test_delayed, self.coef)
            Y_pred_per_delay = delayer.reshape_by_delays(Y_pred_delay, axis=0)
            avg_Y_pred = np.mean(Y_pred_per_delay, axis=0)
            print("avg_Y_pred shape:", avg_Y_pred.shape)

            print("X_test_delayed shape:", X_test_delayed.shape)
            Y_pred_coef = np.matmul(X_test_scaled, self.encoding_model)
            print("Y_pred shape:", Y_pred_coef.shape)
            self.pipeline_predictions.append(Y_pred_pipeline)
            self.coef_predictions.append(avg_Y_pred)  # Y_pred_coef)

        # convert tensors to numpy arrays
        self.pipeline_predictions = [
            pred.detach().cpu().numpy() for pred in self.pipeline_predictions]

    def correlate(self):
        """Calculate the correlation and R-squared between predicted
        and actual fMRI data.

        Done when both test_stim_data and test_fmri_data are provided."""
        self.pipeline_correlations = []
        self.coef_correlations = []
        self.pipeline_r_squared = []
        self.coef_r_squared = []

        assert len(self.pipeline_predictions) == len(self.coef_predictions)
        print("predictions shape", np.array(self.pipeline_predictions).shape)

        for i in range(len(self.pipeline_predictions)):
            pipeline_r2 = utils.r2_score(self.test_fmri_arrays[i],
                                         self.pipeline_predictions[i])
            self.pipeline_r_squared.append(pipeline_r2)

            coef_r2 = utils.r2_score(self.test_fmri_arrays[i],
                                     self.coef_predictions[i])
            self.coef_r_squared.append(coef_r2)

            # # Calculate the correlation
            # test_correlations = utils.calc_corr(
            #     self.predictions[i], self.test_fmri_arrays[i])
            # self.correlations.append(test_correlations)

            pipeline_corr = [
                pearsonr(self.test_fmri_arrays[i][:, j],
                         self.pipeline_predictions[i][:, j])[0] for j in range(
                             self.test_fmri_arrays[i].shape[1])]
            self.pipeline_correlations.append(pipeline_corr)

            coef_corr = [
                pearsonr(self.test_fmri_arrays[i][:, j],
                         self.coef_predictions[i][:, j])[0] for j in range(
                                self.test_fmri_arrays[i].shape[1])]
            self.coef_correlations.append(coef_corr)

            print("Max pipeline correlation:", np.nanmax(pipeline_corr))
            print("Max pipeline R-squared:", np.nanmax(pipeline_r2))
            print("Max coef correlation:", np.nanmax(coef_corr))
            print("Max coef R-squared:", np.nanmax(coef_r2))

        # Take the mean of the correlations
        self.mean_pipeline_corr = np.nanmean(
            np.stack((self.pipeline_correlations)), axis=0)
        # Take the mean of the R-squared values
        self.mean_pipeline_r_squared = np.nanmean(
            np.stack((self.pipeline_r_squared)), axis=0
        )
        self.mean_coef_corr = np.nanmean(
            np.stack((self.coef_correlations)), axis=0
        )
        self.mean_coef_r_squared = np.nanmean(
            np.stack((self.coef_r_squared)), axis=0
        )

    def encoding_pipeline(self, alignment=False, cv=None):
        """The encoding pipeline depends on the kind of data provided."""
        # Define the directory and file name
        directory = f'results/{self.model_handler.layer_name}'

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Loading features and fmri data will always occur before this
        if self.test_stim_dir:
            # In this case we build the full training model
            # and use it to predict data from test_stim_files
            print("Building encoding model and running predictions")
            self.build(cv=cv)

            # check if alignment needed
            if self.train_stim_type != self.test_stim_type:
                alignment = True

            self.predict(alignment=alignment)
            if self.test_fmri_dir:
                # In this case we add on to the last step and
                # calculate correlations between predicted and actual data
                print("Calculating correlations")
                self.correlate()

                # Output is the correlations
                correlation_types = [self.mean_pipeline_corr,
                                     self.mean_coef_corr]
                for i, corr_type in enumerate(correlation_types):
                    self.output = corr_type
                    if i == 0:
                        file_name = 'mean_pipeline_correlations.npy'
                    else:
                        file_name = 'mean_coef_correlations.npy'
                    file_path = os.path.join(directory, file_name)
                    np.save(file_path, self.output)

                r2_types = [self.mean_pipeline_r_squared,
                            self.mean_coef_r_squared]

                for i, r2_type in enumerate(r2_types):
                    self.output = r2_type
                    if i == 0:
                        file_name = 'mean_pipeline_r_squared.npy'
                    else:
                        file_name = 'mean_coef_r_squared.npy'
                    file_path = os.path.join(directory, file_name)
                    np.save(file_path, self.output)

                # # save output
                # np.save(file_path, self.output)

                # self.output = self.mean_r_squared

                # file_name = 'pred_correlations.npy'
                # file_path = os.path.join(directory, file_name)

                # # save output
                # np.save(file_path, self.output)
            else:
                # Output is mean predictions
                prediction_types = [self.pipeline_predictions,
                                    self.coef_predictions]
                for i, pred_type in enumerate(prediction_types):
                    print("before average shape [n_inputs,"
                          " len_inputs, features]",
                          np.array(pred_type).shape)
                    self.output = np.nanmean(np.array(pred_type), axis=0)
                    # Since predictions are still over time,
                    # average over time (first dim)
                    self.output = np.nanmean(self.output, axis=0)
                    print("after average shape", self.output.shape)

                    if i == 0:
                        file_name = 'mean_pipeline_predictions.npy'
                    else:
                        file_name = 'mean_coef_predictions.npy'
                    file_path = os.path.join(directory, file_name)

                    # save output
                    np.save(file_path, self.output)

                # print("before average shape [n_inputs, len_inputs,
                # features]",
                #       np.array(self.predictions).shape)
                # self.output = np.nanmean(np.array(self.predictions), axis=0)
                # # Since predictions are still over time,
                # # average over time (first dim)
                # self.output = np.nanmean(self.output, axis=0)
                # print("after average shape", self.output.shape)

                # file_name = 'predictions.npy'
                # file_path = os.path.join(directory, file_name)

                # # save output
                # np.save(file_path, self.output)
        else:
            # In this case we evaluate the model using leave-one-run-out
            # cross-validation
            print("Evaluating encoding model")
            self.evaluate()

            # Output is the average correlations
            self.output = np.nanmean(np.array(self.correlations), axis=0)

            file_name = 'eval_correlations.npy'
            file_path = os.path.join(directory, file_name)

            # save output
            np.save(file_path, self.output)
