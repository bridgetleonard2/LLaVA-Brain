import os
import numpy as np
from classes import visual_featuresCLASS
from classes import language_featuresCLASS
from sklearn import set_config

import utils


class EncodingModels:
    def __init__(self, model_handler, train_stim_dir, train_fmri_dir,
                 train_stim_type, test_stim_dir=None, test_fmri_dir=None,
                 test_stim_type=None, features_dir=None):
        """Initialize the EncodingModels class.
        train_stim_dir (str): Path to the directory with training stimuli.
        train_fmri_dir (str): Path to the directory with training fMRI data.

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
        self.train_stim_files = os.listdir(train_stim_dir)
        self.train_fmri_files = os.listdir(train_fmri_dir)

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
                    stim_title = stim.split('.')[0]
                    fmri_title = fmri.split('.')[0]
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
                train_stim_features = np.load(feat_path, allow_pickle=True)
            except FileNotFoundError:
                stim_path = os.path.join(self.train_stim_dir, stim_file)
                if self.train_stim_type == "visual":
                    visual_features = visual_featuresCLASS.VisualFeatures(
                        stim_path, self.model_handler)
                    visual_features.load_image()
                    visual_features.get_features()
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
            print("features shape", train_stim_features.shape[0])
            print("fmri shape", train_fmri_shape[0])
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
                        visual_features.get_features()
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

                    np.save(feat_path, test_stim_features)

                # Only resample features if dimensions don't match fmri
                if self.test_fmri_dir:
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

            # if X_train > 2 dimensions, average across the second dimension
            if len(X_train.shape) > 2:
                X_train = np.mean(X_train, axis=1)
                print("X_train shape", X_train.shape)

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

            # if X_test > 2 dimensions, average across the second dimension
            if len(X_test.shape) > 2:
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
        # try loading alignment
        filename = f"{self.model_handler.layer_name}_alignment.npy"
        alignment_path = os.path.join(self.features_dir, filename)
        try:
            align_matrix = np.load(alignment_path)
        except FileNotFoundError:
            # if alignment doesn't exist, create one
            alignment_data = load_dataset("nlphuji/flickr30k", split='test',
                                streaming=True)
            
            alignment_features = []
            
            for item in tqdm(alignment_data):
                image = item['image']
                image_array = np.array(image)
                caption = " ".join(item['caption'])
                
                stim_path = ""
                
                # dont do load_image
                visual_features = visual_featuresCLASS.VisualFeatures(
                        stim_path, self.model_handler)
                visual_features.stim_data = image_array
                visual_features.get_features()
                image_vector = visual_features.visualFeatures

                language_features = (
                        language_featuresCLASS.LanguageFeatures(
                            stim_path, self.model_handler))
                language_features.stim_data = caption
                language_features.get_features(alignment=True)
                caption_vector = language_features.languageFeatures

                alignment_features.append((image_vector, caption_vector))

            alignment_features = np.array(alignment_features)
            
            captions = alignment_features[:, 1]
            images = alignment_features[:, 0]

            # Data should be 2d of shape (n_images/n, num_features)
            # if data is above 2d, average 2nd+ dimensions
            if captions.ndim > 2:
                captions = np.mean(captions, axis=1)
                images = np.mean(images, axis=1)
            
            pipeline, backend = utils.set_pipeline("", cv=5)

            set_config(display='diagram')  # requires scikit-learn 0.23
            pipeline

            _ = pipeline.fit(images, captions)
            self.coef_image_to_caption = backend.to_numpy(pipeline[-1].coef_)

            # Check if zeroes in coef_images_to_captions
            num_zeroes_im_to_cap = np.count_nonzero(
                self.coef_image_to_caption == 0)
            print("image to caption zeros:", num_zeroes_im_to_cap)

            epsilon = 1e-10

            if num_zeroes_im_to_cap > 0:
                self.coef_image_to_caption = self.coef_image_to_caption.astype(
                    float)
                self.coef_image_to_caption[
                    self.coef_image_to_caption == 0] = epsilon

            self.coef_image_to_caption /= np.linalg.norm(
                self.coef_image_to_caption, axis=0)[None]

            _ = pipeline.fit(captions, images)
            self.coef_caption_to_image = backend.to_numpy(pipeline[-1].coef_)

            # Check if zeroes in coef_captions_to_images
            num_zeroes_cap_to_im = np.count_nonzero(
                self.coef_caption_to_image == 0)
            print("caption to image zeros:", num_zeroes_cap_to_im)

            if num_zeroes_cap_to_im > 0:
                self.coef_caption_to_image = self.coef_caption_to_image.astype(
                    float)
                self.coef_caption_to_image[
                    self.coef_caption_to_image == 0] = epsilon

            self.coef_caption_to_image /= np.linalg.norm(
                self.coef_caption_to_image, axis=0)[None]

    def build(self):
        """Build the encoding model."""
        print("Building encoding model using all training data")
        X_train = np.vstack(self.train_feature_arrays)
        Y_train = np.vstack(self.train_fmri_arrays)

        pipeline, backend = utils.set_pipeline(self.train_feature_arrays)

        set_config(display='diagram')  # requires scikit-learn 0.23
        pipeline

        X_train = X_train.astype(np.float32)

        # if X_train > 2 dimensions, average across the second dimension
        if len(X_train.shape) > 2:
            X_train = np.mean(X_train, axis=1)
            print("X_train shape", X_train.shape)

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

    def predict(self, alignment=False):
        """Predict fMRI data using the encoding model."""
        if alignment:
            if self.test_stim_type == "visual":
                self.test_feature_arrays = [
                    np.dot(X, self.coef_image_to_caption) for X
                    in self.test_feature_arrays]
            elif self.test_stim_type == "language":
                self.test_feature_arrays = [
                    np.dot(X, self.coef_caption_to_image) for X
                    in self.test_feature_arrays]
        self.predictions = []
        for i in range(len(self.test_feature_arrays)):
            X_test = self.test_feature_arrays[i]

            # if X_test > 2 dimensions, average across the second dimension
            if len(X_test.shape) > 2:
                X_test = np.mean(X_test, axis=1)
                print("X_test shape", X_test.shape)

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

    def encoding_pipeline(self, alignment=False):
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
            self.build()

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
                self.output = self.mean_correlations

                file_name = 'pred_correlations.npy'
                file_path = os.path.join(directory, file_name)

                # save output
                np.save(file_path, self.output)
            else:
                # Output is mean predictions
                print("before average shape [n_inputs, len_inputs, features]",
                      np.array(self.predictions).shape)
                self.output = np.nanmean(np.array(self.predictions), axis=0)
                # Since predictions are still over time,
                # average over time (first dim)
                self.output = np.nanmean(self.output, axis=0)
                print("after average shape", self.output.shape)

                file_name = 'predictions.npy'
                file_path = os.path.join(directory, file_name)

                # save output
                np.save(file_path, self.output)
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
