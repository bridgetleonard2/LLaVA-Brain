# Basics
import numpy as np
import torch
import os

# Data loading
from torch.nn.functional import pad
from PIL import Image

# Ridge regression
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config

# Model
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig

# Specialize functions
import fun

# Progress bar
from tqdm import tqdm


def setup_model(layer_selected='multi_modal_projector.linear_2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.float16)

    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    layer = model.multi_modal_projector.linear_2.register_forward_hook(get_features(layer_selected))

    return device, model, processor, features, layer


def get_movie_features(movie_data, layer, n=30):
    """Function to average feature vectors over every n inputs.

    Parameters
    ----------
    movie_data: Array
        An array of shape (n_images, 512, 512). Represents frames from
        a color movie.
    n (optional): int
        Number of frames to average over. Set at 30 to mimick an MRI
        TR = 2 with a 15 fps movie.

    Returns
    -------
    data : Dictionary
        Dictionary where keys are the model layer from which activations are
        extracted. Values are lists representing activations of 768 dimensions
        over the course of n_images / 30.
    """
    print("loading HDF array")
    movie_data = fun.load_hdf5_array(movie_data, key='stimuli')

    # Define Model
    device, model, processor, features, layer_selected = setup_model(layer)

    # create overall data structure for average feature vectors
    # a dictionary with layer names as keys and a list of vectors as it values
    data = {}

    # a dictionary to store vectors for n consecutive trials
    avg_data = {}

    print("Running movie through model")
    # loop through all inputs
    for i, image in tqdm(enumerate(movie_data)):

        model_input = processor(image, "", return_tensors="pt")
        # Assuming model_input is a dictionary of tensors
        model_input = {key: value.to(device) for key,
                       value in model_input.items()}

        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in avg_data:
                avg_data[name] = []
            avg_data[name].append(tensor)

        # check if average should be stored
        if (i + 1) % n == 0:
            for name, tensors in avg_data.items():
                first_size = tensors[0].size()

                if all(tensor.size() == first_size for tensor in tensors):
                    avg_feature = torch.mean(torch.stack(tensors), dim=0)
                    avg_feature_numpy = avg_feature.detach().cpu().numpy()
                    # print(len(avg_feature_numpy))
                else:
                    # Find problem dimension
                    for dim in range(tensors[0].dim()):
                        first_dim = tensors[0].size(dim)

                        if not all(tensor.size(dim) == first_dim
                                   for tensor in tensors):
                            # Specify place to pad
                            p_dim = (tensors[0].dim()*2) - (dim + 2)
                            # print(p_dim)
                            max_size = max(tensor.size(dim)
                                           for tensor in tensors)
                            padded_tensors = []

                            for tensor in tensors:
                                # Make a list with length of 2*dimensions - 1
                                # to insert pad later
                                pad_list = [0] * ((2*tensor[0].dim()) - 1)
                                pad_list.insert(
                                    p_dim, max_size - tensor.size(dim))
                                # print(tuple(pad_list))
                                padded_tensor = pad(tensor, tuple(pad_list))
                                padded_tensors.append(padded_tensor)

                    avg_feature = torch.mean(torch.stack(padded_tensors),
                                             dim=0)
                    avg_feature_numpy = avg_feature.detach().cpu().numpy()
                    # print(len(avg_feature_numpy))

                if name not in data:
                    data[name] = []
                data[name].append(avg_feature_numpy)

            avg_data = {}

    layer_selected.remove()

    # Save data
    data = np.array(data[layer])
    print("Got movie features")

    return data


def get_Xdata(layer):
    data_path = 'data/raw_stimuli/shortclips/stimuli/'
    print("Extracting features from data")

    # Extract features from raw stimuli
    train00 = get_movie_features(data_path + 'train_00.hdf', layer)
    train01 = get_movie_features(data_path + 'train_01.hdf', layer)
    train02 = get_movie_features(data_path + 'train_02.hdf', layer)
    train03 = get_movie_features(data_path + 'train_03.hdf', layer)
    train04 = get_movie_features(data_path + 'train_04.hdf', layer)
    train05 = get_movie_features(data_path + 'train_05.hdf', layer)
    train06 = get_movie_features(data_path + 'train_06.hdf', layer)
    train07 = get_movie_features(data_path + 'train_07.hdf', layer)
    train08 = get_movie_features(data_path + 'train_08.hdf', layer)
    train09 = get_movie_features(data_path + 'train_09.hdf', layer)
    train10 = get_movie_features(data_path + 'train_10.hdf', layer)
    train11 = get_movie_features(data_path + 'train_11.hdf', layer)
    test = get_movie_features(data_path + 'test.hdf', layer)

    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11, test]

    X_train = np.vstack(feature_arrays)

    return X_train, feature_arrays


def get_Ydata(subject):
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    def remove_nan(data):
        mask = ~np.isnan(data)

        # Apply the mask and then flatten
        # This will keep only the non-NaN values
        data_reshaped = data[mask].reshape(data.shape[0], -1)

        print("fMRI shape:", data_reshaped.shape)
        return data_reshaped

    # Prep data
    train_fmri = remove_nan(fmri_train)
    test_fmri = remove_nan(fmri_test)

    fmri_arrays = [train_fmri, test_fmri]

    Y_train = np.vstack(fmri_arrays)

    return Y_train


def run_model(X, y, feature_arrays):
    # Define cross-validation
    run_onsets = []
    current_index = 0
    for arr in feature_arrays:
        next_index = current_index + arr.shape[0]
        run_onsets.append(current_index)
        current_index = next_index

    n_samples_train = X.shape[0]
    cv = fun.generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = fun.Delayer(delays=[1, 2, 3, 4])

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X = X.astype("float32")

    alphas = np.logspace(1, 20, 20)

    print("Running linear model")
    ridge_cv = RidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100))

    pipeline = make_pipeline(
        scaler,
        delayer,
        ridge_cv,
    )

    set_config(display='diagram')  # requires scikit-learn 0.23
    pipeline

    _ = pipeline.fit(X, y)

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

    assert average_coef.shape[0] == X.shape[1]*4
    assert average_coef.shape[1] == y.shape[1]

    print("Finished vision encoding model")
    return average_coef


def get_test_data(layer, modality):
    if modality == 'face':
        data_path = 'data/face_stimuli'
    elif modality == 'landscape':
        data_path = 'data/landscape_stimuli'
    else:
        print("Invalid modality. Please choose 'face' or 'landscape'.")
    # Define Model
    device, model, processor, features, layer_selected = setup_model(layer)

    # Initiate data dict
    data = {}

    # Get face features
    for i, image_filename in enumerate(os.listdir(data_path)):
        # Load image as PIL
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(data_path, image_filename)
            try:
                image = Image.open(image_path).convert('RGB')
                model_input = processor(image, "", return_tensors="pt")
                model_input = {key: value.to(device) for key, value in model_input.items()}
            except Exception as e:
                print(f"Failed to process {image_filename}: {str(e)}")

        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            data[name].append(tensor)

        layer_selected.remove()

    # Save data
    data = np.array(data[f"layer_{layer}"])
    print("Got face features")

    return data
