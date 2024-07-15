# Basics
import numpy as np
import os

# Data loading
from PIL import Image
import torch
from torch.nn.functional import pad
from datasets import load_dataset

# Ridge regression
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config

# Model
from transformers import BridgeTowerModel, BridgeTowerProcessor
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration as llava
from transformers import BitsAndBytesConfig

# Specialized functions
import OLDutils

# Progress bar
from tqdm import tqdm


def setup_bridgetower(device, layer):
    """Function to setup transformers model with layer hooks.

    Parameters
    ----------
    layer : int
        A layer reference for the model.
        Sets the forward hook on the relevant layer.
    layer_name : str
        The name of the layer for which the forward hook is set.

    Returns
    -------
    device : str
        The device used for model computation. Can be 'cuda' or 'cpu'.
    model : torch.nn.Module
        The initialized model.
    processor : transformers.PreTrainedProcessor
        The initialized processor.
    features : dict
        A dictionary to store the output features of the forward hook.
    layer_selected : torch.utils.hooks.RemovableHandle
        The handle for the registered forward hook.

    Notes
    -----
    This function sets up a transformers model with layer hooks. It
    initializes the model and processor based on the
    provided model name. It also registers a forward hook on the specified
    layer and stores the output features in the
    `features` dictionary."""
    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    processor = BridgeTowerProcessor.from_pretrained(
        "BridgeTower/bridgetower-base")
    model = model.to(device)

    # Define layers
    model_layers = {
            1: model.cross_modal_text_transform,
            2: model.cross_modal_image_transform,
            3: model.token_type_embeddings,
            4: model.vision_model.visual.ln_post,
            5: model.text_model.encoder.layer[-1].output.LayerNorm,
            6: model.cross_modal_image_layers[-1].output,
            7: model.cross_modal_text_layers[-1].output,
            8: model.cross_modal_image_pooler,
            9: model.cross_modal_text_pooler,
            10: model.cross_modal_text_layernorm,
            11: model.cross_modal_image_layernorm,
            12: model.cross_modal_text_link_tower[-1],
            13: model.cross_modal_image_link_tower[-1],
        }

    # register forward hooks with layers of choice
    layer_selected = model_layers[layer].register_forward_hook(
        get_features(f"layer_{layer}"))

    return model, processor, features, layer_selected


def setup_llava(device, layer):
    """Function to setup transformers model with layer hooks.

    Parameters
    ----------
    layer : int
        A layer reference for the model.
        Sets the forward hook on the relevant layer.
    layer_name : str
        The name of the layer for which the forward hook is set.

    Returns
    -------
    device : str
        The device used for model computation. Can be 'cuda' or 'cpu'.
    model : torch.nn.Module
        The initialized model.
    processor : transformers.PreTrainedProcessor
        The initialized processor.
    features : dict
        A dictionary to store the output features of the forward hook.
    layer_selected : torch.utils.hooks.RemovableHandle
        The handle for the registered forward hook.

    Notes
    -----
    This function sets up a transformers model with layer hooks. It
    initializes the model and processor based on the
    provided model name. It also registers a forward hook on the specified
    layer and stores the output features in the
    `features` dictionary."""
    # placeholder for batch features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # detached_outputs = [tensor.detach() for tensor in output]
            last_output = output[-1].detach()
            features[name] = last_output  # detached_outputs
        return hook

    # Add in layers later
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.float16)
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = llava.from_pretrained(model_id,
                                  quantization_config=quantization_config,
                                  device_map="auto")
    layer = model.multi_modal_projector.linear_2
    layer_selected = layer.register_forward_hook(get_features(layer))

    return model, processor, features, layer_selected


def process_model_input(model, processor, input_data, device):
    model_input = processor(*input_data, return_tensors="pt")
    model_input = {key: value.to(device) for key, value in model_input.items()}
    _ = model(**model_input)
    return model_input


def average_tensors(tensors):
    first_size = tensors[0].size()
    if all(tensor.size() == first_size for tensor in tensors):
        avg_feature = torch.mean(torch.stack(tensors), dim=0)
    else:
        padded_tensors = pad_tensors_to_max(tensors)
        avg_feature = torch.mean(torch.stack(padded_tensors), dim=0)
    return avg_feature.detach().cpu().numpy()


def pad_tensors_to_max(tensors):
    max_size = [max(tensor.size(dim) for tensor in tensors) for dim in
                range(tensors[0].dim())]
    padded_tensors = []
    for tensor in tensors:
        pad_list = [max_size[dim] - tensor.size(dim) for dim in
                    range(tensor.dim())]
        pad_list = [item for sublist in zip(pad_list, [0] * len(pad_list)) for
                    item in sublist]
        padded_tensor = torch.nn.functional.pad(tensor, pad_list)
        padded_tensors.append(padded_tensor)
    return padded_tensors


# Movie Features
def get_movie_features(model_name, movie, subject, layer, layer_name, n=30):
    try:
        return np.load(f"results/features/movie/{subject}/" +
                       f"{layer_name}_{movie}.npy")
    except FileNotFoundError:
        pass

    movie_data = load_movie_data(movie)
    device, model, processor, features, layer_selected = setup_model(model_name, layer, layer_name)
    data = process_movie(movie_data, model, processor, features, n, device)
    layer_selected.remove()
    np.save(data, f"results/features/movie/{subject}/" +
                  f"{layer_name}_{movie}.npy")
    return data[layer_name]


def load_movie_data(movie):
    data_path = 'data/raw_stimuli/shortclips/stimuli/'
    print("Loading HDF array")
    return OLDutils.load_hdf5_array(f"{data_path}{movie}.hdf", key='stimuli')


def process_movie(movie_data, model, processor, features, n, device):
    data = {}
    avg_data = {}
    print("Running movie through model")
    for i, image in tqdm(enumerate(movie_data)):
        _ = process_model_input(model, processor, (image, ""), device)
        for name, tensor in features.items():
            if name not in avg_data:
                avg_data[name] = []
            avg_data[name].append(tensor)
        if (i + 1) % n == 0:
            for name, tensors in avg_data.items():
                avg_feature_numpy = average_tensors(tensors)
                if name not in data:
                    data[name] = []
                data[name].append(avg_feature_numpy)
            avg_data = {}
    return data


# Story Features
def get_story_features(model_name, story, subject, layer, layer_name, n=20):
    try:
        return np.load(f"results/features/story/{subject}/" +
                             f"{layer_name}_{story}.npy")
    except FileNotFoundError:
        pass

    story_data = load_story_data(story)
    device, model, processor, features, layer_selected = setup_model(model_name, layer, layer_name)
    data = process_story(story_data, model, processor, features, n, device)
    layer_selected.remove()
    np.save(data, f"results/features/story/{subject}/" +
                  "{layer_name}_{story}.npy")
    return data[layer_name]


def load_story_data(story):
    data_path = 'data/raw_stimuli/textgrids/stimuli/'
    print("Loading TextGrid")
    return OLDutils.textgrid_to_array(f"{data_path}{story}.TextGrid")


def process_story(story_data, model, processor, features, n, device):
    data = {}
    gray_value = 128
    image_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)
    print("Running story through model")
    for i, word in tqdm(enumerate(story_data)):
        word_with_context = get_word_with_context(story_data, i, n)
        _ = process_model_input(model, processor, (image_array,
                                                   word_with_context), device)
        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            data[name].append(tensor.detach().cpu().numpy())
    return data


def get_word_with_context(story_data, i, n):
    if i < n:
        return ' '.join(story_data[:(i+n)])
    elif i > (len(story_data) - n):
        return ' '.join(story_data[(i-n):])
    else:
        return ' '.join(story_data[(i-n):(i+n)])


def alignment(model_name, layer, layer_name):
    """Function generate matrices for feature alignment. Capture the
    linear relationship between caption features and image features
    output by a specific layer of the BridgeTower model.

    Parameters
    ----------
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer

    Returns
    -------
    coef_images_to_captions : Array
        Array of shape (layer_output_size, layer_output_size) mapping
        the relationship of image features to caption features.
    coef_captions_to_images: Array
        Array of shape (layer_output_size, layer_output_size) mapping
            the relationship of caption features to image features.
    """
    # Check if alignment is already done
    try:
        coef_images_to_captions = np.load(f'results/alignment/{layer_name}/'
                                          'coef_images_to_captions.npy')
        coef_captions_to_images = np.load(f'results/alignment/{layer_name}/'
                                          'coef_captions_to_images.npy')
        print("Alignment already done, retrieving coefficients")
    except FileNotFoundError:
        print("Starting feature alignment")
        # Stream the dataset so it doesn't download to device
        test_dataset = load_dataset("nlphuji/flickr30k", split='test',
                                    streaming=True)

        # Define Model
        device, model, processor, features, _ = setup_model(model_name, layer, layer_name)

        data = []

        print("Running flickr through model")
        # Assuming 'test_dataset' is an IterableDataset from a streaming source
        for item in tqdm(test_dataset):
            # Access data directly from the item, no need for indexing
            image = item['image']
            image_array = np.array(image)
            caption = " ".join(item['caption'])

            # Run image
            image_input = processor(image_array, "", return_tensors="pt")
            image_input = {key: value.to(device)
                           for key, value in image_input.items()}

            _ = model(**image_input)
            image_vector = features[layer_name]

            # Run caption
            # Create a numpy array filled with gray values (128 in this case)
            # This will act as the zero image input
            gray_value = 128
            gray_image_array = np.full((512, 512, 3), gray_value,
                                       dtype=np.uint8)

            caption_input = processor(gray_image_array, caption,
                                      return_tensors="pt")
            caption_input = {key: value.to(device)
                             for key, value in caption_input.items()}
            _ = model(**caption_input)

            caption_vector = features[layer_name]

            data.append([image_vector.detach().cpu().numpy(),
                        caption_vector.detach().cpu().numpy()])

        # Run encoding model
        backend = set_backend("torch_cuda", on_error="warn")
        print(backend)

        data = np.array(data)
        # Test data
        print(data.shape)
        # Variables
        captions = data[:, 1, :]
        images = data[:, 0, :]

        alphas = np.logspace(1, 20, 20)
        scaler = StandardScaler(with_mean=True, with_std=False)

        ridge_cv = RidgeCV(
            alphas=alphas, cv=5,
            solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                               n_targets_batch_refit=100))

        pipeline = make_pipeline(
            scaler,
            ridge_cv
        )

        _ = pipeline.fit(images, captions)
        coef_images_to_captions = backend.to_numpy(pipeline[-1].coef_)
        coef_images_to_captions /= np.linalg.norm(coef_images_to_captions,
                                                  axis=0)[None]

        _ = pipeline.fit(captions, images)
        coef_captions_to_images = backend.to_numpy(pipeline[-1].coef_)
        coef_captions_to_images /= np.linalg.norm(coef_captions_to_images,
                                                  axis=0)[None]

        print("Finished feature alignment, saving coefficients")
        # Save coefficients
        np.save(f'results/alignment/{layer_name}/coef_images_to_captions.npy',
                coef_images_to_captions)
        np.save(f'results/alignment/{layer_name}/coef_captions_to_images.npy',
                coef_captions_to_images)
    return coef_images_to_captions, coef_captions_to_images


def crossmodal_vision_model(model_name, subject, layer):
    """Function to build the vision encoding model. Creates a
    matrix mapping the linear relationship between BridgeTower features
    and brain voxel activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.

    Returns
    -------
    average_coef: Array
        Array of shape (layer_output_size*4, num_voxels) mapping
        the relationship of delayed feature vectors to each voxel
        in the fmri data.
    """
    print("Extracting features from data")

    # Extract features from raw stimuli
    train00 = get_movie_features(model_name, 'train_00', subject, layer)
    train01 = get_movie_features(model_name, 'train_01', subject, layer)
    train02 = get_movie_features(model_name, 'train_02', subject, layer)
    train03 = get_movie_features(model_name, 'train_03', subject, layer)
    train04 = get_movie_features(model_name, 'train_04', subject, layer)
    train05 = get_movie_features(model_name, 'train_05', subject, layer)
    train06 = get_movie_features(model_name, 'train_06', subject, layer)
    train07 = get_movie_features(model_name, 'train_07', subject, layer)
    train08 = get_movie_features(model_name, 'train_08', subject, layer)
    train09 = get_movie_features(model_name, 'train_09', subject, layer)
    train10 = get_movie_features(model_name, 'train_10', subject, layer)
    train11 = get_movie_features(model_name, 'train_11', subject, layer)
    test = get_movie_features(model_name, 'test', subject, layer)

    # Build encoding model
    print("Loading movie fMRI data")
    # Load fMRI data
    # Using all data for cross-modality encoding model
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    train_fmri = OLDutils.remove_nan(fmri_train)
    test_fmri = OLDutils.remove_nan(fmri_test)

    fmri_arrays = [train_fmri, test_fmri]
    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11, test]

    # Combine data
    Y_train = np.vstack(fmri_arrays)
    X_train = np.vstack(feature_arrays)

    # Define cross-validation
    run_onsets = []
    current_index = 0
    for arr in feature_arrays:
        next_index = current_index + arr.shape[0]
        run_onsets.append(current_index)
        current_index = next_index

    n_samples_train = X_train.shape[0]
    cv = OLDutils.generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)

    delayer = OLDutils.Delayer(delays=[1, 2, 3, 4])

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X_train = X_train.astype("float32")

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

    print("Finished vision encoding model")

    return average_coef


def crossmodal_language_model(model_name, subject, layer, layer_name):
    """Function to build the language encoding model. Creates a
    matrix mapping the linear relationship between BridgeTower features
    and brain voxel activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.

    Returns
    -------
    average_coef: Array
        Array of shape (layer_output_size*4, num_voxels) mapping
        the relationship of delayed feature vectors to each voxel
        in the fmri data.
    """
    print("Extracting features from data")

    # Extract features from raw stimuli
    alternateithicatom = get_story_features(model_name, 'alternateithicatom', subject,
                                            layer, layer_name)
    avatar = get_story_features(model_name, 'avatar', subject, layer, layer_name)
    howtodraw = get_story_features(model_name, 'howtodraw', subject, layer, layer_name)
    legacy = get_story_features(model_name, 'legacy', subject, layer, layer_name)
    life = get_story_features(model_name, 'life', subject, layer, layer_name)
    yankees = get_story_features(model_name, 'myfirstdaywiththeyankees', subject,
                                 layer, layer_name)
    naked = get_story_features(model_name, 'naked', subject, layer, layer_name)
    ode = get_story_features(model_name, 'odetostepfather', subject, layer, layer_name)
    souls = get_story_features(model_name, 'souls', subject, layer, layer_name)
    undertheinfluence = get_story_features(model_name, 'undertheinfluence',
                                           subject, layer, layer_name)

    # Build encoding model
    print('Load story fMRI data')
    # Load fmri data
    # Using all data for cross-modality encoding model
    fmri_alternateithicatom = np.load(f"data/storydata/{subject}/" +
                                      "alternateithicatom.npy")
    fmri_avatar = np.load(f"data/storydata/{subject}/avatar.npy")
    fmri_howtodraw = np.load(f"data/storydata/{subject}/howtodraw.npy")
    fmri_legacy = np.load(f"data/storydata/{subject}/legacy.npy")
    fmri_life = np.load(f"data/storydata/{subject}/life.npy")
    fmri_yankees = np.load(f"data/storydata/{subject}/" +
                           "myfirstdaywiththeyankees.npy")
    fmri_naked = np.load(f"data/storydata/{subject}/naked.npy")
    fmri_ode = np.load(f"data/storydata/{subject}/odetostepfather.npy")
    fmri_souls = np.load(f"data/storydata/{subject}/souls.npy")
    fmri_undertheinfluence = np.load(f"data/storydata/{subject}/" +
                                     "undertheinfluence.npy")

    print(alternateithicatom.shape)
    # Prep data
    fmri_ai, ai_features = OLDutils.prep_data(fmri_alternateithicatom,
                                           alternateithicatom)
    fmri_avatar, avatar_features = OLDutils.prep_data(fmri_avatar, avatar)
    fmri_howtodraw, howtodraw_features = OLDutils.prep_data(fmri_howtodraw,
                                                         howtodraw)
    fmri_legacy, legacy_features = OLDutils.prep_data(fmri_legacy, legacy)
    fmri_life, life_features = OLDutils.prep_data(fmri_life, life)
    fmri_yankees, yankees_features = OLDutils.prep_data(fmri_yankees, yankees)
    fmri_naked, naked_features = OLDutils.prep_data(fmri_naked, naked)
    fmri_ode, odetostepfather_features = OLDutils.prep_data(fmri_ode, ode)
    fmri_souls, souls_features = OLDutils.prep_data(fmri_souls, souls)
    fmri_under, under_features = OLDutils.prep_data(fmri_undertheinfluence,
                                                 undertheinfluence)

    fmri_arrays = [fmri_ai, fmri_avatar, fmri_howtodraw,
                   fmri_legacy, fmri_life, fmri_yankees, fmri_naked,
                   fmri_ode, fmri_souls, fmri_under]
    feature_arrays = [ai_features, avatar_features, howtodraw_features,
                      legacy_features, life_features, yankees_features,
                      naked_features, odetostepfather_features,
                      souls_features, under_features]
    # Combine data
    Y_train = np.vstack(fmri_arrays)
    X_train = np.vstack(feature_arrays)

    # Define cross-validation
    run_onsets = []
    current_index = 0
    for arr in feature_arrays:
        next_index = current_index + arr.shape[0]
        run_onsets.append(current_index)
        current_index = next_index

    n_samples_train = X_train.shape[0]
    cv = OLDutils.generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)  # cross-validation splitter into a reusable list

    # Define the model
    scaler = StandardScaler(with_mean=True, with_std=False)

    delayer = OLDutils.Delayer(delays=[1, 2, 3, 4])

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X_train = X_train.astype("float32")

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

    print("Finished language encoding model")

    return average_coef


def story_prediction(model_name, subject, layer, layer_name, vision_encoding_matrix):
    """Function to run the vision encoding model. Predicts brain activity
    to story listening and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    vision_encoding_matrix: array
        Generated by vision_model() function. A matrix mapping the relationship
        between feature vectors and brain activity

    Returns
    -------
    correlations: Array
        Array of shape (num_voxels) representing the correlation between
        predictions and real brain activity for each voxel.
    """
    _, coef_captions_to_images = alignment(model_name, layer, layer_name)

    # Get story features
    alternateithicatom = get_story_features('alternateithicatom', subject,
                                            layer)
    avatar = get_story_features('avatar', subject, layer)
    howtodraw = get_story_features('howtodraw', subject, layer)
    legacy = get_story_features('legacy', subject, layer)
    life = get_story_features('life', subject, layer)
    yankees = get_story_features('myfirstdaywiththeyankees', subject,
                                 layer)
    naked = get_story_features('naked', subject, layer)
    ode = get_story_features('odetostepfather', subject, layer)
    souls = get_story_features('souls', subject, layer)
    undertheinfluence = get_story_features('undertheinfluence',
                                           subject, layer)

    # Project features into opposite space
    alternateithicatom_transformed = np.dot(alternateithicatom,
                                            coef_captions_to_images.T)
    avatar_transformed = np.dot(avatar, coef_captions_to_images.T)
    howtodraw_transformed = np.dot(howtodraw, coef_captions_to_images.T)
    legacy_transformed = np.dot(legacy, coef_captions_to_images.T)
    life_transformed = np.dot(life, coef_captions_to_images.T)
    yankees_transformed = np.dot(yankees, coef_captions_to_images.T)
    naked_transformed = np.dot(naked, coef_captions_to_images.T)
    ode_transformed = np.dot(ode, coef_captions_to_images.T)
    souls_transformed = np.dot(souls, coef_captions_to_images.T)
    undertheinfluence_transformed = np.dot(undertheinfluence,
                                           coef_captions_to_images.T)

    # Load fmri data
    fmri_alternateithicatom = np.load("data/storydata/" + subject +
                                      "/alternateithicatom.npy")
    fmri_avatar = np.load("data/storydata/" + subject + "/avatar.npy")
    fmri_howtodraw = np.load("data/storydata/" + subject + "/howtodraw.npy")
    fmri_legacy = np.load("data/storydata/" + subject + "/legacy.npy")
    fmri_life = np.load("data/storydata/" + subject + "/life.npy")
    fmri_yankees = np.load("data/storydata/" + subject +
                           "/myfirstdaywiththeyankees.npy")
    fmri_naked = np.load("data/storydata/" + subject + "/naked.npy")
    fmri_ode = np.load("data/storydata/" + subject + "/odetostepfather.npy")
    fmri_souls = np.load("data/storydata/" + subject + "/souls.npy")
    fmri_undertheinfluence = np.load("data/storydata/" + subject +
                                     "/undertheinfluence.npy")

    # Prep data
    fmri_ai, ai_features = OLDutils.prep_data(fmri_alternateithicatom,
                                           alternateithicatom_transformed)
    fmri_avatar, avatar_features = OLDutils.prep_data(fmri_avatar,
                                                   avatar_transformed)
    fmri_howtodraw, howtodraw_features = OLDutils.prep_data(fmri_howtodraw,
                                                         howtodraw_transformed)
    fmri_legacy, legacy_features = OLDutils.prep_data(fmri_legacy,
                                                   legacy_transformed)
    fmri_life, life_features = OLDutils.prep_data(fmri_life, life_transformed)
    fmri_yankees, yankees_features = OLDutils.prep_data(fmri_yankees,
                                                     yankees_transformed)
    fmri_naked, naked_features = OLDutils.prep_data(fmri_naked, naked_transformed)
    fmri_ode, odetostepfather_features = OLDutils.prep_data(fmri_ode,
                                                         ode_transformed)
    fmri_souls, souls_features = OLDutils.prep_data(fmri_souls, souls_transformed)
    fmri_under, under_features = OLDutils.prep_data(fmri_undertheinfluence,
                                                 undertheinfluence_transformed)

    # Make fmri predictions
    ai_predictions = np.dot(ai_features, vision_encoding_matrix)
    avatar_predictions = np.dot(avatar_features, vision_encoding_matrix)
    howtodraw_predictions = np.dot(howtodraw_features, vision_encoding_matrix)
    legacy_predictions = np.dot(legacy_features, vision_encoding_matrix)
    life_predictions = np.dot(life_features, vision_encoding_matrix)
    yankees_predictions = np.dot(yankees_features, vision_encoding_matrix)
    naked_predictions = np.dot(naked_features, vision_encoding_matrix)
    odetostepfather_predictions = np.dot(odetostepfather_features,
                                         vision_encoding_matrix)
    souls_predictions = np.dot(souls_features, vision_encoding_matrix)
    under_predictions = np.dot(under_features, vision_encoding_matrix)

    # Calculate correlations
    ai_correlations = OLDutils.calc_correlation(ai_predictions, fmri_ai)
    avatar_correlations = OLDutils.calc_correlation(avatar_predictions,
                                                 fmri_avatar)
    howtodraw_correlations = OLDutils.calc_correlation(howtodraw_predictions,
                                                    fmri_howtodraw)
    legacy_correlations = OLDutils.calc_correlation(legacy_predictions,
                                                 fmri_legacy)
    life_correlations = OLDutils.calc_correlation(life_predictions, fmri_life)
    yankees_correlations = OLDutils.calc_correlation(yankees_predictions,
                                                  fmri_yankees)
    naked_correlations = OLDutils.calc_correlation(naked_predictions, fmri_naked)
    ode_correlations = OLDutils.calc_correlation(odetostepfather_predictions,
                                              fmri_ode)
    souls_correlations = OLDutils.calc_correlation(souls_predictions, fmri_souls)
    under_correlations = OLDutils.calc_correlation(under_predictions, fmri_under)

    # Get mean correlation
    all_correlations = np.stack((ai_correlations, avatar_correlations,
                                 howtodraw_correlations, legacy_correlations,
                                 life_correlations, yankees_correlations,
                                 naked_correlations, ode_correlations,
                                 souls_correlations, under_correlations))

    story_correlations = np.nanmean(all_correlations, axis=0)
    print("Max correlation:", np.nanmax(story_correlations))

    np.save(f'results/movie_to_story/{subject}/' +
            f'layer{str(layer)}_correlations.npy', story_correlations)

    return story_correlations


def movie_prediction(subject, layer, language_encoding_model):
    """Function to run the language encoding model. Predicts brain activity
    to movie watching and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    language_encoding_matrix: array
        Generated by language_model() function. A matrix mapping the
        relationship between feature vectors and brain activity

    Returns
    -------
    correlations: Array
        Array of shape (num_voxels) representing the correlation between
        predictions and real brain activity for each voxel.
    """
    coef_images_to_captions, _ = alignment(layer)

    # Get movie features
    train00 = get_movie_features('train_00', subject, layer)
    train01 = get_movie_features('train_01', subject, layer)
    train02 = get_movie_features('train_02', subject, layer)
    train03 = get_movie_features('train_03', subject, layer)
    train04 = get_movie_features('train_04', subject, layer)
    train05 = get_movie_features('train_05', subject, layer)
    train06 = get_movie_features('train_06', subject, layer)
    train07 = get_movie_features('train_07', subject, layer)
    train08 = get_movie_features('train_08', subject, layer)
    train09 = get_movie_features('train_09', subject, layer)
    train10 = get_movie_features('train_10', subject, layer)
    train11 = get_movie_features('train_11', subject, layer)
    test = get_movie_features('test', subject, layer)

    # Project features into opposite space
    test_transformed = np.dot(test, coef_images_to_captions.T)
    train00_transformed = np.dot(train00, coef_images_to_captions.T)
    train01_transformed = np.dot(train01, coef_images_to_captions.T)
    train02_transformed = np.dot(train02, coef_images_to_captions.T)
    train03_transformed = np.dot(train03, coef_images_to_captions.T)
    train04_transformed = np.dot(train04, coef_images_to_captions.T)
    train05_transformed = np.dot(train05, coef_images_to_captions.T)
    train06_transformed = np.dot(train06, coef_images_to_captions.T)
    train07_transformed = np.dot(train07, coef_images_to_captions.T)
    train08_transformed = np.dot(train08, coef_images_to_captions.T)
    train09_transformed = np.dot(train09, coef_images_to_captions.T)
    train10_transformed = np.dot(train10, coef_images_to_captions.T)
    train11_transformed = np.dot(train11, coef_images_to_captions.T)

    # Load fmri data
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Prep data
    fmri_train = OLDutils.remove_nan(fmri_train)
    fmri_test = OLDutils.remove_nan(fmri_test)

    # Make fmri predictions
    feature_arrays = [train00_transformed, train01_transformed,
                      train02_transformed, train03_transformed,
                      train04_transformed, train05_transformed,
                      train06_transformed, train07_transformed,
                      train08_transformed, train09_transformed,
                      train10_transformed, train11_transformed]

    features_train = np.vstack(feature_arrays)
    features_test = test_transformed

    predictions_train = np.dot(features_train, language_encoding_model)
    predictions_test = np.dot(features_test, language_encoding_model)

    # Calculate correlations
    correlations_train = OLDutils.calc_correlation(predictions_train, fmri_train)
    correlations_test = OLDutils.calc_correlation(predictions_test, fmri_test)

    # Get mean correlation
    all_correlations = np.stack((correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_train, correlations_train,
                                 correlations_test))

    correlations = np.nanmean(all_correlations, axis=0)
    print('max correlation', np.nanmax(correlations))

    np.save(f'results/story_to_movie/{subject}/' +
            f'layer{str(layer)}_correlations.npy', correlations)

    return correlations


def withinmodal_vision_model(subject, layer):
    """Function to build the vision encoding model. Creates a
    matrix mapping the linear relationship between BridgeTower features
    and brain voxel activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.

    Returns
    -------
    average_coef: Array
        Array of shape (layer_output_size*4, num_voxels) mapping
        the relationship of delayed feature vectors to each voxel
        in the fmri data.
    """
    print("Extracting features from data")

    # Extract features from raw stimuli
    train00 = get_movie_features('train_00', subject, layer)
    train01 = get_movie_features('train_01', subject, layer)
    train02 = get_movie_features('train_02', subject, layer)
    train03 = get_movie_features('train_03', subject, layer)
    train04 = get_movie_features('train_04', subject, layer)
    train05 = get_movie_features('train_05', subject, layer)
    train06 = get_movie_features('train_06', subject, layer)
    train07 = get_movie_features('train_07', subject, layer)
    train08 = get_movie_features('train_08', subject, layer)
    train09 = get_movie_features('train_09', subject, layer)
    train10 = get_movie_features('train_10', subject, layer)
    train11 = get_movie_features('train_11', subject, layer)
    test = get_movie_features('test', subject, layer)

    feature_arrays = [train00, train01, train02, train03, train04,
                      train05, train06, train07, train08, train09,
                      train10, train11, test]

    # Build encoding model
    print("Loading movie fMRI data")
    # Load fMRI data
    fmri_train = np.load("data/moviedata/" + subject + "/train.npy")
    fmri_test = np.load("data/moviedata/" + subject + "/test.npy")

    # Split the fmri train data to match features (12 parts)
    fmri_train00 = fmri_train[:300]
    fmri_train01 = fmri_train[300:600]
    fmri_train02 = fmri_train[600:900]
    fmri_train03 = fmri_train[900:1200]
    fmri_train04 = fmri_train[1200:1500]
    fmri_train05 = fmri_train[1500:1800]
    fmri_train06 = fmri_train[1800:2100]
    fmri_train07 = fmri_train[2100:2400]
    fmri_train08 = fmri_train[2400:2700]
    fmri_train09 = fmri_train[2700:3000]
    fmri_train10 = fmri_train[3000:3300]
    fmri_train11 = fmri_train[3300:]

    # Prep data
    train00_fmri = OLDutils.remove_nan(fmri_train00)
    train01_fmri = OLDutils.remove_nan(fmri_train01)
    train02_fmri = OLDutils.remove_nan(fmri_train02)
    train03_fmri = OLDutils.remove_nan(fmri_train03)
    train04_fmri = OLDutils.remove_nan(fmri_train04)
    train05_fmri = OLDutils.remove_nan(fmri_train05)
    train06_fmri = OLDutils.remove_nan(fmri_train06)
    train07_fmri = OLDutils.remove_nan(fmri_train07)
    train08_fmri = OLDutils.remove_nan(fmri_train08)
    train09_fmri = OLDutils.remove_nan(fmri_train09)
    train10_fmri = OLDutils.remove_nan(fmri_train10)
    train11_fmri = OLDutils.remove_nan(fmri_train11)
    test_fmri = OLDutils.remove_nan(fmri_test)

    fmri_arrays = [train00_fmri, train01_fmri, train02_fmri,
                   train03_fmri, train04_fmri, train05_fmri,
                   train06_fmri, train07_fmri, train08_fmri,
                   train09_fmri, train10_fmri, train11_fmri,
                   test_fmri]

    correlations = []

    # For each of the 12 x,y pairs, we will train
    # a model on 11 and test using the held out one
    for i in range(len(feature_arrays)):
        print("leaving out run", i)
        new_feat_arrays = OLDutils.remove_run(feature_arrays, i)
        X_train = np.vstack(new_feat_arrays)
        Y_train = np.vstack(OLDutils.remove_run(fmri_arrays, i))

        print("X_train shape", X_train.shape)
        # Define cross-validation
        run_onsets = []
        current_index = 0
        for arr in new_feat_arrays:
            next_index = current_index + arr.shape[0]
            run_onsets.append(current_index)
            current_index = next_index

        print(run_onsets)
        n_samples_train = X_train.shape[0]
        cv = OLDutils.generate_leave_one_run_out(n_samples_train, run_onsets)
        cv = check_cv(cv)  # cross-validation splitter into a reusable list

        # Define the model
        scaler = StandardScaler(with_mean=True, with_std=False)
        delayer = OLDutils.Delayer(delays=[1, 2, 3, 4])
        backend = set_backend("torch_cuda", on_error="warn")
        print(backend)
        X_train = X_train.astype("float32")
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
        X_test = feature_arrays[i]
        Y_test = fmri_arrays[i]

        # Predict
        Y_pred = np.dot(X_test, average_coef)

        test_correlations = OLDutils.calc_correlation(Y_pred, Y_test)

        print("Max correlation:", np.nanmax(test_correlations))

        correlations.append(test_correlations)

    print("Finished vision encoding model")

    # Make correlations np array
    correlations = np.array(correlations)
    print(correlations.shape)

    # Take average correlations over all runs
    average_correlations = np.nanmean(correlations, axis=0)

    np.save('results/vision_model/' + subject +
            '/layer' + str(layer) + '_correlations.npy', average_correlations)

    return average_correlations


def faceLandscape_prediction(subject, modality, layer, vision_encoding_matrix):
    """Function to run the vision encoding model. Predicts brain activity
    to story listening and return correlations between predictions and real
    brain activity.

    Parameters
    ----------
    subject: string
        A reference to the subject for analysis. Used to load fmri data.
    layer: int
        A layer reference for the BridgeTower model. Set's the forward
        hook on the relevant layer.
    vision_encoding_matrix: array
        Generated by vision_model() function. A matrix mapping the relationship
        between feature vectors and brain activity

    Returns
    -------
    correlations: Array
        Array of shape (num_voxels) representing the correlation between
        predictions and real brain activity for each voxel.
    """
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

    print("number of images:", len(os.listdir(data_path)))
    # Get face features
    for i, image_filename in tqdm(enumerate(os.listdir(data_path))):
        # Load image as PIL
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg',
                                            '.bmp', '.gif')):
            image_path = os.path.join(data_path, image_filename)
            try:
                image = Image.open(image_path).convert('RGB')
                model_input = processor(image, "", return_tensors="pt")
                model_input = {key: value.to(device) for
                               key, value in model_input.items()}
            except Exception as e:
                print(f"Failed to process {image_filename}: {str(e)}")

        _ = model(**model_input)

        for name, tensor in features.items():
            if name not in data:
                data[name] = []
            numpy_tensor = tensor.detach().cpu().numpy()

            data[name].append(numpy_tensor)

        layer_selected.remove()

    # Save data
    data = np.array(data[f"layer_{layer}"])

    # Data should be 2d of shape (n_images/n, num_features)
    # if data is above 2d, average 2nd+ dimensions
    if data.ndim > 2:
        data = np.mean(data, axis=1)

    print(f"Got {modality} features")

    print('encoding matrix shape:', vision_encoding_matrix.shape)
    print('data shape:', data.shape)
    # Make fmri predictions
    fmri_predictions = np.dot(data, vision_encoding_matrix)
    print('predictions shape:', fmri_predictions.shape)

    average_predictions = np.mean(fmri_predictions, axis=0)

    if modality == 'face':
        np.save('results/faces/' + subject +
                '/layer' + str(layer) + '_predictions.npy',
                average_predictions)
    elif modality == 'landscape':
        np.save('results/landscapes/' + subject +
                '/layer' + str(layer) + '_predictions.npy',
                average_predictions)

    return average_predictions