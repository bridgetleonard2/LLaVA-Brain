from classes import model_handlerCLASS
from PIL import Image
import requests
import numpy as np
import h5py

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()


# test feature extraction
def load_hdf5_array(file_name, key=None, slice=slice(0, None)):
    """Function to load data from an hdf file.

    Parameters
    ----------
    file_name: string
        hdf5 file name.
    key: string
        Key name to load. If not provided, all keys will be loaded.
    slice: slice, or tuple of slices
        Load only a slice of the hdf5 array. It will load `array[slice]`.
        Use a tuple of slices to get a slice in multiple dimensions.

    Returns
    -------
    result : array or dictionary
        Array, or dictionary of arrays (if `key` is None).
    """
    with h5py.File(file_name, mode='r') as hf:
        if key is None:
            data = dict()
            for k in hf.keys():
                data[k] = hf[k][slice]
            return data
        else:
            return hf[key][slice]


# prepare image and text prompt, using the appropriate prompt template
path = '../bridgetower-brain/data/raw_stimuli/shortclips/stimuli/train_00.hdf'
movie = load_hdf5_array(path, key='stimuli')
movie = np.array(movie)

# llava-v1.6-34b-hf requires the following format:
# "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
image = movie[0]
# image = Image.fromarray(image)
# Define the prompt and expected input format
prompt = "What's going on?"
formatted_prompt = f"system\nAnswer the questions.\nuser\n<image>\n{prompt}\nassistant\n"

# Process the prompt and image
inputs = model_handler.processor(formatted_prompt, image, return_tensors="pt").to('cuda')

# autoregressively complete prompt
output = model_handler.model.generate(**inputs, max_new_tokens=100)

print(model_handler.processor.decode(output[0], skip_special_tokens=True))

# Extracted features:
features = model_handler.features['layer']  # This will contain the extracted features from the specified layer

# print some of features
print(features[:10])

model_handler.reset_features()  # Reset the features for the next extraction