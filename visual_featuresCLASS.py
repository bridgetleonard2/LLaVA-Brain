import numpy as np
import h5py

# Progress bar
from tqdm import tqdm

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


class VisualFeatures:
    def __init__(self, path, processor, device, model, features):
        self.path = path
        self.data_type = path.split('.')[-1]
        self.processor = processor
        self.device = device
        self.model = model
        self.features = features

    def load_image(self):
        if self.data_type == "hdf":
            self.stim_data = load_hdf5_array(self.path)

    def get_features(self):
        # prepare images for model
        # text is just blank strings for each of the items in stim_data
        text = ["" for i in range(len(self.stim_data))]
        model_inputs = self.processor(self.stim_data, text, return_tensors='pt')

        model_inputs = {key: value.to(self.device) for key,
                       value in model_inputs.items()}

        for i, image in tqdm(enumerate(self.stim_data)):
