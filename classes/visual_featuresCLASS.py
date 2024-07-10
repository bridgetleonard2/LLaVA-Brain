import numpy as np
import torch
import torch.nn.functional as F
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
    def __init__(self, path, ModelHandler):
        self.path = path
        self.data_type = path.split('.')[-1]
        self.ModelHandler = ModelHandler

    def load_image(self):
        if self.data_type == "hdf":
            self.stim_data = load_hdf5_array(self.path)
            if self.ModelHandler.model_name == 'llava':
                # convert list to np.array
                self.stim_data = np.array(self.stim_data)

    def get_features(self, n=30):
        # prepare images for model
        # text is just blank strings for each of the items in stim_data
        text = ["" for i in range(self.stim_data.shape[0])]
        model_inputs = self.ModelHandler.processor(self.stim_data, text, return_tensors='pt')

        model_inputs = {key: value.to(self.ModelHandler.device) for key,
                        value in model_inputs.items()}

        # **RESET FEATURES DICT HERE** #
        self.ModelHandler.reset_features()

        _ = self.ModelHandler.model(**model_inputs)

        # Now features will be a dict with one key: 'layer'
        tensors = self.ModelHandler.features['layer']

        average_tensors = []
        # for every n tensor, take the average
        for i in tqdm(range(0, len(tensors), n)):
            try:
                n_tensors = tensors[i:i+10]
                average_tensors.append(torch.mean(torch.stack(n_tensors), dim=0))
            except Exception as e:
                print(f"Failed to average tensors: {e}")
                n_tensors = tensors[i:i+10]

                # size of first tensor
                first_size = tensors[0].size

                if not all(tensor.size() == first_size for tensor in n_tensors):
                    print("tensor size mismatch")
                    # find tensor with wrong size
                    for j, tensor in enumerate(n_tensors):
                        if tensor.size() != first_size:
                            print(f"Removing tensor: {tensor.size()} from average")
                            n_tensors.pop(j)
                    average_tensors.append(torch.mean(torch.stack(n_tensors), dim=0))

        average_tensors_numpy = [tensor.detach().cpu().numpy() for tensor in average_tensors]

        self.visualFeatures = np.array(average_tensors_numpy)

