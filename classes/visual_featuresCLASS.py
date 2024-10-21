import numpy as np
import torch  # type: ignore
import h5py  # type: ignore
from PIL import Image

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
            self.stim_data = load_hdf5_array(self.path,
                                             key='stimuli')
            self.stim_data = np.array(self.stim_data)
        elif self.data_type in ['png', 'jpg', 'jpeg',
                                'bmp', 'gif']:
            self.stim_data = Image.open(self.path).convert('RGB')

            # Since it's a single image, add a dimension to make it
            # a single frame movie
            self.stim_data = np.expand_dims(np.array(self.stim_data),
                                            axis=0)
            self.stim_data = np.array(self.stim_data)
        elif self.data_type == "npy":
            self.stim_data = np.load(self.path)
        print(f"Loaded {self.stim_data.shape} image data")

    def get_features(self, batch_size=50, n=30):
        print("Batch size:", batch_size)
        prompt = ""
        # prepare images for model
        if self.ModelHandler.model_name == 'llava':
            mod_id = self.ModelHandler.model_id
            # Follow prompt format:
            if mod_id == 'llava-hf/llava-v1.6-mistral-7b-hf':
                formatted_prompt = (
                    f"[INST] <image>\n{prompt}[/INST]"
                )
            elif mod_id == "llava-hf/llava-1.5-7b-hf":
                formatted_prompt = (
                    f"system\nUnderstand this image.\nuser\n<image>\n"
                    f"{prompt}\nassistant\n"
                )
            elif mod_id == "llava-hf/llava-v1.6-34b-hf":
                formatted_prompt = (
                    "<|im_start|>system\nUnderstand this image."
                    "<|im_end|><|im_start|>user"
                    f"\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
                )  # llava1.6-34b-hf
        else:
            formatted_prompt = prompt
        print("self.stim_data.shape[0]:", self.stim_data.shape)
        num_images = self.stim_data.shape[0]
        print("num_images:", num_images)
        # text is just blank strings for each of the items in stim_data
        text = [formatted_prompt for _ in range(self.stim_data.shape[0])]

        # Set number of batches to run through
        # (based on memory constraints vs time benefit)
        num_batches = (num_images + batch_size - 1) // batch_size

        # Make sure features is clean before starting
        self.ModelHandler.reset_features()

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.stim_data.shape[0])
            print("Batch start:", batch_start)
            print("Batch end:", batch_end)

            batch_images = self.stim_data[batch_start:batch_end]
            batch_text = text[batch_start:batch_end]

            # Make images list of np
            batch_images = [img for img in batch_images]

            model_inputs = self.ModelHandler.processor(images=batch_images,
                                                       text=batch_text,
                                                       return_tensors='pt')
            model_inputs = {key: value.to(self.ModelHandler.device) for key,
                            value in model_inputs.items()}

            # Perform model inference on the batch
            with torch.no_grad():
                _ = self.ModelHandler.model.generate(**model_inputs)
            print("Batch number ", batch_idx, ", Number of tensors so far: ",
                  len(self.ModelHandler.features['layer']))

        all_tensors = self.ModelHandler.features['layer']

        # Now features will be a dict with one key: 'layer'
        # tensors = self.ModelHandler.features['layer']
        print(f"Captured {len(all_tensors)} tensors")

        if n > 1:
            average_tensors = []
            # for every n tensor, take the average
            for i in tqdm(range(0, len(all_tensors), n)):
                try:
                    n_tensors = all_tensors[i:i+10]
                    average_tensors.append(torch.mean(torch.stack(n_tensors),
                                                      dim=0))
                except Exception as e:
                    print(f"Failed to average tensors: {e}")
                    n_tensors = all_tensors[i:i+10]

                    # size of first tensor
                    fst_size = all_tensors[0].size

                    if not all(tensor.size() == fst_size for tensor
                               in n_tensors):
                        print("tensor size mismatch")
                        # find tensor with wrong size
                        for j, tensor in enumerate(n_tensors):
                            if tensor.size() != fst_size:
                                print(f"Removing tensor: {tensor.size()}")
                                n_tensors.pop(j)
                        average_tensors.append(torch.mean(
                            torch.stack(n_tensors), dim=0))

            average_tensors_numpy = [tensor.detach().cpu().numpy() for
                                     tensor in average_tensors]
        else:
            average_tensors_numpy = [tensor.detach().cpu().numpy() for
                                     tensor in all_tensors]

        self.visualFeatures = np.array(average_tensors_numpy)

        self.ModelHandler.reset_features()
