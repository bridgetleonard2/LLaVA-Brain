import numpy as np
import torch  # type: ignore
import re

# Progress bar
from tqdm import tqdm


def textgrid_to_array(textgrid):
    """Function to load transcript from textgrid into a list.

    Parameters
    ----------
    textgrid: string
        TextGrid file name.

    Returns
    -------
    full_transcript : Array
        Array with each word in the story.
    """
    if textgrid == 'data/raw_stimuli/textgrids/stimuli/legacy.TextGrid':
        with open(textgrid, 'r')as file:
            data = file.readlines()

        full_transcript = []
        # Important info starts at line 5
        for line in data[5:]:
            if line.startswith('2'):
                index = data.index(line)
                word = re.search(r'"([^"]*)"', data[index+1].strip()).group(1)
                full_transcript.append(word)
    elif textgrid == 'data/raw_stimuli/textgrids/stimuli/life.TextGrid':
        with open(textgrid, 'r') as file:
            data = file.readlines()

        full_transcript = []
        for line in data:
            if "word" in line:
                index = data.index(line)
                words = data[index+6:]  # this is where first word starts

        for i, word in enumerate(words):
            if i % 3 == 0:
                word = re.search(r'"([^"]*)"', word.strip()).group(1)
                full_transcript.append(word)
    else:
        with open(textgrid, 'r') as file:
            data = file.readlines()

        # Important info starts at line 8
        for line in data[8:]:
            # We only want item [2] info because those are the words instead
            # of phonemes
            if "item [2]" in line:
                index = data.index(line)

        summary_info = [line.strip() for line in data[index+1:index+6]]
        print(summary_info)

        word_script = data[index+6:]
        full_transcript = []
        for line in word_script:
            if "intervals" in line:
                # keep track of which interval we're on
                ind = word_script.index(line)
                word = re.search(r'"([^"]*)"',
                                 word_script[ind+3].strip()).group(1)
                full_transcript.append(word)

    return np.array(full_transcript)


class LanguageFeatures:
    def __init__(self, path, ModelHandler):
        self.path = path
        self.data_type = path.split('.')[-1]
        self.ModelHandler = ModelHandler

    def load_text(self):
        if self.data_type == "TextGrid":
            self.stim_data = textgrid_to_array(self.path)
            # if self.ModelHandler.model_name == 'llava':
            #     # convert list to np.array
            #     self.stim_data = np.array(self.stim_data)

    def get_features(self, batch_size=50, n=30):
        prompt = ""
        # prepare images for model
        if self.ModelHandler.model_name == 'llava':
            # Follow prompt format:
            formatted_prompt = (
                f"system\nUnderstand this story.\nuser\n<image>\n"
                f"{prompt}\nassistant\n"
            )
        else:
            formatted_prompt = prompt
        # text is just blank strings for each of the items in stim_data
        text = [formatted_prompt for i in range(self.stim_data.shape[0])]

        # Set number of batches to run through
        # (based on memory constraints vs time benefit)
        num_batches = (self.stim_data.shape[0] + batch_size - 1) // batch_size

        # Make sure features is clean before starting
        self.ModelHandler.reset_features()

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.stim_data.shape[0])

            batch_images = self.stim_data[batch_start:batch_end]
            batch_text = text[batch_start:batch_end]

            model_inputs = self.ModelHandler.processor(images=batch_images,
                                                       text=batch_text,
                                                       return_tensors='pt')
            model_inputs = {key: value.to(self.ModelHandler.device) for key,
                            value in model_inputs.items()}

            # Perform model inference on the batch
            with torch.no_grad():
                _ = self.ModelHandler.model.generate(**model_inputs)

        all_tensors = self.ModelHandler.features['layer']

        # Now features will be a dict with one key: 'layer'
        # tensors = self.ModelHandler.features['layer']
        print(f"Captured {len(all_tensors)} tensors")

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

                if not all(tensor.size() == fst_size for tensor in n_tensors):
                    print("tensor size mismatch")
                    # find tensor with wrong size
                    for j, tensor in enumerate(n_tensors):
                        if tensor.size() != fst_size:
                            print(f"Removing tensor: {tensor.size()} from avg")
                            n_tensors.pop(j)
                    average_tensors.append(torch.mean(torch.stack(n_tensors),
                                                      dim=0))

        average_tensors_numpy = [tensor.detach().cpu().numpy() for
                                 tensor in average_tensors]

        self.visualFeatures = np.array(average_tensors_numpy)

        self.ModelHandler.reset_features()
