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

    def get_features(self, batch_size=20, context=20, alignment=False):
        if alignment:
            words_with_context = [self.stim_data]
        else:
            words_with_context = []
            for i, word in enumerate(self.stim_data):
                # if one of first 20 words, just pad with all the words before
                if i < context:
                    chunk = ' '.join(self.stim_data[:(i+context)])
                # if one of last 20 words, just pad with all the words after it
                elif i > len(self.stim_data) - context:
                    chunk = ' '.join(self.stim_data[(i-context):])
                else:
                    chunk = ' '.join(self.stim_data[(i-context):(i+context)])
                words_with_context.append(chunk)

        # prepare images for model
        if self.ModelHandler.model_name == 'llava':
            # Follow prompt format:
            formatted_prompt = [
                (f"system\nUnderstand this story.\nuser\n<image>"
                 f"\n{prompt}\nassistant\n")
                for prompt in words_with_context
            ]
        else:
            formatted_prompt = words_with_context

        # Create a numpy array filled with gray values (128 in this case)
        # THis will act as tthe zero image input***
        gray_value = 128
        image_array = np.full((512, 512, 3), gray_value, dtype=np.uint8)

        images = [image_array for i in range(len(words_with_context))]

        # Set number of batches to run through
        # (based on memory constraints vs time benefit)
        num_batches = (self.stim_data.shape[0] + batch_size - 1) // batch_size

        # Make sure features is clean before starting
        self.ModelHandler.reset_features()

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.stim_data.shape[0])

            batch_images = images[batch_start:batch_end]
            batch_text = formatted_prompt[batch_start:batch_end]

            model_inputs = self.ModelHandler.processor(images=batch_images,
                                                       text=batch_text,
                                                       return_tensors='pt',
                                                       padding=True)
            model_inputs = {key: value.to(self.ModelHandler.device) for key,
                            value in model_inputs.items()}

            # Perform model inference on the batch
            with torch.no_grad():
                _ = self.ModelHandler.model.generate(**model_inputs,
                                                     max_new_tokens=50)

        all_tensors = self.ModelHandler.features['layer']

        all_tensors_numpy = [tensor.detach().cpu().numpy() for
                             tensor in all_tensors]

        self.languageFeatures = np.array(all_tensors_numpy)

        self.ModelHandler.reset_features()
