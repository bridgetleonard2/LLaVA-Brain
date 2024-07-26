import torch   # type: ignore
# import numpy as np

# models
from transformers import BridgeTowerModel, BridgeTowerProcessor  # type: ignore
from transformers import AutoProcessor  # type: ignore
from transformers import LlavaForConditionalGeneration  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore


class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'llava':
            self.model = LlavaForConditionalGeneration
            self.model_id = "llava-hf/llava-1.5-7b-hf"
            self.processor = AutoProcessor
            self.q = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16)
        elif model_name == 'bridgetower':
            self.model = BridgeTowerModel
            self.model_id = "BridgeTower/bridgetower-base"
            self.processor = BridgeTowerProcessor

    def select_layer(self):
        if self.model_name == 'llava':
            model_layers = [
                (self.model.vision_tower.vision_model.encoder.layers[0]
                 .self_attn.k_proj),
                self.model.multi_modal_projector.linear_2,
                self.model.language_model.model.layers[0].self_attn.q_proj
            ]

            model_options = [
                "Option 1: vision tower",
                "Option 2: multi-modal projector",
                "Option 3: language model"]

        elif self.model_name == 'bridgetower':
            model_layers = [
                self.model.cross_modal_text_transform,
                self.model.cross_modal_image_transform,
                self.model.vision_model.visual.embeddings.patch_embedding,
                self.model.text_model.embeddings.word_embeddings,
                self.model.cross_modal_image_pooler,
                self.model.cross_modal_text_pooler,
                self.model.cross_modal_text_link_tower[-1],
                self.model.cross_modal_image_link_tower[-1]]

            model_options = [
                "Option 1: cross modal text transform",
                "Option 2: cross modal image transform",
                "Option 3: vision model",
                "Option 4: text model",
                "Option 6: cross modal image pooler",
                "Option 7: cross modal text pooler",
                "Option 12: cross modal text link tower",
                "Option 13: cross modal image link tower",
            ]

        print("Select the layer you want features from:")
        for option in model_options:
            print(option)

        selected_option = int(
            input("Enter the number of the layer you want: "))
        self.layer = model_layers[selected_option - 1]

        layer_name = model_options[selected_option-1].split(":")[1]
        # Remove space before characters and then replace spaces with _
        layer_name = layer_name.lstrip().replace(" ", "_")
        self.layer_name = layer_name

        print(f"Selected layer: {model_options[selected_option-1]}")

    def load_model(self):
        self.device = torch.device('cuda')

        if self.q:
            self.model = self.model.from_pretrained(self.model_id,
                                                    quantization_config=self.q,
                                                    device_map="auto")
        else:
            self.model = self.model.from_pretrained(self.model_id,
                                                    device_map="auto")

        # select layer
        self.select_layer()

        self.processor = self.processor.from_pretrained(self.model_id)

        # dictionary of list for layer of interest
        # dict in case multiple layers of interest
        self.features = {'layer': []}

        def get_features(name):
            def hook(model, input, output):
                # print("Batch output:", np.array(output.detach().cpu()).shape)
                output = output.detach().cpu()
                self.features[name].extend(output)  # detached_outputs
            return hook

        self.hook = self.layer.register_forward_hook(get_features('layer'))

    def reset_features(self):
        self.features = {'layer': []}
