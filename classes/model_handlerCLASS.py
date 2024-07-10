import torch

# models
from transformers import BridgeTowerModel, BridgeTowerProcessor
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig


class ModelHandler:
    def __init__(self, model_name):
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

    def load_model(self):
        device = torch.device('cuda')

        if self.q:
            self.model = self.model.from_pretrained(self.model_id,
                                                    quantization_config=self.q,
                                                    device_map="auto")
        else:
            self.model = self.model.from_pretrained(self.model_id,
                                                    device_map="auto")

        # select layer
        self.layer = self.model.multi_modal_projector.linear_2

        self.processor = self.processor.from_pretrained(self.model_id)

        self.features = {}

        def get_features(name):
            def hook(model, input, output):
                # detached_outputs = [tensor.detach() for tensor in output]
                last_output = output[-1].detach()
                self.features[name] = last_output  # detached_outputs
            return hook

        self.hook = self.layer.register_forward_hook(get_features('layer'))
