# confirm that feature extraction works with llava
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests

model_id = 'llava-hf/llava-v1.6-34b-hf'
processor = LlavaNextProcessor.from_pretrained(model_id)

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

features = {}


def get_features(name):
    def hook(model, input, output):
        # detached_outputs = [tensor.detach() for tensor in output]
        last_output = output[-1].detach()
        features[name] = last_output  # detached_outputs
    return hook


layer = model.multi_modal_projector.linear_2.register_forward_hook(get_features('mm_proj_L2'))

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# llava-v1.6-34b-hf requires the following format:
# "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

inputs = processor(prompt, image, return_tensors="pt").to('cuda')

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))

# Extracted features:
mm_feat = features['mm_proj_L2']  # This will contain the extracted features from the specified layer

# print some of mm_feat
print(mm_feat[:10])

layer.remove()  # Remove the hook after extracting features


# Works -- Output:
# What is shown in this image?<|im_start|> assistant
# The image displays a radar chart, also known as a spider chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.

# In this particular chart, there are several datasets represented by different colors and labels, such as "MM-Vet," "LLaVa-Bench," "Seed-Bench," "MMBench-CN," "
# tensor([[ 0.0598, -0.0310, -0.2712,  ..., -0.0336, -0.3164,  0.1720],
#         [-0.1852,  0.1825, -0.3052,  ..., -0.0571, -0.3198,  0.1362],
#         [ 0.3123, -0.2148, -0.1020,  ..., -0.1548, -0.0385, -0.0592],
#         ...,
#         [ 0.2113, -0.2042, -0.1179,  ...,  0.0053, -0.2761, -0.0986],
#         [-0.3074,  0.0357, -0.1349,  ..., -0.2786, -0.3005,  0.1462],
#         [-0.5898,  0.4802, -0.5269,  ...,  0.3064,  0.2454,  0.1152]],
#        device='cuda:0', dtype=torch.float16)