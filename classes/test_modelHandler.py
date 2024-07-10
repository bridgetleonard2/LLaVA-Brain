import model_handlerCLASS
from PIL import Image
import requests

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

# test feature extraction

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# llava-v1.6-34b-hf requires the following format:
# "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

inputs = model_handler.processor(prompt, image, return_tensors="pt").to('cuda')

# autoregressively complete prompt
output = model_handler.model.generate(**inputs, max_new_tokens=100)

print(model_handler.processor.decode(output[0], skip_special_tokens=True))

# Extracted features:
features = model_handler.features['layer']  # This will contain the extracted features from the specified layer

# print some of features
print(features[:10])

model_handler.hook.remove()  # Remove the hook after extracting features