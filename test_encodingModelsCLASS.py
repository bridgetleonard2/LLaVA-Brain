from classes import model_handlerCLASS
import encoding_modelsCLASS

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

# vision model evaluate to start
train_stim_dir = "../bridgetower-brain/data/encodingModels_pipeline/train_stim"
train_fmri_dir = "../bridgetower-brain/data/encodingModels_pipeline/train_fmri"

feature_dir = "results/features"

encoding_model = encoding_modelsCLASS.EncodingModels(
    model_handler, train_stim_dir, train_fmri_dir, feature_dir
    )

encoding_model.load_fmri()
encoding_model.load_features()

encoding_model.encoding_pipeline()