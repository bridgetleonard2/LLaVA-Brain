from classes import model_handlerCLASS
import encoding_modelsCLASS

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

test_type = "pred"
data_dir = "../bridgetower-brain/data/encodingModels_pipeline"

if test_type == "eval":
    # vision model evaluate to start
    train_stim_dir = f"{data_dir}/movie_stim"
    train_fmri_dir = f"{data_dir}/movie_fmri"

    train_stim_type = "visual"

    feat_dir = "results/features"

    encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, features_dir=feat_dir
        )

    encoding_model.load_fmri()
    encoding_model.load_features()

    encoding_model.encoding_pipeline()
elif test_type == "pred":
    # prediction model with faces
    train_stim_dir = f"{data_dir}/movie_stim"
    train_fmri_dir = f"{data_dir}/movie_fmri"

    train_stim_type = "visual"

    test_stim_dir = f"{data_dir}/face_stim"

    feat_dir = "results/features"

    encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        test_stim_type="visual", features_dir=feat_dir
        )

    encoding_model.load_fmri()
    encoding_model.load_features()

    encoding_model.encoding_pipeline()
