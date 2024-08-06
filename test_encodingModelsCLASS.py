from classes import model_handlerCLASS
import classes.encoding_modelsCLASS as encoding_modelsCLASS

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

test_type = "eval"
data_dir = "../bridgetower-brain/data/encodingModels_pipeline"

if test_type == "eval":
    # vision model evaluate to start
    train_stim_dir = f"{data_dir}/story_stim"
    train_fmri_dir = f"{data_dir}/story_fmri"

    train_stim_type = "language"

    feat_dir = f"results/features/{model_handler.layer_name}"

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

    test_stim_dir = f"{data_dir}/landscape_stim"

    feat_dir = f"results/features/{model_handler.layer_name}"

    encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        test_stim_type="visual", features_dir=feat_dir
        )

    encoding_model.load_fmri()
    encoding_model.load_features()

    encoding_model.encoding_pipeline()
elif test_type == "predcorr":
    train_stim_dir = f"{data_dir}/movie_stim"
    train_fmri_dir = f"{data_dir}/movie_fmri"

    train_stim_type = "visual"

    test_stim_dir = f"{data_dir}/story_stim"
    test_fmri_dir = f"{data_dir}/story_fmri"

    test_stim_type = "language"

    feat_dir = f"results/features/{model_handler.layer_name}"

    encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        test_fmri_dir=test_fmri_dir, test_stim_type="language",
        features_dir=feat_dir
        )

    encoding_model.load_fmri()
    encoding_model.load_features()

    encoding_model.encoding_pipeline()
elif test_type == "alignment":
    train_stim_dir = f"{data_dir}/movie_stim"
    train_fmri_dir = f"{data_dir}/movie_fmri"

    train_stim_type = "visual"

    test_stim_dir = f"{data_dir}/story_stim"
    test_fmri_dir = f"{data_dir}/story_fmri"

    feat_dir = f"results/features/{model_handler.layer_name}"

    encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        test_fmri_dir=test_fmri_dir, test_stim_type="language",
        features_dir=feat_dir
        )

    encoding_model.alignment()
