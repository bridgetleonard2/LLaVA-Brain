from classes import model_handlerCLASS
import classes.visual_featuresCLASS as visual_featuresCLASS

model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

movie_path = (
    '../bridgetower-brain/data/raw_stimuli/shortclips/stimuli/train_00.hdf'
)

visual_features = visual_featuresCLASS.VisualFeatures(movie_path,
                                                      model_handler)
visual_features.load_image()
visual_features.get_features()

# Extracted features:
print(visual_features.visualFeatures[:10])
