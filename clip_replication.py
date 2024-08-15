import numpy as np
from classes import model_handlerCLASS
import classes.encoding_modelsCLASS as encoding_modelsCLASS
from sklearn.model_selection import train_test_split
from PIL import Image

# run get_pairs.sh first
subj = 1
project_dir = '../clip2brain'
project_output_dir = 'output'

# stimuli_dir is from config
stimuli_dir = '%s/data/NSD_images/images' % project_dir
all_coco_ids = np.load("%s/%s/coco_ID_of_repeats_subj%02d.npy" %
                       (project_dir, project_output_dir, subj))

all_images_paths = ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
print("Number of Images: {}".format(len(all_images_paths)))

brain_data_path = ('%s/%s/cortical_voxels/'
                   'averaged_cortical_responses_zscored_by_run_subj%02d.npy' %
                   (project_dir, project_output_dir, subj))
br_data = np.load(brain_data_path)

# generate an image array
image_array = np.array([np.array(Image.open(img_path).convert('RGB'))
                        for img_path in all_images_paths])
print("Image array shape: {}".format(image_array.shape))

trial_mask = np.sum(np.isnan(br_data), axis=1) <= 0
br_data = br_data[trial_mask, :]

image_array = image_array[trial_mask, :, :, :]

print("remove dead trials; brain data shape:", br_data.shape)
print("image array shape:", image_array.shape)

X_train, X_test, y_train, y_test = train_test_split(
        image_array, br_data, test_size=0.15, random_state=42)

# Set up directories
# We'll split up the data into test and train set 70/30
np.save('data/clip/train_stim/clip_85.npy',
        X_train)
np.save('data/clip/train_fmri/clip_85.npy',
        y_train)

np.save('data/clip/test_stim/clip_15.npy',
        X_test)
np.save('data/clip/test_fmri/clip_15.npy',
        y_test)

# Load model
model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

data_dir = 'data/clip'

train_stim_dir = f"{data_dir}/train_stim"
train_fmri_dir = f"{data_dir}/train_fmri"

train_stim_type = "visual"

test_dir = '../bridgetower-brain/data/encodingModels_pipeline'

test_stim_dir = f"{test_dir}/landscape_stim"
# test_fmri_dir = f"{data_dir}/test_fmri"

test_stim_type = "visual"

feat_dir = f"results/features/clip_pipeline/{model_handler.layer_name}"

encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        # test_fmri_dir=test_fmri_dir,
        test_stim_type=test_stim_type,
        features_dir=feat_dir
        )

encoding_model.load_fmri()
encoding_model.load_features(n=1)

encoding_model.encoding_pipeline(cv=7)
