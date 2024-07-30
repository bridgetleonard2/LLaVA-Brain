import numpy as np
from classes import model_handlerCLASS
import encoding_modelsCLASS
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

# Set up directories
np.save('data/clip/train_stim/train_01.npy', image_array)
np.save('data/clip/train_fmri/train_01.npy', br_data)

# Load model
model_name = 'llava'
model_handler = model_handlerCLASS.ModelHandler(model_name)
model_handler.load_model()

data_dir = 'data/clip'

train_stim_dir = f"{data_dir}/train_stim"
train_fmri_dir = f"{data_dir}/train_fmri"

train_stim_type = "visual"

test_stim_dir = "../bridgetower-brain/data/encodingModels_pipeline/face_stim"

feat_dir = f"results/features/{model_handler.layer_name}"

encoding_model = encoding_modelsCLASS.EncodingModels(
        model_handler, train_stim_dir, train_fmri_dir,
        train_stim_type, test_stim_dir=test_stim_dir,
        test_stim_type="visual", features_dir=feat_dir
        )

encoding_model.load_fmri()
encoding_model.load_features()

encoding_model.encoding_pipeline()
