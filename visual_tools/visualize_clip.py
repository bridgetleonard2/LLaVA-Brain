import numpy as np
import cortex  # type: ignore
import os

import matplotlib.pyplot as plt

# Check current filestore location
print("Current filestore:", cortex.database.default_filestore)

# Check configuration file location
print("User configuration file:", cortex.options.usercfg)

# Verify that pycortex is using the correct filestore
filestore_path = cortex.database.default_filestore
assert filestore_path == 'nsd_pycortex_db', f"cortex is using {filestore_path}"

# Check if the transformation file exists
subject = 'subj01'
transform_name = 'func1pt8_to_anat0pt8_autoFSbbr'
transform_file = os.path.join('nsd_pycortex_db', subject, 'transforms',
                              transform_name, 'matrices.xfm')

# good correlation mask
corr_data = np.load(
    'results/multi-modal_projector/vision_subj01_8515.npy'
    )
brain_dims = (81, 104, 83)
corr_3d = np.zeros(brain_dims)
mask = np.load("visual_tools/cortical_mask_subj01.npy")
print("Mask shape:", mask.shape)
print("Correlation shape:", corr_3d.shape)
corr_3d[mask] = corr_data

corr_3d = np.transpose(corr_3d, (2, 1, 0))
print("Correlation shape:", corr_3d.shape)

# Make a mask where the correlation is above 0.2
corr_mask = np.where(corr_3d > 0.1, 1, 0)

# # # Create the Volume object
# prediction_data = np.load(
#     'results/multi-modal_projector/subj01_singleface.npy'
#     )

face_data = np.load('results/multi-modal_projector/subj01_singleface.npy')
landscape_data = np.load('results/multi-modal_projector/subj01_singleland.npy')

# normalize data before taking difference
face_data = (face_data - np.mean(face_data)) / np.std(face_data)
landscape_data = (landscape_data - np.mean(landscape_data)) / np.std(landscape_data)

print("Max diff", np.max(face_data - landscape_data))
print("Min diff", np.min(face_data - landscape_data))
print("Mean diff", np.mean(face_data - landscape_data))
prediction_data = face_data - landscape_data

# take absolute value of prediction data
# prediction_data = np.abs(prediction_data)

# # only take positive values above 0.05
# prediction_data = np.where(prediction_data > 0.3, prediction_data, 0)

prediction_3d = np.zeros(brain_dims)
print("Mask shape:", mask.shape)
print("Prediction shape:", prediction_3d.shape)
prediction_3d[mask] = prediction_data

prediction_3d = np.transpose(prediction_3d, (2, 1, 0))
print("Prediction shape:", prediction_3d.shape)

# filter with correlation mask
prediction_3d = prediction_3d * corr_mask

# Flatten the data to 2D (example with max projection)
flat_prediction = np.max(prediction_3d, axis=2)

vmin = -.05
vmax = .05
vol = cortex.Volume(prediction_3d, subject, transform_name, cmap="RdBu_r", #"hot",
                    vmin=vmin, vmax=vmax
                    )
# cmap="inferno")

# Create and display the flatmap
output_name = 'vision_subj01_singlefaceMinusland'
output_png = f'results/multi-modal_projector/{output_name}.png'

fig = cortex.quickflat.make_png(output_png, vol, with_colorbar=True,
                                bgcolor='white')

plt.show()
