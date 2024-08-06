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

# Create the Volume object
prediction_data = np.load('results/vision_tower/subj01.npy')

# take absolute value of prediction data
prediction_data = np.abs(prediction_data)

# keep only results over 0.05
prediction_data[prediction_data < 0.05] = 0

brain_dims = (83, 104, 81)
prediction_3d = np.zeros(brain_dims)
mask = np.load("visual_tools/cortical_mask_subj01.npy")
mask = np.transpose(mask, (2, 1, 0))
print("Mask shape:", mask.shape)
print("Prediction shape:", prediction_3d.shape)
prediction_3d[mask] = prediction_data

# Flatten the data to 2D (example with max projection)
flat_prediction = np.max(prediction_3d, axis=2)

vol = cortex.Volume(prediction_3d, subject, transform_name, cmap="inferno")

# Create and display the flatmap
output_png = 'results/vision_tower/subj01_flatmap.png'
fig = cortex.quickflat.make_png(output_png, vol, with_colorbar=True,
                                bgcolor='white')

plt.show()
