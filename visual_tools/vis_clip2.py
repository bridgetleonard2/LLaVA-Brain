import h5py
import numpy as np
import cortex
import os

import matplotlib.pyplot as plt

cortex.options.config.set('basic', 'filestore', 'nsd_pycortex_db')
print("Current filestore:", cortex.options.config.get('basic', 'filestore'))

beta_file_path = 'visual_tools/betas_session01.hdf5'

# Load original beta file to see shape
with h5py.File(beta_file_path, 'r') as f:
    # Inspect the shape of the '/betas' dataset
    beta_data = f['/betas'][0, :, :, :]  # Load the first trial

# transpose beta_data to match the shape of the cortical mask
beta_data = np.transpose(beta_data, (2, 1, 0))
print(f"Original beta data shape: {beta_data.shape}")

mask = np.load("visual_tools/cortical_mask_subj01.npy")
print(f"Shape of cortical mask: {mask.shape}")

cortical_beta = beta_data[mask]
print(f"Shape of cortical beta data: {cortical_beta.shape}")

pred_data = np.load("results/multi-modal_projector/subj01.npy")

brain_dims = mask.shape
prediction_3d = np.zeros(brain_dims)

prediction_3d[mask] = pred_data

print(f"Shape of prediction 3D: {prediction_3d.shape}")

# Create a volume
subject_name = "subj01"
xfm_name = "func1pt8_to_anat0pt8_autoFSbbr"

transform_dir = 'nsd_pycortex_db/subj01/transforms/func1pt8_to_anat0pt8_autoFSbbr'
assert os.path.exists(transform_dir), f"Directory does not exist: {transform_dir}"

transform_file = os.path.join(transform_dir, 'matrices.xfm')
assert os.path.isfile(transform_file), f"File does not exist: {transform_file}"

vol = cortex.Volume(prediction_3d, subject_name, xfm_name)

# Show the flatmap
cortex.webgl.show(vol)

# Create and display the flatmap
fig = cortex.quickflat.make_figure(vol, with_rois=True, with_colorbar=True)
fig.savefig('results/multi-modal_projector/subj01_flatmap.png')

plt.show()
