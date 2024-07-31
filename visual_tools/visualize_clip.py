import numpy as np
import cortex
import argparse


def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def make_volume(subj):
    mask = cortex.utils.get_cortical_mask(f"subj{subj:02d}", "func1pt8_to_anat0pt8_autoFSbbr")
    cortical_mask = np.load(f"../clip2brain/output/voxels_masks/subj{subj}/cortical_mask_subj{subj:02d}.npy")
    vals = np.load(f"results/multi-modal_projector/subj{subj:02d}.npy")

    # Projecting values back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    # Creating the volume
    vol_data = cortex.Volume(all_vals, f"subj{subj:02d}", "func1pt8_to_anat0pt8_autoFSbbr", mask=mask)
    return vol_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted activations on a flatmap.")
    parser.add_argument("--subj", type=int, required=True, help="Specify the subject number.")
    args = parser.parse_args()

    volume = make_volume(args.subj)
    cortex.webgl.show(data={"Predicted Activations": volume}, recache=False)
