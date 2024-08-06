import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt


def create_flatmap(subject, layer):
    # find file to load
    # filepaths = [f"results/{layer}/pred_correlations.npy",
    #              f"results/{layer}/predictions.npy",
    #              f"results/{layer}/eval_correlations.npy"]

    # for filepath in filepaths:
    #     try:
    #         data = np.load(filepath)
    #         break
    #     except FileNotFoundError:
    #         continue
    try:
        filepath = f"results/{layer}/pred_correlations.npy"
        data = np.load(filepath)

        measure = 'predcorr'
    except FileNotFoundError:
        try:
            filepath = f"results/{layer}/predictions.npy"
            data = np.load(filepath)

            measure = 'pred'
        except FileNotFoundError:
            filepath = f"results/{layer}/eval_correlations.npy"
            data = np.load(filepath)

            measure = 'evalcorr'

    print(f"Loaded data from {filepath}")
    print(f"Data shape: {data.shape}")

    # Load mappers
    map_dir = "../BridgeTower-Brain/data/fmri_data/mappers"

    lh_mapping_matrix = load_npz(f"{map_dir}/{subject}_listening_forVL_lh.npz")
    lh_vertex_data = lh_mapping_matrix @ data
    lh_vertex_coords = np.load(f"{map_dir}/{subject}_vertex_coords_lh.npy")

    rh_mapping_matrix = load_npz(f"{map_dir}/{subject}_listening_forVL_rh.npz")
    rh_vertex_data = rh_mapping_matrix @ data
    rh_vertex_coords = np.load(f"{map_dir}/{subject}_vertex_coords_rh.npy")

    # set vmin, vmax based on measure type
    if measure == 'predcorr':
        vmin = -0.1
        vmax = 0.1
    elif (measure == 'pred') | (measure == 'evalcorr'):
        data_max = np.nanmax(abs(np.concatenate((lh_vertex_data,
                                                 rh_vertex_data))))
        vmin, vmax = -data_max, data_max

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    # Plot the first (lh) flatmap
    sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1],
                         c=lh_vertex_data, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, s=.005)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_frame_on(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot the second (rh) flatmap
    _ = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1],
                       c=rh_vertex_data, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, s=.005)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_frame_on(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(top=0.85, wspace=0)

    cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.03])
    cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')

    # Set the color bar to only display min and max values
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])

    cbar.outline.set_visible(False)

    # Set title
    if measure == 'predcorr':
        title = 'Predicted Correlation (Full train and test data)'
    elif measure == 'evalcorr':
        title = 'Evaluated Correlation (Only train data)'
    elif measure == 'pred':
        title = 'Predictions (Full train and test stimuli data)'

    plt.title(title, fontsize=12)

    plt.savefig(f"results/{layer}/{measure}.png")
    plt.show()


def face_minus_land(subject, layer):
    # Load the data
    face_data = np.load(
        r"results\multi_modal_projector.linear_2\predictions_face.npy")
    land_data = np.load(
        r"results\multi_modal_projector.linear_2\predictions_landscape.npy")

    # Subtract the landscape data from the face data
    face_minus_land = face_data - land_data

    # Reverse flattening and masking with an fmri scan
    fmri_scan = np.load('visual_tools/train_00.npy')

    mask = ~np.isnan(fmri_scan[0])

    # Initialize empty array for reconstruction
    reconstructed_data = np.full((31, 100, 100), np.nan)

    # Flatten the mask to get the indices of the non-NaN data points
    valid_indices = np.where(mask.flatten())[0]

    # Assign the data points to their original spatial positions
    for index, value in zip(valid_indices, face_minus_land):
        # convert 1D index back to 3D index
        z, x, y = np.unravel_index(index, (31, 100, 100))
        reconstructed_data[z, x, y] = value

    flattened_data = reconstructed_data.flatten()

    # Load mappers
    map_dir = "../BridgeTower-Brain/data/fmri_data/mappers"

    lh_mapping_matrix = load_npz(f"{map_dir}/{subject}_listening_forVL_lh.npz")
    lh_vertex_data = lh_mapping_matrix @ flattened_data
    lh_vertex_coords = np.load(f"{map_dir}/{subject}_vertex_coords_lh.npy")

    rh_mapping_matrix = load_npz(f"{map_dir}/{subject}_listening_forVL_rh.npz")
    rh_vertex_data = rh_mapping_matrix @ flattened_data
    rh_vertex_coords = np.load(f"{map_dir}/{subject}_vertex_coords_rh.npy")

    # set vmin, vmax based on measure type
    data_max = np.nanmax(abs(np.concatenate((lh_vertex_data,
                                             rh_vertex_data))))
    vmin, vmax = -data_max, data_max

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    # Plot the first (lh) flatmap
    sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1],
                         c=lh_vertex_data, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, s=.005)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_frame_on(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot the second (rh) flatmap
    _ = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1],
                       c=rh_vertex_data, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, s=.005)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_frame_on(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(top=0.85, wspace=0)

    cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.03])
    cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')

    # Set the color bar to only display min and max values
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])

    cbar.outline.set_visible(False)

    # Set title
    title = 'Face - Landscape Predictions'

    plt.title(title, fontsize=12)

    plt.savefig(f"results/{layer}/faceVSland.png")
    plt.show()


if __name__ == "__main__":
    subject = "S1"
    layer = "multi-modal_projector"

    create_flatmap(subject, layer)
    # face_minus_land(subject, layer)
