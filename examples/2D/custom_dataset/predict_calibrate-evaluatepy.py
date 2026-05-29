"""
Combined MicroSplit prediction + calibration / evaluation workflow.

This file combines:
1. Step 2: Using a trained MicroSplit model for prediction
2. Step 3: Calibration and error estimation

"""

import os
import platform
from pathlib import Path
import wandb
import numpy as np
import tifffile as tiff
import torch
from pathlib import Path
import gc
import matplotlib.pyplot as plt

import pooch
from careamics.lightning import VAEModule
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import get_target
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import full_frame_evaluation
from microsplit_reproducibility.notebook_utils.HT_LIF24 import show_sampling

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.utils.utils import plot_input_patches
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file

# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import (
    get_microsplit_parameters,
)
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data

# Path to your own data

DATA_PATH = Path("/Users/sdasgupt/Documents/microsplit/jump-qc/Yokogawa_images/good_images_test/gaussian_blurred/data")

# Setup the path to the noise models
NM_PATH = Path("./noise_models/")

# Load the image data to be processed

root = Path(DATA_PATH)

bad = []
for p in root.rglob("*"):
    if p.is_file():
        try:
            tiff.imread(p)
        except Exception as e:
            bad.append((str(p), repr(e)))

bad[:20], len(bad)

# Print results from above step and insert a cleanup step for .DS_Store files

# running the next code blocks instead of this one because dataset comprises only two images
# setting up train, validation, and test data configs

# this section will need to be modified because this will error if dataset contains two images
# previously while running in the notebook, this cell was not run if the dataset was small
# for now, commenting out this section
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file


def is_valid_tiff_path(p: Path) -> bool:
    name = p.name
    if name.startswith(".") or name.startswith("._"):   # hidden + AppleDouble
        return False
    if p.suffix.lower() not in (".tif", ".tiff"):
        return False
    if p.stat().st_size == 0:  # empty file
        return False
    return True

def load_data(datadir):
    data_path = Path(datadir)
    channel_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())

    channels_data = []
    for channel_dir in channel_dirs:
        image_files = sorted(
            f for f in channel_dir.iterdir()
            if f.is_file() and is_valid_tiff_path(f)
        )
        channel_images = [load_one_file(image_path) for image_path in image_files]

train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(64, 64), num_channels=2
)

# setting up MicroSplit parametrization
experiment_params = get_microsplit_parameters(
    algorithm = "musplit",
    img_size=(64, 64),
    batch_size=8, # use the same configs as in training
    num_epochs=20,
    multiscale_count=3,
    noise_model_path=NM_PATH,
    target_channels=2,
)

# create the dataset
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA_PATH,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=test_data_config,
    load_data_func=get_train_val_data,
)


# # Run this code block if dataset comprises two images

# def is_valid_tiff_path(p: Path) -> bool:
#     name = p.name
#     if name.startswith(".") or name.startswith("._"):
#         return False
#     if p.suffix.lower() not in (".tif", ".tiff"):
#         return False
#     if p.stat().st_size == 0:
#         return False
#     return True

# def _as_NYXC_1(img: np.ndarray) -> np.ndarray:
#     """
#     Convert whatever load_one_file returns into (N, Y, X, 1).
#     """
#     img = np.asarray(img)

#     if img.ndim == 2:          # (Y, X)
#         img = img[None, ..., None]   # (1, Y, X, 1)
#     elif img.ndim == 3:
#         # could be (N, Y, X) or (Y, X, C)
#         # Heuristic: if last dim is small (<=4), treat as channels; otherwise treat as N
#         if img.shape[-1] <= 4:
#             img = img[None, ...]          # (1, Y, X, C)
#             if img.shape[-1] != 1:
#                 # keep only first channel if multi-channel file (rare here)
#                 img = img[..., :1]
#         else:
#             img = img[..., None]          # (N, Y, X, 1)
#     elif img.ndim == 4:
#         # assume already (N, Y, X, C); keep first channel if needed
#         if img.shape[-1] != 1:
#             img = img[..., :1]
#     else:
#         raise ValueError(f"Unexpected image shape from load_one_file: {img.shape}")

#     return img

# def load_data_two_images(datadir, num_channels=2, n_images=2) -> np.ndarray:
#     """
#     Returns (n_images, Y, X, num_channels).
#     Loads only first `num_channels` channel directories under datadir.
#     """
#     data_path = Path(datadir)
#     channel_dirs_all = sorted([p for p in data_path.iterdir() if p.is_dir()])
#     channel_dirs = channel_dirs_all[:num_channels]  # <-- enforce channel count

#     if len(channel_dirs) < num_channels:
#         raise ValueError(f"Found only {len(channel_dirs)} channel dirs, expected {num_channels}.")

#     per_channel = []
#     for channel_dir in channel_dirs:
#         image_files = sorted(
#             f for f in channel_dir.iterdir()
#             if f.is_file() and is_valid_tiff_path(f)
#         )[:n_images]

#         if len(image_files) < n_images:
#             raise ValueError(f"{channel_dir} has only {len(image_files)} images; expected {n_images}.")

#         imgs = [_as_NYXC_1(load_one_file(f)) for f in image_files]  # each (1,Y,X,1)
#         channel_stack = np.concatenate(imgs, axis=0)               # (N,Y,X,1)
#         per_channel.append(channel_stack)

#     data = np.concatenate(per_channel, axis=-1)  # (N,Y,X,C)
#     return data[:n_images, ...]                  # (2,Y,X,C)


# def _split_name(datasplit_type) -> str:
#     name = getattr(datasplit_type, "name", str(datasplit_type))
#     return name.split(".")[-1].lower()

# def get_train_val_data_two_images(data_config, datadir, datasplit_type, **kwargs):
#     split = _split_name(datasplit_type)

#     data = load_data_two_images(
#         datadir,
#         num_channels=data_config.num_channels,
#         n_images=2
#     )  # (2, Y, X, C)

#     if split == "train":
#         return data[:1]        # image 0
#     elif split in ("val", "valid", "validation"):
#         return data[1:2]       # image 1
#     elif split == "test":
#         return data[1:2]       # <-- IMPORTANT: must not be empty
#         # or use: return data[:1]  # if you'd rather duplicate train
#     else:
#         raise ValueError(f"Unknown datasplit_type: {datasplit_type} (parsed as '{split}')")


# tmp = load_data_two_images(DATA_PATH, num_channels=2, n_images=2)
# print("Loaded shape:", tmp.shape)

# configs
train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(64, 64), num_channels=2
)

# params
experiment_params = get_microsplit_parameters(
    algorithm="musplit",
    img_size=(64, 64),
    batch_size=8,
    num_epochs=20,
    multiscale_count=3,
    noise_model_path=NM_PATH,
    target_channels=2,
)

# create datasets using custom 2-image loader
# train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
#     datapath=DATA_PATH,
#     train_config=train_data_config,
#     val_config=val_data_config,
#     test_config=test_data_config,
#     load_data_func=get_train_val_data_two_images,
# )

# Configure `num_workers`
# In Windows and MacOS, setting `num_workers > 0` for dataloaders would cause out-of-memory issue and might crash the system.

def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3  # or any other number suitable for your system

experiment_params["num_workers"] = get_num_workers()

# Prediction splits to run automatically
# This replaces the old manual toggle:
#     evaluate_on_validation_data = True/False
#
# The script will now run predictions first on Test, then on Val, so it creates
# all files required by the calibration section in a single execution.
prediction_splits = [
    ("Test", test_dset),
    ("Val", val_dset),
]

# Visualizations are always generated and saved by this script.
# Figures are written to the predict_evaluate/ folder.

# Step 2.2: Picking microsplit Model to Use

# # Recursively search for .ckpt files in 'checkpoints' folder
# ckpt_folder = Path("./checkpoints")
# ckpt_folders = set()
# for file in ckpt_folder.rglob("*.ckpt"):
#     ckpt_folders.add(file.parent)
# ckpt_folders = sorted(ckpt_folders)


# def list_available_model_checkpoint_folders():
#     print("These models you have trained have been found:")
#     if len(ckpt_folders) == 0:
#         print(" ❌ None!")
#     else:
#         for file in ckpt_folders:
#             print(" 🟢", file)

#
# list_available_model_checkpoint_folders()

# Run this to select the most recent best epoch .ckpt file

# best_epoch_ckpts = [
#     p for p in ckpt_folder.glob("*.ckpt")
#     if p.name.lower().startswith("best-epoch")
# ]

# if best_epoch_ckpts:
#     selected_ckpt = str(max(best_epoch_ckpts, key=lambda p: p.stat().st_mtime))
# else:
#     # Fallback (e.g., your helper that finds the best checkpoint)
#     selected_ckpt = load_checkpoint_path(str(ckpt_folder), best=True)

# print("✅ Selected model checkpoint:", selected_ckpt)


# :alert: Use this if you want to use a specific checkpoint and not the most recent one
selected_ckpt = "/Users/sdasgupt/Documents/microsplit/microsplit/MicroSplit-reproducibility/examples/2D/custom_dataset/checkpoints/best-epoch=12.ckpt"
print("✅ Selected model checkpoint:", selected_ckpt)

# Step 2.3 Prepare microsplit Model
# making our data_stats known to the experiment (model) we prepare
experiment_params["data_stats"] = data_stats

# setting up model config (using default parameters)
model_config = get_model_config(**experiment_params)

# NOTE: The creation of the following configs are not strictly necessary for prediction,
#     but they ARE currently expected by the create_algorithm_config function below.
#     They act as a placeholder for now and we will work to remove them in a following release
loss_config = get_loss_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)

# finally, assemble the full set of experiment configurations
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
)

# Create model and load checkpoint

model = VAEModule(algorithm_config=experiment_config)

from microsplit_reproducibility.notebook_utils.custom_dataset_2D import load_pretrained_model

load_pretrained_model(model, selected_ckpt)

reduce_data = False

# Step 2.4: Predictions on Uncropped Data

from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions,
    get_target,
    get_input,
)

# Note, this parameter is responsible for how many samples are generated for each patch.
# The default value is 5, but in this case you might see stitching artifacts because
# each patch will be slightly different. You can increase this value to 10 to get a
# smoother image.
experiment_params["mmse_count"] = 10

# Running this code to avoid float64 related error
from torch.utils.data import Dataset

class Float32Wrapper(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        inp, tar = self.dset[idx]

        # Convert numpy -> torch float32 (or cast torch -> float32)
        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)
        if isinstance(tar, np.ndarray):
            tar = torch.from_numpy(tar)

        inp = inp.to(dtype=torch.float32)
        tar = tar.to(dtype=torch.float32)

        return inp, tar

    def set_img_sz(self, tile_size, grid_size):
        return self.dset.set_img_sz(tile_size, grid_size)

    def __getattr__(self, name):
        return getattr(self.dset, name)


# Step 2.6: Saving data required for network calibration and error estimations
# ie. for running the calibration section below.
dataset_prefix = "gaussian_blurred_images_"  # TODO: fill in with your own dataset prefix

# Create folder to store all calibration files.
# The script will create these 6 files automatically:
#   prediction_*Val.tif, std_*Val.tif, target_*Val.tif
#   prediction_*Test.tif, std_*Test.tif, target_*Test.tif
path_for_calibration_data = "calibration_data"
os.makedirs(path_for_calibration_data, exist_ok=True)

# Output folder for qualitative 4-channel OME-TIFFs:
# channel order = GT0, Pred0, GT1, Pred1
out_dir = Path("predictions_out/per_set_4ch_gt_pred/bad_images/user_annotated_blurry_images")
out_dir.mkdir(parents=True, exist_ok=True)

# Output folder for saved prediction/evaluation/calibration figures.
predict_evaluate_dir = Path("predict_evaluate")
predict_evaluate_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_path, dpi=200):
    """Save a matplotlib figure and close it to avoid accumulating open figures."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved figure: {output_path}")


def save_new_figures_from_function(function_call, output_dir, filename_prefix, dpi=200):
    """
    Run a plotting function that may create one or more figures, then save any new figures.

    This is useful for helper functions such as plot_input_patches() and
    full_frame_evaluation(), where the function creates the figure internally.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(plt.get_fignums())
    function_call()
    after = set(plt.get_fignums())
    new_fignums = sorted(after - before)

    if not new_fignums:
        # Fallback: save the current figure if the helper reused an existing figure.
        current_fig = plt.gcf()
        if current_fig is not None and current_fig.number not in before:
            new_fignums = [current_fig.number]

    for idx, fignum in enumerate(new_fignums):
        fig = plt.figure(fignum)
        suffix = f"_{idx + 1:02d}" if len(new_fignums) > 1 else ""
        save_figure(fig, output_dir / f"{filename_prefix}{suffix}.png", dpi=dpi)


def clear_torch_memory():
    """Free memory between Test and Val predictions."""
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_frame_stems(dset, n_frames):
    """Get readable frame names if the dataset exposes paths; otherwise use set_000, set_001, ..."""
    for attr in ["frame_paths", "_frame_paths", "paths", "_paths", "files", "_files"]:
        if hasattr(dset, attr):
            maybe = getattr(dset, attr)
            if isinstance(maybe, (list, tuple)) and len(maybe) == n_frames:
                return [Path(p).stem for p in maybe]
    return [f"set_{i:03d}" for i in range(n_frames)]


def save_normalized_target_for_calibration(split_label, dset):
    """Save target data normalized in the same way expected by the calibration section."""
    data_stats = experiment_params["data_stats"]

    target = np.asarray(dset._data[...])
    sep_mean = np.transpose(data_stats[0].numpy(), axes=(0, 2, 3, 1))
    sep_std = np.transpose(data_stats[1].numpy(), axes=(0, 2, 3, 1))

    target_normalized = (target - sep_mean) / sep_std

    target_filename = f"target_{dataset_prefix}{split_label}"
    tiff.imwrite(
        f"{path_for_calibration_data}/{target_filename}.tif",
        target_normalized,
    )
    print(
        f'✅ Saved target data for {split_label} data at '
        f'"{path_for_calibration_data}/{target_filename}.tif"!'
    )

    return target_normalized


def save_qualitative_ome_tiffs(split_label, dset, stitched_predictions):
    """
    Save 4-channel OME-TIFFs for visual inspection.

    Channel order:
        GT0, Pred0, GT1, Pred1
    """
    pred = np.asarray(stitched_predictions).astype(np.float32)  # (T, Y, X, 2)
    assert pred.ndim == 4 and pred.shape[-1] == 2, pred.shape
    T, Y, X, _ = pred.shape

    gt = np.asarray(get_target(dset)).astype(np.float32)        # (T, Y, X, 2)
    assert (
        gt.ndim == 4
        and gt.shape[0] == T
        and gt.shape[1] == Y
        and gt.shape[2] == X
        and gt.shape[-1] == 2
    ), gt.shape

    names = get_frame_stems(dset, T)

    for i in range(T):
        stem = names[i]

        # Interleave channels: GT0, Pred0, GT1, Pred1 -> (Y, X, 4)
        yxc4 = np.stack(
            [gt[i, ..., 0], pred[i, ..., 0], gt[i, ..., 1], pred[i, ..., 1]],
            axis=-1,
        ).astype(np.float32)

        cyx4 = np.moveaxis(yxc4, -1, 0)  # (C, Y, X)

        out_path = out_dir / f"{stem}_gt_pred_{split_label.lower()}.ome.tif"
        tiff.imwrite(
            out_path,
            cyx4,
            ome=True,
            metadata={"axes": "CYX"},
        )

    print(f'✅ Saved 4-channel GT/prediction OME-TIFFs for {split_label} data in "{out_dir}".')


def make_and_save_visualizations(split_label, dset, stitched_predictions):
    """
    Create and save notebook-style prediction/evaluation visualizations for one split.

    Saved outputs go into:
        predict_evaluate/<split_label>/
    """
    print(f"Creating and saving visualizations for {split_label} data...")

    split_fig_dir = predict_evaluate_dir / split_label.lower()
    split_fig_dir.mkdir(parents=True, exist_ok=True)

    inp = get_input(dset).sum(-1)
    tar = get_target(dset)

    frame_idx = 0
    assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds"

    save_new_figures_from_function(
        lambda: full_frame_evaluation(
            stitched_predictions[frame_idx],
            tar[frame_idx],
            inp[frame_idx],
        ),
        split_fig_dir,
        f"{split_label.lower()}_full_frame_evaluation",
    )

    from microsplit_reproducibility.utils.utils import clean_ax
    from microsplit_reproducibility.notebook_utils.HT_LIF24 import (
        pick_random_patches_with_content,
        pick_random_inputs_with_content,
    )

    # Detailed view on random foreground locations
    img_sz = 128
    rand_locations = pick_random_patches_with_content(tar, img_sz)
    if len(rand_locations) > 0:
        ncols = 5
        nrows = min(len(rand_locations), 5)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

        # Make sure ax is always 2D, even if nrows == 1
        ax = np.asarray(ax)
        if ax.ndim == 1:
            ax = ax[None, :]

        for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
            ax[i, 0].imshow(inp[0, h_start : h_start + img_sz, w_start : w_start + img_sz])
            for j in range(ncols // 2):
                vmin = stitched_predictions[..., j].min()
                vmax = stitched_predictions[..., j].max()
                ax[i, 2 * j + 1].imshow(
                    tar[0, h_start : h_start + img_sz, w_start : w_start + img_sz, j],
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[i, 2 * j + 2].imshow(
                    stitched_predictions[
                        0, h_start : h_start + img_sz, w_start : w_start + img_sz, j
                    ],
                    vmin=vmin,
                    vmax=vmax,
                )

        ax[0, 0].set_title("Primary Input")
        for i in range(2):  # 2-channel splitting
            ax[0, 2 * i + 1].set_title(f"Target Channel {i+1}")
            ax[0, 2 * i + 2].set_title(f"Predicted Channel {i+1}")

        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        clean_ax(ax)
        save_figure(fig, split_fig_dir / f"{split_label.lower()}_random_foreground_patches.png")

    # Manual crop inspection
    y_start = 750
    x_start = 750
    crop_size = 512

    ncols = 3
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax[0, 0].imshow(inp[0, y_start : y_start + crop_size, x_start : x_start + crop_size])
    for i in range(ncols - 1):
        vmin = stitched_predictions[..., i].min()
        vmax = stitched_predictions[..., i].max()
        ax[0, i + 1].imshow(
            tar[0, y_start : y_start + crop_size, x_start : x_start + crop_size, i],
            vmin=vmin,
            vmax=vmax,
        )
        ax[1, i + 1].imshow(
            stitched_predictions[
                0, y_start : y_start + crop_size, x_start : x_start + crop_size, i
            ],
            vmin=vmin,
            vmax=vmax,
        )

    ax[1, 0].axis("off")
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Channel 1")
    ax[0, 2].set_title("Channel 2")
    ax[0, 2].yaxis.set_label_position("right")
    ax[0, 2].set_ylabel("Target")
    ax[1, 2].yaxis.set_label_position("right")
    ax[1, 2].set_ylabel("Predicted")
    save_figure(fig, split_fig_dir / f"{split_label.lower()}_manual_crop_prediction_vs_target.png")

    # Posterior sampling inspection
    imgsz = 3
    ncols = 6
    examplecount = 3
    fig, ax = plt.subplots(
        figsize=(imgsz * ncols, 2 * imgsz * examplecount),
        ncols=ncols,
        nrows=2 * examplecount,
    )

    show_sampling(dset, model, ax=ax[:2])
    show_sampling(dset, model, ax=ax[2:4])
    show_sampling(dset, model, ax=ax[4:6])
    save_figure(fig, split_fig_dir / f"{split_label.lower()}_posterior_sampling_overview.png")

    n_samples = 50
    idx_list = pick_random_inputs_with_content(dset)
    if len(idx_list) == 0:
        print("No content-rich patches found for posterior sampling visualization.")
        return

    idx = idx_list[0]
    inp_patch, tar_patch = dset[idx]

    samples = []
    model.eval()

    for _ in range(n_samples):
        with torch.no_grad():
            pred_patch, _ = model(torch.Tensor(inp_patch).unsqueeze(0).to(model.device))
            samples.append(pred_patch[0, : tar_patch.shape[0]].cpu().numpy())
    samples = np.array(samples)

    nrows = 5
    imgsz = 3
    fig, ax = plt.subplots(figsize=(imgsz * 2, imgsz * nrows + imgsz), ncols=2, nrows=nrows)
    ax[0, 0].imshow(inp_patch[0])
    ax[0, 0].set_title(f"Input (Idx: {idx})")

    ax[1, 0].imshow(samples[0, 0])
    ax[1, 0].set_title("C1: Sample")
    ax[2, 0].imshow(np.mean(samples[:5, 0], axis=0))
    ax[2, 0].set_title("C1: MMSE (5)")
    ax[3, 0].imshow(np.mean(samples[:, 0], axis=0))
    ax[3, 0].set_title(f"C1: MMSE ({len(samples)})")
    ax[4, 0].imshow(tar_patch[0])
    ax[4, 0].set_title("C1: Target")

    ax[1, 1].imshow(samples[0, 1])
    ax[1, 1].set_title("C2: Sample")
    ax[2, 1].imshow(np.mean(samples[:5, 1], axis=0))
    ax[2, 1].set_title("C2: MMSE (5)")
    ax[3, 1].imshow(np.mean(samples[:, 1], axis=0))
    ax[3, 1].set_title(f"C2: MMSE ({len(samples)})")
    ax[4, 1].imshow(tar_patch[1])
    ax[4, 1].set_title("C2: Target")

    ax[0, 1].axis("off")
    save_figure(fig, split_fig_dir / f"{split_label.lower()}_posterior_samples_mmse_target.png")


def run_prediction_for_split(split_label, dset):
    """
    Run tiled MicroSplit prediction for one split and save:
      1. normalized target for calibration,
      2. normalized MMSE prediction for calibration,
      3. posterior-sample standard deviation for calibration,
      4. qualitative 4-channel GT/prediction OME-TIFFs.

    Parameters
    ----------
    split_label : str
        Either "Test" or "Val". This string is used in the calibration filenames.
    dset : Dataset
        The corresponding MicroSplit dataset object.
    """
    print(f"\n==============================")
    print(f"Running prediction for {split_label} data")
    print(f"==============================")

    if reduce_data:
        print("Using REDUCED evaluation data for quick'n'dirty testing!")
        dset.reduce_data([0])
    else:
        print("Using the full set of evaluation data!")
        print(f"(More specifically, I will use {dset.get_num_frames()} frames for evaluations.)")

    print(f"Will use {split_label} data containing {dset.get_num_frames()} frames.")

    split_fig_dir = predict_evaluate_dir / split_label.lower()
    save_new_figures_from_function(
        lambda: plot_input_patches(
            dataset=dset,
            num_channels=2,
            num_samples=3,
            patch_size=128,
        ),
        split_fig_dir,
        f"{split_label.lower()}_input_patches",
    )

    # Save target data for this split before calibration.
    save_normalized_target_for_calibration(split_label, dset)

    dset32 = Float32Wrapper(dset)

    stitched_predictions, norm_stitched_predictions, stitched_stds = get_unnormalized_predictions(
        model,
        dset32,
        data_key=dset32._fpath.name,
        mmse_count=experiment_params["mmse_count"],
        grid_size=32,
        num_workers=get_num_workers(),
        batch_size=8,
    )

    assert (
        norm_stitched_predictions.shape == stitched_stds.shape
    ), "MMSE predictions and pixel-wise stds have incompatible shapes."

    target = np.asarray(dset._data[...])
    assert (
        stitched_stds.shape == target.shape
    ), (
        f"Shape of predictions does not fit the loaded {split_label} target data. "
        f"stitched_stds={stitched_stds.shape}, target={target.shape}"
    )

    pred_filename = f"prediction_{dataset_prefix}{split_label}"
    std_filename = f"std_{dataset_prefix}{split_label}"

    tiff.imwrite(
        f"{path_for_calibration_data}/{pred_filename}.tif",
        norm_stitched_predictions,
    )
    print(
        f'✅ Saved MMSE predictions for {split_label} data at '
        f'"{path_for_calibration_data}/{pred_filename}.tif"!'
    )

    tiff.imwrite(
        f"{path_for_calibration_data}/{std_filename}.tif",
        stitched_stds,
    )
    print(
        f'✅ Saved posterior sample Stds for {split_label} data at '
        f'"{path_for_calibration_data}/{std_filename}.tif"!'
    )

    save_qualitative_ome_tiffs(split_label, dset, stitched_predictions)
    make_and_save_visualizations(split_label, dset, stitched_predictions)

    result = {
        "split_label": split_label,
        "prediction_file": f"{path_for_calibration_data}/{pred_filename}.tif",
        "std_file": f"{path_for_calibration_data}/{std_filename}.tif",
        "target_file": f"{path_for_calibration_data}/target_{dataset_prefix}{split_label}.tif",
    }

    # Free memory before running the next split.
    del dset32, stitched_predictions, norm_stitched_predictions, stitched_stds, target
    clear_torch_memory()

    return result


# Run Test first, then Val, without manual changes or a second script execution.
prediction_results = {}
for split_label, split_dset in prediction_splits:
    prediction_results[split_label] = run_prediction_for_split(split_label, split_dset)

print(
    f'\n✅ Finished generating all calibration files. '
    f'The folder "{path_for_calibration_data}" should now contain prediction, std, and target files for both Test and Val data. '
    f'Saved prediction/evaluation figures will be written to "{predict_evaluate_dir}".'
)

# ========================================================================================
# END OF STEP 2 / START OF STEP 3
# ========================================================================================

# **Step 3:** Calibration and Error Estimation


# This code:
# 1. uses saved MMSE predictions, pixel-wise posterior sample standard deviations, and target images we did not use during training to calibrate an error estimator, and then
# 2. generates calibration plots and learn how to interpret them, and
# 3. computes uncertainty maps that can point users at predicted images or parts of predicted images that likely contain unreliable predictions, to finally
# 4. checks how the true error (wrt. to potentially noisy target data) compares to the pixel-wise variability between posterior samples.

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pooch

from careamics.lvae_training.calibration import (
    Calibration,
    plot_calibration,
)


# **Step 3.1:** Load and Prepare the Data
dataset_prefix = "gaussian_blurred_images_"  # TODO: fill in with your own dataset prefix

calib_data_dir = f"calibration_data"
assert os.path.exists(calib_data_dir)

# Load data triplet for Validation data portion...
pred_val_fname = os.path.join(calib_data_dir, f"prediction_{dataset_prefix}Val.tif")
std_val_fname = os.path.join(calib_data_dir, f"std_{dataset_prefix}Val.tif")
target_val_fname = os.path.join(calib_data_dir, f"target_{dataset_prefix}Val.tif")
assert os.path.exists(
    pred_val_fname
), f"File containing MMSE Predictions ({pred_val_fname}) not found. Run the `02_predict.ipynb` on the Validation data and save this file (as instructed in the notebook)."
assert os.path.exists(
    std_val_fname
), f"File containing the pixel-wise std values of posterior samples ({std_val_fname}) not found. Run the `02_predict.ipynb` on the Validation data and save this file (as instructed in the notebook)."
assert os.path.exists(
    target_val_fname
), f"File containing training-data like target data ({target_val_fname}) not found. Run the `02_predict.ipynb` and save this file (as instructed in the notebook)."
pred_val = tifffile.imread(pred_val_fname)
std_val = tifffile.imread(std_val_fname)
target_val = tifffile.imread(target_val_fname)

# Load data triplet for Test data portion...
pred_test_fname = os.path.join(calib_data_dir, f"prediction_{dataset_prefix}Test.tif")
std_test_fname = os.path.join(calib_data_dir, f"std_{dataset_prefix}Test.tif")
target_test_fname = os.path.join(calib_data_dir, f"target_{dataset_prefix}Test.tif")
assert os.path.exists(
    pred_test_fname
), f"File containing MMSE Predictions ({pred_test_fname}) not found. Run the `02_predict.ipynb` on the Test data and save this file (as instructed in the notebook)."
assert os.path.exists(
    std_test_fname
), f"File containing the pixel-wise std values of posterior samples ({std_test_fname}) not found. Run the `02_predict.ipynb` on the Test data and save this file (as instructed in the notebook)."
assert os.path.exists(
    target_test_fname
), f"File containing training-data like target data ({target_test_fname}) not found. Run the `02_predict.ipynb` and save this file (as instructed in the notebook)."
pred_test = tifffile.imread(pred_test_fname)
std_test = tifffile.imread(std_test_fname)
target_test = tifffile.imread(target_test_fname)

if target_test.shape != pred_test.shape:
    assert target_test.shape[1:] == pred_test.shape[1:]
    assert pred_test.shape[0] == 1
    print('You probably enabled reduce_data=True in the `02_predict.ipynb` notebook. This is fine, but you will get calibration on a smaller data.')
    target_test = target_test[:1]


print("✅ if you see no errors, all required data is now loaded!")

# **Step 3.2:** Compute calibration using Validation data
# In order to evaluate how calibrated our network is we must compare the variability between posterior samples at any pixel with the true error of these pixel predictions with respect to high-SNR 'ground truth'. For this we will load the posterior samples we predicted on the ***validation data*** we did not use during model training.


calib = Calibration(
    num_bins=50, # experiment with number of bins?
)
native_stats = calib.compute_stats(pred=pred_val, pred_std=std_val, target=target_val)
count = np.array(native_stats[0]["bin_count"])
count = count / count.sum()

# Compute calibration factors for the channels
calib_factors, factors_array = calib.get_calibrated_factor_for_stdev()
print(f"Calibration factors: {calib_factors}")

# **Step 3.3:** Use calibration setting computed on Validation data and check how calibrated the Test data is
# Once we know how the inter-sample variability (*ie.* RMV) relates to the true error (*ie.* RMSE), we would like to predict the error <nobr>Micro$\mathbb{S}$plit</nobr> makes on our ***test data***, wich we did not use for computing the calibration above (we used the validation data for that).

# Use calibration factor we previously computed on the Validation data...

# ...on the validation data
print("Compute calibration for validation data...", end="")
calib_val = Calibration(num_bins=50)
stats_val = calib_val.compute_stats(
    pred_val, std_val * factors_array["scalar"] + factors_array["offset"], target_val
)
print("✅")

# ...on the test data
print("Compute calibration for test data...", end="")
calib_test = Calibration(num_bins=50)
stats_test = calib_test.compute_stats(
    pred_test, std_test * factors_array["scalar"] + factors_array["offset"], target_test
)
print("✅")

# Finally, plotting the results!
print("Plotting results...")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

# _,ax = plt.subplots(figsize=(5,5))
# plt.title("Test-data Calibration")
ax[0].set_title("Validation-data Calibration")
plot_calibration(ax[0], stats_val)

# _,ax = plt.subplots(figsize=(5,5))
# plt.title("Test-data Calibration")
ax[1].set_title("Test-data Calibration")
plot_calibration(ax[1], stats_test)
save_figure(fig, predict_evaluate_dir / "calibration" / "validation_and_test_calibration.png")

# On the **x-axis** we plot the [root-mean-variance (RMV)](https://pubs.aip.org/aip/aml/article/1/4/046121/2930395/Calibration-in-machine-learning-uncertainty) between the posterior samples we computed in the notebook `02_predict.ipynb` and stored in the `calibration_data` folder.
# On the **y-axis** we plot the [root-mean-squared-error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation) between the MMSE predictions we also computed in the predictiion notebook and the (potentially even noisy) target data, also both stored in the `calibration_data` folder.
# If the RMV between posterior samples would perfectly scale with the true error (measured via RMSE), the plots (one for each of the two predicted channels) would perfectly lie on the diagonal $y=x$. Now, this is not the case, but it is not all to far off, which is good news!

# Step 3.4: Plotting estimated true error (RMSE maps)

# Compute the predicted true error
pred_error_test = (
    std_test * factors_array["scalar"].squeeze()[0]
    + factors_array["offset"].squeeze()[0]
)

fig, ax = plt.subplots(figsize=(12, 8), ncols=3, nrows=2)
hs = 1200
ws = 250
sz = 400
ax[0, 0].imshow(pred_test[0, hs : hs + sz, ws : ws + sz, 0])
ax[0, 1].imshow(pred_error_test[0, hs : hs + sz, ws : ws + sz, 0], cmap="coolwarm")
ax[0, 2].imshow(target_test[0, hs : hs + sz, ws : ws + sz, 0])

ax[1, 0].imshow(pred_test[0, hs : hs + sz, ws : ws + sz, 1])
ax[1, 1].imshow(pred_error_test[0, hs : hs + sz, ws : ws + sz, 1], cmap="coolwarm")
ax[1, 2].imshow(target_test[0, hs : hs + sz, ws : ws + sz, 1])

ax[0, 0].set_title("Prediction")
ax[0, 1].set_title("Estimated RMSE")
ax[0, 2].set_title("Target")
ax[0, 0].set_ylabel("Channel 1")
ax[1, 0].set_ylabel("Channel 2")
save_figure(fig, predict_evaluate_dir / "calibration" / "test_estimated_rmse_maps.png")

print("Intensity stats for Channel 1:")
print(
    f" > Prediciton (min, max): ({pred_test[...,0].min():6.3f},{pred_test[...,0].max():6.3f})"
)
print(
    f" > Target (min, max):     ({target_test[...,0].min():6.3f},{target_test[...,0].max():6.3f})"
)
print(
    f" > RMSE (min, max):       ({pred_error_test[...,0].min():6.3f},{pred_error_test[...,0].max():6.3f})"
)
print("Intensity stats for Channel 2:")
print(
    f" > Prediciton (min, max): ({pred_test[...,1].min():6.3f},{pred_test[...,1].max():6.3f})"
)
print(
    f" > Target (min, max):     ({target_test[...,1].min():6.3f},{target_test[...,1].max():6.3f})"
)
print(
    f" > RMSE (min, max):       ({pred_error_test[...,1].min():6.3f},{pred_error_test[...,1].max():6.3f})"
)

diff_pred_target = np.abs(pred_test - target_test) ** 2
true_L2_error = np.sqrt(np.mean(diff_pred_target))
estimated_L2_error = np.mean(pred_error_test)

print(f"Esimated error: {estimated_L2_error:8.2f}")
print(f"True error:     {true_L2_error:8.2f}")
print(f"Difference:     {np.abs(estimated_L2_error-true_L2_error)}")


print(f'✅ Saved prediction/evaluation/calibration figures in "{predict_evaluate_dir}".')
