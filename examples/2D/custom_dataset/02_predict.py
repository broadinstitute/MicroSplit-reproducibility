# Step 2: Using a Trained microsplit Model

# This code block:
# * makes full frame predictions and inspect the results,
# * explores the possibility of microsplit to sample predictions from the learned posterior of possible solutions,
# * allows visually inspection the data uncertainty we can deduce from posterior samples, and
# * quantitatively evaluates the model using several metrics. 

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

# Step 2.1: Data Preparation

# ⚠️ Warning: Make sure your data is the same as in the 01_train notebook and is saved in the same format

# Path to your own data

DATA_PATH = Path("/Users/sdasgupt/Documents/microsplit/jump-qc/Yokogawa_images/bad_images/blurry_images/data")


# %% [markdown]
# Setup the path to the noise models
NM_PATH = Path("./noise_models/")

# Load the image data to be processed

from pathlib import Path
import tifffile

root = Path(DATA_PATH)

bad = []
for p in root.rglob("*"):
    if p.is_file():
        # mimic the loader's behavior (but add filters if you want)
        try:
            tifffile.imread(p)
        except Exception as e:
            bad.append((str(p), repr(e)))

bad[:20], len(bad)


# running the next code blocks instead of this one because dataset comprises only two images
# setting up train, validation, and test data configs

# this section will need to be modified because this will error if dataset contains two images
# previously while running in the notebook, this cell was not run if the dataset was small
# for now, commenting out this section
# from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file


# def is_valid_tiff_path(p: Path) -> bool:
#     name = p.name
#     if name.startswith(".") or name.startswith("._"):   # hidden + AppleDouble
#         return False
#     if p.suffix.lower() not in (".tif", ".tiff"):
#         return False
#     if p.stat().st_size == 0:  # empty file
#         return False
#     return True

# def load_data(datadir):
#     data_path = Path(datadir)
#     channel_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())

#     channels_data = []
#     for channel_dir in channel_dirs:
#         image_files = sorted(
#             f for f in channel_dir.iterdir()
#             if f.is_file() and is_valid_tiff_path(f)
#         )
#         channel_images = [load_one_file(image_path) for image_path in image_files]

# train_data_config, val_data_config, test_data_config = get_data_configs(
#     image_size=(64, 64), num_channels=2
# )

# # setting up MicroSplit parametrization
# experiment_params = get_microsplit_parameters(
#     algorithm = "musplit",
#     img_size=(64, 64),
#     batch_size=8, # use the same configs as in training
#     num_epochs=20,
#     multiscale_count=3,
#     noise_model_path=NM_PATH,
#     target_channels=2,
# )

# # create the dataset
# train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
#     datapath=DATA_PATH,
#     train_config=train_data_config,
#     val_config=val_data_config,
#     test_config=test_data_config,
#     load_data_func=get_train_val_data,
# )


# Run this code block if dataset comprises two images

def is_valid_tiff_path(p: Path) -> bool:
    name = p.name
    if name.startswith(".") or name.startswith("._"):
        return False
    if p.suffix.lower() not in (".tif", ".tiff"):
        return False
    if p.stat().st_size == 0:
        return False
    return True

def _as_NYXC_1(img: np.ndarray) -> np.ndarray:
    """
    Convert whatever load_one_file returns into (N, Y, X, 1).
    """
    img = np.asarray(img)

    if img.ndim == 2:          # (Y, X)
        img = img[None, ..., None]   # (1, Y, X, 1)
    elif img.ndim == 3:
        # could be (N, Y, X) or (Y, X, C)
        # Heuristic: if last dim is small (<=4), treat as channels; otherwise treat as N
        if img.shape[-1] <= 4:
            img = img[None, ...]          # (1, Y, X, C)
            if img.shape[-1] != 1:
                # keep only first channel if multi-channel file (rare here)
                img = img[..., :1]
        else:
            img = img[..., None]          # (N, Y, X, 1)
    elif img.ndim == 4:
        # assume already (N, Y, X, C); keep first channel if needed
        if img.shape[-1] != 1:
            img = img[..., :1]
    else:
        raise ValueError(f"Unexpected image shape from load_one_file: {img.shape}")

    return img

def load_data_two_images(datadir, num_channels=2, n_images=2) -> np.ndarray:
    """
    Returns (n_images, Y, X, num_channels).
    Loads only first `num_channels` channel directories under datadir.
    """
    data_path = Path(datadir)
    channel_dirs_all = sorted([p for p in data_path.iterdir() if p.is_dir()])
    channel_dirs = channel_dirs_all[:num_channels]  # <-- enforce channel count

    if len(channel_dirs) < num_channels:
        raise ValueError(f"Found only {len(channel_dirs)} channel dirs, expected {num_channels}.")

    per_channel = []
    for channel_dir in channel_dirs:
        image_files = sorted(
            f for f in channel_dir.iterdir()
            if f.is_file() and is_valid_tiff_path(f)
        )[:n_images]

        if len(image_files) < n_images:
            raise ValueError(f"{channel_dir} has only {len(image_files)} images; expected {n_images}.")

        imgs = [_as_NYXC_1(load_one_file(f)) for f in image_files]  # each (1,Y,X,1)
        channel_stack = np.concatenate(imgs, axis=0)               # (N,Y,X,1)
        per_channel.append(channel_stack)

    data = np.concatenate(per_channel, axis=-1)  # (N,Y,X,C)
    return data[:n_images, ...]                  # (2,Y,X,C)


def _split_name(datasplit_type) -> str:
    name = getattr(datasplit_type, "name", str(datasplit_type))
    return name.split(".")[-1].lower()

def get_train_val_data_two_images(data_config, datadir, datasplit_type, **kwargs):
    split = _split_name(datasplit_type)

    data = load_data_two_images(
        datadir,
        num_channels=data_config.num_channels,
        n_images=2
    )  # (2, Y, X, C)

    if split == "train":
        return data[:1]        # image 0
    elif split in ("val", "valid", "validation"):
        return data[1:2]       # image 1
    elif split == "test":
        return data[1:2]       # <-- IMPORTANT: must not be empty
        # or use: return data[:1]  # if you'd rather duplicate train
    else:
        raise ValueError(f"Unknown datasplit_type: {datasplit_type} (parsed as '{split}')")


tmp = load_data_two_images(DATA_PATH, num_channels=2, n_images=2)
print("Loaded shape:", tmp.shape)

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
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA_PATH,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=test_data_config,
    load_data_func=get_train_val_data_two_images,
)

# Configure `num_workers`
# In Windows and MacOS, setting `num_workers > 0` for dataloaders would cause out-of-memory issue and might crash the system.

def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3  # or any other number suitable for your system

experiment_params["num_workers"] = get_num_workers()

# Pick Validation or Test data to be used

evaluate_on_validation_data = False  # set to True to use validation data instead of test data
if evaluate_on_validation_data:
    print("Will use validation data", end="")
    dset = val_dset
else:
    print("Will use test data", end="")
    dset = test_dset
print(f" (containing a total of {dset.get_num_frames()} frames).")

# Let's look at bits of the data you chose

plot_input_patches(dataset=dset, num_channels=2, num_samples=3, patch_size=128)

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

if reduce_data:
    print("Using REDUCED evaluation data for quick'n'dirty testing!")
    dset.reduce_data([0])
else:
    print("Using the full set of evaluation data!")
    print(
        f"(More specifically, I will use {dset.get_num_frames()} frames for evaluations.)"
    )

# Step 2.4: Predictions on Uncropped Data

from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions,
    get_target,
    get_input,
)

# Note, this parameter is responsible for how many samples are generated for each patch.
# The default value is 5, but in this case you might see stitching artifacts because
# each patch will be slightly different. You can increase this value to 10 to get a 
# smoother image
experiment_params["mmse_count"] = 10

# Here we use a small helper function that returns the final results
# after performing Inner Padding, as mentioned above.
# Note also that it also returns `stitched_stds`, which is the pixel-wise
# standard deviation (std) between the posterior samples we have averaged
# while computing the MMSE per patch during tiled predictions. These
# values will become most useful at the end of this notebook and in even
# more so in `03_calibration.ipynb` for calibration and error estimations.
# stitched_predictions, norm_stitched_predictions, stitched_stds = (
#     get_unnormalized_predictions(
#         model,
#         dset,
#         data_key=dset32._fpath.name,
#         mmse_count=experiment_params["mmse_count"],
#         grid_size=32,
#         num_workers=get_num_workers(),
#         batch_size=8,
#     )
# )


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


# Not to be run - this produces predictions for test/val (2 channels) and saves them as OME-TIFF files in the `predictions_out/per_set_2ch` folder.

# out_dir = Path("predictions_out/per_set_2ch")
# out_dir.mkdir(parents=True, exist_ok=True)

# pred = np.asarray(stitched_predictions).astype(np.float32)  # (T,Y,X,C)
# assert pred.ndim == 4 and pred.shape[-1] == 2, pred.shape
# T = pred.shape[0]

# names = None
# for attr in ["frame_paths", "_frame_paths", "paths", "_paths", "files", "_files"]:
#     if hasattr(dset, attr):
#         maybe = getattr(dset, attr)
#         if isinstance(maybe, (list, tuple)) and len(maybe) == T:
#             names = [Path(p).stem for p in maybe]
#             break

# for i in range(T):
#     stem = names[i] if names else f"set_{i:03d}"
#     cyx = np.moveaxis(pred[i], -1, 0)  # (C,Y,X)

#     tiff.imwrite(
#         out_dir / f"{stem}_pred_CYX.ome.tif",
#         cyx,
#         ome=True,
#         metadata={"axes": "CYX"},
#     )

# Saves 4-channel OME-TIFFs with channel order GT0, Pred0, GT1, Pred1

out_dir = Path("predictions_out/per_set_4ch_gt_pred/bad_images/user_annotated_blurry_images") #set the path where you want the images saved
out_dir.mkdir(parents=True, exist_ok=True)

pred = np.asarray(stitched_predictions).astype(np.float32)  # (T, Y, X, 2)
assert pred.ndim == 4 and pred.shape[-1] == 2, pred.shape
T, Y, X, _ = pred.shape

gt = np.asarray(get_target(dset)).astype(np.float32)        # (T, Y, X, 2)
assert gt.ndim == 4 and gt.shape[0] == T and gt.shape[1] == Y and gt.shape[2] == X and gt.shape[-1] == 2, gt.shape

names = None
for attr in ["frame_paths", "_frame_paths", "paths", "_paths", "files", "_files"]:
    if hasattr(dset, attr):
        maybe = getattr(dset, attr)
        if isinstance(maybe, (list, tuple)) and len(maybe) == T:
            names = [Path(p).stem for p in maybe]
            break

for i in range(T):
    stem = names[i] if names else f"set_{i:03d}"

    # Interleave channels: GT0, Pred0, GT1, Pred1  -> (Y, X, 4)
    yxc4 = np.stack(
        [gt[i, ..., 0], pred[i, ..., 0], gt[i, ..., 1], pred[i, ..., 1]],
        axis=-1
    ).astype(np.float32)

    cyx4 = np.moveaxis(yxc4, -1, 0)  # (C, Y, X)

    tiff.imwrite(
        out_dir / f"{stem}_gt_pred_test.ome.tif", # Change to match the set (test/val) being used
        cyx4,
        ome=True,
        metadata={"axes": "CYX"},
    )

# # Saves 4-channel OME-TIFFs with channel order GT0, Pred0, GT1, Pred1 for all frames in the dataset (train/val/test)

# from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
#     get_unnormalized_predictions,
#     get_target,
# )

# out_dir = Path("predictions_out/all_frames_4ch_gt_pred")
# out_dir.mkdir(parents=True, exist_ok=True)
# experiment_params["mmse_count"] = 5

# def get_frame_stems(ds):
#     T = ds.get_num_frames()
#     for attr in ["frame_paths", "_frame_paths", "paths", "_paths", "files", "_files"]:
#         if hasattr(ds, attr):
#             maybe = getattr(ds, attr)
#             if isinstance(maybe, (list, tuple)) and len(maybe) == T:
#                 return [Path(p).stem for p in maybe]
#     return [f"frame_{i:03d}" for i in range(T)]

# def predict_ds(model, ds, mmse_count, grid_size=32, batch_size=4):
#     # batch_size lowered to reduce peak memory
#     return get_unnormalized_predictions(
#         model,
#         ds,
#         data_key=ds._fpath.name,
#         mmse_count=mmse_count,
#         grid_size=grid_size,
#         num_workers=get_num_workers(),
#         batch_size=batch_size,
#     )

# splits = [("train", train_dset), ("val", val_dset), ("test", test_dset)]

# for split_name, ds in splits:
#     stems = [f"{split_name}_{s}" for s in get_frame_stems(ds)]

#     # GT is usually much smaller than predictions; still load per-split only
#     gt = np.asarray(get_target(ds), dtype=np.float32)  # (T,Y,X,2)

#     # Predictions per split (avoid concatenating across splits)
#     pred = np.asarray(predict_ds(model, 
#                                  ds, 
#                                  mmse_count=experiment_params["mmse_count"], 
#                                  grid_size=32, batch_size=2), 
#                                  dtype=np.float32)  # (T,Y,X,2)

#     assert gt.shape == pred.shape and gt.shape[-1] == 2, (gt.shape, pred.shape)

#     T = pred.shape[0]
#     for i in range(T):
#         # 4-channel order: GT0, Pred0, GT1, Pred1
#         yxc4 = np.stack(
#             [gt[i, ..., 0], pred[i, ..., 0], gt[i, ..., 1], pred[i, ..., 1]],
#             axis=-1
#         ).astype(np.float32)  # (Y,X,4)
#         cyx4 = np.moveaxis(yxc4, -1, 0)  # (C,Y,X)

#         tiff.imwrite(
#             out_dir / f"{stems[i]}_gt_pred.ome.tif",
#             cyx4,
#             ome=True,
#             metadata={"axes": "CYX"},
#         )

#     # Free memory before next split
#     del pred, gt
#     gc.collect()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# Following cells, which are responsible for the visualizations, might need to be changed if you want to adapt the visualization to your data.

# load inputs and noisy targets (needed for plotting later on)
inp = get_input(dset).sum(-1)
tar = get_target(dset)

# Overview: visualize full microsplit predictions


frame_idx = 0
assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds"

full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx])

# Detailed view on some (foreground) locations...
# Below, we show few random foreground locations and the corresponding microsplit predictions.

from microsplit_reproducibility.utils.utils import clean_ax
from microsplit_reproducibility.notebook_utils.HT_LIF24 import (
    pick_random_patches_with_content,
)

img_sz = 128
rand_locations = pick_random_patches_with_content(tar, 128)
h_start = rand_locations[
    2, 0
]  # np.random.randint(stitched_predictions.shape[1] - img_sz)
w_start = rand_locations[
    2, 1
]  # np.random.randint(stitched_predictions.shape[2] - img_sz)

ncols = 5
nrows = min(len(rand_locations), 5)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

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
for i in range(2):  # 2 channel splitting
    ax[0, 2 * i + 1].set_title(f"Target Channel {i+1}")
    ax[0, 2 * i + 2].set_title(f"Predicted Channel {i+1}")

# reduce the spacing between the subplots
plt.subplots_adjust(wspace=0.03, hspace=0.03)
clean_ax(ax)

# *Optional:* manual inspection of the predictions

y_start = 750  # np.random.randint(stitched_predictions.shape[1] - crop_size)
x_start = 750  # np.random.randint(stitched_predictions.shape[2] - crop_size)
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

# disable the axis for ax[1,0]
ax[1, 0].axis("off")
ax[0, 0].set_title("Input")
ax[0, 1].set_title("Channel 1")
ax[0, 2].set_title("Channel 2")
# set y labels on the right for ax[0,2]
ax[0, 2].yaxis.set_label_position("right")
ax[0, 2].set_ylabel("Target")

ax[1, 2].yaxis.set_label_position("right")
ax[1, 2].set_ylabel("Predicted")

print("Here the crop you selected:")

# Optional Step 2.5: Posterior Sampling and MMSE Predictions

imgsz = 3
ncols = 6
examplecount = 3
_, ax = plt.subplots(
    figsize=(imgsz * ncols, 2 * imgsz * examplecount),
    ncols=ncols,
    nrows=2 * examplecount,
)

show_sampling(dset, model, ax=ax[:2])
show_sampling(dset, model, ax=ax[2:4])
show_sampling(dset, model, ax=ax[4:6])
plt.tight_layout()

n_samples = 50  # min: 10
if n_samples < 10:
    n_samples = 10

# choose a random input patch
from microsplit_reproducibility.notebook_utils.HT_LIF24 import (
    pick_random_inputs_with_content,
)

idx_list = pick_random_inputs_with_content(dset)
inp_patch, tar_patch = dset[idx_list[0] + 1]

# compute individual posterior samples
samples = []
model.eval()

for _ in (range(n_samples), "Sampling the posterior"):
    with torch.no_grad():
        pred_patch, _ = model(torch.Tensor(inp_patch).unsqueeze(0).to(model.device))
        samples.append(pred_patch[0, : tar_patch.shape[0]].cpu().numpy())
samples = np.array(samples)

inp_patch, tar_patch = dset[idx_list[0]]

nrows = 5
imgsz = 3
_, ax = plt.subplots(figsize=(imgsz * 2, imgsz * nrows + imgsz), ncols=2, nrows=nrows)
ax[0, 0].imshow(inp_patch[0])
ax[0, 0].set_title("Input (Idx: {})".format(idx_list[0]))

# first channel
ax[1, 0].imshow(samples[0, 0])
ax[1, 0].set_title("C1: Sample")
ax[2, 0].imshow(np.mean(samples[:5, 0], axis=0))
ax[2, 0].set_title("C1: MMSE (5)")
ax[3, 0].imshow(np.mean(samples[:, 0], axis=0))
ax[3, 0].set_title(f"C1: MMSE ({len(samples)})")
ax[4, 0].imshow(tar_patch[0])
ax[4, 0].set_title("C1: Target")

# second channel
ax[1, 1].imshow(samples[0, 1])
ax[1, 1].set_title("C2: Sample")
ax[2, 1].imshow(np.mean(samples[:5, 1], axis=0))
ax[2, 1].set_title("C2: MMSE (5)")
ax[3, 1].imshow(np.mean(samples[:, 1], axis=0))
ax[3, 1].set_title(f"C2: MMSE ({len(samples)})")
ax[4, 1].imshow(tar_patch[1])
ax[4, 1].set_title("C2: Target")

ax[0, 1].axis("off")

# Step 2.6: Saving data required for network calibration and error estimations (ie. for running `03_calibration.ipynb`)
# change this only if you used your own data
dataset_prefix = "user_annotated_blurry_images_"  # TODO: fill in with your own dataset prefix

# create folder to store all the data (6 files in total) for the calibration notebook
path_for_calibration_data = f"calibration_data"
os.makedirs(path_for_calibration_data, exist_ok=True)

# Step 2.6.1: Save target data we need in the calibration notebook.

data_stats = experiment_params["data_stats"]
target_val = val_dset._data[...]
target_test = test_dset._data[...]

sep_mean = np.transpose(data_stats[0].numpy(), axes=(0, 2, 3, 1))
sep_std = np.transpose(data_stats[1].numpy(), axes=(0, 2, 3, 1))

target_val_normalized = (target_val - sep_mean) / sep_std
target_test_normalized = (target_test - sep_mean) / sep_std

# store also the corresponding target data (this is just like the supervision data we used during training)
target_val_filename = "target_" + dataset_prefix + "Val"
target_test_filename = "target_" + dataset_prefix + "Test"
tifffile.imwrite(
    f"{path_for_calibration_data}/{target_val_filename}.tif", target_val_normalized
)
print(
    f'✅ Saved target data for Val data at "{path_for_calibration_data}/{target_val_filename}.tif"!'
)
tifffile.imwrite(
    f"{path_for_calibration_data}/{target_test_filename}.tif", target_test_normalized
)
print(
    f'✅ Saved target data for Test data at "{path_for_calibration_data}/{target_test_filename}.tif"!'
)

# Step 2.6.2: Save the MMSE predictions and Std from Step 2.4.
# We need four more things to have all the data for the calibration notebook together. More specifically, and as mentioned before, we need:
# *(i)* the MMSE predictions for the Validation data, 
# *(ii)* the MMSE predictions for the Test data,
# *(iii)* the standard deviation (Std) of the posterior samples we drew to generate the MMSE predictions for the Validation data, and
# *(iv)* the standard deviation (Std) of the posterior samples we drew to generate the MMSE predictions for the Test data.
# 
# Two of these four missing pieces we have computed above, depending on what you choose to work with, Validation or Training data (you made this choice in Step 2.1).
# If you want to be prepared for the next notebook and haven't already, go back to Step 2.1, and set <i> evaluate_on_validation_data = True </i> and re-run the cells in the remainer of the notebook. Make sure `reduce_data` parameter setting is consistent. 

# Let us make sure:
# (i) are the generated predictions of same shape?

assert (
    norm_stitched_predictions.shape == stitched_stds.shape
), "MMSE predictions and pixel-wise stds have incompatible shape. Please redo Step 2.4 of this notebook!"

# target = test_dset._data[...] # changing this to use the correct split
target = (val_dset if evaluate_on_validation_data else test_dset)._data[...]

assert (
    stitched_stds.shape == target.shape
), "Shape of predictions does not fit to shape of loaded inputs. Please check that the notebook is in a consistent state!"

# check if user predicted on validation or test data
val_or_test = "Val" if evaluate_on_validation_data else "Test"

# print what we found for user
print(
    f"✅ Looks like predictions for {val_or_test} data were created above.\nWe will save those predictions in the next cell..."
)
the_other = "Test" if evaluate_on_validation_data else "Val"



# store the predictions currently available (created in Step 2.4 (predictions))
pred_filename = "prediction_" + dataset_prefix + val_or_test
std_filename = "std_" + dataset_prefix + val_or_test
# save only one prediction
tifffile.imwrite(
    f"{path_for_calibration_data}/{pred_filename}.tif", norm_stitched_predictions
)
print(
    f'✅ Saved MMSE predictions for {val_or_test} data at "{path_for_calibration_data}/{pred_filename}.tif"!'
)
tifffile.imwrite(f"{path_for_calibration_data}/{std_filename}.tif", stitched_stds)
print(
    f'✅ Saved posterior sample Stds for {val_or_test} data at "{path_for_calibration_data}/{pred_filename}.tif"!'
)

print(
    f"\n‼️ Please ensure to also create and save these two files also for the {the_other} data!"
)
print(
    f'You are only ready for the calibration notebook once the folder "{path_for_calibration_data}" contain 6 files in total!'
)



