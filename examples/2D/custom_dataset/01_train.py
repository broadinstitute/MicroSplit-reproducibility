
#   This code trains a Microsplit network for unmixing two superimposed channels

# The data directory should be structured as follows:
# you_data_path/
# └── data
#     ├── channel_1
#     │   ├── image1.tiff
#     │   ├── image2.tiff
#     │   └── image3.tiff
#     └── channel_2
#     │   ├── image1.tiff
#     │   ├── image2.tiff
#     │   └── image3.tiff
#     └── channel_n
#     │   ├── image1.tiff
#     │   ├── image2.tiff
#     │   └── image3.tiff
# 
# The mixed image used for splitting will be obtained artificially by a convex combination of the target channels.
# The loss function for MicroSplit is a weighted average of denoiSplit loss and μSplit loss. For both denoiSplit and μSplit, their loss expression have two terms: KL divergence loss and likelihood loss. For more details, please refer to the respective papers.

# import packages and modules
import torch
# import neptune (if using neptune for logging)
import numpy as np
import platform
from pathlib import Path
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import pooch
import matplotlib.pyplot as plt
from careamics.lightning import VAEModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor    

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_optimizer_config,
    get_training_config,
    get_lr_scheduler_config,
)
from microsplit_reproducibility.utils.callbacks import get_callbacks
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.utils.utils import plot_input_patches

# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import (
    get_microsplit_parameters
)
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data

# Set the path to your own data
# Important: the path should end with `data/`

DATA_PATH = Path("/Users/sdasgupt/Documents/microsplit/jump-qc/Yokogawa_images/good_images_train_100images/data")


# Setup the path to the noise models
# This is the path to the noise models that you trained in the notebook **00_noisemodels.ipynb**
# Not used because we are running musplit, which does not use noise models
# NM_PATH = Path("./noise_models/") 

# Change this based on your dataset
NUM_CHANNELS = 2
"""The number of channels considered for the splitting task."""
BATCH_SIZE = 8 # reduced batch size for wandb testing
"""The batch size for training."""
PATCH_SIZE = (64, 64)
"""The size of the patches fed to the network for training in (Y, X)."""
EPOCHS = 5 # reduced number of epochs for wandb testing
"""The number of epochs to train the network."""

assert len(PATCH_SIZE) == 2, "PATCH_SIZE must be a tuple of length 2 (Y, X) since we are using 2D data."

# Setting up train, validation, and test data configs
train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=PATCH_SIZE,
    num_channels=NUM_CHANNELS
)

# setting up MicroSplit parametrization
experiment_params = get_microsplit_parameters(
    algorithm="musplit", # changed to "musplit"
    img_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    multiscale_count=3,
    noise_model_path=NM_PATH,
    target_channels=NUM_CHANNELS,
)

# create the dataset
train_dset, val_dset, _, data_stats = create_train_val_datasets(
    datapath=DATA_PATH,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=get_train_val_data,
)

# %% [markdown]
# ### Configure `num_workers`
# In Windows and MacOS, setting `num_workers > 0` for dataloaders would cause out-of-memory issue and might crash the system.

# %%
def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3  # or any other number suitable for your system

experiment_params["num_workers"] = get_num_workers()

# %% [markdown]
# ***Optional:*** inspect data configurations and <nobr>Micro$\mathbb{S}$plit</nobr> config

# %%
do_show_configs = True

if do_show_configs:
    print("FYI: train_data_config")
    print("----------------------")
    for cfg in train_data_config:
        print(cfg)

    print("\nFYI: experiment_params")
    print("----------------------")
    print(experiment_params)
else:
    print("You opted out of having all params printed... swiftly moving on... ;)")

# %% [markdown]
# ### Final step: create Dataloaders for network training

# %%
train_dloader = DataLoader(
    train_dset,
    batch_size=experiment_params["batch_size"],
    num_workers=experiment_params["num_workers"],
    shuffle=True,
)
val_dloader = DataLoader(
    val_dset,
    batch_size=experiment_params["batch_size"],
    num_workers=experiment_params["num_workers"],
    shuffle=False,
)

# %% [markdown]
# # **Step 1.2:** Prepare <nobr>Micro$\mathbb{S}$plit</nobr> Training
# Next, we create all the configs for the upcoming network training run. These lines are not very intuitive and if you don't intend to dive really deep into CAREamics and the internals of <nobr>Micro$\mathbb{S}$plit</nobr>, you might just execute these cells and move on.

# %%
# making our data_stas known to the experiment we prepare
experiment_params["data_stats"] = data_stats

# setting up training losses and model config (using default parameters)
loss_config = get_loss_config(**experiment_params)
model_config = get_model_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)
training_config = get_training_config(**experiment_params)

# setting up learning rate scheduler and optimizer (using default parameters)
lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
optimizer_config = get_optimizer_config(**experiment_params)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
    lr_scheduler_config=lr_scheduler_config,
    optimizer_config=optimizer_config,
)

# %% [markdown]
# ## Initialize the <nobr>Micro$\mathbb{S}$plit</nobr> model to be trained..

# %%
model = VAEModule(algorithm_config=experiment_config)

# %% [markdown]
# ### *Load checkpoint (optional and for you to implement)*
# 
# <div class="alert alert-block alert-success">
# <b>Note:</b> If you would like to continue a previous training run or finetune a compatible pre-trained model, here would be a good place. You will need to figure out how to implement this for your use-case, but to give you a head-start, we left three potentially useful lines of code below.
# </div>

# %%
# from microsplit_reproducibility.notebook_utils.HT_LIF24 import load_pretrained_model
# ckpt_path = load_checkpoint_path(f"./pretrained_checkpoints/{EXPOSURE_DURATION}/", best=True)
# load_pretrained_model(model, ckpt_path)

# %% [markdown]
# ## Show some training data for a final check!
# ***Tip:*** we show you a few samples of the prepared training data. In case you don't like what you see, execute the cell again and other randomly chosen patches will be shown!

# %%
plot_input_patches(dataset=train_dset, num_channels=NUM_CHANNELS, num_samples=3, patch_size=128)

# %% [markdown]
# # **Step 1.3:** Train the prepared model!
# ***Note:*** if this takes too long, there were to places above where we gave you options to *(i)* reduce the amount of training data, and *(ii)* chose to train for fewer epochs. Revisit your choices if you want to!
# 
# ***Note:*** Depending on the amount of GPU memory you have available, you might want to adjust the batch size. The default is 32, but you can reduce it to 16 if you run out of memory by changing the <i> batch_size </i> parameter in <i> get_microsplit_parameters </i> above.
# 

# %%
# This code block was for generating the metrics.csv file, which we are not using anymore because we log to Neptune
# from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning import Trainer

# logger = CSVLogger(save_dir="lightning_logs", name=None)
# trainer = Trainer(logger=logger)

# %%
# print(experiment_params.keys())

# %%
wandb.login()

# %%
# create a CAREamics 'Trainer'

wandb_logger = WandbLogger(
    project="jump-qc-microsplit",     
    entity="CiminiLab",    
    job_type="train",
    name="JUMP_QC_WANDB3",                
    tags=["musplit", "epochs=2", "images=100_per_channel", "batch_size=8"],
    save_dir=".",                    
    log_model=True,
    save_code=True,                   
)

trainer = Trainer(
    max_epochs=training_config.num_epochs,
    # NOTE: if you are on a mac swap the accelerator to "mps"
    accelerator="mps",
    log_every_n_steps=1,
    enable_progress_bar=True,
    callbacks=get_callbacks(f"./checkpoints/"),
    precision=training_config.precision,
    gradient_clip_val=training_config.gradient_clip_val, #check clipping
    gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    logger=wandb_logger,
    )

# start the training - yay!
trainer.fit(
    model=model,
    train_dataloaders=train_dloader,
    val_dataloaders=val_dloader,
    datamodule=None, # we are passing dataloaders directly, so no need for a datamodule
)

run = wandb_logger.experiment
run.log_code(root="/Users/sdasgupt/Documents/microsplit/microsplit/MicroSplit-reproducibility")

# %%
# from pandas import read_csv
# from microsplit_reproducibility.notebook_utils.custom_dataset_2D import find_recent_metrics, plot_metrics
# df = read_csv(find_recent_metrics())
# plot_metrics(df)

# %% [markdown]
# # **Step 1.4:** Predict and visualize results for validation data...

# %%
# Optional, reduce the validation dataset to speed up the evaluation
val_dset.reduce_data([0])

# %% [markdown]
# ***Note*** Parameter `mmse_count` is responsible for how many samples are generated for each patch. The default value is 1, but in this case you might see stitching artifacts because each patch will be slightly different. You can increase this value to 10 to get a smoother image

# %%
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions,
    get_target,
    get_input,
)

stitched_predictions, _, _ = get_unnormalized_predictions(
    model,
    val_dset,
    data_key=val_dset._fpath.name,
    mmse_count=experiment_params['mmse_count'],
    num_workers=0,
    batch_size=8
)
tar = get_target(val_dset)

# get input as sum of the two channels
inp = get_input(val_dset).sum(-1)

# %% [markdown]
# ## Overview: visualize predictions on validation data...

# %%
# changed to allow adding the prediction and target images to neptune

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax[0, 0].imshow(tar[0, ..., 0], cmap="gray")
ax[0, 0].set_title("Input ch1")
ax[0, 1].imshow(tar[0, ..., 1], cmap="gray")
ax[0, 1].set_title("Input ch2")
ax[1, 0].imshow(stitched_predictions[0, ..., 0], cmap="gray")
ax[1, 0].set_title("Prediction ch1")
ax[1, 1].imshow(stitched_predictions[0, ..., 1], cmap="gray")
ax[1, 1].set_title("Prediction ch2")

# %%
#Rescale for better visualization

ch0_all = np.concatenate([
    tar[0, ..., 0].ravel(),
    stitched_predictions[0, ..., 0].ravel()
])
vmin0, vmax0 = np.percentile(ch0_all, (1, 99))

# channel 1 scale
ch1_all = np.concatenate([
    tar[0, ..., 1].ravel(),
    stitched_predictions[0, ..., 1].ravel()
])
vmin1, vmax1 = np.percentile(ch1_all, (1, 99))

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax[0, 0].imshow(tar[0, ..., 0], cmap="gray", vmin=vmin0, vmax=vmax0); ax[0, 0].set_title("Input ch1")
ax[1, 0].imshow(stitched_predictions[0, ..., 0], cmap="gray", vmin=vmin0, vmax=vmax0); ax[1, 0].set_title("Prediction ch1")

ax[0, 1].imshow(tar[0, ..., 1], cmap="gray", vmin=vmin1, vmax=vmax1); ax[0, 1].set_title("Input ch2")
ax[1, 1].imshow(stitched_predictions[0, ..., 1], cmap="gray", vmin=vmin1, vmax=vmax1); ax[1, 1].set_title("Prediction ch2")

# exp = neptune_logger.experiment
# exp["train/prediction_vs_target_series_rescaled"].append(File.as_image(fig))

# using tensors, numpy arrays or PIL images
wandb_logger.log_image(key="samples", images=[fig])


# %%
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import full_frame_evaluation

frame_idx = 0
assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds"
full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx])

# %% [markdown]
# ## Detailed view on some (foreground) locations...
# Below, we show few random foreground locations and the corresponding <nobr>Micro$\mathbb{S}$plit</nobr> predictions.
# 
# As before, also here you can execute the cell multiple times and different randomly chosen locations will be plotted.

# %%
import numpy as np
from microsplit_reproducibility.utils.utils import clean_ax
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    pick_random_patches_with_content,
)
import matplotlib.pyplot as plt

img_sz = 128
rand_locations = pick_random_patches_with_content(tar, 128)
h_start = rand_locations[
    2, 0
]  # np.random.randint(stitched_predictions.shape[1] - img_sz)
w_start = rand_locations[
    2, 1
]  # np.random.randint(stitched_predictions.shape[2] - img_sz)

ncols = 4 + 1
nrows = min(len(rand_locations), 5)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
    ax[i, 0].imshow(inp[0, h_start : h_start + img_sz, w_start : w_start + img_sz])
    for j in range(ncols // 2):
        vmin = stitched_predictions[
            0, h_start : h_start + img_sz, w_start : w_start + img_sz, j
        ].min()
        vmax = stitched_predictions[
            0, h_start : h_start + img_sz, w_start : w_start + img_sz, j
        ].max()
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
for i in range(2):
    ax[0, 2 * i + 1].set_title(f"Target Channel {i+1}")
    ax[0, 2 * i + 2].set_title(f"Predicted Channel {i+1}")

# reduce the spacing between the subplots
plt.subplots_adjust(wspace=0.03, hspace=0.03)
clean_ax(ax)

wandb.log({"train/prediction_vs_target_crop": wandb.Image(fig)})

# %% [markdown]
# ## *Optional:* manual inspection of the predictions
# <div class="alert alert-block alert-info">
# <b> Task:</b> Set <i>y_start</i>, <i>x_start</i>, and <i>crop_size</i> to inspect the predictions at a  location of your choice.
# </div>

# %%
import numpy as np

y_start = 600  # np.random.randint(stitched_predictions.shape[1] - crop_size)
x_start = 1150  # np.random.randint(stitched_predictions.shape[2] - crop_size)
crop_size = 128

ncols = 3
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
ax[0, 0].imshow(inp[0, y_start : y_start + crop_size, x_start : x_start + crop_size])
for i in range(ncols - 1):
    vmin = stitched_predictions[
        0, y_start : y_start + crop_size, x_start : x_start + crop_size, i
    ].min()
    vmax = stitched_predictions[
        0, y_start : y_start + crop_size, x_start : x_start + crop_size, i
    ].max()
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

# %% [markdown]
# # ***Optional Step 1.4:*** Posterior Sampling
# For a given input patch, <nobr>Micro$\mathbb{S}$plit</nobr> can generate multiple outputs. This is possible because <nobr>Micro$\mathbb{S}$plit</nobr> is learning a full posterior of possible solutions, which is a quite powerful feature!
# 
# As we elaborate in the <nobr>Micro$\mathbb{S}$plit</nobr> paper and also later in the calibration notebook `03_calibration.ipynb`, this allows users to visually judge and even quantify the (data) uncertainty in the predictions their trained model makes.
# 
# Below, we show two posterior samples and how much they differ for a few random foreground locations. Re-run the cell to see different randomly choosen locations and corresponding posterior samples.

# %%
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import show_sampling

imgsz = 3
ncols = 6
examplecount = 3
_, ax = plt.subplots(
    figsize=(imgsz * ncols, imgsz * 2 * examplecount),
    ncols=ncols,
    nrows=2 * examplecount,
)

show_sampling(val_dset, model, ax=ax[:2])
show_sampling(val_dset, model, ax=ax[2:4])
show_sampling(val_dset, model, ax=ax[4:6])
plt.tight_layout()

# %% [markdown]
# ### You are done here! 👍 Congratulations! 🎉
