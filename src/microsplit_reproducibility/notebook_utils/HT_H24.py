import matplotlib.pyplot as plt
import numpy as np
from careamics.lvae_training.eval_utils import get_predictions, get_device
from careamics.lightning import VAEModule
from microsplit_reproducibility.datasets import create_train_val_datasets, SplittingDataset
from microsplit_reproducibility.datasets.HT_LIF24 import get_train_val_data

import os
from pathlib import Path
import torch
import pooch
import requests
from tqdm.notebook import tqdm
import logging


def load_pretrained_model(model: VAEModule, ckpt_path):
    device = get_device()
    ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_dict['state_dict'], strict=False)
    print(f"Loaded model from {ckpt_path}")


def get_all_channel_list(target_channel_list):
    """
    Adds the input channel index to the target channel list.
    """
    input_channel_index_dict = {
        '01' : 8,
        '02' : 9,
        '03' : 10,
        '12' : 11,
        '13' : 12,
        '23' : 13,
    }
    return target_channel_list + [input_channel_index_dict[''.join([str(i) for i in target_channel_list])]]

def get_unnormalized_predictions(model: VAEModule, dset: SplittingDataset, exposure_duration, target_channel_idx_list,mmse_count, num_workers=4, grid_size=32,
                                 batch_size=8):
    """
    Get the stitched predictions which have been unnormlized.
    """
    # You might need to adjust the batch size depending on the available memory
    stitched_predictions, stitched_stds = get_predictions(
        model=model,
        dset=dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        tile_size=model.model.image_size,
        grid_size=grid_size,
    )
    stitched_predictions = stitched_predictions[exposure_duration]
    stitched_stds = stitched_stds[exposure_duration]

    stitched_predictions = stitched_predictions[...,:len(target_channel_idx_list)]
    stitched_stds = stitched_stds[...,:len(target_channel_idx_list)]
    
    mean_params, std_params = dset.get_mean_std()
    unnorm_stitched_predictions = stitched_predictions*std_params['target'].squeeze().reshape(1,1,1,-1) + mean_params['target'].squeeze().reshape(1,1,1,-1)
    return unnorm_stitched_predictions, stitched_predictions, stitched_stds

def get_target(dset):
    return dset._data[...,:-1].copy()

def get_input(dset):
    return dset._data[...,-1].copy()



def pick_random_inputs_with_content(dset):
    idx_list = []
    std_list = []
    count = min(1000, len(dset))
    rand_idx_list = np.random.choice(len(dset), count, replace=False).tolist()
    for idx in rand_idx_list:
        inp = dset[idx][0]
        std_list.append(inp[0].std())
        idx_list.append(idx)
    # sort by std
    idx_list = np.array(idx_list)[np.argsort(std_list)][-40:]
    return idx_list.tolist()

def pick_random_patches_with_content(tar, patch_size):    
    H, W = tar.shape[1:3]
    std_patches = []
    indices = []
    for i in range(1000):
        h_start = np.random.randint(H - patch_size)
        w_start = np.random.randint(W - patch_size)
        std_tmp= []
        for ch_idx in range(tar.shape[-1]):
            std_tmp.append(tar[0,h_start:h_start+patch_size,w_start:w_start+patch_size,ch_idx].std())
        
        std_patches.append(np.mean(std_tmp))
        indices.append((h_start,w_start))
    
    # sort by std
    indices = np.array(indices)[np.argsort(std_patches)][-40:]

    distances = np.linalg.norm(indices[:,None] - indices[None], axis=-1)
    # pick the indices of the indices that are at least patch_size pixels apart
    final_indices = [0]
    for i in range(1, len(indices)):
        if np.all(distances[i,final_indices] >= patch_size):
            final_indices.append(i)

    final_indices = indices[final_indices,:]
    return final_indices

def full_frame_evaluation(stitched_predictions, tar, inp):

    ncols = tar.shape[-1] + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax[0,0].imshow(inp)
    for i in range(ncols -1):
        vmin = stitched_predictions[...,i].min()
        vmax = stitched_predictions[...,i].max()
        ax[0,i+1].imshow(tar[...,i], vmin=vmin, vmax=vmax)
        ax[1,i+1].imshow(stitched_predictions[...,i], vmin=vmin, vmax=vmax)

    # disable the axis for ax[1,0]
    ax[1,0].axis('off')
    ax[0,0].set_title("Input", fontsize=15)
    ax[0,1].set_title("Channel 1", fontsize=15)
    ax[0,2].set_title("Channel 2", fontsize=15)
    # set y labels on the right for ax[0,2]
    ax[0,2].yaxis.set_label_position("right")
    ax[0,2].set_ylabel("Target", fontsize=15)

    ax[1,2].yaxis.set_label_position("right")
    ax[1,2].set_ylabel("Predicted", fontsize=15)


def find_recent_metrics():
    last = ((Path("lightning_logs") / Path(sorted(os.listdir("lightning_logs")[:10])[-1])).absolute() / "metrics.csv")
    assert last.exists(), f"File {last} does not exist. Please remove the lightning_logs folder and re-run the training."
    return last


def plot_metrics(df):
    _,ax = plt.subplots(figsize=(12,3),ncols=4)
    if 'reconstruction_loss_epoch' in df.columns:
        df['reconstruction_loss_epoch'].to_frame('recons_loss').dropna().reset_index(drop=True).plot(ax=ax[0],marker='o')
    else:
        print("No reconstruction loss found")

    if 'kl_loss_epoch' in df.columns:
        df[['kl_loss_epoch']].dropna().reset_index(drop=True).plot(ax=ax[1],marker='o')
    else:
        print("No kl loss found")

    if 'val_loss' in df.columns:
        df[['val_loss']].dropna().reset_index(drop=True).plot(ax=ax[2],marker='o')
    else:
        print("No validation loss found")
    
    if 'val_psnr' in df.columns:
        df[['val_psnr']].dropna().reset_index(drop=True).plot(ax=ax[3], marker='o')
    else:
        print("No validation psnr found")
    plt.tight_layout()
    # switch on the grid
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    # set background color to gray
    ax[0].set_facecolor('lightgray')
    ax[1].set_facecolor('lightgray')
    ax[2].set_facecolor('lightgray')
    ax[3].set_facecolor('lightgray')
    # set a common x label for all the subplots
    for a in ax:
        a.set_xlabel("Epoch")


def show_sampling(dset, model, ax=None):
    idx_list = pick_random_inputs_with_content(dset)
    # inp, S1, S2, diff, mmse, tar
    ncols=6
    imgsz = 3
    if ax is None:
        _,ax = plt.subplots(figsize=(imgsz*ncols, imgsz*2), ncols=ncols, nrows=2)
    inp_patch, tar_patch = dset[idx_list[0]]
    ax[0,0].imshow(inp_patch[0])
    ax[0,0].set_title("Input (Idx: {})".format(idx_list[0]))

    samples = []
    n_samples = 50
    # get prediction 
    model.eval()
    for _ in range(n_samples):
        with torch.no_grad():
            pred_patch,_ = model(torch.Tensor(inp_patch).unsqueeze(0).to(model.device))
            samples.append(pred_patch[0,:tar_patch.shape[0]].cpu().numpy())
    samples = np.array(samples)

    ax[0,1].imshow(samples[0,0]); ax[0,1].set_title("Sample 1")
    ax[0,2].imshow(samples[1,0]); ax[0,2].set_title("Sample 2")
    ax[0,3].imshow(samples[0,0] - samples[1,0], cmap='coolwarm'); ax[0,3].set_title("S1 - S2")
    ax[0,4].imshow(np.mean(samples[:,0], axis=0)); ax[0,4].set_title("MMSE")
    ax[0,5].imshow(tar_patch[0]); ax[0,5].set_title("Target")
    # second channel
    ax[1,1].imshow(samples[0,1])
    ax[1,2].imshow(samples[1,1])
    ax[1,3].imshow(samples[0,1] - samples[1,1], cmap='coolwarm')
    ax[1,4].imshow(np.mean(samples[:,1], axis=0))
    ax[1,5].imshow(tar_patch[1])

    ax[1,0].axis('off')

def get_highsnr_data(train_data_config, val_data_config, test_data_config, evaluate_on_validation):
    highsnr_exposure_duration = '500ms'

    DATA = pooch.create(
        path=f"./data/",
        base_url=f"https://download.fht.org/jug/msplit/ht_lif24/data/",
        registry={f"ht_lif24_{highsnr_exposure_duration}.zip": None},

    )
    for fname in DATA.registry:
        DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)

    train_data_config.dset_type = highsnr_exposure_duration
    val_data_config.dset_type = highsnr_exposure_duration
    test_data_config.dset_type = highsnr_exposure_duration

    _, highSNR_val_dset, highSNR_test_dset, _ = create_train_val_datasets(
        datapath=DATA.path / f"ht_lif24_{highsnr_exposure_duration}.zip.unzip/{highsnr_exposure_duration}",
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    if evaluate_on_validation:
        return highSNR_val_dset
    else:
        return highSNR_test_dset