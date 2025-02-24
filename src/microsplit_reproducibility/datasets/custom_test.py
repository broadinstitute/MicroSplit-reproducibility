import os

import numpy as np

import tifffile

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples


def load_one_file(fpath):
    data = tifffile.imread(fpath)
    return data


def load_data(datadir, ch1_scale=1, ch2_scale=6):
    channels_list = os.listdir(datadir)
    data_list = []
    for i, path in enumerate(channels_list):
        ch_scale = ch1_scale if i == 0 else ch2_scale
        ch_data = []
        ch_path = os.path.join(datadir, path)
        files = os.listdir(ch_path)
        for file in files:
            fpath = os.path.join(ch_path, file)
            data = load_one_file(fpath)
            ch_data.append(data)
        ch_data = np.stack(ch_data, axis=0)
        data_list.append(ch_data / ch_scale)
    
    data = np.stack(data_list, axis=-1)
    return data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(datadir)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    val_idx = train_idx
    test_idx = train_idx
    # TODO temporary hack
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    return data
