import os

import numpy as np

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples


def load_one_fpath(fpath, channel_list):
    data = load_7D(fpath)
    # old_dataset.shape: (1, 20, 1, 19, 1608, 1608, 1)
    data = data[0, :, 0, :, :, :, 0]
    # old_dataset.shape: (20, 19, 1608, 1608)
    # Here, 20 are different locations and 19 are different channels.
    data = data[:, channel_list, ...]
    # swap the second and fourth axis
    data = np.swapaxes(data[..., None], 1, 4)[:, 0]

    fname_prefix = "_".join(os.path.basename(fpath).split(".")[0].split("_")[:-1])
    if fname_prefix == "uSplit_20022025_001":
        data = np.delete(data, 2, axis=0)
    elif fname_prefix == "uSplit_14022025":
        data = np.delete(data, [17, 19], axis=0)

    # old_dataset.shape: (20, 1608, 1608, C)
    return data


def load_data(datadir, channel_list, dataset_type):
    files_dict = get_raw_files_dict()[dataset_type]
    data_list = []
    for fname in files_dict:
        fpath = os.path.join(datadir, fname)
        data = load_one_fpath(fpath, channel_list)
        data_list.append(data)
    if len(data_list) > 1:
        data = np.concatenate(data_list, axis=0)
    else:
        data = data_list[0]
    return data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(
        datadir,
        channel_list=data_config.channel_idx_list,
        dataset_type=data_config.dset_type,
    )
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
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
