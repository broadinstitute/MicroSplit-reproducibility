from typing import Literal

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType


def get_data_configs() -> tuple[DatasetConfig, DatasetConfig, DatasetConfig]:
    """Get the data configurations to use at training time.
    
    Parameters
    ----------
    dset_type : Literal["high", "mid", "low", "verylow", "2ms", "3ms", "5ms", "20ms", "500ms"]
        The dataset type to use.
    channel_idx_list : list[Literal[1, 2, 3, 17]]
        The channel indices to use.
    
    Returns
    -------
    tuple[HTLIF24DataConfig, HTLIF24DataConfig]
        The train, validation and test data configurations.
    """
    train_data_config = DatasetConfig(
        data_type=DataType.HTH23BData,
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=32,
        num_channels=2,
        multiscale_lowres_count=3,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=100,
        input_has_dependant_noise=True,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        input_is_sum=False,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )
    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    test_data_config = val_data_config.model_copy(
        update=dict(datasplit_type=DataSplitType.Test,)
    )
    return train_data_config, val_data_config, test_data_config
