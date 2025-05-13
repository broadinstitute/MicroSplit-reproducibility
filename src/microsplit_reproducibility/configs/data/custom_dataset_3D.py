from typing import Optional

from careamics.lvae_training.dataset import DataSplitType, DatasetConfig, DataType

def get_data_configs(
    image_size: list[int],
    num_channels: int,
    **kwargs,
) -> tuple[DatasetConfig, DatasetConfig, DatasetConfig]:
    """Get the data configurations to use at training time.
    
    Parameters
    ----------
    image_size : list[int]
        The image size to use for the data.
    num_channels : int
        The number of channels in the data.
    
    Returns
    -------
    tuple[HTLIF24DataConfig, HTLIF24DataConfig]
        The train, validation and test data configurations.
    """
    # if "num_z_slices" in kwargs:
    #     num_z_slices = kwargs["num_z_slices"]
    # else:
    #     raise ValueError("`num_z_slices` must be provided!!!")
    
    patch_overlap = (image_size[0], image_size[1] // 2, image_size[2] // 2)
    
    train_data_config = DatasetConfig(
        data_type=DataType.HTH24Data, # TODO temporary hack
        datasplit_type=DataSplitType.Train,
        image_size=image_size,
        grid_size=patch_overlap,
        num_channels=num_channels,
        multiscale_lowres_count=1,
        depth3D=image_size[0],
        mode_3D=True,
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
        # start_alpha=[0.1] * num_channels,
        # end_alpha=[0.9] * num_channels,
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
