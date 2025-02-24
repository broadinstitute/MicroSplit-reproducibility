from ._base import SplittingParameters


def get_microsplit_parameters(
    img_size: tuple[int, ...],
    target_channels: int = 2,
    multiscale_count: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    lr_scheduler_patience: int = 10,
    earlystop_patience: int = 200,
    num_epochs: int = 50,
    num_workers: int = 4,
    mmse_count: int = 50,
    grid_size: int = 32,
) -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=img_size,
        target_channels=target_channels,
        multiscale_count=multiscale_count,
        predict_logvar="pixelwise",
        loss_type="musplit",
        kl_type="kl_restricted",
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        num_workers=num_workers,
        lr_scheduler_patience=lr_scheduler_patience,
        earlystop_patience=earlystop_patience,
        mmse_count=mmse_count,
        grid_size=grid_size,
    ).model_dump()


def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for HT_LIF24.")
    return SplittingParameters().model_dump()
