import os

from ._base import SplittingParameters


def _get_nm_paths(
    nm_path: str, 
) -> list[str]:
    nm_paths = []
    for channel_idx in range(2):
        fname = f"noise_model_Ch{channel_idx}.npz"
        nm_paths.append(os.path.join(nm_path,fname))
    return nm_paths


def get_microsplit_parameters(
    nm_path: str,
    batch_size: int = 32,
) -> dict:
    nm_paths = _get_nm_paths(nm_path=nm_path)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(9, 64, 64),
        target_channels=2,
        multiscale_count=1,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
        batch_size=batch_size,
    ).model_dump()


def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for HT_LIF24.")
    return SplittingParameters().model_dump()