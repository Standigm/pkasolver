import logging

import torch
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


# PyG Dataset to Dataloader
def dataset_to_dataloader(
    data: list, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Take a PyG Dataset and return a Dataloader object. batch_size must be defined. Optional shuffle (highly discouraged) can be disabled.
    ----------
    data
        list of PyG Paired Data
    batch_size
        size of the batches set in the Dataloader function
    shuffle
        if true: shuffles the order of data in every molecule during training to prevent overfitting
    Returns
    -------
    DataLoader
        input object for training PyG Modells
    """
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, follow_batch=["x_p", "x_d"]
    )


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logger.warning("CUDA is not available, using CPU instead")
            return torch.device("cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device {device_str}")
