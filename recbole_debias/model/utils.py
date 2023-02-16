from typing import Optional

import torch


def length2mask(lengths: torch.Tensor, max_len: Optional[int] = None, dtype=None) -> torch.Tensor:
    """
    Refer to https://stackoverflow.com/a/63187433/8860079
    :param lengths: size can be [B]
    :param max_len:
    :param dtype:
    :return: size -> [B, max_len]
    """
    max_len = max_len or torch.max(lengths)
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask
