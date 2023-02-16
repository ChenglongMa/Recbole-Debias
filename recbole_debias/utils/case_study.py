# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @email  : chenyushuo@ruc.edu.cn

"""
recbole.utils.case_study
#####################################
"""

import numpy as np
import torch
from recbole.data.dataset import Dataset

from recbole.data.interaction import Interaction
from tqdm import tqdm

from recbole_debias.data import split_interaction


@torch.no_grad()
def full_sort_scores(model, test_data, uid_series=None, device=None, batch_size=None):
    """Calculate the scores of all items for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray or list): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.

    Returns:
        torch.Tensor: the scores of all items for each user in uid_series.
    """
    device = device or torch.device("cpu")
    uid_field = test_data.dataset.uid_field
    if uid_series is None:
        uid_series = torch.arange(1, test_data.dataset.user_num)
    uid_series = torch.as_tensor(uid_series)
    batch_size = batch_size or len(uid_series)
    dataset: Dataset = test_data.dataset
    model.eval()

    if not test_data.is_sequential:
        input_interaction = dataset.join(Interaction({uid_field: uid_series}))
        history_item = test_data.uid2history_item[list(uid_series)]
        history_row = torch.cat(
            [torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)]
        )
        history_col = torch.cat(list(history_item))
        history_index = history_row, history_col
    else:
        if uid_series is not None:
            _, index = (dataset.inter_feat[uid_field] == uid_series[:, None]).nonzero(as_tuple=True)
            input_interaction = dataset[index]
        else:
            input_interaction = dataset.inter_feat
        history_index = None

    # Get scores of all items
    input_interaction = input_interaction.to(device)
    try:
        scores = model.full_sort_predict(input_interaction)
    except NotImplementedError:
        scores = []
        iter_data = tqdm(
            split_interaction(input_interaction, batch_size=batch_size),
            total=dataset.item_num * len(uid_series),
            ncols=100,
            desc="Full sort predict",
        )
        for batch_data in iter_data:
            n_users = len(batch_data)  # process len(batch_data) users in each epoch
            batch_data = batch_data.repeat_interleave(dataset.item_num)
            batch_data.update(test_data.dataset.get_item_feature().to(device).repeat(n_users))
            scores.append(model.predict(batch_data))
        scores = torch.cat(scores)

    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    if history_index is not None:
        scores[history_index] = -np.inf  # set scores of history items to -inf

    return scores


def _spilt_predict(interaction, batch_size):
    spilt_interaction = dict()
    for key, tensor in interaction.interaction.items():
        spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
    num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
    result_list = []
    for i in range(num_block):
        current_interaction = dict()
        for key, spilt_tensor in spilt_interaction.items():
            current_interaction[key] = spilt_tensor[i]
        result = self.model.predict(
            Interaction(current_interaction).to(self.device)
        )
        if len(result.shape) == 0:
            result = result.unsqueeze(0)
        result_list.append(result)
    return torch.cat(result_list, dim=0)


def full_sort_topk(model, test_data, uid_series=None, k=10, device=None, batch_size=None):
    """Calculate the top-k items' scores and ids for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray, optional): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        k (int): The top-k items.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.
        batch_size (int, optional): number of users to be processed in each epoch

    Returns:
        tuple:
            - topk_scores (torch.Tensor): The scores of topk items.
            - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
    """
    scores = full_sort_scores(model, test_data, uid_series, device, batch_size)
    return torch.topk(scores, k, largest=True, sorted=True)
