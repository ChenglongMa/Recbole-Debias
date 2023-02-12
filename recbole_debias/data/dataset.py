# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn
import numpy as np
import pandas as pd
import torch
from recbole.data.interaction import Interaction

from recbole.data.dataset import Dataset, SequentialDataset
from recbole.sampler import SeqSampler
from recbole.utils import FeatureType, FeatureSource

from recbole_debias.sampler import DICESampler, MaskedSeqSampler


class DebiasDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.USER_ID = config['USER_ID_FIELD']
        self.n_items = self.num(self.ITEM_ID)
        self.n_users = self.num(self.USER_ID)

        self.pscore_method = config['pscore_method']
        self.eta = config['eta']
        self.device = config['device']

    def estimate_pscore(self):
        r"""
            estimate the propensity score
        """
        interaction_data = self.inter_feat  # interaction for training

        if self.pscore_method == 'item':  # item_id may not be consecutive
            column = 'item_id'
            pscore_id_full = torch.arange(self.n_items)
        elif self.pscore_method == 'user':
            column = 'user_id'
            pscore_id_full = torch.arange(self.n_users)
        elif self.pscore_method == 'nb':  # uniform & explicit feedback
            column = 'rating'
            pscore_id_full = torch.arange(6)
        else:
            raise NotImplementedError(f'Unknown `pscore_method`: {self.pscore_method}')

        pscore = torch.unique(interaction_data[column], return_counts=True)
        pscore_id = pscore[0].tolist()
        pscore_cnt = pscore[1]

        pscore_cnt_full = torch.zeros(pscore_id_full.shape).long()
        pscore_cnt_full[pscore_id] = pscore_cnt

        pscore_cnt_full = pow(pscore_cnt_full / pscore_cnt_full.max(), self.eta)
        pscore_cnt = pscore_cnt_full
        return pscore_cnt.to(self.device), column


class MaskedSequentialDataset(SequentialDataset):
    """:class:`H2NETDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It adds users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored H2NETDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def _get_field_from_config(self):
        super()._get_field_from_config()
        list_suffix = self.config["LIST_SUFFIX"]
        neg_prefix = self.config["NEG_PREFIX"]
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix  # default: neg_item_list
        self.mask_field = self.config['MASK_FIELD']

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        if hasattr(self, 'item_id_list_field'):
            self.inter_feat[self.item_list_length_field] = self.inter_feat[self.item_id_list_field].agg(len)
        else:
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                setattr(self, feat_name, self._dataframe_to_interaction(feat))
            self.data_augmentation()

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        # mcl: added
        # self.seq_sampler = SeqSampler(self)
        seq_sampler = MaskedSeqSampler(self, distribution="uniform", alpha=1.0)
        neg_item_list, neg_item_masks = seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field].numpy())

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        # uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                # H2NET
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                # End H2NET
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # H2NET
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    new_dict[self.mask_field] = torch.zeros(shape, dtype=torch.bool)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = neg_item_list[index]
                        new_dict[self.mask_field][i][:length] = neg_item_masks[index]
                # End H2NET

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
                See :class:`~recbole.config.eval_setting.EvalSetting` for details.

                Args:
                    eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                        Object contains evaluation settings, which guide the data processing procedure.

                Returns:
                    list: List of built :class:`Dataset`.
        """
        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]

            # mcl: save
            for dataset in datasets:
                df = pd.DataFrame(dataset.inter_feat)
            # end save
            return datasets

        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()


H2NETDataset = MaskedSequentialDataset
DIENDataset = MaskedSequentialDataset
GRU4RecDataset = MaskedSequentialDataset
