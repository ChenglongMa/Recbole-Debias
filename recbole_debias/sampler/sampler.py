# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import numpy as np
import torch
import copy
from collections import Counter

from recbole.sampler.sampler import AbstractSampler, SeqSampler


class DICESampler(AbstractSampler):
    """
    Args:
        phases (str or list of str): All the phases of input.
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, datasets, distribution='uniform', alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(f'Phases {phases} and datasets {datasets} should have the same length.')

        self.phases = phases
        self.datasets = datasets

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.user_num = datasets[0].user_num
        self.item_num = datasets[0].item_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _get_candidates_list(self):
        """
        Copy from class Sampler(AbstractSampler)
        """
        candidates_list = []
        for dataset in self.datasets:
            candidates_list.extend(dataset.inter_feat[self.iid_field].numpy())
        return candidates_list

    def get_used_ids(self):
        """
        Copy from class Sampler(AbstractSampler)
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def set_phase(self, phase):
        """
        Copy from class Sampler(AbstractSampler)

        Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """
        -------------
        Copy from class Sampler(AbstractSampler)
        This one added `item_ids`
        -------------
        Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, item_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')

    def sample_by_key_ids(self, key_ids, item_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        item_ids_repeat = np.tile(item_ids, num)  # mcl: added

        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids, pop_mask_ids = self.sampling(total_num, item_ids_repeat)  # mcl: added
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value, pop_mask = self.sampling(len(check_list), item_ids_repeat[check_list])  # mcl: added
                value_ids[check_list] = value
                pop_mask_ids[check_list] = pop_mask_ids  # TODO: pop_mask?
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            pop_mask_ids = np.zeros(total_num, dtype=bool)  # mcl: added
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list], pop_mask_ids[check_list] = self.sampling(len(check_list),
                                                                                item_ids_repeat[check_list])  # size一致   # mcl: added
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids), torch.tensor(pop_mask_ids)  # mcl: added

    def sampling(self, sample_num, positive_item_ids):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        assert sample_num == len(positive_item_ids), "The number of samples must be equal to the number of positive samples"
        if self.distribution == 'popularity':
            return self._pop_sampling(sample_num, positive_item_ids)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def _pop_sampling(self, sample_num, positive_item_ids):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = np.array(list(self.prob.keys()))
        random_index_list = np.random.randint(0, len(keys), sample_num)  # 随机产生total个item_id [1,2,3,4....]

        final_random_list = keys[random_index_list]
        pop_mask_list = np.array([self.prob[i] for i in positive_item_ids]) >= np.array([self.prob[i] for i in final_random_list])

        return final_random_list, pop_mask_list

    def sample_neg_sequence(self, pos_sequence):
        """
        -------------------
        Copy from class SeqSampler(AbstractSampler)
        -------------------
        For each moment, sampling one item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)
        value_ids = np.zeros(total_num, dtype=np.int64)
        pop_mask_ids = np.zeros(total_num, dtype=bool)  # mcl: added
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list], pop_mask_ids[check_list] = self.sampling(len(check_list), pos_sequence[check_list])  # mcl: added
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids), torch.tensor(pop_mask_ids)  # mcl: added


class MaskedSeqSampler(SeqSampler):

    def __init__(self, dataset, distribution="uniform", alpha=1.0):
        super().__init__(dataset, distribution, alpha)
        if distribution == "uniform":
            # NOTE: We build alias table regardless of the distribution.
            # We set this `if` condition is because `AbstractSampler` has built alias table for `popularity` distribution
            self._build_alias_table()

    def _get_candidates_list(self):
        """
        Copy from class RepeatableSampler(AbstractSampler):
        TODO: limit the count up to current timestamp
        """
        return self.dataset.inter_feat[self.iid_field].tolist()

    def _uni_sampling(self, sample_num, positive_item_ids):
        final_random_list = np.random.randint(1, self.item_num, sample_num)
        pop_mask_list = np.array([self.prob[i] for i in positive_item_ids]) >= np.array([self.prob[i] for i in final_random_list])
        return final_random_list, pop_mask_list

    def _pop_sampling(self, sample_num, positive_item_ids):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = np.array(list(self.prob.keys()))
        random_index_list = np.random.randint(0, len(keys), sample_num)  # 随机产生total个item_id [1,2,3,4....]

        final_random_list = keys[random_index_list]
        pop_mask_list = np.array([self.prob[i] for i in positive_item_ids]) >= np.array([self.prob[i] for i in final_random_list])

        return final_random_list, pop_mask_list

    def sampling(self, sample_num, positive_item_ids):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        assert sample_num == len(positive_item_ids), "The number of samples must be equal to the number of positive samples"
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num, positive_item_ids)
        if self.distribution == 'popularity':
            return self._pop_sampling(sample_num, positive_item_ids)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def sample_neg_sequence(self, pos_sequence):
        """
        -------------------
        Copy from class SeqSampler(AbstractSampler)
        -------------------
        For each moment, sampling one item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)
        value_ids = np.zeros(total_num, dtype=np.int64)
        pop_mask_ids = np.zeros(total_num, dtype=bool)  # mcl: added
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list], pop_mask_ids[check_list] = self.sampling(len(check_list), pos_sequence[check_list])  # mcl: added
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids), torch.tensor(pop_mask_ids)  # mcl: added
