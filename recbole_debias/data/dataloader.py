# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import copy

import recbole.data
import torch
from recbole.data.dataloader import NegSampleDataLoader
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction
from recbole.utils import ModelType

DebiasDataloader = TrainDataLoader


class DICEDataloader(TrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=True):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.mask_field = config['MASK_FIELD']

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids, mask = self.sampler.sample_by_user_ids(user_ids, item_ids,
                                                                      self.neg_sample_num * candidate_num)
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids, mask)
        elif (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            # neg_item_ids [user_ids.size()*neg_ratio]   mask [True,False.....]
            neg_item_ids, mask = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )  # 更改
            return self.sampling_func(inter_feat, neg_item_ids, mask)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat: Interaction, neg_item_ids, mask):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_mask = Interaction({self.mask_field: mask})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_mask = self._dataset.join(neg_item_mask)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        inter_feat.update(neg_item_mask)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat: Interaction, neg_item_ids, mask):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)

        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))

        neg_item_mask = torch.zeros(pos_inter_num * self.times, dtype=torch.bool)
        neg_item_mask[pos_inter_num:] = mask
        new_data.update(Interaction({self.mask_field: neg_item_mask}))
        return new_data


class NegSampleEvalDataLoader(recbole.data.dataloader.NegSampleEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        super().__init__(config, dataset, sampler, shuffle)


class DICEEvalDataloader(NegSampleEvalDataLoader):
    """
    Duplicate of DICEDataloader
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle)
        self.mask_field = config['MASK_FIELD']

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids, mask = self.sampler.sample_by_user_ids(user_ids, item_ids,
                                                                      self.neg_sample_num * candidate_num)
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids, mask)
        elif (
                self.neg_sample_args["distribution"] != "none"
                and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            # neg_item_ids [user_ids.size()*neg_ratio]   mask [True,False.....]
            neg_item_ids, mask = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )  # 更改
            return self.sampling_func(inter_feat, neg_item_ids, mask)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat: Interaction, neg_item_ids, mask):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_mask = Interaction({self.mask_field: mask})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_mask = self._dataset.join(neg_item_mask)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        inter_feat.update(neg_item_mask)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat: Interaction, neg_item_ids, mask):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)

        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))

        neg_item_mask = torch.zeros(pos_inter_num * self.times, dtype=torch.bool)
        neg_item_mask[pos_inter_num:] = mask
        new_data.update(Interaction({self.mask_field: neg_item_mask}))
        return new_data
