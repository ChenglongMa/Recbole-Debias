# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn
import os
from collections import defaultdict
from time import time, strftime

import numpy as np
import pandas as pd
import torch
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, EvaluatorType, get_gpu_usage
from tqdm import tqdm


class DebiasTrainer(Trainer):

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        topk = max(self.config["topk"])
        topk_results = defaultdict(list)
        eval_dataset = eval_data.dataset
        dataset_name = eval_dataset.dataset_name
        uid_field, iid_field, score_field = eval_dataset.uid_field, eval_dataset.iid_field, 'score'
        phase = eval_data._sampler.phase

        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)

            # mcl: added
            if phase == 'test':
                topk_scores, topk_idx = torch.topk(
                    scores, topk, dim=-1
                )  # n_users x k
                user_ids, indices = np.unique(interaction[uid_field], return_index=True)
                user_ids = user_ids[indices.argsort()].repeat(topk)

                topk_results[uid_field] += user_ids.tolist()
                topk_results[iid_field] += topk_idx.cpu().detach().flatten().tolist()
                topk_results[score_field] += topk_scores.cpu().detach().flatten().tolist()
            # end of mcl

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        # mcl: add topk result
        if phase == 'test':
            topk_results[uid_field] = eval_dataset.id2token(eval_dataset.uid_field, topk_results[uid_field])
            topk_results[iid_field] = eval_dataset.id2token(eval_dataset.iid_field, topk_results[iid_field])
            topk_results = pd.DataFrame(topk_results)
            now = strftime("%y%m%d%H%M%S")
            filename = os.path.join(self.config['result_dir'], f'topk_{self.config["model"]}_{dataset_name}_{now}.csv')
            topk_results.to_csv(filename, index=False)
        # mcl: add topk result

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result


class DICETrainer(DebiasTrainer):
    r"""

    """

    def __init__(self, config, model):
        super(DICETrainer, self).__init__(config, model)
        self.decay = config['decay']

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

                Args:
                    train_data (DataLoader): the train data
                    valid_data (DataLoader, optional): the valid data, default: None.
                                                       If it's None, the early_stopping is invalid.
                    verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                    saved (bool, optional): whether to save the model parameters, default: True
                    show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                    callback_fn (callable): Optional callback function executed at end of epoch.
                                            Includes (epoch_idx, valid_score) input arguments.

                Returns:
                     (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
                """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (sum(train_loss) if isinstance(train_loss, tuple) else train_loss)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx}, head="train", )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", "green")
                                      + " ["
                                      + set_color("time", "blue")
                                      + ": %.2fs, "
                                      + set_color("valid_score", "blue")
                                      + ": %f]"
                                      ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (set_color("valid result", "blue") + ": \n" + dict2str(valid_result))
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, "valid_step": valid_step}, head="valid")

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                            epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

            # mcl: added for DICE
            if self.config['adaptive']:
                self.adapt_hyperparams()
            # end of addition

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def adapt_hyperparams(self):
        self.model.adapt(self.decay)
        # self.sampler.adapt()
