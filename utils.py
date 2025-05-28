import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_gsnr(task_gradients, args):
    grads = torch.cat([g.flatten() for g in task_gradients])

    # mean_gradient = torch.mean(grads, dim=0)

    epsilon = 1e-5

    grad_squared = grads ** 2

    # import pdb
    # pdb.set_trace()

    sum_grad_squared = [(g ** 2) for g in task_gradients]

    sum_grad_squared = torch.cat([sgs.flatten() for sgs in sum_grad_squared])

    # batch_size = args.train_batchsize

    GSNR = grad_squared / (sum_grad_squared - grad_squared + epsilon)

    # variance = torch.mean((grads - mean_gradient)**2, dim=0)

    # 计算梯度的平方均值（(E[g])^2）
    # mean_gradient_squared = torch.mean(mean_gradient**2)

    # # 计算梯度的方差的均值（E[Var[g]]）
    # mean_variance = torch.mean(variance)

    # 计算 GSNR
    # GSNR = mean_gradient_squared / mean_variance

    return GSNR


class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, device, total_num_inner_loop_steps, init_learning_rate=1e-5, use_learnable_lr=True,
                 lr_of_lr=1e-3, use_learnable_drop_p=True, top_p=0.8):
        super(LSLRGradientDescentLearningRule, self).__init__()
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_lr = use_learnable_lr
        self.lr_of_lr = lr_of_lr
        self.use_learnable_drop_p = use_learnable_drop_p
        self.top_p = torch.tensor(top_p, requires_grad=True)

    def initialize(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        # import pdb
        # pdb.set_trace()
        for key, param in names_weights_dict:
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate,
                requires_grad=self.use_learnable_lr)

    def update_lrs(self, loss, scaler=None, args=None):
        if self.use_learnable_lr:
            if scaler is not None:
                scaled_grads = torch.autograd.grad(scaler.scale(loss), self.names_learning_rates_dict.values())
                inv_scale = 1. / scaler.get_scale()
                grads = [p * inv_scale for p in scaled_grads]
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, adjust scale and zero out gradients')
                    if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                        scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                    for g in grads: g.zero_()
            else:
                grads = torch.autograd.grad(loss, self.names_learning_rates_dict.values())
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, zero out gradients')
                    for g in grads: g.zero_()

            Mean_grads_GSNR = []
            for grad in grads:
                grad_GSNR = compute_gsnr(grad, args)
                # Mean_grad_GSNR = torch.mean(torch.tensor(grad_GSNR))
                Mean_grad_GSNR = torch.mean(grad_GSNR.clone().detach())
                Mean_grads_GSNR.append(Mean_grad_GSNR)

            k = int(len(Mean_grads_GSNR) * self.top_p)
            # # # if Mean_grads_GSNR[idx]<topk_gsnr
            topk_gsnr = sorted(Mean_grads_GSNR)[k]

            # import pdb
            # pdb.set_trace()
            # Mean_grads_GSNR = torch.stack(Mean_grads_GSNR)

            # sorted_values, sorted_indices = torch.sort(Mean_grads_GSNR, descending=True)

            # ranks = torch.argsort(torch.argsort(Mean_grads_GSNR, stable=True), descending=True)

            # learning_rate_max = 1e-5 + 8e-6
            # learning_rate_min = 1e-5
            self.lr_of_lr = 5e-6

            for idx, key in enumerate(self.names_learning_rates_dict.keys()):
                if Mean_grads_GSNR[idx] > topk_gsnr:
                    # self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx] * ranks[idx]/len(ranks))
                    self.names_learning_rates_dict[key] = nn.Parameter(
                        self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx])
                    # self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx] + (learning_rate_max-learning_rate_min)* (ranks[idx]/len(ranks)))
                    # self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] + (learning_rate_max-learning_rate_min)* (ranks[idx]/len(ranks)))

    def update_drop_p(self, loss, scaler=None, args=None):
        if self.use_learnable_drop_p:
            if scaler is not None:

                scaled_grads = torch.autograd.grad(scaler.scale(loss), self.drop_p)
                inv_scale = 1. / scaler.get_scale()

                grads = [p * inv_scale for p in scaled_grads]
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, adjust scale and zero out gradients')
                    if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                        scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                    for g in grads: g.zero_()
            else:
                grads = torch.autograd.grad(loss, self.names_learning_rates_dict.values())
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, zero out gradients')
                    for g in grads: g.zero_()

            # Mean_grads_GSNR = []
            # for grad in grads:
            #     grad_GSNR = compute_gsnr(grad,args)
            #     # Mean_grad_GSNR = torch.mean(torch.tensor(grad_GSNR))
            #     Mean_grad_GSNR = torch.mean(grad_GSNR.clone().detach())
            #     Mean_grads_GSNR.append(Mean_grad_GSNR)

            # import pdb
            # pdb.set_trace()

            self.drop_p = self.drop_p - - self.lr_of_lr * grads[idx]

            # for idx, key in enumerate(self.names_learning_rates_dict.keys()):
            #     # import pdb
            #     # pdb.set_trace()
            #     self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx])  # 201

    def update_params(self, names_weights_dict, grads, num_step):
        # import pdb
        # pdb.set_trace()
        return OrderedDict(
            (
            key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
            for idx, key in enumerate(names_weights_dict.keys()))

    def update_params_perturb(self, names_weights_dict, grads, num_step, args):
        # import pdb
        # pdb.set_trace()
        Mean_grads_GSNR = []
        for grad in grads:
            grad_GSNR = compute_gsnr(grad, args)
            # Mean_grad_GSNR = torch.mean(torch.tensor(grad_GSNR))
            Mean_grad_GSNR = torch.mean(grad_GSNR.clone().detach())
            Mean_grads_GSNR.append(Mean_grad_GSNR)

        # pdb.set_trace()
        k = int(len(Mean_grads_GSNR) * self.drop_p)
        # # if Mean_grads_GSNR[idx]<topk_gsnr
        topk_gsnr = sorted(Mean_grads_GSNR)[k]

        # Mean_grads_GSNR = F.softmax(torch.tensor(Mean_grads_GSNR) / args.softmax_temp, -1)

        new_ordered_dict = OrderedDict()
        for idx, key in enumerate(names_weights_dict.keys()):
            # 检查条件是否满足
            if Mean_grads_GSNR[idx] < topk_gsnr:
                # 替换键中的点为下划线
                # replaced_key = key.replace(".", "-")

                # 计算新的值
                new_value = names_weights_dict[
                    key]  # - self.names_learning_rates_dict[replaced_key][num_step] * grads[idx]
                # new_value = names_weights_dict[key] - 1e-5 * grads[idx]
                # 将新的键值对添加到OrderedDict中
                new_ordered_dict[key] = new_value

            else:
                # replaced_key = key.replace(".", "-")
                # # 计算新的值
                # new_value = names_weights_dict[key] - self.names_learning_rates_dict[replaced_key][num_step] * grads[idx]
                # new_value = names_weights_dict[key] - 9e-5 * grads[idx]
                # 将新的键值对添加到OrderedDict中
                # import pdb
                # pdb.set_trace()
                # new_ordered_dict[key] = torch.zeros_like(names_weights_dict[key]) # names_weights_dict[key].zero_()
                new_ordered_dict[key] = names_weights_dict[key] * 0  # names_weights_dict[key].zero_()
        return new_ordered_dict

    def update_params_modify(self, names_weights_dict, grads, num_step, args):
        # import pdb

        Mean_grads_GSNR = []
        for grad in grads:
            grad_GSNR = compute_gsnr(grad, args)
            # Mean_grad_GSNR = torch.mean(torch.tensor(grad_GSNR))
            Mean_grad_GSNR = torch.mean(grad_GSNR.clone().detach())
            Mean_grads_GSNR.append(Mean_grad_GSNR)

        # pdb.set_trace()
        k = int(len(Mean_grads_GSNR) * 0.8)
        # # if Mean_grads_GSNR[idx]<topk_gsnr
        topk_gsnr = sorted(Mean_grads_GSNR)[k]

        # Mean_grads_GSNR = F.softmax(torch.tensor(Mean_grads_GSNR) / args.softmax_temp, -1)

        new_ordered_dict = OrderedDict()
        for idx, key in enumerate(names_weights_dict.keys()):
            # 检查条件是否满足
            if Mean_grads_GSNR[idx] < topk_gsnr:
                # 替换键中的点为下划线
                replaced_key = key.replace(".", "-")
                # import pdb
                # pdb.set_trace()
                # 计算新的值
                new_value = names_weights_dict[key] - self.names_learning_rates_dict[replaced_key][num_step] * grads[
                    idx]
                # new_value = names_weights_dict[key] - 1e-5 * grads[idx]
                # 将新的键值对添加到OrderedDict中
                new_ordered_dict[key] = new_value

            else:
                replaced_key = key.replace(".", "-")
                # 计算新的值
                new_value = names_weights_dict[key] - self.names_learning_rates_dict[replaced_key][num_step] * grads[
                    idx]
                # new_value = names_weights_dict[key] - 9e-5 * grads[idx]
                # 将新的键值对添加到OrderedDict中
                new_ordered_dict[key] = 0
        return new_ordered_dict

        # return OrderedDict(
        #     (key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
        #     for idx, key in enumerate(names_weights_dict.keys()) if Mean_grads_GSNR[idx]<topk_gsnr)