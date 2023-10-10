import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy

from utils import import_attr
from .rl import compute_returns


class FreeRecallSumMSELoss(nn.Module):
    def __init__(self, var_weight=1.0) -> None:
        super().__init__()
        self.var_weight = var_weight
    
    def forward(self, output, gt):
        loss = torch.sum((torch.sum(output, dim=0) - torch.sum(gt, dim=0)) ** 2)
        gt_argmax = torch.argmax(gt, dim=2)
        for i in range(gt_argmax.shape[1]):
            loss -= torch.sum(torch.var(output[:, i, gt_argmax[:, i]], dim=0) / (torch.mean(output[:, i, gt_argmax[:, i]], dim=0) 
                + torch.finfo(torch.float32).eps)) * self.var_weight
        return loss
    

class FreeRecallSumMSEReciprocalVarLoss(nn.Module):
    def __init__(self, var_weight=1.0) -> None:
        super().__init__()
        self.var_weight = var_weight
    
    def forward(self, output, gt):
        loss = torch.sum((torch.sum(output, dim=0) - torch.sum(gt, dim=0)) ** 2)
        gt_argmax = torch.argmax(gt, dim=2)
        for i in range(gt_argmax.shape[1]):
            loss += 1.0 / torch.sum(torch.var(output[:, i, gt_argmax[:, i]], dim=0) / (torch.mean(output[:, i, gt_argmax[:, i]], dim=0) 
                + torch.finfo(torch.float32).eps)) * self.var_weight
        return loss


class FreeRecallSumMSEMultipleOutputLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5]) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
        # if loss == "FreeRecallSumMSELoss":
        #     self.loss_class = FreeRecallSumMSELoss(var_weight=var_weight)
        # elif loss == "FreeRecallSumMSEReciprocalVarLoss":
        #     self.loss_class = FreeRecallSumMSEReciprocalVarLoss(var_weight=var_weight)
        # else:
        #     raise AttributeError("invalid loss")
    
    def forward(self, output, gt):
        loss = 0.0
        assert len(output) == len(self.output_weight)
        for i in range(len(output)):
            loss += self.loss_class(output[i], gt) * self.output_weight[i]
        return loss


class FreeRecallSumMSETrainEncodeLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5], encode_weight=1.0, only_encode=False) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.encode_weight = encode_weight
        self.only_encode = only_encode
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
        # if loss == "FreeRecallSumMSELoss":
        #     self.loss_class = FreeRecallSumMSELoss(var_weight=var_weight)
        # elif loss == "FreeRecallSumMSEReciprocalVarLoss":
        #     self.loss_class = FreeRecallSumMSEReciprocalVarLoss(var_weight=var_weight)
        # else:
        #     raise AttributeError("invalid loss")
    
    def forward(self, output, gt):
        loss = 0.0
        if len(output) == 2:
            free_recall_output_index = [0]
            encode_output_index = 1
        elif len(output) == 4:
            free_recall_output_index = [0, 1]
            encode_output_index = 2
        else:
            raise AttributeError("output length must be 2 or 4")
        if not self.only_encode:
            for i in free_recall_output_index:
                loss += self.loss_class(output[i], gt) * self.output_weight[i]
        loss += mse_loss(output[encode_output_index], gt) * self.encode_weight
        return loss


class FreeRecallSumMSETrainEncodeTwoDecisionLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5], encode_weight=1.0, only_encode=False) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.encode_weight = encode_weight
        self.only_encode = only_encode
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
        # if loss == "FreeRecallSumMSELoss":
        #     self.loss_class = FreeRecallSumMSELoss(var_weight=var_weight)
        # elif loss == "FreeRecallSumMSEReciprocalVarLoss":
        #     self.loss_class = FreeRecallSumMSEReciprocalVarLoss(var_weight=var_weight)
        # else:
        #     raise AttributeError("invalid loss")
    
    def forward(self, output, gt):
        loss = 0.0
        if len(output) == 6:
            if not self.only_encode:
                for i in range(2):
                    loss += self.loss_class(output[i], gt) * self.output_weight[i]
            loss += mse_loss(output[3], gt) * self.encode_weight
            loss += mse_loss(output[5][1:], gt[:-1]) * self.encode_weight
        if len(output) == 4:
            if not self.only_encode:
                loss += self.loss_class(output[0], gt) * self.output_weight[0]
            loss += mse_loss(output[2], gt) * self.encode_weight
            loss += mse_loss(output[3][1:], gt[:-1]) * self.encode_weight
        return loss


class FreeRecallSumCETrainEncodeLoss(nn.Module):
    def __init__(self, loss="FreeRecallSumMSELoss", var_weight=1.0, output_weight=[0.5, 0.5], encode_weight=1.0, only_encode=False) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.output_weight = output_weight
        self.encode_weight = encode_weight
        self.only_encode = only_encode
        self.loss_class = import_attr("train.criterions.{}".format(loss))(var_weight=var_weight)
        # if loss == "FreeRecallSumMSELoss":
        #     self.loss_class = FreeRecallSumMSELoss(var_weight=var_weight)
        # elif loss == "FreeRecallSumMSEReciprocalVarLoss":
        #     self.loss_class = FreeRecallSumMSEReciprocalVarLoss(var_weight=var_weight)
        # else:
        #     raise AttributeError("invalid loss")
    
    def forward(self, output, gt):
        loss = 0.0
        if len(output) == 2:
            free_recall_output_index = [0]
            encode_output_index = 1
        elif len(output) == 4:
            free_recall_output_index = [0, 1]
            encode_output_index = 2
        else:
            raise AttributeError("output length must be 2 or 4")
        if not self.only_encode:
            for i in free_recall_output_index:
                loss += self.loss_class(output[i], gt) * self.output_weight[i]
        loss += cross_entropy(output[encode_output_index].reshape(-1, output[encode_output_index].shape[-1]), torch.argmax(gt, dim=2).reshape(-1)) * self.encode_weight
        return loss


# class FreeRecallSumMSEValueLoss(nn.Module):
#     def __init__(self, normalize=True) -> None:
#         super().__init__()
#         self.normalize = normalize

#     def forward(self, output, gt, values, rewards, device):
#         loss = torch.sum((torch.sum(output, dim=0) - torch.sum(gt, dim=0)) ** 2)
#         gt_argmax = list(torch.argmax(gt, dim=1).cpu().numpy())
#         loss -= torch.sum(torch.var(output[:, gt_argmax], dim=0))

#         returns = compute_returns(rewards, normalize=self.normalize)
#         value_losses = []
#         for v_t, R_t in zip(values, returns):
#             value_losses.append(0.5 * mse_loss(torch.squeeze(v_t.to(device)), torch.squeeze(R_t.to(device))))
#         value_loss = torch.stack(value_losses).mean()
#         loss += 5e-2 * value_loss

#         return loss


# class FreeRecallMSELoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, outputs, gt):
#         # rand_index = np.arange(0, gt.shape[0])
#         # np.random.shuffle(rand_index)
#         # return torch.mean((outputs - gt[rand_index]) ** 2)
#         gt_argmax = torch.argmax(gt, dim=1)
#         outputs_argmax = torch.argmax(outputs, dim=1)
#         reordered_gt = torch.zeros(gt.size()).to(gt.device)
#         recalled = torch.zeros(gt.shape[0])
#         answer_correct = torch.zeros(gt.shape[0])
#         for i in range(outputs.shape[0]):
#             for j in range(gt.shape[0]):
#                 if outputs_argmax[i] == gt_argmax[j] and recalled[j] == 0:
#                     reordered_gt[i] = gt[j]
#                     recalled[j] = 1
#                     answer_correct[i] = 1
#                     break
#         not_recalled = []
#         not_correct = []
#         for i in range(gt.shape[0]):
#             if not recalled[i]:
#                 not_recalled.append(i)
#         for i in range(outputs.shape[0]):
#             if not answer_correct[i]:
#                 not_correct.append(i)
#         not_recalled = np.array(not_recalled)
#         not_correct = np.array(not_correct)
#         # np.random.shuffle(not_correct)
#         reordered_gt[not_correct] = gt[not_recalled]
#         # print(outputs_argmax, gt_argmax, torch.argmax(reordered_gt, dim=1))
#         return torch.mean((outputs - reordered_gt) ** 2)


# class FreeRecallL1Loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, outputs, gt):
#         allowed_targets = []
#         gt_argmax = list(torch.argmax(gt, dim=1).cpu().numpy())
#         allowed_targets.append(deepcopy(gt_argmax))
#         for i in range(outputs.shape[0]-1):
#             choice = torch.argmax(outputs[i])
#             if choice in gt_argmax:
#                 gt_argmax.pop(gt_argmax.index(choice))
#             allowed_targets.append(deepcopy(gt_argmax))

#         normalized_outputs = outputs / torch.sum(outputs, dim=1, keepdim=True)
#         loss = 0
#         # print(allowed_targets)
#         for i in range(outputs.shape[0]):
#             loss -= torch.sum(normalized_outputs[i][allowed_targets[i]])
#         loss += 1e-1 * torch.sum(torch.abs(outputs))
#         return loss


# class FreeRecallL1SubtractionLoss(nn.Module):
#     """
#     sum of allowed targets - sum of others
#     """
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, outputs, gt):
#         allowed = torch.zeros(outputs.shape)
#         allowed_targets = []
#         gt_argmax = list(torch.argmax(gt, dim=1).cpu().numpy())
#         allowed_targets.append(deepcopy(gt_argmax))
#         for i in range(outputs.shape[0]-1):
#             choice = torch.argmax(outputs[i])
#             if choice in gt_argmax:
#                 gt_argmax.pop(gt_argmax.index(choice))
#             allowed_targets.append(deepcopy(gt_argmax))

#         normalized_outputs = outputs / torch.sum(outputs, dim=1, keepdim=True)
#         # print(allowed_targets)
#         for i in range(outputs.shape[0]):
#             allowed[i][allowed_targets[i]] = 1
#         loss = torch.mean(normalized_outputs[allowed == 0]) - torch.mean(normalized_outputs[allowed == 1])
#         loss += 1e-2 * torch.sum(torch.abs(outputs))
#         return loss
