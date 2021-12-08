import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))
 

class CrossEntropyLoss(nn.Module):
    def forward(self, pred, label):
        if pred.shape[1] == 1:
            cross_entropy_loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='none')
        else:
            cross_entropy_loss = F.cross_entropy(pred, torch.argmax(label, dim=1), reduction='none')
        return cross_entropy_loss.mean()
    
def FocalTverskyLoss(inputs, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=1, ignore=None, activation=True):
    if activation:
        if inputs.shape[1] == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = torch.softmax(inputs, dim=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp

    ft_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft_loss = (1 - ft_score) ** gamma

    return ft_loss.mean()


class NeoUNetLoss(nn.Module):
    __name__ = 'neounet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        loss = 0
        neo_gt = mask[:, [0], ...]
        non_gt = mask[:, [1], ...]
        polyp_gt = neo_gt + non_gt
        cross_entropy_loss = CrossEntropyLoss()

        # tinh loss theo 4 tang dau ra 
        for y_pr in y_prs:
            neo_pr = y_pr[:, [0], ...]
            non_pr = y_pr[:, [1], ...]
          
            polyp_pr = (neo_pr > non_pr) * neo_pr + (non_pr > neo_pr) * non_pr
            
            aux_loss = cross_entropy_loss(polyp_pr, polyp_gt) + FocalTverskyLoss(polyp_pr, polyp_gt)
            
            main_loss = (cross_entropy_loss(neo_pr, neo_gt) + FocalTverskyLoss(neo_pr, neo_gt) +\
                          cross_entropy_loss(non_pr, non_gt) + FocalTverskyLoss(non_pr, non_gt))
            loss += 0.75 * main_loss + 0.25 * aux_loss

        return loss / 4

        
def dice_score(input, target): # input: 16x2x352x352 target 16x2x352x352
    batch_size = target.size(0)
    dice_score = 0 
    smooth = 1.
    neo_tflat = target[:, [0], ...]
    non_neo_tflat = target[:, [1], ...]
    seg_tflat = (neo_tflat + non_neo_tflat).view(batch_size, -1)
    for pre in input:
        neo_iflat = torch.sigmoid(pre[:, [0], ...])
        non_neo_iflat = torch.sigmoid(pre[:, [1], ...])
        seg_iflat = ((neo_iflat > non_neo_iflat) * neo_iflat + (non_neo_iflat > neo_iflat) * non_neo_iflat).view(batch_size, -1)
        intersection_neo = (seg_iflat * seg_tflat).sum(axis=1)

        dice_score += (((2. * intersection_neo + smooth) /
                  (seg_iflat.sum(axis=1) + seg_tflat.sum(axis=1) + smooth)) / len(input)).mean()

    return dice_score