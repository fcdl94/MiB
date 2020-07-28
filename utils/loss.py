import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
