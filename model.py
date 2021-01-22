import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_part = nn.Sequential(
            nn.Conv2d(3, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self._fc_part = nn.Sequential(
            nn.Linear(16*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self._pred_part = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self._loss = nn.BCEWithLogitsLoss()

    def get_repr(self, batch):
        feats_in = batch.feats_in
        feats_out = batch.feats_out
        if next(self.parameters()).is_cuda:
            feats_in = feats_in.cuda()
            feats_out = feats_out.cuda()

        n_batch, n_ex, c, w, h = feats_in.shape
        conv_in = self._conv_part(feats_in.view(n_batch * n_ex, c, w, h))
        fc_in = self._fc_part(conv_in.view(n_batch * n_ex, 16*5*5))
        conv_out = self._conv_part(feats_out)
        rep_out = self._fc_part(conv_out.view(n_batch, 16*5*5))
        final_feat = torch.cat([fc_in, rep_out], dim=0)
        return final_feat

    def forward(self, batch):
        feats_in = batch.feats_in
        feats_out = batch.feats_out
        label_out = batch.label_out
        if next(self.parameters()).is_cuda:
            feats_in = feats_in.cuda()
            feats_out = feats_out.cuda()
            label_out = label_out.cuda()

        n_batch, n_ex, c, w, h = feats_in.shape
        conv_in = self._conv_part(feats_in.view(n_batch * n_ex, c, w, h))
        fc_in = self._fc_part(conv_in.view(n_batch * n_ex, 16*5*5))
        predictor = self._pred_part(fc_in.view(n_batch, n_ex, 64).sum(dim=1))

        conv_out = self._conv_part(feats_out)
        rep_out = self._fc_part(conv_out.view(n_batch, 16*5*5))

        score = (predictor * rep_out).sum(dim=1)
        labels = (score > 0).float()
        loss = self._loss(score, label_out)

        return loss, (labels == label_out).float().mean(), labels, predictor