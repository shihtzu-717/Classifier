import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class softLabelLoss(nn.Module):
    def __init__(self, args, mixup_fn) -> None:
        super().__init__()
        lossfn = args.lossfn
        self.ratio = args.soft_label_ratio
        self.label_ratio = args.label_ratio
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            if lossfn == 'BCE':
                self.criterion = nn.BCEWithLogitsLoss()
            elif lossfn == 'CE':
                self.criterion = nn.CrossEntropyLoss()
            elif lossfn == 'L1':
                self.criterion = nn.L1Loss()
            elif lossfn == 'MSE':
                self.criterion = nn.MSELoss()

    def partial_onehot(self, target, size=2):
        onehot = torch.zeros(len(target), size).to(target.device)
        for i, t in enumerate(target):
            if t == 2:
                onehot[i][0] = self.label_ratio
                onehot[i][1] = 1-self.label_ratio
            elif t == 3:
                onehot[i][0] = 1-self.label_ratio
                onehot[i][1] = self.label_ratio
            elif t == 0:
                onehot[i][0] = self.ratio
                onehot[i][1] = 1-self.ratio
            elif t == 1:
                onehot[i][0] = 1-self.ratio
                onehot[i][1] = self.ratio
        return onehot

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        leninp = input.shape
        lentar = torch.nn.functional.one_hot(target).shape

        if leninp != lentar:
            target = self.partial_onehot(target)
        else:
            target = torch.nn.functional.one_hot(target).float()
        loss = self.criterion(input, target)
        return loss