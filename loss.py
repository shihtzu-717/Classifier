import torch
import torch.nn as nn


class softLabelLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.CEloss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.L1loss = nn.L1Loss()

    def partial_onehot(self, target, size=2):
        onehot = torch.zeros(len(target), size).to(target.device)
        for i, t in enumerate(target):
            if t == 2:
                onehot[i][0] = 0.9
                onehot[i][1] = 0.1
            elif t == 3:
                onehot[i][0] = 0.1
                onehot[i][1] = 0.9
            elif t == 0:
                onehot[i][0] = 0.6
                onehot[i][1] = 0.4
            elif t == 1:
                onehot[i][0] = 0.6
                onehot[i][1] = 0.4
        return onehot

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        leninp = input.shape
        lentar = torch.nn.functional.one_hot(target).shape

        if leninp == lentar:
            loss = self.CEloss(input, target)
        else:
            onehot = self.partial_onehot(target)
            loss = self.L1loss(input, onehot)
    
        return loss