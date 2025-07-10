import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDModule(nn.Module):
    """Random Network Distillation used for intrinsic reward."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state: torch.Tensor):
        with torch.no_grad():
            target_output = self.target(state)
        pred_output = self.predictor(state)
        intrinsic_reward = F.mse_loss(pred_output, target_output, reduction="none").mean(dim=1)
        return intrinsic_reward, pred_output, target_output
