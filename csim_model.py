from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as tfunctional


class PermInvPolicyNet(nn.Module):
    def __init__(self, obs_width: int, cxt_width: int, device: str or torch.device):
        super(PermInvPolicyNet, self).__init__()
        self.obs_width = obs_width
        self.cxt_width = cxt_width
        self.device = device
        modules = [
            nn.Linear(self.obs_width, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        self.mapper = nn.Sequential(*modules)

    def forward(self, observation: Dict[str, torch.Tensor]) -> int:
        context = observation['context']
        options = observation['options']
        assert context.shape[-1] == self.cxt_width
        aux = context.view(1, context.shape[-1]).repeat(options.shape[0], 1)
        obs = torch.cat((options, aux), dim=-1)
        assert obs.shape[-1] == self.obs_width
        raw_pi = self.mapper(obs).squeeze(dim=-1)
        pi_cpu = tfunctional.softmax(raw_pi, dim=-1).cpu()
        return torch.argmax(pi_cpu, dim=-1).item()
