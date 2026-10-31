import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = temperatures == 0
        safe_temperatures = torch.where(greedy, 1.0, temperatures)
        logits = logits.float().div_(safe_temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        sampled = (
            probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sampled)
