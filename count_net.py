"""CountNet: a Layer I copy trained end to end for the number of fixed points only.

Layer I's count head reads frozen, detached features; on composition branches it saturates
around 78% for k = 3 and 54% for k = 5 however the head is retrained. CountNet starts from
the Layer I weights with count_detach = False, so the count loss also trains the encoder and
the slot decoder. At inference it only replaces count_logit: slots, positions and stability
still come from the original Layer I, so fixed-point positions cannot regress.

Under constant forcing a forced SCC is an autonomous SCC with production P * u, so one
CountNet serves root and forced SCCs (InferenceOptions(forced_model="l1", count_model=net)).
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from model import SccFixedPointNet


class CountNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = SccFixedPointNet(cfg)
        self.net.count_detach = False

    def forward(self, batch):
        """count_logit [B, max_fp + 1] (all counts; odd masking is the caller's)."""
        return self.net(batch)["count_logit"]


def load_count_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = CountNet(SimpleNamespace(**ck["cfg"]))
    net.load_state_dict(ck.get("ema", ck["model"]))
    return net.to(device).eval()
