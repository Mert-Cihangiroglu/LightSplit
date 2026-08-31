"""Split-learning forward/backward helper.

Wraps a SplitModel from train_split.py to enforce a true client/server cut at
the smashed payload z. The client owns head + (learned compress, if any) +
nothing else; the server owns (learned decompress, if any) + backbone + tail.
Only z (forward) and grad_z (backward) cross the boundary.

This is NOT a pull request to train_split.py — we do the cut here so train_split
stays untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from train_split import (
    FixedRandomConvBottleneck,
    FixedRandomMLPLiftBottleneck,
    Learned1x1Bottleneck,
    SplitModel,
)


# Bottlenecks whose forward/lift_back operate on spatial (B, C, H, W) tensors.
SPATIAL_BOTTLENECKS = (Learned1x1Bottleneck, FixedRandomConvBottleneck)


def _client_bottleneck_params(bottleneck: nn.Module) -> List[nn.Parameter]:
    """Trainable parameters that belong to the client side of the bottleneck.

    Only Learned1x1Bottleneck.compress is client-owned and learned.
    FixedRandomConvBottleneck.compress is client-owned but fixed (requires_grad=False).
    """
    if isinstance(bottleneck, Learned1x1Bottleneck):
        return [p for p in bottleneck.compress.parameters() if p.requires_grad]
    return []


def _server_bottleneck_params(bottleneck: nn.Module) -> List[nn.Parameter]:
    """Trainable parameters that belong to the server side of the bottleneck."""
    params: List[nn.Parameter] = []
    if isinstance(bottleneck, FixedRandomMLPLiftBottleneck):
        params.extend(p for p in bottleneck.learned_lift.parameters() if p.requires_grad)
    elif isinstance(bottleneck, (Learned1x1Bottleneck, FixedRandomConvBottleneck)):
        params.extend(p for p in bottleneck.decompress.parameters() if p.requires_grad)
    return params


def build_optimizers(
    model: SplitModel,
    lr: float,
    weight_decay: float,
) -> Tuple[torch.optim.Adam, torch.optim.Adam]:
    """Construct (opt_client, opt_server) Adam optimizers with disjoint param sets."""
    client_params: List[nn.Parameter] = []
    if model.client_heads:
        client_params.extend(p for p in model.heads.parameters() if p.requires_grad)
    elif model.head is not None:
        client_params.extend(p for p in model.head.parameters() if p.requires_grad)
    client_params.extend(p for p in model.tail.parameters() if p.requires_grad)
    client_params.extend(_client_bottleneck_params(model.bottleneck))

    server_params: List[nn.Parameter] = []
    server_params.extend(p for p in model.backbone.parameters() if p.requires_grad)
    server_params.extend(_server_bottleneck_params(model.bottleneck))

    opt_client = torch.optim.Adam(client_params, lr=lr, weight_decay=weight_decay)
    opt_server = torch.optim.Adam(server_params, lr=lr, weight_decay=weight_decay)
    return opt_client, opt_server


class SplitForward:
    """Performs the manual client/server forward + backward across the cut at z.

    Usage per batch:
        sf.zero_grad()
        z, info, ctx = sf.client_forward(x, cid)
        loss, logits, r = sf.server_forward(z, info, ctx, y)
        sf.backward_and_step(z, loss)
        # observe ctx['z_obs'] (= z.detach()) and ctx['r_obs'] (= r.detach())
    """

    def __init__(self, model: SplitModel, opt_client, opt_server):
        self.model = model
        self.opt_client = opt_client
        self.opt_server = opt_server

    def zero_grad(self) -> None:
        self.opt_client.zero_grad(set_to_none=True)
        self.opt_server.zero_grad(set_to_none=True)

    def client_forward(
        self, x: torch.Tensor, client_id: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Run head + bottleneck.forward on the client. Returns (z, info, ctx).

        ctx carries the original spatial shape for flat bottlenecks so the server
        can reshape z_approx before the backbone.
        """
        if self.model.client_heads:
            z_spatial = self.model.heads[client_id](x)
        else:
            z_spatial = self.model.head(x)

        if isinstance(self.model.bottleneck, SPATIAL_BOTTLENECKS):
            z, info = self.model.bottleneck(z_spatial)
            ctx = {"is_spatial": True, "spatial_shape": z_spatial.shape}
        else:
            z_flat = z_spatial.flatten(1)
            z, info = self.model.bottleneck(z_flat)
            ctx = {"is_spatial": False, "spatial_shape": z_spatial.shape}

        ctx["z_client"] = z  # the client's tensor with autograd graph
        return z, info, ctx

    def server_forward(
        self,
        z_client: torch.Tensor,
        info: Dict[str, Any],
        ctx: Dict[str, Any],
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cut the graph at z, then run lift_back + backbone + tail + cross_entropy.

        Returns (loss, logits, r, z_server) where z_server is the server-side
        tensor whose .grad will be the message sent back to the client.
        """
        z_server = z_client.detach().clone().requires_grad_(True)

        z_approx = self.model.bottleneck.lift_back(z_server, info)
        if not ctx["is_spatial"]:
            z_approx = z_approx.reshape(ctx["spatial_shape"])

        r = self.model.backbone(z_approx)
        logits = self.model.tail(r)
        loss = F.cross_entropy(logits, y)
        return loss, logits, r, z_server

    def backward_and_step(
        self,
        z_client: torch.Tensor,
        z_server: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Server backward (populates z_server.grad), then client backward
        through z_client using grad_z as the upstream gradient. One .step on each.
        """
        # Server: backward from loss to z_server (and through tail+backbone+decompress)
        loss.backward()
        self.opt_server.step()

        # Client: backprop grad_z through compress + head
        grad_z = z_server.grad
        if grad_z is not None:
            z_client.backward(gradient=grad_z)
        self.opt_client.step()
