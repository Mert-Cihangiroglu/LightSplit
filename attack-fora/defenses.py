"""Client-side defenses bolted onto FORA's target_bottom.

Three Bottleneck classes are verbatim copies from
utility/train_split.py (Learned1x1Bottleneck,
FixedRandomBottleneck). DefenseWrapper presents them under a uniform
spatial-in / spatial-out signature so they can be appended to the client
model via nn.Sequential without any change to the rest of the FORA stack.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn




class Learned1x1Bottleneck(nn.Module):
    def __init__(self, in_channels: int, k_channels: int):
        super().__init__()
        self.k_channels = k_channels
        self.compress = nn.Conv2d(in_channels, k_channels, 1)
        self.decompress = nn.Conv2d(k_channels, in_channels, 1)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.compress(z_spatial), {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decompress(payload)




class FixedRandomBottleneck(nn.Module):
    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


# Honest transmitted-byte accounting for sparse payloads:
# each kept coordinate costs a 4-byte float32 value + a 2-byte uint16 index.
# (uint16 index covers activation_dim up to 65535; max d here is 64*16*16=16384.)
VALUE_BYTES = 4
INDEX_BYTES = 2


class TopKSparseBottleneck(nn.Module):
    """True magnitude-based top-k sparsification (matched-byte compression baseline).

    Keeps the k_s largest-magnitude coordinates per sample and zeroes the rest. The
    payload stays dense (d-dim) so DefenseWrapper's flatten/reshape path is reused
    unchanged. Identity lift-back (no reconstruction). Not compatible with WCC (it
    emits no projected z), which is correct: WCC is a LightSplit component, not part
    of this baseline. Gradients flow straight through the kept coordinates
    (mask-multiply), i.e. trained with the sparsifier in place.
    """

    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, dim = h.shape
        k = min(self.k_s, dim)
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)   # 0/1 keep-mask, per sample
        return h * mask, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


class BlockRandomBottleneck(nn.Module):
    """Block-local fixed orthonormal random projection: a privacy/utility dial.

    Tiles the (C,H,W) cut into block x block spatial tiles (each tile = C*block*block
    activations spanning ALL channels), projects every tile with ONE shared orthonormal
    random basis R_b (in_per_block -> out_per_block), and lifts back with R_b^T. The
    macro grid (H/block x W/block) is preserved, so the random mixing is confined to a
    block x block receptive field -- the knob between privacy and utility:
        block = 1   -> per-position channel-only mix (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)

    The full effective projection is assembled as ONE buffer R of shape
    (activation_dim, payload_floats): a permuted block-diagonal of R_b whose columns are
    orthonormal. FORA flattens the cut channel-major (flat idx = c*H*W + y*W + x), so R
    scatters each tile's R_b into exactly the flat rows that tile owns; then z = h @ R
    reproduces the tile-wise projection exactly and z @ R.T = P h is the rank-payload
    orthogonal projector. This is fixed_random with a block-structured R, so it reuses
    DefenseWrapper's flatten path unchanged and exposes bottleneck_z for WCC. Verbatim
    (minus DP clip/noise) from privileged_attacks/train_split.py; numerically identical
    to the unfold reference in the utility harness. R is client-only (never trained/sent).
    """

    def __init__(self, channels: int, hw_h: int, hw_w: int, block: int, k: int):
        super().__init__()
        if block < 1 or hw_h % block != 0 or hw_w % block != 0:
            raise ValueError(
                f"block={block} must be >=1 and divide spatial dims ({hw_h},{hw_w}).")
        self.channels, self.h, self.w, self.block = channels, hw_h, hw_w, block
        self.grid_h, self.grid_w = hw_h // block, hw_w // block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * block * block
        # floor-divide the float budget across tiles (exact for square 2^n grids); the
        # honest transmitted size is payload_floats = n_blocks * out_per_block.
        self.out_per_block = max(1, k // self.n_blocks)
        self.payload_floats = self.n_blocks * self.out_per_block
        self.activation_dim = channels * hw_h * hw_w

        # One shared orthonormal basis per tile (in_per_block -> out_per_block).
        rand_matrix = torch.randn(self.in_per_block, self.out_per_block)
        q, _ = torch.linalg.qr(rand_matrix)            # orthonormal columns

        # Assemble the full (activation_dim, payload) projection R by scattering R_b into
        # the flat rows owned by each tile. Within-tile element q decodes channel-major
        # to (c, by, bx) = (q // block^2, (q % block^2) // block, q % block); it sits at
        # spatial position (gh*block + by, gw*block + bx) for tile t = gh*grid_w + gw,
        # hence flat row c*H*W + y*W + x. Disjoint tiles touch disjoint rows, so R's
        # columns stay orthonormal and R.T @ R = I.
        R_full = torch.zeros(self.activation_dim, self.payload_floats)
        bb = block * block
        q_idx = torch.arange(self.in_per_block)
        c = q_idx // bb
        by = (q_idx % bb) // block
        bx = q_idx % block
        for t in range(self.n_blocks):
            gh, gw = t // self.grid_w, t % self.grid_w
            rows = c * (hw_h * hw_w) + (gh * block + by) * hw_w + (gw * block + bx)
            cols = torch.arange(t * self.out_per_block, (t + 1) * self.out_per_block)
            R_full[rows.unsqueeze(1), cols.unsqueeze(0)] = q
        self.register_buffer("R", R_full)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


class DefenseWrapper(nn.Module):
    """Spatial-in / spatial-out wrapper around a bottleneck.

    forward(h_spatial: (B, C, H, W)) -> (B, C, H, W)
    The compressed payload is stashed (along with whatever info the
    bottleneck returns) on self.last_info so the training loop can pull
    bottleneck_z for WCC without changing pseudo_training's signature.
    """

    def __init__(self, defense_type: str, in_channels: int, H: int, W: int, k: int,
                 block: int = 4, clip_norm: float = 0.0, noise_multiplier: float = 0.0):
        super().__init__()
        self.defense_type = defense_type
        self.C, self.H, self.W = in_channels, H, W
        self.flat_dim = in_channels * H * W
        self.k = k
        self.block = block      # only used by block_random
        # DP on the transmitted payload: per-sample L2 clip (bounds sensitivity) then
        # calibrated Gaussian noise (std = noise_multiplier * clip_norm). Applied on every
        # forward (train AND attack/eval) so the server truly receives a noised payload.
        # Gated on clip_norm > 0 -> byte-identical to the un-noised path when off (default).
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.last_info: Dict[str, torch.Tensor] = {}

        if defense_type == "none":
            self.module = None
        elif defense_type == "learned_1x1":
            self.module = Learned1x1Bottleneck(in_channels, k)
        elif defense_type == "fixed_random":
            self.module = FixedRandomBottleneck(self.flat_dim, k)
        elif defense_type == "topk_sparse":
            # matched-byte vs fixed_random at the same k: each kept coord costs
            # 6 bytes (4 value + 2 index), so k_s*6 ~= k*4 (the LightSplit budget).
            k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))
            self.k_s = k_s
            print(f"[DefenseWrapper] topk_sparse: budget k={k} -> keep k_s={k_s}/"
                  f"{self.flat_dim} coords ({k_s * (VALUE_BYTES + INDEX_BYTES)} B "
                  f"~= {k * VALUE_BYTES} B LightSplit)")
            self.module = TopKSparseBottleneck(k_s=k_s)
        elif defense_type == "block_random":
            # fixed_random with a block-structured R; matched-byte to fixed_random at the
            # same k (payload = n_blocks * out_per_block ~= k). block is the locality dial.
            self.module = BlockRandomBottleneck(channels=in_channels, hw_h=H, hw_w=W,
                                                block=block, k=k)
            m = self.module
            eff_cr = self.flat_dim / m.payload_floats
            print(f"[DefenseWrapper] block_random: block={block} k={k} -> "
                  f"{m.n_blocks} blocks x {m.out_per_block} = {m.payload_floats} floats "
                  f"(in_per_block={m.in_per_block}, eff CR={eff_cr:.1f})")
        else:
            raise ValueError(f"unknown defense_type {defense_type!r}")

    def _apply_dp(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample L2 clip + Gaussian noise on the transmitted payload (train+eval).

        Verbatim from utility/train_split.py:_apply_dp. Inert (returns payload
        unchanged) when clip_norm <= 0, so the un-noised paths stay byte-identical.
        """
        if self.clip_norm <= 0:
            return payload
        flat = payload.flatten(1)
        norms = flat.norm(dim=1, keepdim=True)
        clip_factor = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        flat = flat * clip_factor
        info["dp_clip_fraction"] = (norms.squeeze() > self.clip_norm).float().mean().detach()
        if self.noise_multiplier > 0:
            flat = flat + torch.randn_like(flat) * (self.noise_multiplier * self.clip_norm)
        return flat.reshape_as(payload)

    def forward(self, h_spatial: torch.Tensor) -> torch.Tensor:
        if self.module is None:
            # raw / no-defense: the transmitted payload IS the full activation. With DP on,
            # clip+noise it (flattened) and reshape back; otherwise pass through untouched.
            if self.clip_norm > 0:
                info: Dict[str, torch.Tensor] = {}
                B = h_spatial.size(0)
                dp = self._apply_dp(h_spatial.reshape(B, -1), info)
                self.last_info = info
                return dp.view_as(h_spatial)
            self.last_info = {}
            return h_spatial

        if self.defense_type == "learned_1x1":
            payload, info = self.module(h_spatial)
            payload = self._apply_dp(payload, info)
            lifted = self.module.lift_back(payload, info)
            self.last_info = info
            return lifted

        # fixed_random / block_random: flatten -> compress -> [DP] -> lift -> reshape
        B = h_spatial.size(0)
        h_flat = h_spatial.reshape(B, self.flat_dim)
        payload, info = self.module(h_flat)
        payload = self._apply_dp(payload, info)
        lifted_flat = self.module.lift_back(payload, info)
        self.last_info = info
        return lifted_flat.view(B, self.C, self.H, self.W)
