"""Per-client per-epoch activation observer.

Buckets per-batch stats during the training loop and emits one row per client
at end of epoch. The runner is responsible for handing in (z_obs, r_obs, loss,
n_correct, bsz) for each batch (see run_backdoor.py).

z = smashed payload (post-bottleneck, pre-backbone)
r = backbone output (post-backbone, pre-tail)
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch

from backdoor_metrics import (
    compute_cosine_to_global,
    compute_moment_stats,
    compute_norm_stats,
    compute_spectral_stats,
)


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """Ensure (B, D) layout. Spatial activations get flattened to (B, C*H*W)."""
    return t.flatten(1) if t.dim() > 2 else t


def _per_batch_stats(t: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute moment + norm + spectral stats for one batch."""
    t = _flatten(t).detach().float()
    out: Dict[str, float] = {}
    out.update(compute_moment_stats(t, prefix))
    out.update(compute_norm_stats(t, prefix))
    out.update(compute_spectral_stats(t, prefix))
    return out


def _accumulate(
    bucket: Optional[Dict[str, float]], new: Dict[str, float]
) -> Dict[str, float]:
    if bucket is None:
        return dict(new)
    for k, v in new.items():
        bucket[k] = bucket.get(k, 0.0) + v
    return bucket


# Metric column suffixes (per-signal). Used in finalize_epoch to ensure every row
# has every column even when SVD on tiny batches falls back to defaults.
_METRIC_SUFFIXES = (
    "mean", "var", "skewness", "kurtosis",
    "l1_norm", "l2_norm", "linf_norm",
    "top_sv", "spectral_energy_ratio", "effective_rank",
)


class EpochObserver:
    """Per-epoch streaming aggregator.

    State per client:
      n_batches[cid]:   batch count (used for averaging)
      stat_sum[cid]:    running sum of per-batch stat dicts (one for z, one for r)
      act_sum[cid]:     running sum of (D,) per-client mean activation vectors
      act_count[cid]:   number of batches contributing to act_sum
      total_loss[cid] / total_correct[cid] / total_samples[cid]: task metrics

    finalize_epoch averages the running sums, computes per-client cosine to the
    global mean, and returns one dict per client.
    """

    def __init__(self, num_clients: int, malicious_ids: Iterable[int]):
        self.num_clients = num_clients
        self.malicious_ids = set(int(m) for m in malicious_ids)
        self.reset()

    def reset(self) -> None:
        self.n_batches: Dict[int, int] = {cid: 0 for cid in range(self.num_clients)}
        self.stat_sum: Dict[str, Dict[int, Optional[Dict[str, float]]]] = {
            "z": {cid: None for cid in range(self.num_clients)},
            "r": {cid: None for cid in range(self.num_clients)},
        }
        self.act_sum: Dict[str, Dict[int, Optional[torch.Tensor]]] = {
            "z": {cid: None for cid in range(self.num_clients)},
            "r": {cid: None for cid in range(self.num_clients)},
        }
        self.act_count: Dict[str, Dict[int, int]] = {
            "z": {cid: 0 for cid in range(self.num_clients)},
            "r": {cid: 0 for cid in range(self.num_clients)},
        }
        self.total_loss: Dict[int, float] = {cid: 0.0 for cid in range(self.num_clients)}
        self.total_correct: Dict[int, int] = {cid: 0 for cid in range(self.num_clients)}
        self.total_samples: Dict[int, int] = {cid: 0 for cid in range(self.num_clients)}

    def observe(
        self,
        client_id: int,
        z: torch.Tensor,
        r: torch.Tensor,
        loss: float,
        n_correct: int,
        bsz: int,
    ) -> None:
        """Record one batch."""
        cid = int(client_id)
        self.n_batches[cid] += 1

        for sig, t in (("z", z), ("r", r)):
            self.stat_sum[sig][cid] = _accumulate(
                self.stat_sum[sig][cid], _per_batch_stats(t, sig)
            )
            mean_vec = _flatten(t).detach().float().mean(dim=0).cpu()
            if self.act_sum[sig][cid] is None:
                self.act_sum[sig][cid] = mean_vec
            else:
                self.act_sum[sig][cid] = self.act_sum[sig][cid] + mean_vec
            self.act_count[sig][cid] += 1

        self.total_loss[cid] += float(loss) * bsz
        self.total_correct[cid] += int(n_correct)
        self.total_samples[cid] += int(bsz)

    def finalize_epoch(self, epoch: int) -> List[Dict[str, Any]]:
        """Return one dict per client. Caller adds run-level meta and CA/ASR."""
        # Per-client mean activation vectors
        client_means: Dict[str, Dict[int, torch.Tensor]] = {"z": {}, "r": {}}
        for sig in ("z", "r"):
            for cid, vec in self.act_sum[sig].items():
                if vec is None or self.act_count[sig][cid] == 0:
                    continue
                client_means[sig][cid] = vec / self.act_count[sig][cid]

        # Global mean = mean of client means (skipping clients with no data)
        global_means: Dict[str, Optional[torch.Tensor]] = {"z": None, "r": None}
        for sig in ("z", "r"):
            vecs = list(client_means[sig].values())
            if vecs:
                global_means[sig] = torch.stack(vecs).mean(dim=0)

        rows: List[Dict[str, Any]] = []
        for cid in range(self.num_clients):
            row: Dict[str, Any] = {
                "epoch": int(epoch),
                "client_id": cid,
                "is_malicious": int(cid in self.malicious_ids),
            }

            n = max(self.n_batches[cid], 1)
            for sig in ("z", "r"):
                stats = self.stat_sum[sig][cid]
                if stats is None:
                    for suf in _METRIC_SUFFIXES:
                        row[f"{sig}_{suf}"] = 0.0
                else:
                    for k, v in stats.items():
                        row[k] = v / n

                # Cosine to global mean
                if cid in client_means[sig] and global_means[sig] is not None:
                    cos = compute_cosine_to_global(
                        client_means[sig][cid], global_means[sig], sig
                    )
                    row.update(cos)
                else:
                    row[f"{sig}_cosine_to_global"] = 0.0

            # Task metrics
            samples = max(self.total_samples[cid], 1)
            row["loss"] = self.total_loss[cid] / samples
            row["accuracy"] = self.total_correct[cid] / samples
            rows.append(row)

        self.reset()
        return rows
