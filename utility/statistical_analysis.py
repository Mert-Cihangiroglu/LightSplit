#!/usr/bin/env python3
"""Comprehensive post-hoc statistical analysis for backdoor client detection.

Loads CSV from backdoor_detection.py and runs:
  B. Cross-client comparative tests (KS, t-test, Mann-Whitney, effect sizes, MAD)
  C. Multivariate analysis (clustering, outlier detection, spectral signature, PCA)
  D. Temporal detection (CUSUM, EWMA, change-point, windowed)
  G. Hypothesis testing framework (multiple comparison correction, bootstrap CIs, power)
  H. Aggregate detection scores (composite, ROC/AUC, precision-recall)

Usage:
    python statistical_analysis.py --csv-path results/detection_stats.csv \
        --output-dir results/analysis/ --run-all
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy
from scipy import stats as sp_stats
from scipy.spatial.distance import mahalanobis

# sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# statsmodels (for multiple comparison correction)
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Optional: HDBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Optional: ruptures for change-point detection
try:
    import ruptures
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


# Columns that are metadata, not metrics
META_COLS = {
    "round_idx", "epoch", "client_id", "is_malicious",
    "dataset", "mode", "k", "num_clients", "alpha",
    "malicious_ratio", "poison_rate", "target_class", "seed",
}

# Top metrics for composite scoring (highest expected discriminative power)
TOP_METRICS = [
    "z_l2_norm", "z_spectral_energy_ratio", "z_cosine_to_global",
    "grad_z_l2_norm", "r_spectral_energy_ratio", "r_cosine_to_global",
    "loss", "z_linf_norm", "r_l2_norm",
]
TOP_METRIC_WEIGHTS = [0.13, 0.13, 0.12, 0.12, 0.10, 0.12, 0.08, 0.10, 0.10]


class DetectionAnalyzer:
    """Post-hoc statistical analysis of collected per-client per-round metrics."""

    def __init__(
        self,
        csv_path: str,
        malicious_ids: Optional[List[int]] = None,
    ):
        self.df = pd.read_csv(csv_path)
        # Compatibility: backdoor runner writes 'epoch'; older code expects 'round_idx'.
        if "round_idx" not in self.df.columns and "epoch" in self.df.columns:
            self.df["round_idx"] = self.df["epoch"]
        # Convert is_malicious to bool
        if "is_malicious" in self.df.columns:
            self.df["is_malicious"] = self.df["is_malicious"].astype(bool)

        self.num_clients = self.df["client_id"].nunique()
        self.num_rounds = self.df["round_idx"].max() + 1
        self.client_ids = sorted(self.df["client_id"].unique())

        # Determine malicious IDs from data or parameter
        if malicious_ids is not None:
            self.malicious_ids = set(malicious_ids)
        elif "is_malicious" in self.df.columns:
            self.malicious_ids = set(
                self.df.loc[self.df["is_malicious"], "client_id"].unique()
            )
        else:
            self.malicious_ids = set()

        # Identify metric columns
        self.metrics = [
            c for c in self.df.columns
            if c not in META_COLS and self.df[c].dtype in [np.float64, np.int64, np.float32]
        ]

    def _get_client_series(
        self, metric: str, client_id: int, window: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Get time series of a metric for a specific client."""
        mask = self.df["client_id"] == client_id
        if window is not None:
            mask = mask & (self.df["round_idx"] >= window[0]) & (self.df["round_idx"] < window[1])
        return self.df.loc[mask, metric].values.astype(float)

    def _get_population_series(
        self, metric: str, exclude_client: int, window: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Get pooled metric values from all clients except one."""
        mask = self.df["client_id"] != exclude_client
        if window is not None:
            mask = mask & (self.df["round_idx"] >= window[0]) & (self.df["round_idx"] < window[1])
        return self.df.loc[mask, metric].values.astype(float)

    # -----------------------------------------------------------------------
    # B.1  Distributional Tests
    # -----------------------------------------------------------------------

    def run_distributional_tests(
        self, metric: str, window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """KS test and Anderson-Darling for each client vs population."""
        rows = []
        for cid in self.client_ids:
            client_vals = self._get_client_series(metric, cid, window)
            pop_vals = self._get_population_series(metric, cid, window)
            if len(client_vals) < 3 or len(pop_vals) < 3:
                continue

            ks_stat, ks_p = sp_stats.ks_2samp(client_vals, pop_vals)

            try:
                ad_result = sp_stats.anderson_ksamp([client_vals, pop_vals])
                ad_stat, ad_p = ad_result.statistic, ad_result.pvalue
            except Exception:
                ad_stat, ad_p = np.nan, np.nan

            try:
                cvm = sp_stats.cramervonmises_2samp(client_vals, pop_vals)
                cvm_stat, cvm_p = cvm.statistic, cvm.pvalue
            except Exception:
                cvm_stat, cvm_p = np.nan, np.nan

            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "metric": metric,
                "ks_stat": ks_stat, "ks_p": ks_p,
                "ad_stat": ad_stat, "ad_p": ad_p,
                "cvm_stat": cvm_stat, "cvm_p": cvm_p,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # B.2  Parametric Tests
    # -----------------------------------------------------------------------

    def run_parametric_tests(
        self, metric: str, window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Welch's t-test for each client vs rest."""
        rows = []
        for cid in self.client_ids:
            client_vals = self._get_client_series(metric, cid, window)
            pop_vals = self._get_population_series(metric, cid, window)
            if len(client_vals) < 2 or len(pop_vals) < 2:
                continue

            t_stat, t_p = sp_stats.ttest_ind(client_vals, pop_vals, equal_var=False)
            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "metric": metric,
                "t_stat": t_stat, "t_p": t_p,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # B.3  Non-Parametric Tests
    # -----------------------------------------------------------------------

    def run_nonparametric_tests(
        self, metric: str, window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Mann-Whitney U and Kruskal-Wallis tests."""
        rows = []
        for cid in self.client_ids:
            client_vals = self._get_client_series(metric, cid, window)
            pop_vals = self._get_population_series(metric, cid, window)
            if len(client_vals) < 2 or len(pop_vals) < 2:
                continue

            try:
                u_stat, u_p = sp_stats.mannwhitneyu(
                    client_vals, pop_vals, alternative="two-sided"
                )
                # Rank-biserial correlation
                n1, n2 = len(client_vals), len(pop_vals)
                rank_biserial = 1.0 - 2.0 * u_stat / (n1 * n2)
            except Exception:
                u_stat, u_p, rank_biserial = np.nan, np.nan, np.nan

            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "metric": metric,
                "u_stat": u_stat, "u_p": u_p,
                "rank_biserial": rank_biserial,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # B.4  Effect Sizes
    # -----------------------------------------------------------------------

    def compute_effect_sizes(
        self, metric: str, window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cohen's d, Hedges' g, Glass's delta for each client vs rest."""
        rows = []
        for cid in self.client_ids:
            client_vals = self._get_client_series(metric, cid, window)
            pop_vals = self._get_population_series(metric, cid, window)
            if len(client_vals) < 2 or len(pop_vals) < 2:
                continue

            n1, n2 = len(client_vals), len(pop_vals)
            m1, m2 = np.mean(client_vals), np.mean(pop_vals)
            s1, s2 = np.std(client_vals, ddof=1), np.std(pop_vals, ddof=1)

            # Pooled std
            pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            pooled_std = max(pooled_std, 1e-10)

            cohens_d = (m1 - m2) / pooled_std
            # Hedges' g correction
            correction = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
            hedges_g = cohens_d * correction
            # Glass's delta (uses population std)
            glass_delta = (m1 - m2) / max(s2, 1e-10)

            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "metric": metric,
                "cohens_d": cohens_d,
                "hedges_g": hedges_g,
                "glass_delta": glass_delta,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # B.5  Robust Statistics
    # -----------------------------------------------------------------------

    def compute_robust_scores(
        self, metric: str, window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """MAD z-scores and trimmed means for each client."""
        # Compute population statistics
        all_vals = []
        client_medians = {}
        for cid in self.client_ids:
            vals = self._get_client_series(metric, cid, window)
            if len(vals) > 0:
                client_medians[cid] = np.median(vals)
                all_vals.extend(vals.tolist())

        if not all_vals:
            return pd.DataFrame()

        pop_array = np.array(all_vals)
        pop_median = np.median(pop_array)
        mad = sp_stats.median_abs_deviation(pop_array, scale="normal")
        mad = max(mad, 1e-10)

        rows = []
        for cid in self.client_ids:
            vals = self._get_client_series(metric, cid, window)
            if len(vals) < 2:
                continue
            client_med = client_medians.get(cid, 0.0)
            mad_zscore = (client_med - pop_median) / mad
            trimmed = sp_stats.trim_mean(vals, proportiontocut=0.1)
            pop_trimmed = sp_stats.trim_mean(pop_array, proportiontocut=0.1)

            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "metric": metric,
                "mad_zscore": mad_zscore,
                "client_median": client_med,
                "pop_median": pop_median,
                "pop_mad": mad,
                "trimmed_mean": trimmed,
                "pop_trimmed_mean": pop_trimmed,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # C.1  Client Feature Matrix
    # -----------------------------------------------------------------------

    def build_client_feature_matrix(
        self, window: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """Aggregate metrics per client → feature matrix [n_clients, n_features].

        For each client, computes mean and std of each metric across rounds.
        Returns (matrix, client_ids, feature_names).
        """
        feature_names = []
        for m in self.metrics:
            feature_names.extend([f"{m}_mean", f"{m}_std"])

        matrix = []
        cids = []
        for cid in self.client_ids:
            mask = self.df["client_id"] == cid
            if window is not None:
                mask = mask & (self.df["round_idx"] >= window[0]) & (self.df["round_idx"] < window[1])
            sub = self.df.loc[mask, self.metrics]
            if len(sub) == 0:
                continue
            row = []
            for m in self.metrics:
                vals = sub[m].values.astype(float)
                row.append(np.mean(vals))
                row.append(np.std(vals))
            matrix.append(row)
            cids.append(cid)

        return np.array(matrix), cids, feature_names

    # -----------------------------------------------------------------------
    # C.2  Clustering
    # -----------------------------------------------------------------------

    def run_clustering(
        self,
        method: str = "kmeans",
        window: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Dict:
        """Run clustering on client feature vectors.

        Methods: 'kmeans', 'hdbscan', 'gmm', 'spectral'.
        Returns dict with labels, method, and evaluation metrics.
        """
        X, cids, feat_names = self.build_client_feature_matrix(window)
        if len(X) < 2:
            return {"error": "not enough clients"}

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0.0))

        if method == "kmeans":
            n_clusters = kwargs.get("n_clusters", 2)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        elif method == "hdbscan":
            if not HAS_HDBSCAN:
                return {"error": "hdbscan not installed"}
            min_cluster_size = kwargs.get("min_cluster_size", max(2, len(cids) // 5))
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
            labels = model.fit_predict(X_scaled)
        elif method == "gmm":
            n_components = kwargs.get("n_components", 2)
            model = GaussianMixture(n_components=n_components, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif method == "spectral":
            n_clusters = kwargs.get("n_clusters", 2)
            model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                       affinity="nearest_neighbors",
                                       n_neighbors=min(5, len(cids) - 1))
            labels = model.fit_predict(X_scaled)
        else:
            return {"error": f"unknown method: {method}"}

        # Map labels to client_ids and check if malicious clients cluster together
        client_labels = {cid: int(lbl) for cid, lbl in zip(cids, labels)}
        mal_labels = [client_labels[c] for c in cids if c in self.malicious_ids]
        clean_labels = [client_labels[c] for c in cids if c not in self.malicious_ids]

        # Detection: malicious clients should be in a minority cluster
        return {
            "method": method,
            "client_labels": client_labels,
            "malicious_labels": mal_labels,
            "clean_labels": clean_labels,
            "n_clusters": len(set(labels) - {-1}),
        }

    # -----------------------------------------------------------------------
    # C.3  Outlier Detection
    # -----------------------------------------------------------------------

    def run_outlier_detection(
        self,
        method: str = "lof",
        window: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Dict:
        """LOF, Isolation Forest, or Mahalanobis outlier detection.

        Returns dict with outlier scores per client and predicted labels.
        """
        X, cids, _ = self.build_client_feature_matrix(window)
        if len(X) < 3:
            return {"error": "not enough clients"}

        X_clean = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        if method == "lof":
            n_neighbors = min(kwargs.get("n_neighbors", 5), len(cids) - 1)
            model = LocalOutlierFactor(n_neighbors=max(n_neighbors, 1), novelty=False)
            preds = model.fit_predict(X_scaled)
            scores = -model.negative_outlier_factor_
        elif method == "isolation_forest":
            contamination = kwargs.get("contamination", 0.1)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X_scaled)
            scores = -model.decision_function(X_scaled)
        elif method == "mahalanobis":
            mean = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled, rowvar=False)
            # Regularize covariance
            cov += np.eye(cov.shape[0]) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
                scores = np.array([
                    mahalanobis(X_scaled[i], mean, cov_inv)
                    for i in range(len(X_scaled))
                ])
            except np.linalg.LinAlgError:
                scores = np.zeros(len(cids))
            # Threshold: 1.5 * MAD from median
            med = np.median(scores)
            mad = np.median(np.abs(scores - med)) * 1.4826
            mad = max(mad, 1e-10)
            preds = np.where(np.abs(scores - med) > 1.5 * mad, -1, 1)
        else:
            return {"error": f"unknown method: {method}"}

        client_scores = {cid: float(s) for cid, s in zip(cids, scores)}
        client_preds = {cid: int(p) for cid, p in zip(cids, preds)}
        flagged = [cid for cid, p in client_preds.items() if p == -1]

        return {
            "method": method,
            "scores": client_scores,
            "predictions": client_preds,
            "flagged": flagged,
            "flagged_malicious": [c for c in flagged if c in self.malicious_ids],
        }

    # -----------------------------------------------------------------------
    # C.5  Spectral Signature Analysis (Tran et al.)
    # -----------------------------------------------------------------------

    def run_spectral_signature(
        self,
        signal: str = "r",
        window: Optional[Tuple[int, int]] = None,
        threshold_factor: float = 1.5,
    ) -> Dict:
        """Spectral signature detection (Tran et al., NeurIPS 2018).

        1. Compute mean representation per client (across rounds).
        2. Center, SVD, project onto top singular vector.
        3. Flag clients with |projection - median| > threshold_factor * MAD.
        """
        # Find metric columns that start with the signal prefix
        signal_metrics = [
            m for m in self.metrics
            if m.startswith(f"{signal}_") and "cosine" not in m
        ]
        if not signal_metrics:
            return {"error": f"no metrics found for signal '{signal}'"}

        # Build per-client mean vectors
        vectors = []
        cids = []
        for cid in self.client_ids:
            mask = self.df["client_id"] == cid
            if window is not None:
                mask = mask & (self.df["round_idx"] >= window[0]) & (self.df["round_idx"] < window[1])
            sub = self.df.loc[mask, signal_metrics]
            if len(sub) < 2:
                continue
            vectors.append(sub.mean().values.astype(float))
            cids.append(cid)

        if len(vectors) < 3:
            return {"error": "not enough clients for spectral signature"}

        V = np.array(vectors)  # [n_clients, n_metrics]
        # Center
        global_mean = V.mean(axis=0)
        V_centered = V - global_mean

        # SVD
        U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)
        top_sv = Vt[0]  # top right singular vector

        # Project each client
        projections = V_centered @ top_sv  # [n_clients]

        # MAD-based thresholding
        med = np.median(projections)
        mad = np.median(np.abs(projections - med)) * 1.4826
        mad = max(mad, 1e-10)

        flagged = []
        client_scores = {}
        for i, cid in enumerate(cids):
            score = abs(projections[i] - med) / mad
            client_scores[cid] = float(score)
            if score > threshold_factor:
                flagged.append(cid)

        return {
            "signal": signal,
            "singular_values": S[:min(5, len(S))].tolist(),
            "projections": {cid: float(p) for cid, p in zip(cids, projections)},
            "mad_scores": client_scores,
            "flagged": flagged,
            "flagged_malicious": [c for c in flagged if c in self.malicious_ids],
            "threshold": threshold_factor,
        }

    # -----------------------------------------------------------------------
    # C.4  Dimensionality Reduction
    # -----------------------------------------------------------------------

    def run_dimensionality_reduction(
        self,
        method: str = "pca",
        n_components: int = 2,
        window: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """PCA (or t-SNE) on client feature matrix for visualization."""
        X, cids, _ = self.build_client_feature_matrix(window)
        if len(X) < 2:
            return {"error": "not enough clients"}

        X_clean = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        if method == "pca":
            n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
            model = PCA(n_components=n_comp)
            X_reduced = model.fit_transform(X_scaled)
            explained_var = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(5, len(cids) - 1)
            model = TSNE(n_components=n_components, perplexity=max(perplexity, 1), random_state=42)
            X_reduced = model.fit_transform(X_scaled)
            explained_var = []
        else:
            return {"error": f"unknown method: {method}"}

        coords = {cid: X_reduced[i].tolist() for i, cid in enumerate(cids)}
        return {
            "method": method,
            "coordinates": coords,
            "explained_variance": explained_var,
            "is_malicious": {cid: cid in self.malicious_ids for cid in cids},
        }

    # -----------------------------------------------------------------------
    # D.1  CUSUM
    # -----------------------------------------------------------------------

    def run_cusum(
        self,
        metric: str,
        client_id: int,
        k_factor: float = 0.5,
        h_factor: float = 4.0,
        window: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """CUSUM control chart for one client on one metric.

        k = k_factor * population_std, h = h_factor * population_std.
        Returns alarm round (if any) and CUSUM values.
        """
        client_vals = self._get_client_series(metric, client_id, window)
        pop_vals = self._get_population_series(metric, client_id, window)

        if len(client_vals) < 5 or len(pop_vals) < 5:
            return {"error": "insufficient data"}

        mu0 = np.mean(pop_vals)
        sigma = max(np.std(pop_vals), 1e-10)
        k = k_factor * sigma
        h = h_factor * sigma

        # One-sided CUSUM (positive shift)
        S_pos = np.zeros(len(client_vals))
        # One-sided CUSUM (negative shift)
        S_neg = np.zeros(len(client_vals))
        alarm_round = None

        for t in range(len(client_vals)):
            if t == 0:
                S_pos[t] = max(0, client_vals[t] - mu0 - k)
                S_neg[t] = max(0, mu0 - client_vals[t] - k)
            else:
                S_pos[t] = max(0, S_pos[t - 1] + client_vals[t] - mu0 - k)
                S_neg[t] = max(0, S_neg[t - 1] + mu0 - client_vals[t] - k)

            if alarm_round is None and (S_pos[t] > h or S_neg[t] > h):
                alarm_round = t

        return {
            "metric": metric,
            "client_id": client_id,
            "alarm_round": alarm_round,
            "mu0": mu0,
            "sigma": sigma,
            "k": k,
            "h": h,
            "max_S_pos": float(np.max(S_pos)),
            "max_S_neg": float(np.max(S_neg)),
        }

    # -----------------------------------------------------------------------
    # D.2  EWMA
    # -----------------------------------------------------------------------

    def run_ewma(
        self,
        metric: str,
        client_id: int,
        lam: float = 0.1,
        L: float = 3.0,
        window: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """EWMA control chart."""
        client_vals = self._get_client_series(metric, client_id, window)
        pop_vals = self._get_population_series(metric, client_id, window)

        if len(client_vals) < 5 or len(pop_vals) < 5:
            return {"error": "insufficient data"}

        mu0 = np.mean(pop_vals)
        sigma = max(np.std(pop_vals), 1e-10)

        ewma = np.zeros(len(client_vals))
        alarm_round = None

        for t in range(len(client_vals)):
            prev = mu0 if t == 0 else ewma[t - 1]
            ewma[t] = lam * client_vals[t] + (1 - lam) * prev
            # Control limit
            factor = lam / (2 - lam) * (1 - (1 - lam) ** (2 * (t + 1)))
            ucl = mu0 + L * sigma * np.sqrt(factor)
            lcl = mu0 - L * sigma * np.sqrt(factor)
            if alarm_round is None and (ewma[t] > ucl or ewma[t] < lcl):
                alarm_round = t

        return {
            "metric": metric,
            "client_id": client_id,
            "alarm_round": alarm_round,
            "lambda": lam,
            "L": L,
        }

    # -----------------------------------------------------------------------
    # D.3  Change-Point Detection
    # -----------------------------------------------------------------------

    def run_changepoint(
        self,
        metric: str,
        client_id: int,
        method: str = "pelt",
        window: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Change-point detection using ruptures library."""
        if not HAS_RUPTURES:
            return {"error": "ruptures not installed"}

        vals = self._get_client_series(metric, client_id, window)
        if len(vals) < 10:
            return {"error": "insufficient data"}

        if method == "pelt":
            algo = ruptures.Pelt(model="rbf").fit(vals)
            penalty = np.log(len(vals)) * np.var(vals)
            result = algo.predict(pen=max(penalty, 0.01))
        elif method == "binseg":
            algo = ruptures.Binseg(model="rbf").fit(vals)
            result = algo.predict(n_bkps=3)
        else:
            return {"error": f"unknown method: {method}"}

        return {
            "metric": metric,
            "client_id": client_id,
            "method": method,
            "changepoints": [int(cp) for cp in result if cp < len(vals)],
        }

    # -----------------------------------------------------------------------
    # D.4  Windowed Analysis
    # -----------------------------------------------------------------------

    def run_windowed_analysis(
        self, n_windows: int = 3,
    ) -> pd.DataFrame:
        """Compare key metrics across temporal windows (early/mid/late)."""
        window_size = self.num_rounds // n_windows
        rows = []

        for w in range(n_windows):
            w_start = w * window_size
            w_end = (w + 1) * window_size if w < n_windows - 1 else self.num_rounds
            window = (w_start, w_end)

            for metric in TOP_METRICS:
                if metric not in self.metrics:
                    continue
                for cid in self.client_ids:
                    vals = self._get_client_series(metric, cid, window)
                    if len(vals) == 0:
                        continue
                    rows.append({
                        "window": w,
                        "window_start": w_start,
                        "window_end": w_end,
                        "client_id": cid,
                        "is_malicious": cid in self.malicious_ids,
                        "metric": metric,
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                        "median": np.median(vals),
                    })

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # G.1  Multiple Comparison Correction
    # -----------------------------------------------------------------------

    def apply_multiple_comparison_correction(
        self, p_values: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05,
    ) -> Dict:
        """Apply Bonferroni, Holm, or BH-FDR correction."""
        if not HAS_STATSMODELS:
            # Manual Bonferroni fallback
            n = len(p_values)
            corrected = np.minimum(p_values * n, 1.0)
            rejected = corrected < alpha
            return {
                "method": "bonferroni_manual",
                "rejected": rejected.tolist(),
                "corrected_p": corrected.tolist(),
                "alpha": alpha,
            }

        rejected, corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
        return {
            "method": method,
            "rejected": rejected.tolist(),
            "corrected_p": corrected.tolist(),
            "alpha": alpha,
        }

    # -----------------------------------------------------------------------
    # G.2  Bootstrap CIs
    # -----------------------------------------------------------------------

    def compute_bootstrap_ci(
        self,
        metric: str,
        client_id: int,
        n_boot: int = 5000,
        confidence: float = 0.95,
        window: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Percentile bootstrap confidence interval for client mean."""
        vals = self._get_client_series(metric, client_id, window)
        if len(vals) < 3:
            return {"error": "insufficient data"}

        rng = np.random.default_rng(42)
        boot_means = np.array([
            np.mean(rng.choice(vals, size=len(vals), replace=True))
            for _ in range(n_boot)
        ])

        alpha = (1 - confidence) / 2
        ci_low = np.percentile(boot_means, alpha * 100)
        ci_high = np.percentile(boot_means, (1 - alpha) * 100)

        return {
            "metric": metric,
            "client_id": client_id,
            "mean": float(np.mean(vals)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "confidence": confidence,
            "n_boot": n_boot,
        }

    # -----------------------------------------------------------------------
    # G.4  Power Analysis
    # -----------------------------------------------------------------------

    def power_analysis(
        self, metric: str, alpha: float = 0.05, power: float = 0.8,
    ) -> Dict:
        """Estimate required rounds for reliable detection based on observed effect sizes."""
        # Compute effect size from malicious vs clean
        mal_vals = []
        clean_vals = []
        for cid in self.client_ids:
            vals = self._get_client_series(metric, cid)
            if cid in self.malicious_ids:
                mal_vals.extend(vals.tolist())
            else:
                clean_vals.extend(vals.tolist())

        if len(mal_vals) < 2 or len(clean_vals) < 2:
            return {"error": "insufficient data"}

        m1, m2 = np.mean(mal_vals), np.mean(clean_vals)
        s1, s2 = np.std(mal_vals, ddof=1), np.std(clean_vals, ddof=1)
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
        effect_size = abs(m1 - m2) / max(pooled, 1e-10)

        # Approximate required n per group for two-sample t-test
        # n ≈ (z_alpha/2 + z_beta)^2 * 2 / d^2
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        if effect_size > 1e-5:
            required_n = int(np.ceil((z_alpha + z_beta) ** 2 * 2 / effect_size ** 2))
        else:
            required_n = float("inf")

        return {
            "metric": metric,
            "observed_effect_size": float(effect_size),
            "required_rounds_per_client": required_n if required_n != float("inf") else -1,
            "alpha": alpha,
            "power": power,
        }

    # -----------------------------------------------------------------------
    # H.1  Composite Anomaly Scores
    # -----------------------------------------------------------------------

    def compute_composite_scores(
        self,
        method: str = "weighted_mad",
        window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Composite anomaly score per client combining multiple metrics."""
        rows = []

        # Collect MAD z-scores for each top metric
        metric_scores = {}  # metric -> {client_id: mad_zscore}
        for metric in TOP_METRICS:
            if metric not in self.metrics:
                continue
            robust_df = self.compute_robust_scores(metric, window)
            if robust_df.empty:
                continue
            metric_scores[metric] = dict(
                zip(robust_df["client_id"], robust_df["mad_zscore"].abs())
            )

        if not metric_scores:
            return pd.DataFrame()

        for cid in self.client_ids:
            scores = []
            weights = []
            for i, metric in enumerate(TOP_METRICS):
                if metric not in metric_scores:
                    continue
                if cid in metric_scores[metric]:
                    scores.append(metric_scores[metric][cid])
                    w = TOP_METRIC_WEIGHTS[i] if i < len(TOP_METRIC_WEIGHTS) else 0.1
                    weights.append(w)

            if not scores:
                continue

            if method == "weighted_mad":
                composite = float(np.average(scores, weights=weights))
            elif method == "max":
                composite = float(max(scores))
            elif method == "mean":
                composite = float(np.mean(scores))
            else:
                composite = float(np.mean(scores))

            rows.append({
                "client_id": cid,
                "is_malicious": cid in self.malicious_ids,
                "composite_score": composite,
                "n_metrics": len(scores),
            })

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # H.2  Evaluation (ROC/AUC, Precision-Recall)
    # -----------------------------------------------------------------------

    def evaluate_detection(
        self,
        scores: pd.DataFrame,
        score_col: str = "composite_score",
    ) -> Dict:
        """Evaluate detection performance using ground truth labels.

        Args:
            scores: DataFrame with 'client_id', 'is_malicious', and score_col.
        """
        if scores.empty or "is_malicious" not in scores.columns:
            return {"error": "no data or labels"}

        y_true = scores["is_malicious"].astype(int).values
        y_scores = scores[score_col].values

        # Need at least one positive and one negative
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return {"error": "need both positive and negative labels"}

        roc_auc = float(roc_auc_score(y_true, y_scores))
        auprc = float(average_precision_score(y_true, y_scores))

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # FPR at TPR=1.0
        idx_full_tpr = np.where(tpr >= 1.0)[0]
        fpr_at_full_tpr = float(fpr[idx_full_tpr[0]]) if len(idx_full_tpr) > 0 else 1.0

        return {
            "roc_auc": roc_auc,
            "auprc": auprc,
            "fpr_at_tpr_1": fpr_at_full_tpr,
            "n_malicious": int(y_true.sum()),
            "n_clean": int((1 - y_true).sum()),
        }

    # -----------------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------------

    def run_full_analysis(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run all analyses and return comprehensive results."""
        results = {}

        # B: Cross-client tests on top metrics
        for metric in TOP_METRICS:
            if metric not in self.metrics:
                continue
            key = f"dist_tests_{metric}"
            results[key] = self.run_distributional_tests(metric).to_dict("records")
            results[f"effect_sizes_{metric}"] = self.compute_effect_sizes(metric).to_dict("records")
            results[f"robust_{metric}"] = self.compute_robust_scores(metric).to_dict("records")

        # C: Clustering and outlier detection
        for method in ["kmeans", "hdbscan"]:
            results[f"clustering_{method}"] = self.run_clustering(method)
        for method in ["lof", "isolation_forest", "mahalanobis"]:
            results[f"outlier_{method}"] = self.run_outlier_detection(method)

        # C5: Spectral signature
        for signal in ["z", "r", "grad_z"]:
            results[f"spectral_sig_{signal}"] = self.run_spectral_signature(signal)

        # D: Temporal
        for cid in self.client_ids:
            for metric in ["z_l2_norm", "grad_z_l2_norm", "loss"]:
                if metric not in self.metrics:
                    continue
                results[f"cusum_{metric}_c{cid}"] = self.run_cusum(metric, cid)
                results[f"ewma_{metric}_c{cid}"] = self.run_ewma(metric, cid)

        # D4: Windowed
        windowed_df = self.run_windowed_analysis()
        results["windowed"] = windowed_df.to_dict("records") if not windowed_df.empty else []

        # H: Composite scores + evaluation
        for method in ["weighted_mad", "max"]:
            scores_df = self.compute_composite_scores(method)
            results[f"composite_{method}"] = scores_df.to_dict("records") if not scores_df.empty else []
            if not scores_df.empty:
                eval_result = self.evaluate_detection(scores_df)
                results[f"eval_{method}"] = eval_result

        # G4: Power analysis
        for metric in TOP_METRICS:
            if metric not in self.metrics:
                continue
            results[f"power_{metric}"] = self.power_analysis(metric)

        # Save if output dir provided
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save full results as JSON
            def _sanitize(obj):
                """Recursively convert numpy types to native Python for JSON."""
                if isinstance(obj, dict):
                    return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return obj

            with (out_path / "full_results.json").open("w") as f:
                json.dump(_sanitize(results), f, indent=2)

            # Save composite scores CSV
            for method in ["weighted_mad", "max"]:
                scores_df = self.compute_composite_scores(method)
                if not scores_df.empty:
                    scores_df.to_csv(out_path / f"composite_{method}.csv", index=False)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Post-hoc statistical analysis for backdoor detection")
    p.add_argument("--csv-path", required=True, help="Path to detection_stats.csv")
    p.add_argument("--output-dir", default="./analysis_results", help="Output directory")
    p.add_argument("--malicious-ids", nargs="+", type=int, default=None)
    p.add_argument("--run-all", action="store_true", help="Run full analysis")
    return p.parse_args()


def main():
    args = parse_args()
    analyzer = DetectionAnalyzer(args.csv_path, args.malicious_ids)

    print(f"Loaded {len(analyzer.df)} rows, {analyzer.num_clients} clients, "
          f"{analyzer.num_rounds} rounds, {len(analyzer.metrics)} metrics")
    print(f"Malicious clients: {analyzer.malicious_ids}")

    if args.run_all:
        results = analyzer.run_full_analysis(args.output_dir)

        # Print summary
        for method in ["weighted_mad", "max"]:
            key = f"eval_{method}"
            if key in results:
                ev = results[key]
                print(f"\n{method} composite score evaluation:")
                print(f"  ROC-AUC: {ev.get('roc_auc', 'N/A'):.4f}")
                print(f"  AUPRC:   {ev.get('auprc', 'N/A'):.4f}")
                print(f"  FPR@TPR=1: {ev.get('fpr_at_tpr_1', 'N/A'):.4f}")

        # Spectral signature results
        for signal in ["z", "r", "grad_z"]:
            key = f"spectral_sig_{signal}"
            if key in results and "flagged" in results[key]:
                res = results[key]
                print(f"\nSpectral signature ({signal}):")
                print(f"  Flagged: {res['flagged']}")
                print(f"  Flagged malicious: {res['flagged_malicious']}")

        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
