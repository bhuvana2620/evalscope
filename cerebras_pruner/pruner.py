"""
cerebras_pruner/pruner.py
Universal stratified difficulty pruner for evalscope benchmarks.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PruningConfig:
    prune_ratio: float = 0.2
    n_tiers: int = 3
    seed: int = 42
    noise_aware: bool = True
    noise_band: Tuple[float, float] = (0.35, 0.65)
    noise_upweight: float = 1.5


class StratifiedPruner:
    def __init__(self, predictions_dir, reviews_dir, config=None):
        self.predictions_dir = Path(predictions_dir)
        self.reviews_dir = Path(reviews_dir)
        self.cfg = config or PruningConfig()
        self._rng = random.Random(self.cfg.seed)

    def prune(self) -> List[str]:
        scores_by_index = self._load_scores()
        if not scores_by_index:
            raise ValueError(f"No review files found under {self.reviews_dir}.")
        indices, mean_scores, variance_scores = self._compute_statistics(scores_by_index)
        tiers = self._assign_tiers(mean_scores)
        return self._stratified_sample(indices, mean_scores, variance_scores, tiers)

    def compare_runs(self, full_scores, pruned_scores):
        models = list(full_scores.keys())
        full_agg = {m: self._aggregate(full_scores[m]) for m in models}
        pruned_agg = {m: self._aggregate(pruned_scores[m]) for m in models}
        rank_corr = self._spearman_rank_correlation(
            [full_agg[m] for m in models], [pruned_agg[m] for m in models])
        deltas = {m: abs(full_agg[m] - pruned_agg[m]) for m in models}
        pruned_indices = set(idx for m in pruned_scores.values() for idx in m.keys())
        n_full = sum(len(v) for v in full_scores.values()) // max(len(models), 1)
        return {
            "n_full": n_full,
            "n_pruned": len(pruned_indices),
            "prune_ratio_actual": len(pruned_indices) / max(n_full, 1),
            "rank_correlation": round(rank_corr, 4),
            "max_accuracy_delta": round(max(deltas.values(), default=0), 4),
            "mean_accuracy_delta": round(sum(deltas.values()) / max(len(deltas), 1), 4),
            "model_scores_full": {m: round(full_agg[m], 4) for m in models},
            "model_scores_pruned": {m: round(pruned_agg[m], 4) for m in models},
            "rank_order_preserved": rank_corr >= 0.9,
        }

    def _load_scores(self):
        scores = {}
        for fpath in self.reviews_dir.glob("**/*.jsonl"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                        idx = str(rec.get("index", ""))
                        raw = rec.get("sample_score", None)
                        if idx and raw is not None:
                            scores.setdefault(idx, []).append(float(raw))
                    except Exception:
                        continue
        return scores

    def _compute_statistics(self, scores_by_index):
        indices = list(scores_by_index.keys())
        mean_scores, variance_scores = {}, {}
        for idx in indices:
            s = scores_by_index[idx]
            mean = sum(s) / len(s)
            var = sum((x - mean) ** 2 for x in s) / max(len(s), 1)
            mean_scores[idx] = mean
            variance_scores[idx] = var
        return indices, mean_scores, variance_scores

    def _assign_tiers(self, mean_scores):
        scores = list(mean_scores.values())
        n = self.cfg.n_tiers
        sorted_scores = sorted(scores)
        boundaries = [sorted_scores[int(i * len(sorted_scores) / n)] for i in range(1, n)]
        tiers = {}
        for idx, score in mean_scores.items():
            tier = sum(1 for b in boundaries if score > b)
            tiers[idx] = tier
        return tiers

    def _stratified_sample(self, indices, mean_scores, variance_scores, tiers):
        tier_groups = {}
        for idx in indices:
            tier_groups.setdefault(tiers[idx], []).append(idx)
        n_keep = max(1, int(len(indices) * self.cfg.prune_ratio))
        selected = []
        for tier_id in sorted(tier_groups.keys()):
            group = tier_groups[tier_id]
            budget = max(1, round(n_keep * len(group) / len(indices)))
            def sort_key(idx):
                var = variance_scores[idx]
                if self.cfg.noise_aware:
                    ms = mean_scores[idx]
                    lo, hi = self.cfg.noise_band
                    if lo <= ms <= hi:
                        var *= self.cfg.noise_upweight
                return -var
            group_sorted = sorted(group, key=sort_key)
            candidates = group_sorted[:max(budget * 2, len(group))]
            self._rng.shuffle(candidates)
            selected.extend(candidates[:budget])
        self._rng.shuffle(selected)
        return selected[:n_keep]

    @staticmethod
    def _aggregate(scores):
        return sum(scores.values()) / len(scores) if scores else 0.0

    @staticmethod
    def _spearman_rank_correlation(x, y):
        n = len(x)
        if n < 2: return 1.0
        def rank(lst):
            sorted_idx = sorted(range(n), key=lambda i: lst[i])
            ranks = [0.0] * n
            for r, idx in enumerate(sorted_idx):
                ranks[idx] = r + 1.0
            return ranks
        rx, ry = rank(x), rank(y)
        d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1.0 - (6.0 * d_sq) / max(n * (n ** 2 - 1), 1)
