"""
Pruned LiveCodeBench v5 adapter with stratified difficulty sampling.

Usage:
    evalscope eval --model MODEL --datasets live_code_bench_pruned \
        --dataset-args '{"live_code_bench_pruned": {"reviews_dir": "Evals/Part 1/reviews", "prune_ratio": 0.2}}'
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import WeightedAverageAccuracy

logger = logging.getLogger(__name__)


@Benchmark.register(
    name="live_code_bench_pruned",
    dataset_id="livecodebench/code_generation_lite",
    subset_list=["default"],
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=0,
    train_split=None,
    eval_split="test",
    prompt_template=(
        "You are an expert programmer. Solve the following coding problem. "
        "Provide only the complete, runnable solution code without explanation."
    ),
)
class LiveCodeBenchPrunedAdapter(DataAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        extra = self.extra_params or {}
        self._predictions_dir = Path(extra.get("predictions_dir", "Evals/Part 1/predictions"))
        self._reviews_dir = Path(extra.get("reviews_dir", "Evals/Part 1/reviews"))
        self._prune_ratio = float(extra.get("prune_ratio", 0.2))
        self._n_tiers = int(extra.get("n_tiers", 3))
        self._seed = int(extra.get("seed", 42))
        self._pruned_indices: Optional[List[str]] = None

    def load_from_disk(self, dataset_name_or_path, subset_name, split):
        full_dataset = super().load_from_disk(dataset_name_or_path, subset_name, split)
        pruned_indices = self._get_pruned_indices()
        if not pruned_indices:
            logger.warning("No pruned indices — returning full dataset.")
            return full_dataset
        pruned_set = set(str(i) for i in pruned_indices)
        filtered = [s for s in full_dataset if str(s.get("index", s.get("id", ""))) in pruned_set]
        logger.info(f"LiveCodeBenchPruned: {len(full_dataset)} -> {len(filtered)} samples")
        return filtered

    def get_gold_answer(self, input_d):
        return input_d.get("expected_output", input_d.get("answer", ""))

    def match(self, gold, pred):
        if pred is None: return 0.0
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

    def compute_metric(self, review_res_list):
        correct = sum(1 for r in review_res_list if r.get("sample_score", 0) == 1.0)
        total = len(review_res_list)
        return {"accuracy": correct / total if total else 0.0, "n_samples": total}

    def _get_pruned_indices(self):
        if self._pruned_indices is not None:
            return self._pruned_indices
        try:
            from cerebras_pruner import PruningConfig, StratifiedPruner
            cfg = PruningConfig(prune_ratio=self._prune_ratio, n_tiers=self._n_tiers,
                                seed=self._seed, noise_aware=False)
            pruner = StratifiedPruner(self._predictions_dir, self._reviews_dir, cfg)
            self._pruned_indices = pruner.prune()
            logger.info(f"Pruned to {len(self._pruned_indices)} LCB samples")
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            self._pruned_indices = []
        return self._pruned_indices
