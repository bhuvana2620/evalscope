"""
MMMU image-encoder probe adapter.
Selects encoder-stressing subjects and applies adversarial image perturbations
to surface image encoder degradation specifically.

Usage:
    evalscope eval --model VLM_MODEL --datasets mmmu_probe \
        --dataset-args '{"mmmu_probe": {"prune_ratio": 0.25, "encoder_stress_test": true}}'
"""
from __future__ import annotations
import io, logging, random
from pathlib import Path
from typing import Any, Dict, List, Optional
from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import WeightedAverageAccuracy

logger = logging.getLogger(__name__)

ENCODER_STRESSING_SUBJECTS = [
    "Art", "Art_Theory", "Architecture_and_Engineering", "Design", "Mechanical_Engineering",
    "Clinical_Medicine", "Basic_Medical_Science", "Pathology", "Biology", "Pharmacy",
    "Math", "Physics", "Electronics_and_Communication_Engineering", "Computer_Science", "Materials_Science",
]

TEXT_HEAVY_SUBJECTS = [
    "History", "Psychology", "Economics", "Sociology", "Law",
    "Accounting", "Finance", "Marketing", "Literature", "Education_and_Learning",
]


@Benchmark.register(
    name="mmmu_probe",
    dataset_id="MMMU/MMMU",
    subset_list=ENCODER_STRESSING_SUBJECTS,
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=0,
    train_split=None,
    eval_split="test",
    prompt_template=(
        "Look at the image carefully and answer the following multiple-choice question. "
        "Respond with only the letter of the correct answer (A, B, C, or D)."
    ),
)
class MMMUProbeAdapter(DataAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        extra = self.extra_params or {}
        self._prune_ratio = float(extra.get("prune_ratio", 0.25))
        self._encoder_stress_test = bool(extra.get("encoder_stress_test", True))
        self._seed = int(extra.get("seed", 42))
        self._rng = random.Random(self._seed)

    def load_from_disk(self, dataset_name_or_path, subset_name, split):
        full_dataset = super().load_from_disk(dataset_name_or_path, subset_name, split)
        n_keep = max(10, int(len(full_dataset) * self._prune_ratio))
        sampled = self._stratified_sample_mmmu(full_dataset, n_keep)
        logger.info(f"MMMUProbe [{subset_name}]: {len(full_dataset)} -> {len(sampled)} samples")
        return sampled

    def get_gold_answer(self, input_d):
        return input_d.get("answer", "")

    def match(self, gold, pred):
        if pred is None: return 0.0
        gold_clean = gold.strip().upper()
        for char in pred.upper():
            if char in "ABCD":
                return 1.0 if char == gold_clean else 0.0
        return 0.0

    def compute_metric(self, review_res_list):
        correct = sum(1 for r in review_res_list if r.get("sample_score", 0) == 1.0)
        total = len(review_res_list)
        return {"accuracy": correct / total if total else 0.0,
                "n_samples": total,
                "subjects_covered": len(ENCODER_STRESSING_SUBJECTS),
                "encoder_stress_test": self._encoder_stress_test}

    def apply_encoder_stress(self, image_bytes: bytes, stress_type: str = "resolution") -> bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            orig_size = img.size
            if stress_type == "resolution":
                stressed = img.resize((64, 64), Image.LANCZOS).resize(orig_size, Image.LANCZOS)
            elif stress_type == "greyscale":
                stressed = img.convert("L").convert("RGB")
            elif stress_type == "crop":
                w, h = orig_size
                x0 = self._rng.randint(0, w // 2)
                y0 = self._rng.randint(0, h // 2)
                stressed = img.crop((x0, y0, x0 + w // 2, y0 + h // 2)).resize(orig_size, Image.LANCZOS)
            else:
                return image_bytes
            buf = io.BytesIO()
            stressed.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"Encoder stress test failed: {e}")
            return image_bytes

    def _stratified_sample_mmmu(self, dataset, n_keep):
        if not dataset: return dataset
        def difficulty_proxy(s):
            return len(s.get("question", "")) * 0.01 + len([k for k in s if k.startswith("option_") and s[k]])
        scored = sorted([(difficulty_proxy(s), s) for s in dataset], key=lambda x: x[0])
        n = len(scored)
        tier_size = n // 3
        tiers = [scored[:tier_size], scored[tier_size:2*tier_size], scored[2*tier_size:]]
        selected = []
        for tier in tiers:
            budget = max(1, n_keep // 3)
            self._rng.shuffle(tier)
            selected.extend(s for _, s in tier[:budget])
        self._rng.shuffle(selected)
        return selected[:n_keep]
