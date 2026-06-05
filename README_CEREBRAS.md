# evalscope — Cerebras Fork
## Task 2: Benchmark Compression


## Video Walkthrough
[Watch here](https://www.loom.com/share/9636e76bc86f4feca8540995e94ad29d)

**Pinned commit SHA:** `85c89b1f182f2fa2aa94366e5bb3b8805e536ac9`

## Install
```bash
git clone https://github.com/bhuvana2620/evalscope.git
cd evalscope
pip install -e ".[all]"
```

## Quick Start

### Prune LiveCodeBench
```bash
python scripts/run_pruner.py prune --benchmark lcb --predictions-dir "Evals/Part 1/predictions" --reviews-dir "Evals/Part 1/reviews" --prune-ratio 0.2 --output pruned_lcb.json
```

### Prune AA-LCR (noise-aware)
```bash
python scripts/run_pruner.py prune --benchmark aalcr --predictions-dir "Evals/Part 1/predictions" --reviews-dir "Evals/Part 1/reviews" --prune-ratio 0.3 --noise-aware --output pruned_aalcr.json
```

### Compare full vs pruned
```bash
python scripts/run_pruner.py compare --full-reviews "Evals/Part 1/reviews" --pruned-indices pruned_lcb.json
```

### Run evalscope with pruned benchmark
```bash
evalscope eval --model YOUR_MODEL --datasets live_code_bench_pruned --dataset-args '{"live_code_bench_pruned": {"reviews_dir": "Evals/Part 1/reviews", "prune_ratio": 0.2}}'
```

### MMMU Encoder Probe
```bash
evalscope eval --model YOUR_VLM --datasets mmmu_probe --dataset-args '{"mmmu_probe": {"prune_ratio": 0.25, "encoder_stress_test": true}}'
```

## Strategy: Stratified Difficulty Sampling
1. Estimate difficulty from cross-model mean score
2. Bin into tiers using equal-frequency quantiles
3. Sample proportionally, prioritising high-variance samples

**Result:** 78% cost reduction (415 -> 93 samples) with rank correlation >= 0.90

## Handouts
- [HANDOUT_A.md](HANDOUT_A.md) - Technical (engineering audience)
- [HANDOUT_B.md](HANDOUT_B.md) - Business (mixed audience)
