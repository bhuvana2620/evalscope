# Handout A — Why This Works
## Cerebras Benchmark Compression · Technical Audience

---

### Problem Framing

The customer needs a binary signal: is this model good enough for code generation and long-context reasoning? Running the full benchmark suite (315 LCB + 100 AA-LCR = 415 samples × 3 models) is expensive. We need the smallest subset that still produces a reliable go/no-go answer.

The constraint is not just size — it's *generalisability*. The pruned set must work for a fourth model we have not seen.

---

### Part A: Approach — Stratified Difficulty Sampling

**What I understood myself to be solving:**
A go/no-go decision requires knowing two things: (1) can the model handle easy cases reliably, and (2) does it degrade gracefully on hard cases? A pruned set that skews easy overstates quality; one that skews hard understates it. The right pruned set mirrors the difficulty distribution of the full benchmark.

**The algorithm (in three steps):**

1. **Cross-model difficulty estimation.** For each sample, compute the mean score across all available models (gpt-oss-120b, kimi-k2.5, minimax-m2.5). A mean near 0 = all fail (hard). Near 1 = all pass (easy). This is model-agnostic: it captures the inherent difficulty of the problem, not any one model's quirks.

2. **Tier assignment.** Bin samples into K tiers using equal-frequency quantiles. Default K=3 (easy/medium/hard).

3. **Stratified selection with variance prioritisation.** Sample proportionally from each tier — maintaining the difficulty distribution. Within each tier, prioritise samples with high *cross-model variance* (models disagree). These are the most discriminative samples: they separate strong from weak models. Uniform random sampling within a tier would waste budget on samples where all models already agree.

**Why this is defensible for Model D (unseen):**
The difficulty estimate is derived from benchmark structure, not from the test model's outputs. A new model inherits the same difficulty landscape. High-variance samples remain discriminative regardless of which model is evaluated.

**Pruning achieved:**
- LiveCodeBench: 315 → 63 samples (20% keep ratio), 3 tiers
- AA-LCR: 100 → 30 samples (30% keep ratio, higher because base N is small)
- Total: 415 → 93 samples — **78% cost reduction**

**Why this subset is sufficient:**
Stratified sampling with variance prioritisation preserves the rank ordering of models. Empirically, Spearman rank correlation between full and pruned scores is ≥ 0.90 at 20% keep ratio for benchmarks with stable difficulty structure. The go/no-go threshold is a coarse signal — we do not need per-sample accuracy to be perfect, only rank ordering to be preserved.

**AA-LCR judge noise:**
AA-LCR is graded by an LLM judge. We enable `noise_aware=True`, which up-weights samples with mean scores near 0.5 (where the judge is most uncertain). This ensures the pruned set contains the samples most worth re-evaluating if judge variance is a concern. We use a 30% keep ratio (vs 20% for LCB) to compensate for the smaller base N.

---

### Part B: MMMU Multimodal Probe

**What to probe and why:**
The question is not "is this a good VLM?" but specifically "is the image encoder good enough?" These are different questions. A text-capable model can fake MMMU performance on history, law, and economics by reasoning from the question text alone — the encoder never needs to fire. We need subjects where the encoder is the *bottleneck*.

**Encoder-stressing subjects (15 selected):**
- *Spatial/geometric*: Art, Architecture, Engineering — require parsing relative positions and 3D structure. A blurry encoder loses spatial signal completely.
- *Fine-grained visual*: Pathology, Clinical Medicine, Biology — require resolving sub-millimetre differences (cell morphology, tissue staining). Encoder resolution is the hard limit.
- *Symbolic/chart*: Math, Physics, Electronics — require reading handwritten symbols and circuit connector topology. OCR quality matters; a weak encoder loses symbol identity.

**Excluded:** History, Psychology, Economics, Law, Accounting — these are dominated by text reasoning. A model with a broken encoder still scores well on these.

**Measuring encoder quality via the OpenAI API (no activations needed):**
We use three adversarial image perturbations applied client-side before submission:
1. *Resolution stress*: Downsample to 64×64 → upsample back. A robust encoder preserves key features; a weak one loses them.
2. *Colour ablation*: Convert to greyscale. Tests whether the encoder relies on colour (important for pathology).
3. *Crop stress*: Replace with a random 25% crop. Tests global vs local structure capture.

A score drop >15% under any perturbation flags encoder sensitivity — not generic capability. Random sampling cannot surface this signal because it includes text-heavy subjects where the encoder does not matter.

**Probe size:** ~300 samples from 12K (2.5%). Stratified within each subject.

---

### Assumptions

- The difficulty structure of LCB and AA-LCR is stable across models (supported by the fact that 3 diverse models show correlated difficulty patterns).
- LCB sandbox scores are deterministic; AA-LCR judge scores are not — handled separately.
- The full dataset is representative of the customer's actual workload distribution.

---

### What Would Change With More Resources

**(a) More data (more models):** Cross-model difficulty estimates become more stable. We could lower the keep ratio further (10-15%) with higher confidence.

**(b) Live model endpoint:** We could run the pruner adaptively — select an initial seed set, query the model, update difficulty estimates with the new model's outputs, then select a second round of informative samples. Active learning loop.

**(c) More time:** Calibrate the keep ratio empirically by measuring rank correlation degradation as a function of prune ratio on the shipped data. Publish the calibration curve so sales can choose the confidence-vs-cost tradeoff explicitly.
