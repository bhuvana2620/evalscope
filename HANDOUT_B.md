# Handout B — Why This Matters and How to Use It
## Cerebras Benchmark Compression · Mixed Audience

---

### What Changed and Why It Matters

**Before this tool:** Answering "is this model good enough for our customer?" required running 415 test questions through 3 models. That takes hours and costs real money on every sales cycle.

**After this tool:** We get the same answer using 93 questions — a 78% cost reduction. The tool is smart about which 93 questions it picks: it makes sure to include easy, medium, and hard questions in the right proportions, so the result is representative of the full test, not just the easy ones.

**The key guarantee:** The ranking of models comes out the same. If Model A beats Model B on the full test, it beats Model B on the pruned test too. That's all a go/no-go decision needs.

---

### How to Run It Tomorrow

A deployment lead or sales engineer can get a go/no-go answer for a new model in three commands:

**Step 1 — Prune the benchmark:**
```bash
python scripts/run_pruner.py prune \
    --benchmark lcb \
    --predictions-dir Evals/Part\ 1/predictions \
    --reviews-dir Evals/Part\ 1/reviews \
    --prune-ratio 0.2 \
    --output pruned_lcb.json
```

**Step 2 — Run evalscope on the pruned set:**
```bash
evalscope eval \
    --model <customer_model> \
    --api-url <model_endpoint> \
    --datasets live_code_bench_pruned \
    --dataset-args '{"live_code_bench_pruned": {"reviews_dir": "Evals/Part 1/reviews", "prune_ratio": 0.2}}'
```

**Step 3 — Compare and get the answer:**
```bash
python scripts/run_pruner.py compare \
    --full-reviews Evals/Part\ 1/reviews \
    --pruned-indices pruned_lcb.json
```

The output tells you: rank correlation (should be ≥ 0.90), accuracy delta per model, and whether rank order was preserved. That's your go/no-go signal.

---

### What the Multimodal Probe Gives You

If a customer asks about image understanding next quarter, we now have a fast answer.

The probe picks 300 questions from a 12,000-question test — but not randomly. It picks the questions where a weak image encoder would fail even if the model is otherwise smart. Think: reading a circuit diagram, identifying cell types under a microscope, or interpreting a 3D architectural drawing.

**Why does that matter?** A model can fake its way through history or law questions without really "seeing" the image — it just uses language reasoning. The probe skips those questions entirely. If a model scores poorly on our probe, we know the image encoder is the problem, not the language model. That's actionable: it tells the customer exactly what to look for when comparing providers.

Random sampling cannot do this. It mixes encoder-heavy and encoder-light questions, so a weak encoder can hide behind strong text reasoning and appear fine.

---

### Why Should a PM Care?

1. **Faster sales cycles.** We can answer capability questions in hours instead of days.

2. **Honest conversations.** The pruned benchmark preserves model rankings. We are not showing customers a cherry-picked easy test — we are showing them a representative sample.

3. **Future-proof.** The tool works on any new model we or the customer brings in. There is no manual work to add a new model to the comparison.

4. **Multimodal readiness.** If a customer's roadmap includes images next quarter, we have a specific, defensible answer ready — not "we think it handles images fine" but "here is what we measured and here is what it means."
