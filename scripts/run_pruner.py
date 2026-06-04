"""
scripts/run_pruner.py - CLI for Cerebras benchmark pruner.

Usage:
    python scripts/run_pruner.py prune --benchmark lcb --predictions-dir "Evals/Part 1/predictions" --reviews-dir "Evals/Part 1/reviews" --prune-ratio 0.2 --output pruned_lcb.json
    python scripts/run_pruner.py compare --full-reviews "Evals/Part 1/reviews" --pruned-indices pruned_lcb.json
    python scripts/run_pruner.py mmmu-probe --prune-ratio 0.25
"""
import argparse, json, sys
from pathlib import Path


def cmd_prune(args):
    from cerebras_pruner import PruningConfig, StratifiedPruner
    cfg = PruningConfig(prune_ratio=args.prune_ratio, n_tiers=args.n_tiers,
                        seed=args.seed, noise_aware=args.noise_aware)
    pruner = StratifiedPruner(Path(args.predictions_dir), Path(args.reviews_dir), cfg)
    indices = pruner.prune()
    print(f"Pruned to {len(indices)} samples (ratio={args.prune_ratio})")
    print(f"First 10: {indices[:10]}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"pruned_indices": indices}, f, indent=2)
        print(f"Saved to {args.output}")


def cmd_compare(args):
    from cerebras_pruner import PruningConfig, StratifiedPruner
    with open(args.pruned_indices) as f:
        pruned_idx_set = set(str(i) for i in json.load(f)["pruned_indices"])
    reviews_dir = Path(args.full_reviews)
    full_scores, pruned_scores = {}, {}
    for fpath in reviews_dir.glob("**/*.jsonl"):
        model_name = fpath.parent.name
        full_scores.setdefault(model_name, {})
        pruned_scores.setdefault(model_name, {})
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    idx = str(rec.get("index", ""))
                    score = float(rec.get("sample_score", 0))
                    full_scores[model_name][idx] = score
                    if idx in pruned_idx_set:
                        pruned_scores[model_name][idx] = score
                except Exception:
                    continue
    if not full_scores:
        print("No review files found. Check --full-reviews path.")
        sys.exit(1)
    cfg = PruningConfig()
    pruner = StratifiedPruner(reviews_dir, reviews_dir, cfg)
    report = pruner.compare_runs(full_scores, pruned_scores)
    print(f"Full: {report['n_full']} | Pruned: {report['n_pruned']} | Rank corr: {report['rank_correlation']} | Rank preserved: {report['rank_order_preserved']}")
    for m in report["model_scores_full"]:
        print(f"  {m}: {report['model_scores_full'][m]:.4f} -> {report['model_scores_pruned'].get(m,0):.4f}")


def cmd_mmmu_probe(args):
    from evalscope.benchmarks.mmmu_probe.mmmu_probe_adapter import ENCODER_STRESSING_SUBJECTS, TEXT_HEAVY_SUBJECTS
    print(f"Encoder-stressing subjects: {len(ENCODER_STRESSING_SUBJECTS)}")
    print(f"Excluded text-heavy: {len(TEXT_HEAVY_SUBJECTS)}")
    print(f"Estimated probe size: ~{int(12000 * 0.25 * args.prune_ratio)} samples")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"subjects": ENCODER_STRESSING_SUBJECTS, "excluded": TEXT_HEAVY_SUBJECTS}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("prune")
    p.add_argument("--benchmark", choices=["lcb","aalcr"], default="lcb")
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--reviews-dir", required=True)
    p.add_argument("--prune-ratio", type=float, default=0.2)
    p.add_argument("--n-tiers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-aware", action="store_true")
    p.add_argument("--output", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--full-reviews", required=True)
    c.add_argument("--pruned-indices", required=True)
    m = sub.add_parser("mmmu-probe")
    m.add_argument("--prune-ratio", type=float, default=0.25)
    m.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.command == "prune": cmd_prune(args)
    elif args.command == "compare": cmd_compare(args)
    elif args.command == "mmmu-probe": cmd_mmmu_probe(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
