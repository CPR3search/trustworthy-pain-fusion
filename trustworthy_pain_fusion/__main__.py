"""Command-line entry point: python -m trustworthy_pain_fusion <stage> [args]."""
import runpy
import sys

from trustworthy_pain_fusion.paths import PROJECT_ROOT

STAGES = {
    "decodability": ["trustworthy_pain_fusion.experiments.decodability_ai4pain2026", "trustworthy_pain_fusion.experiments.decodability_ai4pain2025",
                     "trustworthy_pain_fusion.experiments.decodability_biovid", "trustworthy_pain_fusion.experiments.per_modality"],
    "independence": ["trustworthy_pain_fusion.methods.independence"],
    "fusion": ["trustworthy_pain_fusion.methods.fusion"],
    "resampling": ["trustworthy_pain_fusion.experiments.resampling"],
    "normwear": ["trustworthy_pain_fusion.experiments.normwear_ai4pain2026", "trustworthy_pain_fusion.experiments.normwear_biovid"],
    "evaluation": ["trustworthy_pain_fusion.experiments.biovid_loso", "trustworthy_pain_fusion.experiments.ai4pain2025_holdout",
                   "trustworthy_pain_fusion.experiments.transfer", "trustworthy_pain_fusion.experiments.condition_effects",
                   "trustworthy_pain_fusion.experiments.benchmark"],
    "eda-search": ["trustworthy_pain_fusion.methods.eda_search"],
    "eda-null": ["trustworthy_pain_fusion.methods.eda_search_null"],
    "partitions": ["trustworthy_pain_fusion.experiments.partitions"],
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in STAGES:
        print("usage: python -m trustworthy_pain_fusion <stage> [args]\nstages: " + ", ".join(STAGES))
        return 2
    for sub in ("results", "cache"):
        (PROJECT_ROOT / sub).mkdir(exist_ok=True)
    stage, rest = argv[0], argv[1:]
    for module in STAGES[stage]:
        print(f"== {module}", flush=True)
        sys.argv = [module] + rest
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
