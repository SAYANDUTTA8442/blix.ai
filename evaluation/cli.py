"""
Blix v0.3 evaluation CLI.

Usage
-----
    python -m blix.evaluation.cli --dataset path/to/dataset.json --output report.json

Dataset JSON format
-------------------
{
  "name": "my_benchmark",
  "version": "1.0",
  "cases": [
    {
      "case_id": "case_001",
      "query": "What is gradient descent?",
      "relevant_memory_ids": [1, 3, 5],
      "ground_truth_facts": ["Gradient descent minimises loss by following the negative gradient."],
      "ground_truth_profile": {"name": "Sayan"},
      "ground_truth_edges": [["sayan", "works_on", "blix"]],
      "reference_summary": "Discussed gradient descent optimisation."
    }
  ]
}

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python -m blix.evaluation.cli`` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import EvalCase, EvalDataset, EvalReport, MemoryEvaluator


def _load_dataset(path: Path) -> EvalDataset:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cases = [EvalCase(**c) for c in raw["cases"]]
    return EvalDataset(
        name=raw.get("name", path.stem),
        cases=cases,
        version=raw.get("version", "1.0"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blix v0.3 memory evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to dataset JSON file.")
    parser.add_argument("--output", type=Path, default=Path("eval_report.json"), help="Output JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    args = parser.parse_args(argv)

    try:
        dataset = _load_dataset(args.dataset)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    evaluator = MemoryEvaluator()
    # Without live retrieval/profile hooks, produce a metadata-only report
    report = evaluator.evaluate(dataset)

    if not args.quiet:
        evaluator.print_report(report)

    evaluator.save_report(report, args.output)
    if not args.quiet:
        print(f"Report saved → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
