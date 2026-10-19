"""Build the difficulty-ladder question sets (B2). No API calls.

Writes data/ladder/<benchmark>.jsonl for each benchmark in LADDER_BENCHMARKS
(format and sources in helpers/ladder.py) and data/ladder/build_report.json
(sizes, and for HLE how many questions were dropped and why). ARC comes from
Part A's sampled data; MMLU, GPQA Diamond and HLE are downloaded from
HuggingFace. GPQA and HLE are gated: accept their terms on the HuggingFace
website, then log in once (`huggingface-cli login`) on the machine that runs this.

Run from this folder (on a machine that can reach HuggingFace):
    python 10_build_ladder.py [--benchmarks arce arcc mmlu gpqa hle]
"""

import argparse
import json
from collections import Counter

from helpers.bigfive import write_records
from helpers.config import LADDER_BENCHMARKS, LADDER_DIR, LADDER_MIN_HLE, LADDER_NAMES
from helpers.ladder import ladder_path, load_arc, load_gpqa, load_hle, load_mmlu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmarks", nargs="+", choices=LADDER_BENCHMARKS, default=list(LADDER_BENCHMARKS))
    args = parser.parse_args()

    report_path = LADDER_DIR / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for benchmark in args.benchmarks:
        info: dict = {}
        if benchmark in ("arce", "arcc"):
            items = load_arc(benchmark)
        elif benchmark == "mmlu":
            items = load_mmlu()
        elif benchmark == "gpqa":
            items = load_gpqa()
        else:
            items, info = load_hle()
            if info["kept"] < LADDER_MIN_HLE:
                info["exploratory"] = True
                print(f"  HLE: only {info['kept']} text-only multiple-choice questions (< {LADDER_MIN_HLE}); "
                      "report it as exploratory (PLAN B2)")
        write_records(ladder_path(benchmark), items)
        answers = Counter(item["answer"] for item in items)
        options = Counter(len(item["options"]) for item in items)
        report[benchmark] = {"n": len(items), "answer_letters": dict(sorted(answers.items())),
                             "n_options": dict(sorted(options.items())), **info}
        print(f"{LADDER_NAMES[benchmark]}: {len(items)} questions -> {ladder_path(benchmark)}; "
              f"options per question {dict(sorted(options.items()))}")
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
