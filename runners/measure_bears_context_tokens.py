import json, shutil, tempfile, argparse, subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from tokens import count_tokens
from slice_utils import extract_context

ASSETS_PATH = Path("generated_assets/Bears/Bears.jsonl")
project_root: Path = Path(__file__).parent.parent
benchmarks_root: Path = project_root / "benchmarks"
bears_root: Path = benchmarks_root / "Bears"


def checkout_bears_bug(branch: str, checkout_dir: Path) -> bool:
    try:
        subprocess.run(["git","-C",str(bears_root),"worktree","add",str(checkout_dir),branch], check=True)
        subprocess.run(["git","-C",str(checkout_dir),"checkout","HEAD~2"], check=True)
    except subprocess.CalledProcessError:
        return False
    git_file = checkout_dir / ".git"
    if git_file.exists():
        git_file.unlink()
    return True


def cleanup_bears_worktree(checkout_dir: Path) -> None:
    shutil.rmtree(checkout_dir, ignore_errors=True)
    subprocess.run(["git","-C",str(bears_root),"worktree","prune"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context_type", type=str, default="method")
    ap.add_argument("--model_name", type=str, default="gpt-5.1-2025-11-13")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--start_task_idx", type=int, default=0)
    args = ap.parse_args()

    # bug -> branch
    bug_branch = {}
    with open("benchmarks/Bears/scripts/data/bug_id_and_branch.json") as f:
        for e in json.load(f):
            bug_branch[e["bugId"]] = e["bugBranch"]

    results = []
    seen, taken = 0, 0

    # -------------------------
    # NEW: global totals
    # -------------------------
    global_totals = {
        "eligible_seen": 0,         # after filtering to single-hunk + single-line
        "processed": 0,             # contexts successfully extracted and counted
        "checkout_failed": 0,
        "missing_file": 0,
        "context_failed": 0,
        "total_context_tokens": 0,
        "total_context_chars": 0,
        "total_context_lines": 0,
    }

    with ASSETS_PATH.open() as f:
        for raw in tqdm(f, desc="Measuring contexts"):
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))
            if len(hunks) != 1:
                continue
            hunk = hunks[0]
            if hunk["added_lines"].count("\n") > 1:
                continue

            global_totals["eligible_seen"] += 1

            seen += 1
            if seen <= args.start_task_idx:
                continue

            branch = bug_branch[bug_id]
            source_path = Path(hunk["source_path"])
            line_no, _bug_len = hunk["removed_line_numbers_range"]

            checkout_dir = Path(tempfile.mkdtemp(prefix="bears_ctx_"))
            try:
                if not checkout_bears_bug(branch, checkout_dir):
                    results.append({"bug_id": bug_id, "error": "checkout_failed"})
                    global_totals["checkout_failed"] += 1
                    continue

                java_file = checkout_dir / source_path
                if not java_file.exists():
                    results.append({"bug_id": bug_id, "error": "missing_file"})
                    global_totals["missing_file"] += 1
                    continue

                try:
                    context = extract_context(java_file, line_no, context_type=args.context_type)
                except Exception as e:
                    results.append({"bug_id": bug_id, "error": f"context_failed: {e}"})
                    global_totals["context_failed"] += 1
                    continue

                # NEW: handle None / empty context
                if not context:
                    results.append({"bug_id": bug_id, "error": "empty_or_none_context"})
                    global_totals["context_failed"] += 1
                    continue

                ctx_tokens = count_tokens(context, args.model_name)
                ctx_chars = len(context)
                ctx_lines = context.count("\n") + 1

                # NEW: accumulate totals
                global_totals["processed"] += 1
                global_totals["total_context_tokens"] += ctx_tokens
                global_totals["total_context_chars"] += ctx_chars
                global_totals["total_context_lines"] += ctx_lines

                results.append({
                    "bug_id": bug_id,
                    "context_type": args.context_type,
                    "model_name_for_tokenizer": args.model_name,
                    "context_tokens": ctx_tokens,
                    "context_chars": ctx_chars,
                    "context_lines": ctx_lines,
                })

            finally:
                cleanup_bears_worktree(checkout_dir)

            taken += 1
            if args.max_tasks is not None and taken >= args.max_tasks:
                break

    # -------------------------
    # NEW: add summary row
    # -------------------------
    processed = max(global_totals["processed"], 1)
    summary = {
        "__summary__": {
            "context_type": args.context_type,
            "model_name_for_tokenizer": args.model_name,
            **global_totals,
            "avg_context_tokens": global_totals["total_context_tokens"] / processed,
            "avg_context_chars": global_totals["total_context_chars"] / processed,
            "avg_context_lines": global_totals["total_context_lines"] / processed,
        }
    }
    results.append(summary)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bears_context_tokens_{args.context_type}_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with out_path.open("w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    # NEW: print global totals
    print("\n=== GLOBAL CONTEXT TOTALS ===")
    for k, v in summary["__summary__"].items():
        print(f"{k}: {v}")

    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
