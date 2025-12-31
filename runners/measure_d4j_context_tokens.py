#!/usr/bin/env python3
import json
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from tokens import count_tokens
from slice_utils import extract_context
from d4j_utils import checkout_bug, get_src_root


ASSETS_PATH_DEFAULT = Path("generated_assets/Defects4J/Defects4J.jsonl")
print(f"Path: {ASSETS_PATH_DEFAULT}")


def cleanup_dir(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)


def resolve_java_file(checkout_dir: Path, src_root: Path, source_path: Path) -> Path:
    """
    Matches your working Defects4J runner logic:
    - assets sometimes store paths relative to checkout root (e.g. 'src/java/...')
    - src_root might already be '.../src/java'
    - avoid src/java/src/java/...
    """
    source_path = Path(str(source_path).lstrip("/"))

    # Try joining with src_root first
    candidate = Path(src_root) / source_path
    if candidate.exists():
        return candidate

    # If source_path already includes src_root’s relative path, trim it
    try:
        rel = Path(src_root).relative_to(checkout_dir)  # e.g. 'src/java' or 'source'
        parts = list(source_path.parts)
        rel_parts = list(rel.parts)
        if parts[: len(rel_parts)] == rel_parts:
            trimmed = Path(*parts[len(rel_parts):])
            candidate2 = Path(src_root) / trimmed
            if candidate2.exists():
                return candidate2
    except Exception:
        pass

    # Fallback: relative to checkout root
    candidate3 = checkout_dir / source_path
    if candidate3.exists():
        return candidate3

    return candidate  # best-effort for debugging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_path", type=str, default=str(ASSETS_PATH_DEFAULT))
    ap.add_argument("--context_type", type=str, default="method")
    ap.add_argument("--model_name", type=str, default="gpt-5.1-2025-11-13")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--start_task_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    assets_path = Path(args.assets_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    results: list[dict] = []

    # Global totals
    totals = {
        "eligible_seen": 0,          # after filters (single-hunk + single-line replace)
        "processed": 0,              # extracted + counted
        "checkout_failed": 0,
        "missing_file": 0,
        "context_failed": 0,         # exception in extract_context
        "empty_or_none_context": 0,  # extract_context returned None/""
        "total_context_tokens": 0,
        "total_context_chars": 0,
        "total_context_lines": 0,
    }

    seen, taken = 0, 0

    with assets_path.open() as f:
        for raw in tqdm(f, desc="Measuring Defects4J contexts"):
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))

            # Mirror your Defects4J runner filters
            if not isinstance(hunks, list) or len(hunks) != 1:
                continue
            hunk = hunks[0]

            # single-line bug (your runner uses: removed_line_numbers_range[1] == 1)
            if hunk["removed_line_numbers_range"][1] != 1:
                continue

            # single-line replacement
            if hunk["added_lines"].count("\n") > 1:
                continue

            totals["eligible_seen"] += 1

            seen += 1
            if seen <= args.start_task_idx:
                continue

            # read fields
            source_path = Path(hunk["source_path"])
            line_no, _bug_len = hunk["removed_line_numbers_range"]

            # Your runner strips leading "source/"
            if source_path.parts and source_path.parts[0] == "source":
                source_path = Path(*source_path.parts[1:])

            checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_ctx_"))
            try:
                # ✅ use your known-good checkout helper
                try:
                    checkout_bug(*bug_id.split(), checkout_dir)
                except Exception as e:
                    results.append({
                        "bug_id": bug_id,
                        "error": f"checkout_failed: {type(e).__name__}: {e}",
                    })
                    totals["checkout_failed"] += 1
                    continue

                # ✅ use your known-good src root helper
                try:
                    src_root = get_src_root(checkout_dir)
                except Exception as e:
                    results.append({
                        "bug_id": bug_id,
                        "error": f"get_src_root_failed: {type(e).__name__}: {e}",
                    })
                    totals["missing_file"] += 1
                    continue

                java_file = resolve_java_file(checkout_dir, src_root, source_path)
                if not java_file.exists():
                    results.append({
                        "bug_id": bug_id,
                        "error": "missing_file",
                        "source_path": str(source_path),
                        "resolved_java_file": str(java_file),
                        "src_root": str(src_root),
                    })
                    totals["missing_file"] += 1
                    continue

                # extract context
                try:
                    context = extract_context(java_file, line_no, context_type=args.context_type)
                except Exception as e:
                    results.append({
                        "bug_id": bug_id,
                        "error": f"context_failed: {type(e).__name__}: {e}",
                        "source_path": str(source_path),
                        "line_no": line_no,
                        "java_file": str(java_file),
                    })
                    totals["context_failed"] += 1
                    continue

                if not context:
                    results.append({
                        "bug_id": bug_id,
                        "error": "empty_or_none_context",
                        "source_path": str(source_path),
                        "line_no": line_no,
                        "java_file": str(java_file),
                    })
                    totals["empty_or_none_context"] += 1
                    continue

                # count tokens (proxy = tiktoken via your tokens.py)
                ctx_tokens = count_tokens(context, args.model_name)
                ctx_chars = len(context)
                ctx_lines = context.count("\n") + 1

                totals["processed"] += 1
                totals["total_context_tokens"] += ctx_tokens
                totals["total_context_chars"] += ctx_chars
                totals["total_context_lines"] += ctx_lines

                results.append({
                    "bug_id": bug_id,
                    "context_type": args.context_type,
                    "model_name_for_tokenizer": args.model_name,
                    "source_path": str(source_path),
                    "line_no": line_no,
                    "context_tokens": ctx_tokens,
                    "context_chars": ctx_chars,
                    "context_lines": ctx_lines,
                })

            finally:
                cleanup_dir(checkout_dir)

            taken += 1
            if args.max_tasks is not None and taken >= args.max_tasks:
                break

    processed = max(totals["processed"], 1)
    summary = {
        "__summary__": {
            "assets_path": str(assets_path),
            "context_type": args.context_type,
            "model_name_for_tokenizer": args.model_name,
            **totals,
            "avg_context_tokens": totals["total_context_tokens"] / processed,
            "avg_context_chars": totals["total_context_chars"] / processed,
            "avg_context_lines": totals["total_context_lines"] / processed,
        }
    }
    results.append(summary)

    out_path = out_dir / (
        f"d4j_context_tokens_{args.context_type}_{args.model_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with out_path.open("w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    print("\n=== GLOBAL TOTALS ===")
    for k, v in summary["__summary__"].items():
        print(f"{k}: {v}")
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
