import os
import json
import shutil
import tempfile
import argparse
import traceback
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from tokens import count_tokens
from patch_utils import insert_patch, strip_outer_quotes_once
from slice_utils import extract_context
from score_utils import calculate_pass_at_k

from model import Model

ASSETS_PATH = Path("generated_assets/Bears/Bears.jsonl")
project_root: Path = Path(__file__).parent.parent
benchmarks_root: Path = project_root / "benchmarks"
bears_root: Path = benchmarks_root / "Bears"


def checkout_bears_bug(branch: str, checkout_dir: Path) -> bool:
    """
    Create a BEARS buggy checkout using git worktrees.
    Returns False if the worktree already exists or checkout fails.
    """
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(bears_root),
                "worktree",
                "add",
                str(checkout_dir),
                branch,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] git worktree add failed for {branch}: {e}")
        return False

    # BEARS convention: buggy version is HEAD~2
    try:
        subprocess.run(
            ["git", "-C", str(checkout_dir), "checkout", "HEAD~2"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] git checkout failed for {branch}: {e}")
        return False

    # Detach worktree from git metadata (important!)
    git_file = checkout_dir / ".git"
    if git_file.exists():
        git_file.unlink()

    return True


def run_bears_tests(checkout_dir: Path) -> bool:
    """
    Run BEARS tests using Maven.
    Returns True if tests pass.
    """
    try:
        result = subprocess.run(
            ["mvn", "test", "-q"],
            cwd=checkout_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # print(f"Result: {result}")
    except subprocess.TimeoutExpired:
        return False

    # Maven convention: non-zero = failure
    return result.returncode == 0


def cleanup_bears_worktree(checkout_dir: Path) -> None:
    """
    Remove BEARS git worktree safely.
    """
    try:
        shutil.rmtree(checkout_dir, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(bears_root), "worktree", "prune"],
            check=False,
        )
    except Exception:
        pass


def call_llm(
    llm: Model,
    buggy_line: str,
    context: str,
    n: int,
) -> list[str]:
    LANG = "java"  # or infer from task

    prompt = f"""You are fixing a Java bug.

    BUGGY LINE (replace this line exactly):
    {buggy_line}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Output ONLY the replacement Java line.
    - Do NOT include explanations, comments, or extra text.
    - You MUST wrap the output in a fenced code block labeled ```{LANG}```.

    EXAMPLE OUTPUT:
    ```{LANG}
    <replacement line>
    ```
    """.strip()

    # Token counts (local; accurate for OpenAI tokenizers)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    context_tokens = count_tokens(context, model_name)
    prompt_tokens_est = count_tokens(prompt, model_name)

    patches: list[str] = []
    completion_tokens_est: list[int] = []
    usage_from_api: list[dict[str, Any]] = []  # if your wrapper returns usage

    for _ in range(n):
        # ---- If your Model.generate supports returning usage, prefer it ----
        # Try a couple patterns without breaking older wrappers.
        out = llm.generate(prompt)

        # Pattern A: wrapper returns {"text": "...", "usage": {...}}
        if isinstance(out, dict) and "text" in out:
            patch_text = str(out["text"])
            usage = out.get("usage")
            if isinstance(usage, dict):
                usage_from_api.append(usage)
        else:
            patch_text = str(out)

        patch_text = patch_text.strip()
        patch_text = strip_outer_quotes_once(patch_text)

        patches.append(patch_text)
        completion_tokens_est.append(count_tokens(patch_text, model_name))

    return {
        "prompt": prompt,  # optional; remove if you don’t want to save it
        "patches": patches,
        "token_estimates": {
            "model_name_for_tokenizer": model_name,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens_per_sample": completion_tokens_est,
            "completion_tokens_sum": sum(completion_tokens_est),
            "total_tokens_est": prompt_tokens_est * n + sum(completion_tokens_est),
        },
        "api_usage_per_sample": usage_from_api,  # empty unless wrapper provides it
    }


def evaluate_patch(
    patch: str,
    branch: str,
    source_path: Path,
    hunk: dict,
    line_no: int,
) -> int:
    sample_checkout_dir = Path(tempfile.mkdtemp(prefix="bears_sample_"))
    try:
        if not checkout_bears_bug(branch, sample_checkout_dir):
            return 0

        sample_java_file = sample_checkout_dir / source_path
        if not sample_java_file.exists():
            return 0

        indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
        indent = hunk["added_lines"][:indent_size]

        insert_patch(
            patch,
            sample_java_file,
            sample_java_file,
            line_no,
            hunk["removed_line_numbers_range"][1],
            indent,
        )

        return 1 if run_bears_tests(sample_checkout_dir) else 0

    finally:
        cleanup_bears_worktree(sample_checkout_dir)


def process_bears_bug(task):
    """
    Worker function for a single BEARS bug.
    task: tuple containing all information needed for one bug
    """
    (bug_id, branch, source_path, buggy_line, expected_fix, hunk, line_no, bug_len, model_name, context_type) = task

    NUM_SAMPLES = 20
    llm = Model(model_name=model_name, temperature=1)

    # -------------------------
    # STEP 1: checkout buggy version (context only)
    # -------------------------
    checkout_dir = Path(tempfile.mkdtemp(prefix="bears_ctx_"))
    try:
        ok = checkout_bears_bug(branch, checkout_dir)
        if not ok:
            return {"bug_id": bug_id, "error": "checkout failed"}
        java_file = checkout_dir / source_path
        assert java_file.exists(), f"Missing file: {java_file}"

        # -------------------------
        # STEP 2: extract context
        # -------------------------
        try:
            context = extract_context(java_file, line_no, context_type=context_type)
        except Exception as e:
            return {"bug_id": bug_id, "error": f"context extraction failed: {e}"}
    finally:
        cleanup_bears_worktree(checkout_dir)

    # -------------------------
    # STEP 3: generate patches
    # -------------------------
    try:
        # patches = call_llm(llm, buggy_line, context, n=NUM_SAMPLES)
        llm_out = call_llm(llm, buggy_line, context, n=NUM_SAMPLES)
        patches = llm_out["patches"]
        token_estimates = llm_out["token_estimates"]
        api_usage_per_sample = llm_out.get("api_usage_per_sample", [])
    except Exception as e:
        # return {"bug_id": bug_id, "error": f"LLM failed: {e}"}
        tb = traceback.format_exc()
        return {
            "bug_id": bug_id,
            "error": f"{type(e).__name__}: {repr(e)}",
            "traceback": tb,
        }

    scores = []
    all_patches = []

    for patch in patches:
        all_patches.append(patch)
        sample_checkout_dir = Path(tempfile.mkdtemp(prefix="bears_sample_"))
        try:
            checkout_bears_bug(branch, sample_checkout_dir)
            sample_java_file = sample_checkout_dir / source_path

            insert_patch(
                patch,
                sample_java_file,
                sample_java_file,
                line_no,
                bug_len,
                hunk["added_lines"][: len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip())],
            )

            passed = run_bears_tests(sample_checkout_dir)
            scores.append(1 if passed else 0)
        except Exception:
            scores.append(0)
        finally:
            cleanup_bears_worktree(sample_checkout_dir)

    # return {
    #     "bug_id": bug_id,
    #     "line_no": line_no,
    #     "source_path": str(source_path),
    #     "buggy_line": buggy_line,
    #     "expected_fix": expected_fix,
    #     "patches": all_patches,
    #     "num_samples": NUM_SAMPLES,
    #     "scores": scores,
    #     "num_passed": sum(scores),
    #     "pass_at_1": calculate_pass_at_k(1, scores),
    #     "pass_at_5": calculate_pass_at_k(5, scores),
    #     "pass_at_10": calculate_pass_at_k(10, scores),
    #     "pass_at_20": calculate_pass_at_k(20, scores),
    # }

    return {
        "bug_id": bug_id,
        "line_no": line_no,
        "source_path": str(source_path),
        "buggy_line": buggy_line,
        "expected_fix": expected_fix,
        "patches": all_patches,
        "num_samples": NUM_SAMPLES,
        "scores": scores,
        "num_passed": sum(scores),
        "pass_at_1": calculate_pass_at_k(1, scores),
        "pass_at_5": calculate_pass_at_k(5, scores),
        "pass_at_10": calculate_pass_at_k(10, scores),
        "pass_at_20": calculate_pass_at_k(20, scores),

        # NEW
        "token_estimates": token_estimates,
        "api_usage_per_sample": api_usage_per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of single-line bugs (tasks) to run"
    )
    parser.add_argument(
        "--context_type",
        type=str,
        default="method",
        help="Type of context to extract (e.g., 'method', 'backward_slice')"
    )
    parser.add_argument(
        "--start_task_idx",
        type=int,
        default=0,
        help="Index of the first eligible BEARS task to run (0-based)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name to use for patch generation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel BEARS tasks to run"
    )

    # --- argument parsing ---
    args = parser.parse_args()
    context_type = args.context_type
    start_task_idx = args.start_task_idx
    max_tasks = args.max_tasks
    model_name = args.model_name

    print(f"Context type: {context_type}")

    # --- load BUG_BRANCH_MAP ---
    BUG_BRANCH_MAP = {}
    with open("benchmarks/Bears/scripts/data/bug_id_and_branch.json") as f:
        data = json.load(f)
        for entry in data:
            BUG_BRANCH_MAP[entry["bugId"]] = entry["bugBranch"]

    # --- Prepare tasks and run in parallel ---
    tasks = []
    seen_tasks = 0
    task_count = 0

    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))
            if len(hunks) != 1:
                continue
            hunk = hunks[0]
            if hunk["added_lines"].count("\n") > 1:
                continue

            seen_tasks += 1
            if seen_tasks <= start_task_idx:
                continue

            branch = BUG_BRANCH_MAP[bug_id]
            source_path = Path(hunk["source_path"])
            buggy_line = hunk["removed_lines"].rstrip()
            expected_fix = hunk["added_lines"].rstrip()
            line_no, bug_len = hunk["removed_line_numbers_range"]

            tasks.append(
                (bug_id, branch, source_path, buggy_line, expected_fix, hunk,
                 line_no, bug_len, model_name, context_type)
            )
            task_count += 1
            if max_tasks is not None and task_count >= max_tasks:
                break

    # --- run all bugs in parallel ---
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_bears_bug, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # --- write results ---
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"bears_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with out_path.open("w") as out_f:
        for r in results:
            out_f.write(json.dumps(r) + "\n")

    print(f"\nResults written to {out_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--max_tasks",
#         type=int,
#         default=None,
#         help="Maximum number of single-line bugs (tasks) to run"
#     )
#     parser.add_argument(
#         "--context_type",
#         type=str,
#         default="method",
#         help="Type of context to extract (e.g., 'method', 'backward_slice')"
#     )
#     parser.add_argument(
#         "--start_task_idx",
#         type=int,
#         default=0,
#         help="Index of the first eligible BEARS task to run (0-based)"
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="gpt-4o-mini",
#         help="LLM model name to use for patch generation"
#     )
#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=1,
#         help="Number of parallel BEARS tasks to run"
#     )

#     args = parser.parse_args()

#     context_type = args.context_type
#     start_task_idx = args.start_task_idx
#     max_tasks = args.max_tasks
#     model_name = args.model_name

#     task_count = 0
#     seen_tasks = 0

#     NUM_SAMPLES = 20  # number of patches per bug

#     llm = Model(
#         model_name=model_name,
#         temperature=1,  # IMPORTANT for pass@k
#     )

#     # -------------------------
#     # Output file
#     # -------------------------
#     results_dir = Path("results")
#     results_dir.mkdir(exist_ok=True)
#     out_path = results_dir / f"bears_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
#     out_f = out_path.open("w")

#     # -------------------------
#     # Bug → branch map
#     # -------------------------
#     BUG_BRANCH_MAP: dict[str, str] = {}
#     with open("benchmarks/Bears/scripts/data/bug_id_and_branch.json") as f:
#         data = json.load(f)
#         for entry in data:
#             BUG_BRANCH_MAP[entry["bugId"]] = entry["bugBranch"]

#     # -------------------------
#     # Loop over BEARS JSON
#     # -------------------------
#     pbar = tqdm(desc="Processing BEARS tasks", unit="bug")
#     with ASSETS_PATH.open() as f:
#         for raw in f:
#             entry = json.loads(raw)
#             bug_id, hunks = next(iter(entry.items()))

#             # Only process bugs that have exactly one hunk
#             if len(hunks) != 1:
#                 continue

#             hunk = hunks[0]

#             # Only process single-line fixes
#             if hunk["added_lines"].count("\n") > 1:
#                 continue

#             seen_tasks += 1

#             # Skip until we reach the start index
#             if seen_tasks <= start_task_idx:
#                 continue

#             task_count += 1
#             pbar.update(1)

#             if max_tasks is not None and task_count >= max_tasks:
#                 print(f"\nReached max_tasks={max_tasks}, stopping.")
#                 out_f.close()
#                 print(f"Results written to {out_path}")
#                 return

#             branch = BUG_BRANCH_MAP[bug_id]
#             source_path = Path(hunk["source_path"])
#             buggy_line = hunk["removed_lines"].rstrip()
#             expected_fix = hunk["added_lines"].rstrip()
#             line_no, bug_len = hunk["removed_line_numbers_range"]

#             print(f"\n=== {bug_id}:{line_no} ===")
#             print(f"Buggy line:    {buggy_line}")
#             print(f"Expected fix: {expected_fix}")

#             # -------------------------
#             # STEP 1: checkout buggy version (context only)
#             # -------------------------
#             checkout_dir = Path(tempfile.mkdtemp(prefix="bears_ctx_"))
#             try:
#                 ok = checkout_bears_bug(branch, checkout_dir)
#                 if not ok:
#                     print(f"[WARN] Skipping Bears bug {bug_id} due to worktree conflict")
#                     continue

#                 java_file = checkout_dir / source_path
#                 assert java_file.exists(), f"Missing file: {java_file}"

#                 # -------------------------
#                 # STEP 2: extract context
#                 # -------------------------
#                 try:
#                     context = extract_context(java_file, line_no, context_type=context_type)
#                     print(f"Context: {context}")
#                 except Exception as e:
#                     print(f"Skipping bug {bug_id}:{line_no} — context extraction failed: {e}")
#                     continue

#             finally:
#                 cleanup_bears_worktree(checkout_dir)

#             # -------------------------
#             # STEP 3: LLM repair
#             # -------------------------
#             patches = call_llm(
#                 llm=llm,
#                 buggy_line=buggy_line,
#                 context=context,
#                 n=NUM_SAMPLES,
#             )

#             scores = []  # 1 = pass, 0 = fail
#             all_patches = []  # store patches

#             for i, patch in enumerate(patches):
#                 all_patches.append(patch)
#                 print(f"\n--- Sample {i+1}/{NUM_SAMPLES} ---")
#                 print("Patch:", patch)
#                 print(f"Expected fix: {expected_fix}")

#                 sample_checkout_dir = Path(tempfile.mkdtemp(prefix="bears_sample_"))
#                 try:
#                     try:
#                         checkout_bears_bug(branch, sample_checkout_dir)
#                     except subprocess.CalledProcessError as e:
#                         print(f"⚠️ Failed to checkout branch for sample {i+1}: {e}")
#                         scores.append(0)  # mark as failed
#                         continue  # skip this sample and go to the next patch

#                     sample_java_file = sample_checkout_dir / source_path
#                     if not sample_java_file.exists():
#                         print(f"⚠️ Missing file: {sample_java_file}")
#                         scores.append(0)
#                         continue

#                     # -------------------------
#                     # STEP 4: insert patch
#                     # -------------------------
#                     indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
#                     indent = hunk["added_lines"][:indent_size]

#                     insert_patch(
#                         patch,
#                         sample_java_file,
#                         sample_java_file,
#                         line_no,
#                         bug_len,
#                         indent,
#                     )

#                     # -------------------------
#                     # STEP 5: run tests
#                     # -------------------------
#                     passed = run_bears_tests(sample_checkout_dir)
#                     scores.append(1 if passed else 0)
#                     print("LLM result:", passed)

#                 finally:
#                     cleanup_bears_worktree(sample_checkout_dir)

#             # -------------------------
#             # Record results
#             # -------------------------
#             result = {
#                 "bug_id": bug_id,
#                 "line_no": line_no,
#                 "source_path": str(source_path),
#                 "buggy_line": buggy_line,
#                 "expected_fix": expected_fix,
#                 "patches": all_patches,
#                 "num_samples": NUM_SAMPLES,
#                 "scores": scores,
#                 "num_passed": sum(scores),
#                 "pass_at_1": calculate_pass_at_k(1, scores),
#                 "pass_at_5": calculate_pass_at_k(5, scores),
#                 "pass_at_10": calculate_pass_at_k(10, scores),
#                 "pass_at_20": calculate_pass_at_k(20, scores),
#             }

#             pbar.close()
#             out_f.write(json.dumps(result) + "\n")
#             out_f.flush()
#             print(f"Scores: {scores}")

#     out_f.close()
#     print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
