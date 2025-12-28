import json
import time
import shutil
import tempfile
import argparse
import traceback
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


from model import Model
from patch_utils import insert_patch
from slice_utils import extract_context
from score_utils import calculate_pass_at_k
from d4j_utils import (
    checkout_bug,
    get_src_root,
    run_tests
)


ASSETS_PATH = Path("generated_assets/Defects4J/Defects4J.jsonl")


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

    patches = []
    for _ in range(n):
        patch = llm.generate(prompt)
        print(f"Appending: {patch}")
        patches.append(repr(patch))

    return patches


# def process_task(task: dict) -> dict:
#     """
#     Runs one bug/hunk end-to-end (NUM_SAMPLES patches), returns JSON-serializable result dict.
#     IMPORTANT: must be top-level for multiprocessing.
#     """
#     bug_id = task["bug_id"]
#     hunk = task["hunk"]
#     context_type = task["context_type"]
#     model_name = task["model_name"]
#     num_samples = task["num_samples"]

#     llm = Model(model_name=model_name, temperature=1)

#     buggy_line = hunk["removed_lines"].rstrip()
#     expected_fix = hunk["added_lines"].rstrip()
#     line_no, bug_len = hunk["removed_line_numbers_range"]

#     source_path = Path(hunk["source_path"])
#     if source_path.parts and source_path.parts[0] == "source":
#         source_path = Path(*source_path.parts[1:])

#     all_patches, scores = [], []

#     for i in range(num_samples):
#         checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_"))
#         try:
#             checkout_bug(*bug_id.split(), checkout_dir)
#             src_root = get_src_root(checkout_dir)

#             java_file = src_root / source_path
#             if not java_file.exists():
#                 all_patches.append("")
#                 scores.append(0)
#                 continue

#             context = extract_context(java_file, line_no, context_type=context_type)

#             patch = call_llm(llm=llm, buggy_line=buggy_line, context=context).strip()
#             all_patches.append(patch)

#             indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
#             indent = hunk["added_lines"][:indent_size]
#             insert_patch(patch, java_file, java_file, line_no, bug_len, indent)

#             passed = run_tests(checkout_dir)
#             scores.append(1 if passed else 0)

#         finally:
#             shutil.rmtree(checkout_dir, ignore_errors=True)

#     return {
#         "bug_id": bug_id,
#         "line_no": line_no,
#         "source_path": str(source_path),
#         "buggy_line": buggy_line,
#         "expected_fix": expected_fix,
#         "patches": all_patches,
#         "num_samples": num_samples,
#         "scores": scores,
#         "num_passed": sum(scores),
#         "pass_at_1": calculate_pass_at_k(1, scores),
#         "pass_at_5": calculate_pass_at_k(5, scores),
#         "pass_at_10": calculate_pass_at_k(10, scores),
#         "pass_at_20": calculate_pass_at_k(20, scores),
#     }

def _ts():
    return time.strftime("%H:%M:%S")


def eval_one_patch(args) -> tuple[int, str, int]:
    """
    Evaluate a single patch in an isolated checkout dir.
    Returns (index, patch, score).
    Must be top-level for ProcessPoolExecutor.
    """
    (
        idx,
        bug_id,
        patch,
        source_path_str,
        line_no,
        bug_len,
        indent,
        timeout_sec,
    ) = args

    patch = (patch or "").strip()

    sample_dir = Path(tempfile.mkdtemp(prefix="d4j_sample_"))
    try:
        checkout_bug(*bug_id.split(), sample_dir)
        src_root = get_src_root(sample_dir)

        sample_java_file = src_root / Path(source_path_str)

        if not sample_java_file.exists() or not patch:
            return (idx, patch, 0)

        insert_patch(patch, sample_java_file, sample_java_file, line_no, bug_len, indent)

        passed = run_tests(sample_dir, timeout_sec=timeout_sec)
        print(f"Passed: {passed}")
        return (idx, patch, 1 if passed else 0)

    except Exception as e:
        print(f"Exception: {e}")
        return (idx, patch, 0)

    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)


def resolve_java_file(checkout_dir: Path, src_root: Path, source_path: Path) -> Path:
    """
    Defects4J assets sometimes store paths relative to checkout root (e.g. 'src/java/...'),
    while src_root may already be '.../src/java'. This normalizes to avoid 'src/java/src/java/...'.
    """
    source_path = Path(str(source_path).lstrip("/"))

    # Try joining with src_root first (your current behavior)
    candidate = src_root / source_path
    if candidate.exists():
        return candidate

    # If source_path already includes src_root’s relative path (e.g. source_path starts with 'src/java')
    try:
        rel = src_root.relative_to(checkout_dir)  # e.g. 'src/java' or 'src'
        parts = list(source_path.parts)
        rel_parts = list(rel.parts)

        if parts[: len(rel_parts)] == rel_parts:
            trimmed = Path(*parts[len(rel_parts):])
            candidate2 = src_root / trimmed
            if candidate2.exists():
                return candidate2
    except ValueError:
        # src_root not under checkout_dir, ignore
        pass

    # As a fallback: try interpreting source_path as relative to checkout root
    candidate3 = checkout_dir / source_path
    if candidate3.exists():
        return candidate3

    # Give the best candidate for debugging
    return candidate


# def process_task(task: dict) -> dict:
#     """
#     Runs one bug end-to-end:
#       1) checkout once + extract context
#       2) call LLM to get N candidate replacement lines
#       3) evaluate those patches in parallel (fresh checkout per patch)
#     Returns a JSON-serializable dict.

#     IMPORTANT:
#       - Must be top-level for multiprocessing (so ProcessPoolExecutor can pickle it).
#       - Requires a top-level eval_one_patch() worker (also pickleable).
#     """
#     bug_id = task["bug_id"]
#     hunk = task["hunk"]
#     context_type = task["context_type"]
#     model_name = task["model_name"]
#     num_samples = int(task.get("num_samples", 20))
#     num_workers = int(task.get("num_workers", 4))
#     timeout_sec = int(task.get("timeout_sec", 900))

#     # Small per-bug pool for test runs (DO NOT make this huge)
#     TEST_WORKERS = max(1, min(4, num_workers))

#     # Model is used only for patch generation (not for testing)
#     llm = Model(model_name=model_name, temperature=1)

#     buggy_line = hunk["removed_lines"].rstrip()
#     expected_fix = hunk["added_lines"].rstrip()
#     line_no, bug_len = hunk["removed_line_numbers_range"]

#     source_path = Path(hunk["source_path"])
#     if source_path.parts and source_path.parts[0] == "source":
#         source_path = Path(*source_path.parts[1:])

#     # Indent for insertion
#     # indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
#     # indent = hunk["added_lines"][:indent_size]
#     raw_buggy = hunk["removed_lines"]
#     indent = raw_buggy[: len(raw_buggy) - len(raw_buggy.lstrip(" \t"))]
#     print(f"Indent: {len(indent)}")

#     # -------------------------
#     # STEP 1: checkout once to extract context
#     # -------------------------
#     ctx_checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_ctx_"))
#     try:
#         checkout_bug(*bug_id.split(), ctx_checkout_dir)
#         src_root = get_src_root(ctx_checkout_dir)

#         # java_file = src_root / source_path
#         # if not java_file.exists():
#         #     return {"bug_id": bug_id, "error": f"Missing file: {java_file}"}
#         java_file = resolve_java_file(ctx_checkout_dir, src_root, source_path)
#         print(f"Java File: {java_file}")
#         if not java_file.exists():
#             return {
#                 "bug_id": bug_id,
#                 "error": f"Missing file: {java_file}",
#                 "debug": {
#                     "checkout_dir": str(ctx_checkout_dir),
#                     "src_root": str(src_root),
#                     "source_path": str(source_path),
#                     "src_root_rel": str(src_root.relative_to(ctx_checkout_dir)) if src_root.is_relative_to(ctx_checkout_dir) else None,
#                 },
#             }

#         try:
#             context = extract_context(java_file, line_no, context_type=context_type)
#         except Exception as e:
#             return {"bug_id": bug_id, "error": f"context extraction failed: {type(e).__name__}: {e}"}

#     except Exception as e:
#         return {"bug_id": bug_id, "error": f"checkout failed: {type(e).__name__}: {e}"}
#     finally:
#         shutil.rmtree(ctx_checkout_dir, ignore_errors=True)

#     # -------------------------
#     # STEP 2: generate patches
#     # -------------------------
#     try:
#         patches = call_llm(llm=llm, buggy_line=buggy_line, context=context, n=num_samples)
#     except Exception as e:
#         tb = traceback.format_exc()
#         return {
#             "bug_id": bug_id,
#             "error": f"LLM failed: {type(e).__name__}: {repr(e)}",
#             "traceback": tb,
#         }

#     # Normalize patches
#     # all_patches = [(p or "").strip() for p in patches]
#     all_patches = [(p or "") for p in patches]
#     scores = [0] * len(all_patches)

#     # -------------------------
#     # STEP 3: score patches in parallel (fresh checkout per patch)
#     # -------------------------
#     # Package args for workers (KEEP THESE PICKLEABLE)
#     work = [
#         (
#             i,                  # index so we can restore ordering
#             bug_id,              # for logging/debug
#             all_patches[i],      # the candidate replacement line
#             str(source_path),    # serialize Path for pickling
#             line_no,
#             bug_len,
#             indent,              # string
#             timeout_sec,
#         )
#         for i in range(len(all_patches))
#     ]

#     print(
#         f"[{_ts()}] {bug_id} evaluating {len(work)} patches with TEST_WORKERS={TEST_WORKERS}",
#         flush=True,
#     )

#     from concurrent.futures import ProcessPoolExecutor, as_completed

#     try:
#         with ProcessPoolExecutor(max_workers=TEST_WORKERS) as ex:
#             futures = [ex.submit(eval_one_patch, w) for w in work]

#             for fut in as_completed(futures):
#                 # eval_one_patch MUST return: (idx, patch, score)
#                 idx, patch, score = fut.result()
#                 scores[idx] = int(score)

#                 # Progress logging (optional)
#                 print(
#                     f"[{_ts()}] {bug_id} sample {idx+1}/{len(work)} done "
#                     f"score={score} patch_len={len(patch or '')}",
#                     flush=True,
#                 )

#     except Exception as e:
#         # If the per-bug test pool itself crashed, return an error record for this bug
#         tb = traceback.format_exc()
#         return {
#             "bug_id": bug_id,
#             "line_no": line_no,
#             "source_path": str(source_path),
#             "buggy_line": buggy_line,
#             "expected_fix": expected_fix,
#             "patches": all_patches,
#             "num_samples": len(all_patches),
#             "scores": scores,
#             "num_passed": sum(scores),
#             "pass_at_1": calculate_pass_at_k(1, scores),
#             "pass_at_5": calculate_pass_at_k(5, scores),
#             "pass_at_10": calculate_pass_at_k(10, scores),
#             "pass_at_20": calculate_pass_at_k(20, scores),
#             "error": f"test pool failed: {type(e).__name__}: {e}",
#             "traceback": tb,
#         }

#     # -------------------------
#     # Final result
#     # -------------------------
#     return {
#         "bug_id": bug_id,
#         "line_no": line_no,
#         "source_path": str(source_path),
#         "buggy_line": buggy_line,
#         "expected_fix": expected_fix,
#         "patches": all_patches,
#         "num_samples": len(all_patches),
#         "scores": scores,
#         "num_passed": sum(scores),
#         "pass_at_1": calculate_pass_at_k(1, scores),
#         "pass_at_5": calculate_pass_at_k(5, scores),
#         "pass_at_10": calculate_pass_at_k(10, scores),
#         "pass_at_20": calculate_pass_at_k(20, scores),
#     }


def process_task(task: dict) -> dict:
    bug_id = task["bug_id"]
    hunk = task["hunk"]
    context_type = task["context_type"]
    model_name = task["model_name"]
    num_samples = int(task["num_samples"])

    llm = Model(model_name=model_name, temperature=1)

    buggy_line = hunk["removed_lines"].rstrip()
    expected_fix = hunk["added_lines"].rstrip()
    line_no, bug_len = hunk["removed_line_numbers_range"]

    source_path = Path(hunk["source_path"])
    if source_path.parts and source_path.parts[0] == "source":
        source_path = Path(*source_path.parts[1:])

    indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
    indent = hunk["added_lines"][:indent_size]

    print(f"[{_ts()}] START task bug={bug_id} line={line_no} file={source_path} model={model_name} ctx={context_type}", flush=True)

    # -------------------------
    # STEP 1: checkout once to extract context
    # -------------------------
    t0 = time.time()
    ctx_checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_ctx_"))
    try:
        print(f"[{_ts()}] {bug_id} ctx checkout -> {ctx_checkout_dir}", flush=True)
        checkout_bug(*bug_id.split(), ctx_checkout_dir)

        src_root = get_src_root(ctx_checkout_dir)
        java_file = resolve_java_file(ctx_checkout_dir, src_root, source_path)
        print(f"[{_ts()}] {bug_id} ctx java_file={java_file} exists={java_file.exists()}", flush=True)

        if not java_file.exists():
            return {
                "bug_id": bug_id,
                "error": f"Missing file: {java_file}",
                "debug": {
                    "checkout_dir": str(ctx_checkout_dir),
                    "src_root": str(src_root),
                    "source_path": str(source_path),
                    "src_root_rel": str(src_root.relative_to(ctx_checkout_dir)) if src_root.is_relative_to(ctx_checkout_dir) else None,
                },
            }

        print(f"[{_ts()}] {bug_id} extracting context...", flush=True)
        context = extract_context(java_file, line_no, context_type=context_type)
        print(f"[{_ts()}] {bug_id} extracted context chars={len(context)} in {time.time()-t0:.2f}s", flush=True)

    except Exception as e:
        return {"bug_id": bug_id, "error": f"ctx step failed: {type(e).__name__}: {repr(e)}", "traceback": traceback.format_exc()}
    finally:
        shutil.rmtree(ctx_checkout_dir, ignore_errors=True)

    # -------------------------
    # STEP 2: generate patches (all at once)
    # -------------------------
    t1 = time.time()
    try:
        print(f"[{_ts()}] {bug_id} calling LLM for n={num_samples}...", flush=True)
        patches = call_llm(llm=llm, buggy_line=buggy_line, context=context, n=num_samples)
        print(f"[{_ts()}] {bug_id} LLM returned {len(patches)} patches in {time.time()-t1:.2f}s", flush=True)
    except Exception as e:
        return {"bug_id": bug_id, "error": f"LLM failed: {type(e).__name__}: {repr(e)}", "traceback": traceback.format_exc()}

    # -------------------------
    # STEP 3: score patches (fresh checkout per patch)
    # -------------------------
    scores = []
    all_patches = []

    for i, patch in enumerate(patches):
        patch = (patch or "").strip()
        all_patches.append(patch)

        print(f"[{_ts()}] {bug_id} sample {i+1}/{num_samples} patch_len={len(patch)}", flush=True)

        sample_dir = Path(tempfile.mkdtemp(prefix="d4j_sample_"))
        try:
            t2 = time.time()
            print(f"[{_ts()}] {bug_id} sample {i+1} checkout -> {sample_dir}", flush=True)
            checkout_bug(*bug_id.split(), sample_dir)

            src_root = get_src_root(sample_dir)
            sample_java_file = src_root / source_path
            print(f"[{_ts()}] {bug_id} sample {i+1} java_file={sample_java_file} exists={sample_java_file.exists()}", flush=True)

            if not sample_java_file.exists() or not patch:
                scores.append(0)
                print(f"[{_ts()}] {bug_id} sample {i+1} SKIP (missing file or empty patch)", flush=True)
                continue

            print(f"[{_ts()}] {bug_id} sample {i+1} PATCH repr={repr(patch)}", flush=True)
            print(f"[{_ts()}] {bug_id} sample {i+1} INDENT repr={repr(indent)}", flush=True)
            insert_patch(patch, sample_java_file, sample_java_file, line_no, bug_len, indent)

            # THIS is usually where "hangs" happen
            print(f"[{_ts()}] {bug_id} sample {i+1} RUN TESTS...", flush=True)
            passed = run_tests(sample_dir, timeout_sec=900)  # 15 min hard timeout
            print(f"[{_ts()}] {bug_id} sample {i+1} tests done passed={passed} in {time.time()-t2:.2f}s", flush=True)

            scores.append(1 if passed else 0)

        except Exception as e:
            scores.append(0)
            print(f"[{_ts()}] {bug_id} sample {i+1} ERROR {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(sample_dir, ignore_errors=True)

    result = {
        "bug_id": bug_id,
        "line_no": line_no,
        "source_path": str(source_path),
        "buggy_line": buggy_line,
        "expected_fix": expected_fix,
        "patches": all_patches,
        "num_samples": num_samples,
        "scores": scores,
        "num_passed": sum(scores),
        "pass_at_1": calculate_pass_at_k(1, scores),
        "pass_at_5": calculate_pass_at_k(5, scores),
        "pass_at_10": calculate_pass_at_k(10, scores),
        "pass_at_20": calculate_pass_at_k(20, scores),
    }

    print(f"[{_ts()}] DONE task bug={bug_id} pass@1={result['pass_at_1']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--context_type", type=str, default="backward_slice")
    parser.add_argument("--start_task_idx", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    context_type = args.context_type
    start_task_idx = args.start_task_idx
    max_tasks = args.max_tasks
    model_name = args.model_name
    num_workers = args.num_workers
    NUM_SAMPLES = args.num_samples

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / (
        f"defects4J_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with out_path.open("w") as out_f:
        # -------------------------
        # Build list of tasks first
        # -------------------------
        tasks = []
        seen_tasks = 0
        task_count = 0

        with ASSETS_PATH.open() as f:
            for raw in f:
                entry = json.loads(raw)
                bug_id, hunks = next(iter(entry.items()))

                # ✅ Only bugs with exactly ONE hunk
                if len(hunks) != 1:
                    continue

                hunk = hunks[0]

                # ✅ Only single-line bugs (removed range length must be 1)
                if hunk["removed_line_numbers_range"][1] != 1:
                    continue

                # ✅ Only single-line replacements (added_lines must be <= 1 line)
                if hunk["added_lines"].count("\n") > 1:
                    continue

                seen_tasks += 1
                if seen_tasks <= start_task_idx:
                    continue

                tasks.append(
                    {
                        "bug_id": bug_id,
                        "hunk": hunk,
                        "context_type": context_type,
                        "model_name": model_name,
                        "num_samples": NUM_SAMPLES,
                    }
                )

                task_count += 1
                if max_tasks is not None and task_count >= max_tasks:
                    break

        print(f"Eligible tasks: {len(tasks)}")

        # -------------------------
        # Run tasks in parallel (threads)
        # -------------------------
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_task = {ex.submit(process_task, t): t for t in tasks}

            for fut in as_completed(future_to_task):
                t = future_to_task[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "bug_id": t.get("bug_id"),
                        "error": f"{type(e).__name__}: {repr(e)}",
                        "traceback": traceback.format_exc(),
                    }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                if "error" in result:
                    print(f"FAILED: {result.get('bug_id')} -> {result.get('error')}")
                else:
                    print(
                        f"Done: {result['bug_id']}:{result['line_no']} "
                        f"pass@1={result['pass_at_1']}"
                    )

    print(f"\nResults written to {out_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_tasks", type=int, default=None)
#     parser.add_argument("--context_type", type=str, default="backward_slice")
#     parser.add_argument("--start_task_idx", type=int, default=0)
#     parser.add_argument("--model_name", type=str, default="gpt-4")
#     parser.add_argument("--num_workers", type=int, default=1)
#     args = parser.parse_args()

#     context_type = args.context_type
#     start_task_idx = args.start_task_idx
#     max_tasks = args.max_tasks
#     model_name = args.model_name
#     num_workers = args.num_workers

#     NUM_SAMPLES = 20

#     results_dir = Path("results")
#     results_dir.mkdir(exist_ok=True)
#     out_path = results_dir / (
#         f"defects4J_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
#     )
#     out_f = out_path.open("w")

#     # -------------------------
#     # Build list of tasks first
#     # -------------------------
#     tasks = []
#     seen_tasks = 0
#     task_count = 0

#     with ASSETS_PATH.open() as f:
#         for raw in f:
#             entry = json.loads(raw)
#             bug_id, hunks = next(iter(entry.items()))

#             # ✅ Only bugs with exactly ONE hunk
#             if len(hunks) != 1:
#                 continue

#             hunk = hunks[0]

#             # ✅ Only single-line bugs (removed range length must be 1)
#             if hunk["removed_line_numbers_range"][1] != 1:
#                 continue

#             # ✅ Only single-line replacements (added_lines must be <= 1 line)
#             if hunk["added_lines"].count("\n") > 1:
#                 continue

#             seen_tasks += 1
#             if seen_tasks <= start_task_idx:
#                 continue

#             tasks.append({
#                 "bug_id": bug_id,
#                 "hunk": hunk,
#                 "context_type": context_type,
#                 "model_name": model_name,
#                 "num_samples": NUM_SAMPLES,
#             })

#             task_count += 1
#             if max_tasks is not None and task_count >= max_tasks:
#                 break

#     print(f"Eligible tasks: {len(tasks)}")

#     # -------------------------
#     # Run tasks in parallel
#     # -------------------------
#     with ThreadPoolExecutor(max_workers=num_workers) as ex:
#         futures = [ex.submit(process_task, t) for t in tasks]
#         for fut in as_completed(futures):
#             try:
#                 result = fut.result()
#             except Exception as e:
#                 result = {
#                     "bug_id": None,
#                     "error": f"{type(e).__name__}: {e}",
#                     "traceback": traceback.format_exc(),
#                 }

#             out_f.write(json.dumps(result) + "\n")
#             out_f.flush()

#             if "error" in result:
#                 print(f"FAILED: {result.get('bug_id')} -> {result.get('error')}")
#             else:
#                 print(f"Done: {result['bug_id']}:{result['line_no']} pass@1={result['pass_at_1']}")

#     out_f.close()
#     print(f"\nResults written to {out_path}")


    # task_count = 0
    # seen_tasks = 0
    # # NUM_SAMPLES = 5  # number of patches to generate per bu
    # NUM_SAMPLES = 20

    # llm = Model(model_name=model_name, temperature=1)

    # results_dir = Path("results")
    # results_dir.mkdir(exist_ok=True)
    # out_path = results_dir / f"defects4J_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    # out_f = out_path.open("w")

    # with ASSETS_PATH.open() as f:
    #     for raw in f:
    #         entry = json.loads(raw)
    #         bug_id, hunks = next(iter(entry.items()))

    #         for hunk in hunks:
    #             # Only single-line bugs
    #             if hunk["removed_line_numbers_range"][1] != 1:
    #                 continue
    #             if hunk["added_lines"].count("\n") > 1:
    #                 continue

    #             seen_tasks += 1
    #             if seen_tasks <= start_task_idx:
    #                 continue
    #             task_count += 1
    #             if max_tasks is not None and task_count > max_tasks:
    #                 print(f"\nReached max_tasks={max_tasks}, stopping.")
    #                 out_f.close()
    #                 return

    #             buggy_line = hunk["removed_lines"].rstrip()
    #             expected_fix = hunk["added_lines"].rstrip()
    #             line_no, bug_len = hunk["removed_line_numbers_range"]
    #             source_path = Path(hunk["source_path"])

    #             print(f"\n=== {bug_id}:{line_no} ===")
    #             print(f"Buggy line: {buggy_line}")
    #             print(f"Expected fix: {expected_fix}")

    #             all_patches = []
    #             scores = []

    #             for i in range(NUM_SAMPLES):
    #                 checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_"))
    #                 checkout_dir.mkdir(parents=True, exist_ok=True)
    #                 try:
    #                     # Checkout buggy version
    #                     checkout_bug(*bug_id.split(), checkout_dir)

    #                     # Resolve source file
    #                     src_root = get_src_root(checkout_dir)

    #                     source_path = Path(hunk["source_path"])
    #                     # Remove leading "source" if present
    #                     if source_path.parts[0] == "source":
    #                         source_path = Path(*source_path.parts[1:])

    #                     java_file = src_root / source_path
    #                     print(f"Java file: {java_file}")

    #                     if not java_file.exists():
    #                         # debug info
    #                         print(f"Missing file: {java_file}")
    #                         continue

    #                     # Extract context
    #                     context = extract_context(java_file, line_no, context_type=context_type)
    #                     print(f"Context: {context}")

    #                     # Call LLM
    #                     patch = call_llm(llm=llm, buggy_line=buggy_line, context=context).strip()
    #                     all_patches.append(patch)

    #                     # Insert patch
    #                     indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
    #                     indent = hunk["added_lines"][:indent_size]
    #                     insert_patch(patch, java_file, java_file, line_no, bug_len, indent)

    #                     # Run tests
    #                     passed = run_tests(checkout_dir)
    #                     scores.append(1 if passed else 0)
    #                     print(f"Sample {i+1}/{NUM_SAMPLES} - Patch: {patch} - Passed: {passed}")

    #                 finally:
    #                     shutil.rmtree(checkout_dir, ignore_errors=True)

    #             # Write results
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
    #             out_f.write(json.dumps(result) + "\n")
    #             out_f.flush()

    # out_f.close()
    # print(f"\nResults written to {out_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_tasks", type=int, default=None)
#     parser.add_argument("--context_type", type=str, default="backward_slice")
#     parser.add_argument("--start_task_idx", type=int, default=0)
#     parser.add_argument("--model_name", type=str, default="gpt-4")
#     parser.add_argument("--num_workers", type=int, default=1)
#     args = parser.parse_args()

#     context_type = args.context_type
#     start_task_idx = args.start_task_idx
#     max_tasks = args.max_tasks
#     model_name = args.model_name
#     num_workers = args.num_workers

#     NUM_SAMPLES = 20

#     # -------------------------
#     # Collect tasks (ONE per bug)
#     # -------------------------
#     tasks = []
#     seen_tasks = 0
#     task_count = 0

#     with ASSETS_PATH.open() as f:
#         for raw in f:
#             entry = json.loads(raw)
#             bug_id, hunks = next(iter(entry.items()))

#             for hunk in hunks:
#                 # Only single-line bugs
#                 if hunk["removed_line_numbers_range"][1] != 1:
#                     continue

#                 if hunk["added_lines"].count("\n") > 1:
#                     print(f"Skipping: {bug_id}", flush=True)
#                     continue

#                 print(f"Processing: {bug_id}")

#                 seen_tasks += 1
#                 if seen_tasks <= start_task_idx:
#                     continue

#                 task_count += 1
#                 if max_tasks is not None and task_count > max_tasks:
#                     break

#                 tasks.append(
#                     (
#                         bug_id,
#                         hunk,
#                         context_type,
#                         model_name,
#                         NUM_SAMPLES,
#                     )
#                 )

#         print(f"\nSubmitting {len(tasks)} Defects4J bugs with {num_workers} workers")

#         # -------------------------
#         # Parallel execution
#         # -------------------------
#         results = []

#         with ProcessPoolExecutor(max_workers=num_workers) as executor:
#             futures = [
#                 executor.submit(process_defects4j_bug, task)
#                 for task in tasks
#             ]

#             for future in tqdm(as_completed(futures), total=len(futures)):
#                 try:
#                     result = future.result()
#                     results.append(result)
#                 except Exception as e:
#                     print("Worker failed:", e)

#         # -------------------------
#         # Write results (single writer)
#         # -------------------------
#         results_dir = Path("results")
#         results_dir.mkdir(exist_ok=True)

#         out_path = results_dir / (
#             f"defects4J_results_"
#             f"{context_type}_"
#             f"{model_name}_"
#             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
#         )

#         with out_path.open("w") as out_f:
#             for r in results:
#                 out_f.write(json.dumps(r) + "\n")

#         print(f"\nResults written to {out_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--max_tasks", type=int, default=None)
#     parser.add_argument("--context_type", type=str, default="backward_slice")
#     parser.add_argument("--start_task_idx", type=int, default=0)
#     parser.add_argument("--model_name", type=str, default="gpt-4")
#     parser.add_argument("--num_workers", type=int, default=1)
#     args = parser.parse_args()

#     context_type = args.context_type
#     start_task_idx = args.start_task_idx
#     max_tasks = args.max_tasks
#     model_name = args.model_name
#     num_workers = args.num_workers

#     NUM_SAMPLES = 20

#     # -------------------------
#     # Collect tasks (ONE per bug)
#     # -------------------------
#     tasks = []
#     seen_tasks = 0
#     task_count = 0

#     with ASSETS_PATH.open() as f:
#         for raw in f:
#             entry = json.loads(raw)
#             bug_id, hunks = next(iter(entry.items()))

#             for hunk in hunks:
#                 # Only single-line bugs
#                 if hunk["removed_line_numbers_range"][1] != 1:
#                     print(f"Skipping: {bug_id}. More than one removed line.")
#                     continue

#                 if hunk["added_lines"].count("\n") > 1:
#                     print(f"Skipping: {bug_id}. More than one added line.")
#                     continue

#                 seen_tasks += 1
#                 if seen_tasks <= start_task_idx:
#                     continue

#                 task_count += 1
#                 if max_tasks is not None and task_count > max_tasks:
#                     break

#                 tasks.append(
#                     (
#                         bug_id,
#                         hunk,
#                         context_type,
#                         model_name,
#                         NUM_SAMPLES,
#                     )
#                 )

#     print(f"\nSubmitting {len(tasks)} Defects4J bugs with {num_workers} workers")

#     # -------------------------
#     # Prepare output file for incremental writing
#     # -------------------------
#     results_dir = Path("results")
#     results_dir.mkdir(exist_ok=True)
#     out_path = results_dir / (
#         f"defects4J_results_"
#         f"{context_type}_"
#         f"{model_name}_"
#         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
#     )

#     # -------------------------
#     # Parallel execution + incremental writing
#     # -------------------------
#     with out_path.open("w") as out_f, ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(process_defects4j_bug, task) for task in tasks]

#         for future in tqdm(as_completed(futures), total=len(futures)):
#             try:
#                 result = future.result()
#                 out_f.write(json.dumps(result) + "\n")
#                 out_f.flush()  # ensures it's written to disk immediately
#             except Exception as e:
#                 print("Worker failed:", e)

#     print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
