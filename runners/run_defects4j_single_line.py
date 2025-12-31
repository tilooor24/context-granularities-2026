import os
import json
import time
import shutil
import tempfile
import argparse
import traceback
from tqdm import tqdm
from typing import Any
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


from model import Model
from tokens import count_tokens
from patch_utils import insert_patch, strip_outer_quotes_once
from slice_utils import extract_context
from score_utils import calculate_pass_at_k
from d4j_utils import (
    checkout_bug,
    get_src_root,
    run_tests
)


ASSETS_PATH = Path("generated_assets/Defects4J/Defects4J.jsonl")
NUM_SAMPLES = 20

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

    # patches = []
    # for _ in range(n):
    #     patch = llm.generate(prompt)
    #     patch = strip_outer_quotes_once(patch)
    #     print(f"Appending: {patch}")
    #     patches.append(patch)

    # return patches

    # Token counts (local; same style as BEARS)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    context_tokens = count_tokens(context, model_name)
    prompt_tokens_est = count_tokens(prompt, model_name)

    patches: list[str] = []
    completion_tokens_est: list[int] = []
    api_usage_per_sample: list[dict[str, Any]] = []  # only filled if wrapper returns usage

    for _ in range(n):
        out = llm.generate(prompt)

        # Pattern A: wrapper returns {"text": "...", "usage": {...}}
        if isinstance(out, dict) and ("text" in out or "output" in out or "completion" in out):
            patch_text = str(out.get("text") or out.get("output") or out.get("completion") or "")
            usage = out.get("usage") or out.get("token_usage") or {}
            if isinstance(usage, dict):
                api_usage_per_sample.append(usage)

        # Pattern B: wrapper returns (text, usage_dict)
        elif isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], str) and isinstance(out[1], dict):
            patch_text = out[0]
            api_usage_per_sample.append(out[1])

        # Pattern C: plain string
        else:
            patch_text = str(out)

        patch_text = patch_text.strip()
        patch_text = strip_outer_quotes_once(patch_text)

        patches.append(patch_text)
        completion_tokens_est.append(count_tokens(patch_text, model_name))

    return {
        "patches": patches,
        "token_estimates": {
            "model_name_for_tokenizer": model_name,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens_per_sample": completion_tokens_est,
            "completion_tokens_sum": sum(completion_tokens_est),
            "total_tokens_est": (prompt_tokens_est * n) + sum(completion_tokens_est),
        },
        "api_usage_per_sample": api_usage_per_sample,
    }


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


def score_one_patch_worker(
    *,
    bug_id: str,
    patch_idx: int,
    patch: str,
    num_samples: int,
    source_path_str: str,
    line_no: int,
    bug_len: int,
    indent: str,
    timeout_sec: int,
) -> dict[str, Any]:
    """
    Returns a dict so caller can log errors without crashing the whole task.
    """
    import os

    patch = (patch or "")
    source_path = Path(source_path_str)

    pid = os.getpid()
    sample_dir = Path(tempfile.mkdtemp(prefix="d4j_sample_"))

    def _log(msg: str) -> None:
        print(f"[{_ts()}] [pid={pid}] {bug_id} sample {patch_idx+1}/{num_samples}: {msg}", flush=True)

    def _safe_read_tail(p: Path, n: int = 60) -> str:
        try:
            # read as text, replace undecodable bytes
            s = p.read_text(encoding="utf-8", errors="replace")
            lines = s.splitlines()
            return "\n".join(lines[-n:])
        except Exception as e:
            return f"<could not read {p}: {type(e).__name__}: {e}>"

    def _snippet_around_line(p: Path, line_no_1based: int, radius: int = 6) -> str:
        try:
            s = p.read_text(encoding="utf-8", errors="replace")
            lines = s.splitlines()
            i = max(0, line_no_1based - 1)  # convert to 0-based index
            a = max(0, i - radius)
            b = min(len(lines), i + radius + 1)
            out = []
            for j in range(a, b):
                prefix = ">>" if j == i else "  "
                out.append(f"{prefix} {j+1:5d}: {lines[j]}")
            return "\n".join(out)
        except Exception as e:
            return f"<could not snippet {p}: {type(e).__name__}: {e}>"

    t0 = time.time()
    try:
        _log(f"START sample_dir={sample_dir}")
        _log(f"source_path={source_path} line_no={line_no} bug_len={bug_len} indent_repr={repr(indent)} patch_len={len(patch)}")
        _log(f"cwd={os.getcwd()}")
        _log(f"PATH(head)={os.environ.get('PATH','')[:160]}...")

        # 1) checkout
        t_checkout = time.time()
        _log("checkout_bug(...)")
        checkout_bug(*bug_id.split(), sample_dir)
        _log(f"checkout DONE in {time.time()-t_checkout:.2f}s exists={sample_dir.exists()}")

        # quick sanity: directory listing (top-level)
        try:
            top = sorted([p.name for p in sample_dir.iterdir()])[:30]
            _log(f"sample_dir entries (first 30)={top}")
        except Exception as e:
            _log(f"WARNING could not list sample_dir: {type(e).__name__}: {e}")

        # 2) resolve source file
        t_root = time.time()
        _log("get_src_root(...)")
        src_root = get_src_root(sample_dir)
        _log(f"get_src_root DONE in {time.time()-t_root:.2f}s src_root={src_root} exists={Path(src_root).exists()}")

        _log("resolve_java_file(...)")
        sample_java_file = resolve_java_file(sample_dir, src_root, source_path)
        _log(f"resolved java_file={sample_java_file} exists={sample_java_file.exists()}")

        if not sample_java_file.exists():
            _log("MISSING java file -> returning 0")
            return {
                "patch_idx": patch_idx,
                "score": 0,
                "passed": False,
                "reason": "missing_file",
                "elapsed_sec": time.time() - t0,
                "debug": {
                    "pid": pid,
                    "sample_dir": str(sample_dir),
                    "src_root": str(src_root),
                    "source_path": str(source_path),
                    "resolved_java_file": str(sample_java_file),
                },
            }

        if not patch:
            _log("EMPTY patch -> returning 0")
            return {
                "patch_idx": patch_idx,
                "score": 0,
                "passed": False,
                "reason": "empty_patch",
                "elapsed_sec": time.time() - t0,
                "debug": {
                    "pid": pid,
                    "sample_dir": str(sample_dir),
                },
            }

        # 3) show pre-patch context around target line (helps diagnose off-by-one / wrong file)
        _log("pre-patch snippet around target line:")
        print(_snippet_around_line(sample_java_file, line_no, radius=6), flush=True)

        # 4) apply patch
        t_patch = time.time()
        _log(f"insert_patch(...) patch_repr={repr(patch[:200])}{'...' if len(patch)>200 else ''}")
        insert_patch(patch, sample_java_file, sample_java_file, line_no, bug_len, indent)
        _log(f"insert_patch DONE in {time.time()-t_patch:.2f}s")

        # 5) show post-patch snippet
        _log("post-patch snippet around target line:")
        print(_snippet_around_line(sample_java_file, line_no, radius=6), flush=True)

        # 6) run tests
        _log(f"RUN TESTS timeout={timeout_sec}s")
        t_tests = time.time()
        passed = run_tests(sample_dir, timeout_sec=timeout_sec)
        _log(f"TESTS DONE passed={passed} in {time.time()-t_tests:.2f}s")

        return {
            "patch_idx": patch_idx,
            "score": 1 if passed else 0,
            "passed": bool(passed),
            "reason": None if passed else "tests_failed",
            "elapsed_sec": time.time() - t0,
            "debug": {
                "pid": pid,
                "sample_dir": str(sample_dir),
                "src_root": str(src_root),
                "resolved_java_file": str(sample_java_file),
            },
        }

    except Exception as e:
        _log(f"ERROR {type(e).__name__}: {e}")
        return {
            "patch_idx": patch_idx,
            "score": 0,
            "passed": False,
            "reason": f"error:{type(e).__name__}:{e}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - t0,
            "debug": {
                "pid": pid,
                "sample_dir": str(sample_dir),
                "source_path": str(source_path),
            },
        }
    finally:
        try:
            _log("cleanup rm -rf sample_dir")
        except Exception:
            pass
        shutil.rmtree(sample_dir, ignore_errors=True)


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
        if not context:
            return {
                "bug_id": bug_id,
                "error": "ctx step failed: empty or None context",
                "debug": {
                    "context_type": context_type,
                    "line_no": line_no,
                    "java_file": str(java_file),
                },
            }

        print(
            f"[{_ts()}] {bug_id} extracted context chars={len(context)} "
            f"in {time.time()-t0:.2f}s",
            flush=True,
        )

    except Exception as e:
        return {"bug_id": bug_id, "error": f"ctx step failed: {type(e).__name__}: {repr(e)}", "traceback": traceback.format_exc()}
    finally:
        shutil.rmtree(ctx_checkout_dir, ignore_errors=True)

    # -------------------------
    # STEP 2: generate patches (all at once)
    # -------------------------
    t1 = time.time()
    try:
        # print(f"[{_ts()}] {bug_id} calling LLM for n={num_samples}...", flush=True)
        # patches = call_llm(llm=llm, buggy_line=buggy_line, context=context, n=num_samples)
        # print(f"[{_ts()}] {bug_id} LLM returned {len(patches)} patches in {time.time()-t1:.2f}s", flush=True)
        llm_out = call_llm(llm=llm, buggy_line=buggy_line, context=context, n=num_samples)
        patches = llm_out["patches"]
        token_estimates = llm_out["token_estimates"]
        api_usage_per_sample = llm_out.get("api_usage_per_sample", [])
    except Exception as e:
        return {"bug_id": bug_id, "error": f"LLM failed: {type(e).__name__}: {repr(e)}", "traceback": traceback.format_exc()}

    # -------------------------
    # STEP 3: score patches (parallel)
    # -------------------------
    scores = [0] * len(patches)
    all_patches = []
    timeout_sec = 900

    # keep logs deterministic by storing patch strings first
    for p in patches:
        all_patches.append((p or "").strip())

    # Tune this carefully: too high can thrash CPU/disk and slow you down.
    # Rule of thumb: 2–4 concurrent test runs per machine.
    max_test_workers = int(os.getenv("D4J_TEST_WORKERS", "3"))

    print(f"[{_ts()}] {bug_id} scoring {len(all_patches)} patches with workers={max_test_workers}", flush=True)

    with ProcessPoolExecutor(max_workers=max_test_workers) as ex:
        futs = []
        for i, patch in enumerate(all_patches):
            print(f"[{_ts()}] {bug_id} enqueue sample {i+1}/{num_samples} patch_len={len(patch)}", flush=True)

            futs.append(
                ex.submit(
                    score_one_patch_worker,
                    bug_id=bug_id,
                    patch_idx=i,
                    patch=patch,
                    num_samples=num_samples,
                    source_path_str=str(source_path),
                    line_no=line_no,
                    bug_len=bug_len,
                    indent=indent,
                    timeout_sec=timeout_sec,
                )
            )

        for fut in as_completed(futs):
            r = fut.result()
            i = r["patch_idx"]
            scores[i] = r["score"]

            # Optional logging
            if r.get("reason"):
                print(f"[{_ts()}] {bug_id} sample {i+1} done score={scores[i]} reason={r['reason']}", flush=True)
            else:
                print(f"[{_ts()}] {bug_id} sample {i+1} done score={scores[i]} elapsed={r.get('elapsed_sec', 0):.2f}s", flush=True)

    result = {
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
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Run only a specific Defects4J bug id (e.g. 'Cli 8', 'Chart 1')"
    )
    args = parser.parse_args()

    context_type = args.context_type
    start_task_idx = args.start_task_idx
    max_tasks = args.max_tasks
    model_name = args.model_name
    num_workers = args.num_workers
    task_id = args.task_id
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

                if task_id is not None and bug_id != task_id:
                    continue

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


if __name__ == "__main__":
    main()
