import json
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from patch_utils import insert_patch
from slice_utils import extract_context
from score_utils import calculate_pass_at_k

from model import Model

ASSETS_PATH = Path("generated_assets/Bears/Bears.jsonl")
project_root: Path = Path(__file__).parent.parent
benchmarks_root: Path = project_root / "benchmarks"
bears_root: Path = benchmarks_root / "Bears"


# def checkout_bears_bug(branch: str, checkout_dir: Path) -> None:
#     """
#     Create a BEARS buggy checkout using git worktrees.
#     """
#     subprocess.run(
#         [
#             "git",
#             "-C",
#             str(bears_root),
#             "worktree",
#             "add",
#             str(checkout_dir),
#             branch],
#         check=True,
#     )

#     # BEARS convention: buggy version is HEAD~2
#     subprocess.run(
#         ["git", "-C", str(checkout_dir), "checkout", "HEAD~2"],
#         check=True,
#     )

#     # Detach worktree from git metadata (important!)
#     git_file = checkout_dir / ".git"
#     if git_file.exists():
#         git_file.unlink()


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
    prompt = f"""
    Buggy line:
    {buggy_line}

    Context:
    {context}

    Return ONLY the corrected Java line.
    """

    patches = []
    for _ in range(n):
        patch = llm.generate(prompt).strip()
        patches.append(patch)

    return patches


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

    args = parser.parse_args()

    context_type = args.context_type
    start_task_idx = args.start_task_idx
    max_tasks = args.max_tasks
    model_name = args.model_name

    task_count = 0
    seen_tasks = 0

    NUM_SAMPLES = 20  # number of patches per bug

    llm = Model(
        model_name=model_name,
        temperature=0.7,  # IMPORTANT for pass@k
    )

    # -------------------------
    # Output file
    # -------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"bears_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_f = out_path.open("w")

    # -------------------------
    # Bug → branch map
    # -------------------------
    BUG_BRANCH_MAP: dict[str, str] = {}
    with open("benchmarks/Bears/scripts/data/bug_id_and_branch.json") as f:
        data = json.load(f)
        for entry in data:
            BUG_BRANCH_MAP[entry["bugId"]] = entry["bugBranch"]

    # -------------------------
    # Loop over BEARS JSON
    # -------------------------
    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))

            # Only process bugs that have exactly one hunk
            if len(hunks) != 1:
                continue

            hunk = hunks[0]

            # Only process single-line fixes
            if hunk["added_lines"].count("\n") > 1:
                continue

            seen_tasks += 1

            # Skip until we reach the start index
            if seen_tasks <= start_task_idx:
                continue

            task_count += 1
            if max_tasks is not None and task_count >= max_tasks:
                print(f"\nReached max_tasks={max_tasks}, stopping.")
                out_f.close()
                print(f"Results written to {out_path}")
                return

            branch = BUG_BRANCH_MAP[bug_id]
            source_path = Path(hunk["source_path"])
            buggy_line = hunk["removed_lines"].rstrip()
            expected_fix = hunk["added_lines"].rstrip()
            line_no, bug_len = hunk["removed_line_numbers_range"]

            print(f"\n=== {bug_id}:{line_no} ===")
            print(f"Buggy line:    {buggy_line}")
            print(f"Expected fix: {expected_fix}")

            # -------------------------
            # STEP 1: checkout buggy version (context only)
            # -------------------------
            checkout_dir = Path(tempfile.mkdtemp(prefix="bears_ctx_"))
            try:
                ok = checkout_bears_bug(branch, checkout_dir)
                if not ok:
                    print(f"[WARN] Skipping Bears bug {bug_id} due to worktree conflict")
                    continue

                java_file = checkout_dir / source_path
                assert java_file.exists(), f"Missing file: {java_file}"

                # -------------------------
                # STEP 2: extract context
                # -------------------------
                try:
                    context = extract_context(java_file, line_no, context_type=context_type)
                    print(f"Context: {context}")
                except Exception as e:
                    print(f"Skipping bug {bug_id}:{line_no} — context extraction failed: {e}")
                    continue

            finally:
                cleanup_bears_worktree(checkout_dir)

            # -------------------------
            # STEP 3: LLM repair
            # -------------------------
            patches = call_llm(
                llm=llm,
                buggy_line=buggy_line,
                context=context,
                n=NUM_SAMPLES,
            )

            scores = []  # 1 = pass, 0 = fail
            all_patches = []  # store patches

            for i, patch in enumerate(patches):
                all_patches.append(patch)
                print(f"\n--- Sample {i+1}/{NUM_SAMPLES} ---")
                print("Patch:", patch)
                print(f"Expected fix: {expected_fix}")

                sample_checkout_dir = Path(tempfile.mkdtemp(prefix="bears_sample_"))
                try:
                    try:
                        checkout_bears_bug(branch, sample_checkout_dir)
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ Failed to checkout branch for sample {i+1}: {e}")
                        scores.append(0)  # mark as failed
                        continue  # skip this sample and go to the next patch

                    sample_java_file = sample_checkout_dir / source_path
                    if not sample_java_file.exists():
                        print(f"⚠️ Missing file: {sample_java_file}")
                        scores.append(0)
                        continue

                    # -------------------------
                    # STEP 4: insert patch
                    # -------------------------
                    indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
                    indent = hunk["added_lines"][:indent_size]

                    insert_patch(
                        patch,
                        sample_java_file,
                        sample_java_file,
                        line_no,
                        bug_len,
                        indent,
                    )

                    # -------------------------
                    # STEP 5: run tests
                    # -------------------------
                    passed = run_bears_tests(sample_checkout_dir)
                    scores.append(1 if passed else 0)
                    print("LLM result:", passed)

                finally:
                    cleanup_bears_worktree(sample_checkout_dir)

            # -------------------------
            # Record results
            # -------------------------
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
            }

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            print(f"Scores: {scores}")

    out_f.close()
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
