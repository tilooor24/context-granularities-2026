import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from model import Model
from patch_utils import insert_patch
from slice_utils import extract_context

MANYBUGS_ROOT = Path("benchmarks/ManyBugs/scenarios")
ASSETS_PATH = Path("generated_assets/ManyBugs/ManyBugs.jsonl")


def call_llm(llm: Model, buggy_line: str, context: str) -> str:
    prompt = f"""
You are repairing a single buggy line in a C program.

Buggy line:
{buggy_line}

Context:
{context}

Return ONLY the corrected C line.
- Do NOT include explanations
- Do NOT include comments
- Do NOT include backticks
- Do NOT include surrounding code
"""
    return llm.generate(prompt)


def get_file_path(
    bug_dir: Path, modified_file: str, metadata: list[str]
) -> Path:
    """
    Resolve the buggy file path from ManyBugs diffs directory.
    Returns ONLY the buggy version.
    """
    buggy_file_path = bug_dir / "diffs" / f"{modified_file}"
    print(f"Buggy file path: {buggy_file_path}")

    if not buggy_file_path.exists():
        candidates = list(buggy_file_path.parent.glob(f"{buggy_file_path.name}*"))
        assert len(candidates) == 1, (
            f"Could not uniquely resolve buggy file for {modified_file}\n"
            f"Candidates: {candidates}"
        )
        buggy_file_path = candidates[0]

    return buggy_file_path


def run_manybugs_tests(bug_dir: Path):
    test_script = bug_dir / "test.sh"

    if not test_script.exists():
        return False, "Missing test.sh"

    try:
        proc = subprocess.run(
            ["bash", "test.sh"],
            cwd=bug_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )

        output = proc.stdout + proc.stderr
        passed = proc.returncode == 0
        return passed, output

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


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

    llm = Model(
        model_name="gpt-4",
        temperature=1,
    )

    # -------------------------
    # Output file
    # -------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"many_bugs_results_{context_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_f = out_path.open("w")

    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))

            for hunk in hunks:
                # single-line bugs only
                if hunk["removed_line_numbers_range"][1] != 1:
                    continue
                if hunk["added_lines"].count("\n") > 1:
                    continue

                seen_tasks += 1
                if seen_tasks <= start_task_idx:
                    continue

                task_count += 1
                if max_tasks is not None and task_count >= max_tasks:
                    print(f"\nReached max_tasks={max_tasks}, stopping.")
                    out_f.close()
                    print(f"Results written to {out_path}")
                    return

                source_path = hunk["source_path"]
                buggy_line = hunk["removed_lines"].rstrip()
                line_no, bug_len = hunk["removed_line_numbers_range"]

                print(f"\n=== {bug_id}:{line_no} ===")
                print(f"Buggy line: {buggy_line}")

                # ----------------------------------
                # STEP 1: resolve buggy file
                # ----------------------------------
                bug_dir = MANYBUGS_ROOT / bug_id
                assert bug_dir.exists(), f"Missing bug dir: {bug_dir}"

                metadata = bug_id.split("-")

                try:
                    buggy_file = get_file_path(
                        bug_dir,
                        source_path,
                        metadata,
                    )
                except AssertionError as e:
                    print(f"[SKIP] {bug_id} – cannot resolve file for {source_path} - {e}")
                    continue

                print(f"Using buggy file: {buggy_file}")

                # ----------------------------------
                # STEP 2: extract context
                # ----------------------------------
                try:
                    context = extract_context(
                        buggy_file,
                        line_no,
                        context_type=context_type,
                    ) or ""
                    print(f"Context: {context}")
                except ValueError as e:
                    print(f"[CONTEXT FALLBACK] {bug_id}:{line_no} – {e}")
                    context = buggy_line  # or empty / window context

                # ----------------------------------
                # STEP 3: LLM repair
                # ----------------------------------
                patch = call_llm(
                    llm=llm,
                    buggy_line=buggy_line,
                    context=context,
                ).strip()

                print("LLM patch:", patch)

                # # ----------------------------------
                # # STEP 4: (optional) write patched copy
                # # ----------------------------------
                # patched_file = buggy_file.with_suffix(".patched.c")
                # insert_patch(
                #     patch,
                #     buggy_file,
                #     patched_file,
                #     line_no,
                #     bug_len,
                #     indent="",
                # )

                # ----------------------------------
                # STEP 4: apply patch + run tests
                # ----------------------------------
                original_src = buggy_file.read_text()

                try:
                    insert_patch(
                        patch,
                        buggy_file,
                        buggy_file,   # overwrite IN PLACE
                        line_no,
                        bug_len,
                        indent="",
                    )

                    passed, test_log = run_manybugs_tests(bug_dir)
                    print(f"Logs: {test_log}", flush=True)
                    exit(0)

                finally:
                    buggy_file.write_text(original_src)

                # ----------------------------------
                # WRITE RESULT
                # ----------------------------------
                # result = {
                #     "bug_id": bug_id,
                #     "line_no": line_no,
                #     "buggy_line": buggy_line,
                #     "context_type": context_type,
                #     "context": context,
                #     "patch": patch,
                #     "model_name": model_name,
                # }
                result = {
                    "bug_id": bug_id,
                    "line_no": line_no,
                    "buggy_line": buggy_line,
                    "context_type": context_type,
                    "context": context,
                    "patch": patch,
                    "model_name": model_name,
                    "tests_passed": passed,
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()   # important for long runs / crashes

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
