import json
import shutil
import tempfile
from pathlib import Path
from model import Model
from patch_utils import insert_patch
from slice_utils import extract_context
from d4j_utils import (
    checkout_bug,
    get_src_root,
    run_tests
)


ASSETS_PATH = Path("generated_assets/Defects4J/Defects4J.jsonl")


def call_llm(llm: Model, buggy_line: str, context: str) -> str:
    prompt = f"""
    Buggy line:
    {buggy_line}

    Context:
    {context}

    Return ONLY the corrected Java line.
    """
    return llm.generate(prompt)


def main():
    llm = Model(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            (bug_id, hunks) = next(iter(entry.items()))

            for hunk in hunks:
                # only single-line bugs
                if hunk["removed_line_numbers_range"][1] != 1:
                    continue

                pid, bid = bug_id.split()
                buggy_line = hunk["removed_lines"].rstrip()
                line_no, bug_len = hunk["removed_line_numbers_range"]

                print(f"\n=== {pid}-{bid}:{line_no} ===")
                print(f"Buggy line: {buggy_line}")

                checkout_dir = Path(tempfile.mkdtemp(prefix="d4j_"))

                try:
                    # --------------------------------------------------
                    # STEP 1: checkout buggy version
                    # --------------------------------------------------
                    checkout_bug(pid, bid, checkout_dir)

                    # --------------------------------------------------
                    # STEP 2: resolve source file
                    # --------------------------------------------------
                    src_root = get_src_root(checkout_dir)

                    source_path = Path(hunk["source_path"])
                    if source_path.parts[0] == "source":
                        source_path = Path(*source_path.parts[1:])

                    java_file = src_root / source_path
                    assert java_file.exists(), f"Missing file: {java_file}"

                    # --------------------------------------------------
                    # STEP 3: extract context
                    # --------------------------------------------------
                    context = extract_context(
                        java_file,
                        line_no,
                        context_type="backward_slice",
                    )

                    # --------------------------------------------------
                    # STEP 4: call LLM (single-line repair)
                    # --------------------------------------------------
                    patch = call_llm(
                        llm=llm,
                        buggy_line=buggy_line,
                        context=context,
                    ).strip()

                    print("LLM patch:", patch)

                    # --------------------------------------------------
                    # STEP 5: insert patch
                    # --------------------------------------------------
                    indent_size = (
                        len(hunk["added_lines"])
                        - len(hunk["added_lines"].lstrip(" \t"))
                    )
                    indent = hunk["added_lines"][:indent_size]

                    target_file_path = checkout_dir / hunk["source_path"]
                    source_file_path = target_file_path  # same file

                    insert_patch(
                        patch,
                        source_file_path,
                        target_file_path,
                        line_no,
                        bug_len,
                        indent,
                    )

                    # --------------------------------------------------
                    # STEP 6: run Defects4J tests
                    # --------------------------------------------------
                    passed = run_tests(checkout_dir)
                    print("LLM result:", passed)

                finally:
                    shutil.rmtree(checkout_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
