import json
from pathlib import Path

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


def main():
    llm = Model(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            bug_id, hunks = next(iter(entry.items()))

            for hunk in hunks:
                # single-line bugs only
                if hunk["removed_line_numbers_range"][1] != 1:
                    continue

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
                context = extract_context(
                    buggy_file,
                    line_no,
                    context_type="method",
                ) or ""
                print(f"Context: {context}")

                # ----------------------------------
                # STEP 3: LLM repair
                # ----------------------------------
                patch = call_llm(
                    llm=llm,
                    buggy_line=buggy_line,
                    context=context,
                ).strip()

                print("LLM patch:", patch)

                # ----------------------------------
                # STEP 4: (optional) write patched copy
                # ----------------------------------
                patched_file = buggy_file.with_suffix(".patched.c")
                insert_patch(
                    patch,
                    buggy_file,
                    patched_file,
                    line_no,
                    bug_len,
                    indent="",
                )


if __name__ == "__main__":
    main()
