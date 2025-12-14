import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from patch_utils import insert_patch
from slice_utils import extract_context

from model import Model

ASSETS_PATH = Path("generated_assets/Bears/Bears.jsonl")
project_root: Path = Path(__file__).parent.parent
benchmarks_root: Path = project_root / "benchmarks"
bears_root: Path = benchmarks_root / "Bears"


def checkout_bears_bug(branch: str, checkout_dir: Path) -> None:
    """
    Create a BEARS buggy checkout using git worktrees.
    """
    subprocess.run(
        [
            "git",
            "-C",
            str(bears_root),
            "worktree",
            "add",
            str(checkout_dir),
            branch],
        check=True,
    )

    # BEARS convention: buggy version is HEAD~2
    subprocess.run(
        ["git", "-C", str(checkout_dir), "checkout", "HEAD~2"],
        check=True,
    )

    # Detach worktree from git metadata (important!)
    git_file = checkout_dir / ".git"
    if git_file.exists():
        git_file.unlink()


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

    BUG_BRANCH_MAP: dict[str, str] = {}
    with open("benchmarks/Bears/scripts/data/bug_id_and_branch.json") as f:
        data = json.load(f)
        for entry in data:
            BUG_BRANCH_MAP[entry["bugId"]] = entry["bugBranch"]

    with ASSETS_PATH.open() as f:
        for raw in f:
            entry = json.loads(raw)
            (bug_id, hunks) = next(iter(entry.items()))

            for hunk in hunks:
                # Only single-line bugs
                if hunk["removed_line_numbers_range"][1] != 1:
                    continue

                branch = BUG_BRANCH_MAP[bug_id]
                source_path = Path(hunk["source_path"])
                buggy_line = hunk["removed_lines"].rstrip()
                line_no, bug_len = hunk["removed_line_numbers_range"]

                print(f"\n=== {bug_id}:{line_no} ===")
                print(f"Buggy line: {buggy_line}")

                checkout_dir = Path(tempfile.mkdtemp(prefix="bears_"))

                try:
                    # ---------------------------------------------
                    # STEP 1: checkout buggy version
                    # ---------------------------------------------
                    checkout_bears_bug(branch, checkout_dir)

                    java_file = checkout_dir / source_path
                    assert java_file.exists(), f"Missing file: {java_file}"

                    # ---------------------------------------------
                    # STEP 2: extract context
                    # ---------------------------------------------
                    context = extract_context(
                        java_file,
                        line_no,
                        context_type="backward_slice",
                    )
                    print(f"Context: {context}")

                    # ---------------------------------------------
                    # STEP 3: LLM repair
                    # ---------------------------------------------
                    patch = call_llm(
                        llm=llm,
                        buggy_line=buggy_line,
                        context=context,
                    ).strip()

                    print("LLM patch:", patch)

                    # ---------------------------------------------
                    # STEP 4: insert patch
                    # ---------------------------------------------
                    indent_size = (
                        len(hunk["added_lines"])
                        - len(hunk["added_lines"].lstrip(" \t"))
                    )
                    indent = hunk["added_lines"][:indent_size]

                    insert_patch(
                        patch,
                        java_file,
                        java_file,
                        line_no,
                        bug_len,
                        indent,
                    )

                    # ---------------------------------------------
                    # STEP 5: run tests
                    # ---------------------------------------------
                    passed = run_bears_tests(checkout_dir)
                    print("LLM result:", passed)

                finally:
                    cleanup_bears_worktree(checkout_dir)


if __name__ == "__main__":
    main()
