import json
import shutil
import tempfile
from pathlib import Path
from program_slicing.graph.parse import Lang
from program_slicing.graph.manager import ProgramGraphsManager
from program_slicing.decomposition.program_slice import ProgramSlice
from model import Model
from patch_utils import insert_patch
from d4j_utils import (
    checkout_bug,
    get_src_root,
    run_tests
)
from ast_utils import (
    get_java_parser,
    find_enclosing,
    find_node_at_line,
    find_stmt_at_line_number,
    preorder_traversal
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


def extract_forward_slice(source_code: str,
                          line_no: int,
                          ellipses: bool,
                          lang: Lang = Lang.JAVA) -> str:
    source_lines = source_code.splitlines()

    manager_by_source = ProgramGraphsManager(source_code, lang)
    manager_pdg = manager_by_source.program_dependence_graph
    target_stmt = find_stmt_at_line_number(
        list(manager_pdg.nodes),
        line_number=line_no)
    succs = dict(manager_pdg.succ)

    if target_stmt and target_stmt in succs:
        target_succs = preorder_traversal(
            target_stmt,
            set(),
            succs,
            source_lines)
        slice_nodes = set(target_succs) | {target_stmt}
        slice = ProgramSlice(
            source_lines=source_lines,
            context=manager_by_source).from_statements(slice_nodes)
        return slice

    return None


def extract_backward_slice(source_code: str,
                           line_no: int,
                           ellipses: bool) -> str:
    source_lines = source_code.splitlines()
    lang: Lang = Lang.JAVA

    manager_by_source = ProgramGraphsManager(source_code, lang)
    manager_pdg = manager_by_source.program_dependence_graph
    target_stmt = find_stmt_at_line_number(
        list(manager_pdg.nodes),
        line_number=line_no
    )
    preds = dict(manager_pdg.pred)

    if target_stmt and target_stmt in preds:
        target_preds = preorder_traversal(
            target_stmt,
            set(),
            preds,
            source_lines)
        slice_nodes = set(target_preds) | {target_stmt}
        slice = ProgramSlice(
            source_lines=source_lines,
            context=manager_by_source).from_statements(slice_nodes)
        return slice

    return None


def extract_context(
    file_path: Path,
    line_no: int,
    context_type: str = "method",
) -> str:
    """
    context_type ∈ {line, window, method, class}
    """
    parser = get_java_parser()

    code = file_path.read_text()
    lines = code.splitlines()

    if context_type == "line":
        return lines[line_no - 1]

    if context_type == "fixed":
        start = max(0, line_no - 6)
        end = min(len(lines), line_no + 5)
        return "\n".join(lines[start:end])

    # ---- Tree-sitter contexts ----
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    stmt_node = find_node_at_line(root, line_no)
    if stmt_node is None:
        raise ValueError(f"No AST node found at line {line_no}")

    if context_type == "method":
        method_node = find_enclosing(stmt_node, {"method_declaration"})
        if not method_node:
            raise ValueError("No enclosing method found")
        return code[method_node.start_byte:method_node.end_byte]

    if context_type == "class":
        class_node = find_enclosing(stmt_node, {"class_declaration"})
        if not class_node:
            raise ValueError("No enclosing class found")
        return code[class_node.start_byte:class_node.end_byte]

    if context_type == "backward_slice":
        slice = extract_backward_slice(code, line_no-1, False)
        print(f"Backward slice: {slice}\n\n")
        return slice

    if context_type == "forward_slice":
        slice = extract_forward_slice(code, line_no-1, False)
        print(f"Backward slice: {slice}\n\n")
        return slice

    raise ValueError(f"Unknown context_type: {context_type}")


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
