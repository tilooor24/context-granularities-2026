from pathlib import Path
from tree_sitter import Parser
from program_slicing.graph.parse import Lang
from program_slicing.graph.manager import ProgramGraphsManager
from program_slicing.decomposition.program_slice import ProgramSlice
from ast_utils import (
    get_java_parser,
    get_c_parser,
    find_enclosing,
    find_node_at_line,
    find_stmt_at_line_number,
    preorder_traversal
)


def infer_language(file_path: Path) -> str:
    """
    Infer source language from ManyBugs / Defects4J style filenames.
    """
    name = file_path.name.lower()

    if ".java" in name:
        return "java"

    if ".c" in name or ".h" in name:
        return "c"

    raise ValueError(f"Cannot infer language from filename: {file_path}")


def get_parser_for_file(file_path: Path) -> Parser | None:
    lang = infer_language(file_path)

    if lang == "java":
        return get_java_parser()
    elif lang == "c":
        return get_c_parser()
    else:
        raise ValueError(f"Unsupported language: {lang}")


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
    lang = infer_language(file_path)
    parser = get_parser_for_file(file_path)

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
        target_types = (
            {"method_declaration"} if lang == "java"
            else {"function_definition"}  # C
        )

        enclosing = find_enclosing(stmt_node, target_types)
        if not enclosing:
            raise ValueError("No enclosing method/function found")

        return code[enclosing.start_byte : enclosing.end_byte]

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
