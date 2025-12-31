from tree_sitter import Parser, Node, Language
from typing import List, Dict, Set, Optional
from program_slicing.graph.statement import Statement, StatementType


JAVA_LANGUAGE = Language(
    str("tools/tree-sitter-lib/" "build/my-languages.so"), "java")


# C_LANGUAGE = Language(
#     "tools/tree-sitter-lib/build/my-languages.so",
#     "c",
# )


def get_java_parser() -> Parser:
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    return parser


# def get_c_parser() -> Parser:
#     parser = Parser()
#     parser.set_language(C_LANGUAGE)
#     return parser


def find_enclosing(node: Node, target_types: set[str]) -> Node | None:
    cur = node
    while cur is not None:
        if cur.type in target_types:
            return cur
        cur = cur.parent
    return None


def find_node_at_line(root: Node, line_no: int) -> Node | None:
    """
    Find the smallest node that spans line_no (1-based).
    """
    target_row = line_no - 1  # tree-sitter uses 0-based rows

    def visit(node: Node) -> Node | None:
        start, end = node.start_point[0], node.end_point[0]
        if start <= target_row <= end:
            for child in node.children:
                found = visit(child)
                if found:
                    return found
            return node
        return None

    return visit(root)


def find_stmt_at_line_number(
    stmts: List[Statement],
    line_number: int
) -> Optional[Node]:
    min_start_point_col = float("inf")
    captured_stmt = None
    for stmt in stmts:
        if stmt.start_point[0] == line_number:
            if (
                stmt.start_point[1] < min_start_point_col
                and
                stmt.statement_type != StatementType.UNKNOWN
            ):
                min_start_point_col = stmt.start_point[1]
                captured_stmt = stmt
    return captured_stmt


def preorder_traversal(
    stmt: Statement,
    visited: Set[Statement],
    predecessors: Dict[Statement, List[Statement]],
    source_lines: List[str]
) -> None:
    if stmt in visited:
        return visited

    visited.add(stmt)

    preds = predecessors.get(stmt)
    if preds:
        for pred in preds:
            preorder_traversal(pred, visited, predecessors, source_lines)

    return visited
