# def insert_patch(patch,
#                  source_file_path,
#                  target_file_path,
#                  bug_line,
#                  bug_len,
#                  indent):
#     print(f"[insert_patch] writing line repr={repr(indent + patch)}", flush=True)

#     with open(source_file_path, encoding="cp1256") as file:
#         lines = file.readlines()
#     if bug_len == 0:
#         lines.insert(bug_line, indent + patch + "\n")
#     else:
#         lines[bug_line - 1: (bug_line - 1) + bug_len] = indent + patch + "\n"

#     with open(target_file_path, "w", encoding="cp1256") as file:
#         file.writelines(lines)


def insert_patch(
    patch,
    source_file_path,
    target_file_path,
    bug_line,
    bug_len,
    indent,
):
    # Keep leading whitespace in patch if it somehow exists; only remove trailing newlines.
    patch = "" if patch is None else patch
    patch = patch.rstrip("\r\n")

    # If the model ever returns a fenced block, extract the inside (single line).
    # (Safe no-op if no fences.)
    import re
    m = re.search(r"```(?:java)?\s*\n(.*?)\n```", patch, flags=re.DOTALL | re.IGNORECASE)
    if m:
        patch = m.group(1).rstrip("\r\n")

    # If patch got wrapped in quotes, remove exactly one outer pair.
    if len(patch) >= 2 and ((patch[0] == "'" and patch[-1] == "'") or (patch[0] == '"' and patch[-1] == '"')):
        patch = patch[1:-1]

    new_line = indent + patch + "\n"
    print(f"[insert_patch] indent repr={repr(indent)}", flush=True)
    print(f"[insert_patch] patch  repr={repr(patch)}", flush=True)
    print(f"[insert_patch] writing line repr={repr(new_line)}", flush=True)

    with open(source_file_path, encoding="cp1256") as file:
        lines = file.readlines()

    # bug_line is 1-based in your code (you use bug_line - 1 elsewhere)
    idx0 = bug_line - 1

    if bug_len == 0:
        # Insert BEFORE idx0
        lines.insert(idx0, new_line)
    else:
        # Replace bug_len lines with EXACTLY ONE line (must be a list!)
        lines[idx0: idx0 + bug_len] = [new_line]

    with open(target_file_path, "w", encoding="cp1256") as file:
        file.writelines(lines)