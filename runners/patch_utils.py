def insert_patch(patch,
                 source_file_path,
                 target_file_path,
                 bug_line,
                 bug_len,
                 indent):
    with open(source_file_path, encoding="cp1256") as file:
        lines = file.readlines()
    if bug_len == 0:
        lines.insert(bug_line, indent + patch + "\n")
    else:
        lines[bug_line - 1: (bug_line - 1) + bug_len] = indent + patch + "\n"

    with open(target_file_path, "w", encoding="cp1256") as file:
        file.writelines(lines)
