import shlex
import subprocess
from pathlib import Path

DEFECTS4J = "defects4j"


def run(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True
    )


def checkout_bug(pid: str, bid: str, workdir: Path):
    run([DEFECTS4J, "checkout", "-p", pid, "-v", f"{bid}b", "-w", str(workdir)])


def get_src_root(workdir: Path) -> Path:
    res = run([DEFECTS4J, "export", "-p", "dir.src.classes"], cwd=workdir)
    return workdir / res.stdout.strip()


def run_tests(workdir: Path) -> bool:
    res = subprocess.run(
        [DEFECTS4J, "test"],
        cwd=workdir,
        text=True,
        capture_output=True
    )
    return "Failing tests: 0" in res.stdout


def run_d4j_cmd(cmd: str) -> str:
    project_root: Path = Path(__file__).parent.parent
    benchmarks_root: Path = project_root / "benchmarks"
    d4j_root: Path = benchmarks_root / "Defects4J"
    d4j_bin: Path = d4j_root / "framework/bin/defects4j"
    d4j_cmd = f"perl {d4j_bin} {cmd}"
    args = shlex.split(d4j_cmd)
    result = subprocess.run(args, capture_output=True, check=True, text=True)
    return result.stdout
