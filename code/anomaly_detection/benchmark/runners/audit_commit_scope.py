from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


DEFAULT_ROOT = Path("code/anomaly_detection/benchmark")
FORBIDDEN_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "runs",
    "checkpoints",
    "logs",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pth",
    ".pt",
    ".pkl",
    ".npy",
    ".npz",
    ".ckpt",
    ".onnx",
}
SECRET_PATTERNS = [
    re.compile(r"\broot@[\w.\-]+", re.IGNORECASE),
    re.compile(r"\bssh\s+-p\s+\d+", re.IGNORECASE),
    re.compile(r"\b(connect|login)\.[\w.\-]+", re.IGNORECASE),
    re.compile(r"-----BEGIN (OPENSSH|RSA|DSA|EC|PRIVATE) PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._\-]{10,}", re.IGNORECASE),
    re.compile(r"\b(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET[_-]?KEY)\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}['\"]?", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(password|passwd|token|secret)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class AuditIssue:
    path: str
    kind: str
    detail: str


def audit_paths(paths: Iterable[Path], repo_root: Path | None = None) -> list[AuditIssue]:
    """Check candidate commit paths for experiment artifacts and obvious secrets."""

    root = repo_root.resolve() if repo_root is not None else None
    issues: list[AuditIssue] = []
    for raw_path in paths:
        path = raw_path.resolve()
        display_path = _display_path(path, root)
        parts = set(path.parts)
        forbidden_dirs = sorted(parts.intersection(FORBIDDEN_DIRS))
        if forbidden_dirs:
            issues.append(
                AuditIssue(
                    path=display_path,
                    kind="forbidden_directory",
                    detail=",".join(forbidden_dirs),
                )
            )
            continue

        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(
                AuditIssue(
                    path=display_path,
                    kind="forbidden_suffix",
                    detail=path.suffix.lower(),
                )
            )
            continue

        if _looks_binary(path):
            issues.append(
                AuditIssue(path=display_path, kind="binary_file", detail="binary content")
            )
            continue

        text = _read_text_safely(path)
        if text is None:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(
                    AuditIssue(
                        path=display_path,
                        kind="possible_secret",
                        detail=pattern.pattern,
                    )
                )
                break
    return issues


def git_candidate_files(repo_root: Path, benchmark_root: Path) -> list[Path]:
    """Return tracked or untracked non-ignored benchmark files that Git can commit."""

    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        str(benchmark_root),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [repo_root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit benchmark files before committing them to GitHub."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    benchmark_root = args.benchmark_root
    if not benchmark_root.is_absolute():
        benchmark_root = repo_root / benchmark_root

    candidates = git_candidate_files(repo_root, benchmark_root)
    issues = audit_paths(candidates, repo_root=repo_root)
    payload = {
        "benchmark_root": _display_path(benchmark_root.resolve(), repo_root),
        "num_candidate_files": len(candidates),
        "num_issues": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if issues else 0


def _display_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return False
    return b"\0" in chunk


def _read_text_safely(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
