from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.audit_commit_scope import audit_paths


def test_audit_paths_allows_source_policy_and_config_files(tmp_path: Path) -> None:
    allowed = tmp_path / "code" / "anomaly_detection" / "benchmark" / "policies" / "policy.md"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("# Policy\n\nNo private values here.\n", encoding="utf-8")

    assert audit_paths([allowed], repo_root=tmp_path) == []


def test_audit_paths_rejects_artifacts_and_possible_secrets(tmp_path: Path) -> None:
    output_file = tmp_path / "code" / "anomaly_detection" / "benchmark" / "outputs" / "scores.jsonl"
    checkpoint = tmp_path / "code" / "anomaly_detection" / "benchmark" / "model.pth"
    suspicious_file = tmp_path / "code" / "anomaly_detection" / "benchmark" / "notes.txt"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    suspicious_file.write_text("ssh" + " -p " + "12345 " + "root@" + "example.invalid\n", encoding="utf-8")

    issues = audit_paths([output_file, checkpoint, suspicious_file], repo_root=tmp_path)

    assert [issue.kind for issue in issues] == [
        "forbidden_directory",
        "forbidden_suffix",
        "possible_secret",
    ]


def test_audit_paths_rejects_common_api_credentials(tmp_path: Path) -> None:
    private_key = tmp_path / "private_key.txt"
    bearer = tmp_path / "bearer.txt"
    api_key = tmp_path / "api_key.txt"
    aws_key = tmp_path / "aws_key.txt"
    private_key.write_text("-----BEGIN " + "OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    bearer.write_text("Authorization: " + "Bearer " + "abcde12345abcde12345\n", encoding="utf-8")
    api_key.write_text("API_KEY=" + "abcde12345abcde12345\n", encoding="utf-8")
    aws_key.write_text("AKIA" + "ABCDEFGHIJKLMNOP\n", encoding="utf-8")

    issues = audit_paths([private_key, bearer, api_key, aws_key], repo_root=tmp_path)

    assert [issue.kind for issue in issues] == [
        "possible_secret",
        "possible_secret",
        "possible_secret",
        "possible_secret",
    ]


def test_commit_scope_audit_cli_reports_clean_current_candidates() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/audit_commit_scope.py")

    result = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"num_issues": 0' in result.stdout
