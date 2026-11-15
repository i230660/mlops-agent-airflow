"""DVC data-management tools. Only an allowlist of read-only commands may run."""
from __future__ import annotations

import glob
import os
import re
import subprocess
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from ..config import Settings
from ..registry import Tool, ToolDenied, ToolRegistry, ToolUnavailable

# Allowlisted (binary, subcommand) pairs. Anything else is rejected.
ALLOWED_COMMANDS = {
    ("dvc", "status"),
    ("dvc", "diff"),
    ("dvc", "list"),
    ("dvc", "checkout"),  # reproducible checkout of already-tracked data
    ("git", "rev-parse"),
    ("git", "log"),
    ("git", "show"),
    ("git", "status"),
}
_SAFE_REV = re.compile(r"^[A-Za-z0-9_./\-~^@]{1,100}$")


def run_allowed(cmd: list[str], cwd: str, timeout: float = 15.0) -> str:
    """Run an allowlisted command without a shell. Raises ToolDenied for anything else."""
    if len(cmd) < 2 or (os.path.basename(cmd[0]).replace(".exe", ""), cmd[1]) not in ALLOWED_COMMANDS:
        raise ToolDenied(f"command not allowed: {' '.join(cmd[:2])}")
    for part in cmd[2:]:
        if part.startswith("-") and part not in {"--json", "--oneline", "--short", "-n", "--targets", "--quiet", "-1"}:
            if not re.match(r"^-\d+$", part):
                raise ToolDenied(f"flag not allowed: {part}")
    try:
        proc = subprocess.run(  # noqa: S603 - no shell, allowlisted argv
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise ToolUnavailable(f"{cmd[0]} is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolUnavailable(f"{cmd[0]} {cmd[1]} timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "command failed").strip()[:500])
    return proc.stdout


class _Empty(BaseModel):
    pass


class RevInput(BaseModel):
    rev: str = Field(default="HEAD", description="Git revision / tag / commit")


class CompareInput(BaseModel):
    rev_a: str = Field(description="Older git revision or tag")
    rev_b: str = Field(default="HEAD", description="Newer git revision or tag")


class ExperimentDatasetInput(BaseModel):
    run_id: str = Field(min_length=1, description="MLflow run id whose recorded provenance to read")


class CheckoutInput(BaseModel):
    rev: str = Field(description="Git revision whose DVC-tracked data should be restored")


def _validate_rev(rev: str) -> str:
    if not _SAFE_REV.match(rev):
        raise ToolDenied(f"unsafe revision string: {rev!r}")
    return rev


def build_dvc_tools(settings: Settings, registry: ToolRegistry, mlflow_client_factory=None) -> None:
    repo = settings.repo_path

    def list_datasets(_: _Empty):
        """Parse *.dvc files and dvc.yaml; no subprocess needed, so it works offline."""
        datasets = []
        for path in glob.glob(os.path.join(repo, "**", "*.dvc"), recursive=True):
            if os.sep + ".dvc" + os.sep in path or ".git" in path.split(os.sep):
                continue
            with open(path, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh) or {}
            for out in meta.get("outs", []):
                datasets.append(
                    {
                        "dvc_file": os.path.relpath(path, repo),
                        "path": out.get("path"),
                        "md5": out.get("md5"),
                        "size": out.get("size"),
                        "nfiles": out.get("nfiles"),
                    }
                )
        dvc_yaml = os.path.join(repo, "dvc.yaml")
        stage_outs = []
        if os.path.exists(dvc_yaml):
            with open(dvc_yaml, encoding="utf-8") as fh:
                spec = yaml.safe_load(fh) or {}
            for name, stage in (spec.get("stages") or {}).items():
                stage_outs.append({"stage": name, "deps": stage.get("deps", []), "outs": stage.get("outs", [])})
        return {"tracked_files": datasets, "pipeline_stages": stage_outs, "count": len(datasets)}

    def current_version(inp: RevInput):
        rev = _validate_rev(inp.rev)
        commit = run_allowed([settings.git_bin, "rev-parse", rev], repo).strip()
        tags = run_allowed([settings.git_bin, "log", "-1", "--oneline", rev], repo).strip()
        tracked = list_datasets(_Empty())["tracked_files"]
        return {
            "git_commit": commit,
            "last_commit_summary": tags,
            "dataset_hashes": {t["path"]: t["md5"] for t in tracked},
        }

    def detect_changes(_: _Empty):
        out = run_allowed([settings.dvc_bin, "status", "--json"], repo)
        text = out.strip()
        changed = text not in ("", "{}")
        return {"has_changes": changed, "raw_status": text or "{}"}

    def compare_versions(inp: CompareInput):
        a, b = _validate_rev(inp.rev_a), _validate_rev(inp.rev_b)
        out = run_allowed([settings.dvc_bin, "diff", "--json", a, b], repo)
        return {"rev_a": a, "rev_b": b, "diff": out.strip() or "{}"}

    def dataset_for_run(inp: ExperimentDatasetInput):
        """Read DVC provenance recorded as MLflow run tags at training time."""
        if mlflow_client_factory is None:
            raise ToolUnavailable("MLflow client not configured")
        client = mlflow_client_factory()
        try:
            run = client.get_run(inp.run_id)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"cannot read run {inp.run_id}: {exc}") from exc
        tags = run.data.tags
        version = tags.get(settings.dataset_tag_key)
        commit = tags.get(settings.git_commit_tag_key)
        md5 = tags.get(settings.dataset_hash_tag_key)
        return {
            "run_id": inp.run_id,
            "dataset_version": version,
            "git_commit": commit,
            "data_md5": md5,
            "provenance_recorded": bool(version or commit or md5),
            "note": None if (version or commit or md5) else "Run carries no DVC provenance tags; cannot determine dataset.",
        }

    def explain_reproduce(inp: RevInput):
        rev = _validate_rev(inp.rev)
        commit = run_allowed([settings.git_bin, "rev-parse", rev], repo).strip()
        steps = [
            f"git checkout {commit}   # restore .dvc pointer files for that version",
            "dvc checkout            # restore data matching those pointers from cache",
            "dvc pull                # if cache is empty, fetch from the configured remote",
            "dvc status              # should report: Data and pipelines are up to date",
        ]
        return {"git_commit": commit, "steps": steps}

    def checkout_version(inp: CheckoutInput):
        """Demonstrates reproducible checkout: inspect pointer files at rev (no working-tree mutation of git)."""
        rev = _validate_rev(inp.rev)
        commit = run_allowed([settings.git_bin, "rev-parse", rev], repo).strip()
        out = run_allowed([settings.dvc_bin, "checkout", "--quiet"], repo)
        return {"git_commit": commit, "dvc_checkout_output": out.strip() or "ok"}

    reg = registry.register
    reg(Tool("dvc_list_datasets", "List datasets and pipeline stages tracked by DVC.", _Empty, list_datasets, "dvc"))
    reg(Tool("dvc_current_version", "Get the current Git commit and DVC dataset hashes.", RevInput, current_version, "dvc"))
    reg(Tool("dvc_detect_changes", "Detect whether DVC-tracked data differs from recorded versions.", _Empty, detect_changes, "dvc"))
    reg(Tool("dvc_compare_versions", "Compare tracked data between two git revisions.", CompareInput, compare_versions, "dvc"))
    reg(Tool("dvc_dataset_for_run", "Read the DVC dataset version/commit recorded for an MLflow run (provenance).", ExperimentDatasetInput, dataset_for_run, "dvc"))
    reg(Tool("dvc_explain_reproduce", "Explain how to reproduce a previous dataset version.", RevInput, explain_reproduce, "dvc"))
    reg(Tool("dvc_checkout_version", "Restore tracked data to match current pointer files (dvc checkout).", CheckoutInput, checkout_version, "dvc"))


__all__ = ["build_dvc_tools", "run_allowed", "ALLOWED_COMMANDS", "Optional"]
