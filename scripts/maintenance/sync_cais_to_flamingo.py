#!/usr/bin/env python3
"""Sync results files from CAIS server to Kubernetes pod."""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tqdm import tqdm

SSH_ALIAS = "cais"
SRC_DIR = "/data/saad_hossain/SafeTuneBed/results"
K8S_NAMESPACE = "u-tamperbench"
# Defaults to Tom's devbox
DEFAULT_K8S_JOB = "job/devbox-t-cpu"
DEST_DIR = "/tamperbench_data/cais_archive"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync results files from CAIS server to Kubernetes pod.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually transferring files",
    )
    parser.add_argument(
        "--job",
        default=DEFAULT_K8S_JOB,
        help=f"Kubernetes job to sync to (default: {DEFAULT_K8S_JOB})",
    )
    return parser.parse_args()


@dataclass
class _FileInfo:
    path: str
    size: int
    mtime: int  # Unix timestamp


class _Colors:
    RED: Final[str] = "\033[0;31m"
    GREEN: Final[str] = "\033[0;32m"
    YELLOW: Final[str] = "\033[1;33m"
    CYAN: Final[str] = "\033[0;36m"
    NC: Final[str] = "\033[0m"


def _log_info(msg: str) -> None:
    print(f"{_Colors.GREEN}[INFO]{_Colors.NC} {msg}")


def _log_warn(msg: str) -> None:
    print(f"{_Colors.YELLOW}[WARN]{_Colors.NC} {msg}")


def _log_error(msg: str) -> None:
    print(f"{_Colors.RED}[ERROR]{_Colors.NC} {msg}")


def _log_section(msg: str) -> None:
    print(f"\n{_Colors.CYAN}{'=' * 60}")
    print(f" {msg}")
    print(f"{'=' * 60}{_Colors.NC}\n")


def _run_ssh(command: str) -> str:
    """Run a command on the SSH server and return stdout."""
    result = subprocess.run(
        ["ssh", SSH_ALIAS, command],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _run_kubectl(command: str, k8s_job: str) -> str:
    """Run a command in the Kubernetes pod and return stdout."""
    result = subprocess.run(
        ["kubectl", "exec", "--namespace", K8S_NAMESPACE, k8s_job, "--", "sh", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _parse_file_list(output: str) -> dict[str, _FileInfo]:
    """Parse output from find command. Returns dict mapping path to _FileInfo."""
    files = {}
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            size, mtime, path = parts
            # mtime from find -printf '%T@' is a float like "1699123456.789"
            files[path] = _FileInfo(path=path, size=int(size), mtime=int(float(mtime)))
    return files


def _get_source_files() -> dict[str, _FileInfo]:
    """Get file list from source SSH server."""
    _log_info("Fetching file list from source server. This may take a while.")
    # Format: size (bytes), mtime (unix timestamp), relative path
    cmd = f"cd '{SRC_DIR}' && find . -type f -printf '%s %T@ %P\\n'"
    output = _run_ssh(cmd)
    files = _parse_file_list(output)
    _log_info(f"Found {len(files)} files on source")
    return files


def _get_dest_files(k8s_job: str) -> dict[str, _FileInfo]:
    """Get file list from Kubernetes pod."""
    _log_info("Fetching file list from Kubernetes pod...")
    cmd = f"cd '{DEST_DIR}' 2>/dev/null && find . -type f -printf '%s %T@ %P\\n' || true"
    output = _run_kubectl(cmd, k8s_job)
    files = _parse_file_list(output)
    _log_info(f"Found {len(files)} files on destination")
    return files


@dataclass
class _SyncPlan:
    new_files: list[str]  # Files to add (exist on source, not on dest)
    modified_files: list[str]  # Files to update (different size or newer mtime)
    orphaned_files: list[str]  # Files to potentially delete (exist on dest, not on source)
    unchanged_files: list[str]  # Files that are the same


def _compute_sync_plan(
    source: dict[str, _FileInfo],
    dest: dict[str, _FileInfo],
) -> _SyncPlan:
    """Compare source and destination to determine what needs syncing."""
    new_files = []
    modified_files = []
    orphaned_files = []
    unchanged_files = []

    # Check all source files
    for path, src_info in source.items():
        if path not in dest:
            new_files.append(path)
        else:
            dest_info = dest[path]
            # Consider modified if size differs OR source is newer
            # (using 1-second tolerance for mtime due to filesystem differences)
            if src_info.size != dest_info.size or src_info.mtime > dest_info.mtime + 1:
                modified_files.append(path)
            else:
                unchanged_files.append(path)

    # Check for orphaned files (on dest but not on source)
    for path in dest:
        if path not in source:
            orphaned_files.append(path)

    return _SyncPlan(
        new_files=sorted(new_files),
        modified_files=sorted(modified_files),
        orphaned_files=sorted(orphaned_files),
        unchanged_files=sorted(unchanged_files),
    )


def _transfer_files(
    files: list[str],
    source_files: dict[str, _FileInfo],
    description: str,
    k8s_job: str,
) -> None:
    """Stream files from source to destination via tar pipe."""
    if not files:
        return

    total_bytes = sum(source_files[f].size for f in files)
    _log_info(f"Transferring {len(files)} {description} ({total_bytes / 1e9:.2f} GB)...")

    # Create a temporary file list on the source server
    file_list_content = "\n".join(files)

    # Create temp file on remote
    remote_tmp = _run_ssh("mktemp").strip()

    local_tmp = None
    try:
        # Copy file list to source server
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(file_list_content)
            local_tmp = f.name

        subprocess.run(
            ["scp", "-q", local_tmp, f"{SSH_ALIAS}:{remote_tmp}"],
            check=True,
        )

        # Stream via tar: ssh tar -> kubectl tar
        ssh_proc = subprocess.Popen(
            ["ssh", SSH_ALIAS, f"cd '{SRC_DIR}' && tar --create --file=- --files-from='{remote_tmp}'"],
            stdout=subprocess.PIPE,
        )

        kubectl_proc = subprocess.Popen(
            [
                "kubectl",
                "exec",
                "--stdin",
                "--namespace",
                K8S_NAMESPACE,
                k8s_job,
                "--",
                "tar",
                "--extract",
                "--file=-",
                "--directory",
                DEST_DIR,
            ],
            stdin=subprocess.PIPE,
        )

        # Stream data through with progress tracking
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        ) as pbar:
            assert ssh_proc.stdout is not None
            assert kubectl_proc.stdin is not None
            while True:
                chunk = ssh_proc.stdout.read(65536)  # 64KB chunks
                if not chunk:
                    break
                kubectl_proc.stdin.write(chunk)
                pbar.update(len(chunk))
            kubectl_proc.stdin.close()

        ssh_proc.wait()
        kubectl_proc.wait()

        if ssh_proc.returncode != 0 or kubectl_proc.returncode != 0:
            _log_error("Transfer failed!")
            sys.exit(1)

    finally:
        # Clean up temp files
        if local_tmp:
            Path(local_tmp).unlink(missing_ok=True)
        _run_ssh(f"rm --force '{remote_tmp}'")

    _log_info(f"Successfully transferred {len(files)} files")


def _print_file_list(files: list[str], max_display: int = 20) -> None:
    """Print a list of files, truncating if too long."""
    for f in files[:max_display]:
        print(f"    {f}")
    if len(files) > max_display:
        print(f"    ... and {len(files) - max_display} more")


def _report_sync_plan(plan: _SyncPlan) -> None:
    """Print a summary of what will be synced."""
    _log_section("Sync Plan Summary")

    print(f"  New files to transfer:      {len(plan.new_files)}")
    print(f"  Modified files to update:   {len(plan.modified_files)}")
    print(f"  Unchanged files:            {len(plan.unchanged_files)}")
    print(f"  Orphaned files on dest:     {len(plan.orphaned_files)}")

    if plan.new_files:
        print(f"\n{_Colors.GREEN}New files:{_Colors.NC}")
        _print_file_list(plan.new_files)

    if plan.modified_files:
        print(f"\n{_Colors.YELLOW}Modified files:{_Colors.NC}")
        _print_file_list(plan.modified_files)

    if plan.orphaned_files:
        print(f"\n{_Colors.RED}Orphaned files (exist on dest but not source):{_Colors.NC}")
        print(f"{_Colors.RED}These would be deleted in a full sync:{_Colors.NC}")
        _print_file_list(plan.orphaned_files)


def _check_connectivity(k8s_job: str) -> None:
    """Verify we can reach both the SSH server and the Kubernetes pod."""
    _log_info("Checking connectivity...")

    # Check CAIS cluster
    try:
        subprocess.run(
            ["ssh", SSH_ALIAS, "echo ok"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError:
        _log_error(f"Cannot connect to SSH server '{SSH_ALIAS}'")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        _log_error(f"SSH connection to '{SSH_ALIAS}' timed out")
        sys.exit(1)

    # Check k8s job
    try:
        subprocess.run(
            ["kubectl", "exec", "--namespace", K8S_NAMESPACE, k8s_job, "--", "echo", "ok"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        _log_error(f"Cannot connect to Kubernetes pod '{k8s_job}' in namespace '{K8S_NAMESPACE}'")
        _log_error(f"kubectl error: {e.stderr.decode() if e.stderr else 'unknown'}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        _log_error("kubectl connection timed out")
        sys.exit(1)

    _log_info("Connectivity OK")


def main() -> None:
    """Entry point for the sync script."""
    args = _parse_args()
    dry_run = args.dry_run
    k8s_job = args.job

    if dry_run:
        _log_section(f"DRY RUN: Syncing {SSH_ALIAS}:{SRC_DIR} → k8s:{k8s_job}:{DEST_DIR}")
        _log_warn("No files will be transferred (dry-run mode)")
    else:
        _log_section(f"Syncing {SSH_ALIAS}:{SRC_DIR} → k8s:{k8s_job}:{DEST_DIR}")

    _check_connectivity(k8s_job)

    # Gather file information
    source_files = _get_source_files()
    dest_files = _get_dest_files(k8s_job)

    # Compute what needs to be done
    plan = _compute_sync_plan(source_files, dest_files)
    _report_sync_plan(plan)

    files_to_transfer = plan.new_files + plan.modified_files
    if not files_to_transfer:
        _log_info("Nothing to sync. Destination is up to date!")
        return

    if dry_run:
        _log_section("Dry Run Summary")
        _log_info(f"Would transfer {len(files_to_transfer)} files:")
        _log_info(f"  - {len(plan.new_files)} new files")
        _log_info(f"  - {len(plan.modified_files)} modified files")
        _log_info("Run without --dry-run to perform the actual sync.")
        return

    _run_kubectl(f"mkdir -p '{DEST_DIR}'", k8s_job)

    _log_section("Transferring Files")

    if plan.new_files:
        _transfer_files(plan.new_files, source_files, "new files", k8s_job)

    if plan.modified_files:
        _transfer_files(plan.modified_files, source_files, "modified files", k8s_job)

    _log_section("Sync Complete")
    _log_info(f"Transferred {len(files_to_transfer)} files total")

    if plan.orphaned_files:
        _log_warn(
            f"{len(plan.orphaned_files)} orphaned files remain on destination. "
            "Review the list above if cleanup is needed."
        )


if __name__ == "__main__":
    main()
