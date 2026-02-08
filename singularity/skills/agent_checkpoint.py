#!/usr/bin/env python3
"""
AgentCheckpointSkill - Full agent state snapshots for crash recovery, migration, and rollback.

This skill enables the agent to serialize its entire runtime state (all skill data files,
configuration, learned behaviors, experiments, goals) into versioned checkpoint snapshots.
Checkpoints can be restored to roll back after failed self-modifications, transferred to
replica agents for warm-start replication, or compared to measure progress over time.

Key capabilities:
- Create full or incremental checkpoints of all agent state
- Restore agent state from any checkpoint (rollback)
- List and compare checkpoints to see what changed
- Export checkpoint as a portable archive for replica warm-start
- Auto-checkpoint before risky operations (self-modify, deploy)
- Prune old checkpoints to manage disk usage
- Tag checkpoints with human-readable labels

Serves all four pillars:
- Self-Improvement: Save state before risky self-modifications, rollback on failure
- Revenue: Resume interrupted customer tasks from last good checkpoint
- Replication: Export checkpoint to seed new replicas with learned state
- Goal Setting: Compare checkpoints over time to measure concrete progress

Actions:
- save: Create a new checkpoint with optional label
- restore: Restore agent state from a specific checkpoint
- list: List all available checkpoints with metadata
- diff: Compare two checkpoints to see what changed
- export: Package a checkpoint for transfer to another agent
- import_checkpoint: Import a checkpoint from another agent
- prune: Remove old checkpoints based on retention policy
- auto_policy: Configure automatic checkpoint triggers
"""

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_INDEX = CHECKPOINT_DIR / "index.json"

# Files/directories to include in checkpoints
CHECKPOINT_SOURCES = [
    "*.json",  # All skill data files in data/
]

# Files to exclude from checkpoints (these are transient)
EXCLUDE_PATTERNS = [
    "checkpoints",  # Don't checkpoint the checkpoints
    "activity.json",  # Transient activity log
    "__pycache__",
]


def _load_index() -> Dict:
    """Load the checkpoint index."""
    if CHECKPOINT_INDEX.exists():
        try:
            with open(CHECKPOINT_INDEX, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"checkpoints": [], "auto_policy": {"enabled": False, "triggers": [], "max_checkpoints": 20}}


def _save_index(index: Dict):
    """Save the checkpoint index."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (IOError, OSError):
        return ""


def _collect_state_files(data_dir: Path) -> Dict[str, Dict]:
    """Collect all state files from the data directory with their hashes."""
    files = {}
    if not data_dir.exists():
        return files

    for item in sorted(data_dir.rglob("*")):
        if not item.is_file():
            continue
        rel = str(item.relative_to(data_dir))
        # Skip excluded patterns
        if any(excl in rel for excl in EXCLUDE_PATTERNS):
            continue
        files[rel] = {
            "size": item.stat().st_size,
            "hash": _hash_file(item),
            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
        }
    return files


class AgentCheckpointSkill(Skill):
    """
    Full agent state checkpointing for crash recovery, migration, and rollback.

    Creates versioned snapshots of all agent data (skill state, goals, experiments,
    learned behaviors) that can be restored, compared, or exported to replicas.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._data_dir = DATA_DIR
        self._checkpoint_dir = CHECKPOINT_DIR
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_checkpoint",
            name="Agent Checkpoint",
            version="1.0.0",
            category="core",
            description=(
                "Full agent state checkpointing for crash recovery, migration, "
                "rollback, and replica warm-start. Snapshots all skill data, "
                "goals, experiments, and learned behaviors."
            ),
            actions=[
                SkillAction(
                    name="save",
                    description="Create a checkpoint of current agent state",
                    parameters={
                        "label": {"type": "string", "required": False, "description": "Human-readable label for this checkpoint"},
                        "reason": {"type": "string", "required": False, "description": "Why this checkpoint was created"},
                    },
                ),
                SkillAction(
                    name="restore",
                    description="Restore agent state from a checkpoint",
                    parameters={
                        "checkpoint_id": {"type": "string", "required": True, "description": "ID of checkpoint to restore"},
                        "dry_run": {"type": "bool", "required": False, "description": "Preview what would change without actually restoring"},
                    },
                ),
                SkillAction(
                    name="list",
                    description="List all available checkpoints",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max checkpoints to return"},
                        "label_filter": {"type": "string", "required": False, "description": "Filter by label substring"},
                    },
                ),
                SkillAction(
                    name="diff",
                    description="Compare two checkpoints to see what changed",
                    parameters={
                        "checkpoint_a": {"type": "string", "required": True, "description": "First checkpoint ID"},
                        "checkpoint_b": {"type": "string", "required": True, "description": "Second checkpoint ID (or 'current' for live state)"},
                    },
                ),
                SkillAction(
                    name="export",
                    description="Package a checkpoint for transfer to another agent",
                    parameters={
                        "checkpoint_id": {"type": "string", "required": True, "description": "Checkpoint to export"},
                        "export_path": {"type": "string", "required": False, "description": "Where to write the export (default: data/exports/)"},
                    },
                ),
                SkillAction(
                    name="import_checkpoint",
                    description="Import a checkpoint from another agent",
                    parameters={
                        "import_path": {"type": "string", "required": True, "description": "Path to the exported checkpoint"},
                        "label": {"type": "string", "required": False, "description": "Label for the imported checkpoint"},
                    },
                ),
                SkillAction(
                    name="prune",
                    description="Remove old checkpoints based on retention policy",
                    parameters={
                        "keep_count": {"type": "int", "required": False, "description": "Number of recent checkpoints to keep (default: 10)"},
                        "keep_labeled": {"type": "bool", "required": False, "description": "Keep all labeled checkpoints (default: true)"},
                        "dry_run": {"type": "bool", "required": False, "description": "Preview what would be pruned"},
                    },
                ),
                SkillAction(
                    name="auto_policy",
                    description="Configure automatic checkpoint triggers",
                    parameters={
                        "enabled": {"type": "bool", "required": False, "description": "Enable/disable auto checkpoints"},
                        "triggers": {"type": "list", "required": False, "description": "Events that trigger auto-checkpoint (e.g. pre_self_modify, pre_deploy, hourly)"},
                        "max_checkpoints": {"type": "int", "required": False, "description": "Max auto-checkpoints to retain"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        """Execute a checkpoint action."""
        params = params or {}
        handlers = {
            "save": self._save,
            "restore": self._restore,
            "list": self._list,
            "diff": self._diff,
            "export": self._export,
            "import_checkpoint": self._import_checkpoint,
            "prune": self._prune,
            "auto_policy": self._auto_policy,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Checkpoint {action} failed: {e}")

    async def _save(self, params: Dict) -> SkillResult:
        """Create a new checkpoint."""
        label = params.get("label", "")
        reason = params.get("reason", "manual")

        checkpoint_id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        cp_dir = self._checkpoint_dir / checkpoint_id

        # Collect current state files
        state_files = _collect_state_files(self._data_dir)
        if not state_files:
            return SkillResult(
                success=True,
                message="Checkpoint created (empty state - no data files found)",
                data={"checkpoint_id": checkpoint_id, "files_count": 0},
            )

        # Copy all state files into checkpoint directory
        cp_dir.mkdir(parents=True, exist_ok=True)
        files_saved = 0
        total_size = 0
        for rel_path, info in state_files.items():
            src = self._data_dir / rel_path
            dst = cp_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(src), str(dst))
                files_saved += 1
                total_size += info["size"]
            except (IOError, OSError):
                continue

        # Create checkpoint metadata
        metadata = {
            "id": checkpoint_id,
            "label": label,
            "reason": reason,
            "created_at": datetime.now().isoformat(),
            "files_count": files_saved,
            "total_size_bytes": total_size,
            "file_manifest": state_files,
        }

        # Save metadata inside checkpoint
        with open(cp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update index
        index = _load_index()
        index["checkpoints"].append({
            "id": checkpoint_id,
            "label": label,
            "reason": reason,
            "created_at": metadata["created_at"],
            "files_count": files_saved,
            "total_size_bytes": total_size,
        })
        _save_index(index)

        return SkillResult(
            success=True,
            message=f"Checkpoint '{checkpoint_id}' saved with {files_saved} files ({total_size} bytes)",
            data={
                "checkpoint_id": checkpoint_id,
                "label": label,
                "files_count": files_saved,
                "total_size_bytes": total_size,
                "files": list(state_files.keys()),
            },
        )

    async def _restore(self, params: Dict) -> SkillResult:
        """Restore agent state from a checkpoint."""
        checkpoint_id = params.get("checkpoint_id", "")
        dry_run = params.get("dry_run", False)

        if not checkpoint_id:
            return SkillResult(success=False, message="checkpoint_id is required")

        cp_dir = self._checkpoint_dir / checkpoint_id
        meta_file = cp_dir / "metadata.json"

        if not cp_dir.exists() or not meta_file.exists():
            return SkillResult(success=False, message=f"Checkpoint '{checkpoint_id}' not found")

        with open(meta_file, "r") as f:
            metadata = json.load(f)

        manifest = metadata.get("file_manifest", {})

        # Calculate what would change
        current_files = _collect_state_files(self._data_dir)
        changes = {"restored": [], "overwritten": [], "removed": []}

        for rel_path in manifest:
            src = cp_dir / rel_path
            if not src.exists():
                continue
            if rel_path in current_files:
                if current_files[rel_path]["hash"] != manifest[rel_path]["hash"]:
                    changes["overwritten"].append(rel_path)
            else:
                changes["restored"].append(rel_path)

        # Files in current state but not in checkpoint
        for rel_path in current_files:
            if rel_path not in manifest:
                changes["removed"].append(rel_path)

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would restore {len(changes['restored'])} files, overwrite {len(changes['overwritten'])}, remove {len(changes['removed'])}",
                data={"changes": changes, "checkpoint": metadata},
            )

        # Auto-save current state before restoring
        pre_restore_result = await self._save({"label": f"pre-restore-{checkpoint_id}", "reason": "auto_pre_restore"})

        # Perform the restore
        restored_count = 0
        for rel_path in manifest:
            src = cp_dir / rel_path
            if not src.exists():
                continue
            dst = self._data_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(src), str(dst))
                restored_count += 1
            except (IOError, OSError):
                continue

        return SkillResult(
            success=True,
            message=f"Restored {restored_count} files from checkpoint '{checkpoint_id}'",
            data={
                "checkpoint_id": checkpoint_id,
                "restored_count": restored_count,
                "changes": changes,
                "pre_restore_checkpoint": pre_restore_result.data.get("checkpoint_id", ""),
            },
        )

    async def _list(self, params: Dict) -> SkillResult:
        """List all available checkpoints."""
        limit = params.get("limit", 50)
        label_filter = params.get("label_filter", "")

        index = _load_index()
        checkpoints = index.get("checkpoints", [])

        if label_filter:
            checkpoints = [cp for cp in checkpoints if label_filter.lower() in cp.get("label", "").lower()]

        # Sort by created_at descending
        checkpoints = sorted(checkpoints, key=lambda x: x.get("created_at", ""), reverse=True)
        checkpoints = checkpoints[:limit]

        total_size = sum(cp.get("total_size_bytes", 0) for cp in index.get("checkpoints", []))

        return SkillResult(
            success=True,
            message=f"Found {len(checkpoints)} checkpoints (total: {total_size} bytes)",
            data={
                "checkpoints": checkpoints,
                "total_count": len(index.get("checkpoints", [])),
                "total_size_bytes": total_size,
            },
        )

    async def _diff(self, params: Dict) -> SkillResult:
        """Compare two checkpoints."""
        cp_a_id = params.get("checkpoint_a", "")
        cp_b_id = params.get("checkpoint_b", "")

        if not cp_a_id or not cp_b_id:
            return SkillResult(success=False, message="Both checkpoint_a and checkpoint_b are required")

        # Load manifest for checkpoint A
        if cp_a_id == "current":
            manifest_a = _collect_state_files(self._data_dir)
        else:
            meta_a = self._checkpoint_dir / cp_a_id / "metadata.json"
            if not meta_a.exists():
                return SkillResult(success=False, message=f"Checkpoint '{cp_a_id}' not found")
            with open(meta_a, "r") as f:
                manifest_a = json.load(f).get("file_manifest", {})

        # Load manifest for checkpoint B
        if cp_b_id == "current":
            manifest_b = _collect_state_files(self._data_dir)
        else:
            meta_b = self._checkpoint_dir / cp_b_id / "metadata.json"
            if not meta_b.exists():
                return SkillResult(success=False, message=f"Checkpoint '{cp_b_id}' not found")
            with open(meta_b, "r") as f:
                manifest_b = json.load(f).get("file_manifest", {})

        # Compute diff
        all_files = set(list(manifest_a.keys()) + list(manifest_b.keys()))
        added = []
        removed = []
        modified = []
        unchanged = []

        for f in sorted(all_files):
            in_a = f in manifest_a
            in_b = f in manifest_b
            if in_a and not in_b:
                removed.append(f)
            elif not in_a and in_b:
                added.append(f)
            elif manifest_a[f]["hash"] != manifest_b[f]["hash"]:
                modified.append({
                    "file": f,
                    "size_a": manifest_a[f]["size"],
                    "size_b": manifest_b[f]["size"],
                })
            else:
                unchanged.append(f)

        return SkillResult(
            success=True,
            message=f"Diff: {len(added)} added, {len(removed)} removed, {len(modified)} modified, {len(unchanged)} unchanged",
            data={
                "added": added,
                "removed": removed,
                "modified": modified,
                "unchanged_count": len(unchanged),
                "checkpoint_a": cp_a_id,
                "checkpoint_b": cp_b_id,
            },
        )

    async def _export(self, params: Dict) -> SkillResult:
        """Export a checkpoint as a portable archive."""
        checkpoint_id = params.get("checkpoint_id", "")
        export_path = params.get("export_path", "")

        if not checkpoint_id:
            return SkillResult(success=False, message="checkpoint_id is required")

        cp_dir = self._checkpoint_dir / checkpoint_id
        if not cp_dir.exists():
            return SkillResult(success=False, message=f"Checkpoint '{checkpoint_id}' not found")

        # Default export path
        if not export_path:
            export_dir = self._data_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = str(export_dir / f"{checkpoint_id}.tar.gz")

        # Ensure parent directory exists
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)

        # Create a compressed archive
        import tarfile
        try:
            with tarfile.open(export_path, "w:gz") as tar:
                tar.add(str(cp_dir), arcname=checkpoint_id)
        except Exception as e:
            return SkillResult(success=False, message=f"Export failed: {e}")

        export_size = os.path.getsize(export_path) if os.path.exists(export_path) else 0

        return SkillResult(
            success=True,
            message=f"Exported checkpoint '{checkpoint_id}' to {export_path} ({export_size} bytes)",
            data={
                "checkpoint_id": checkpoint_id,
                "export_path": export_path,
                "export_size_bytes": export_size,
            },
        )

    async def _import_checkpoint(self, params: Dict) -> SkillResult:
        """Import a checkpoint from an exported archive."""
        import_path = params.get("import_path", "")
        label = params.get("label", "imported")

        if not import_path or not os.path.exists(import_path):
            return SkillResult(success=False, message=f"Import file not found: {import_path}")

        import tarfile
        try:
            with tarfile.open(import_path, "r:gz") as tar:
                # Security: validate no path traversal
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        return SkillResult(success=False, message="Import rejected: archive contains unsafe paths")

                # Extract to checkpoint dir
                tar.extractall(path=str(self._checkpoint_dir))

                # Get the checkpoint ID from the archive root
                top_dirs = {m.name.split("/")[0] for m in tar.getmembers()}
                if len(top_dirs) != 1:
                    return SkillResult(success=False, message="Import rejected: archive must contain exactly one checkpoint")

                checkpoint_id = top_dirs.pop()
        except Exception as e:
            return SkillResult(success=False, message=f"Import failed: {e}")

        # Load metadata and register in index
        meta_path = self._checkpoint_dir / checkpoint_id / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"id": checkpoint_id, "files_count": 0, "total_size_bytes": 0}

        # Update index
        index = _load_index()
        # Avoid duplicates
        existing_ids = {cp["id"] for cp in index["checkpoints"]}
        if checkpoint_id not in existing_ids:
            index["checkpoints"].append({
                "id": checkpoint_id,
                "label": label,
                "reason": "imported",
                "created_at": metadata.get("created_at", datetime.now().isoformat()),
                "files_count": metadata.get("files_count", 0),
                "total_size_bytes": metadata.get("total_size_bytes", 0),
            })
            _save_index(index)

        return SkillResult(
            success=True,
            message=f"Imported checkpoint '{checkpoint_id}' from {import_path}",
            data={
                "checkpoint_id": checkpoint_id,
                "label": label,
                "metadata": metadata,
            },
        )

    async def _prune(self, params: Dict) -> SkillResult:
        """Remove old checkpoints based on retention policy."""
        keep_count = params.get("keep_count", 10)
        keep_labeled = params.get("keep_labeled", True)
        dry_run = params.get("dry_run", False)

        index = _load_index()
        checkpoints = index.get("checkpoints", [])

        # Sort by created_at ascending (oldest first)
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x.get("created_at", ""))

        to_keep = []
        to_prune = []

        # Work from newest to oldest, keeping the most recent keep_count
        for cp in reversed(checkpoints_sorted):
            if len(to_keep) < keep_count:
                to_keep.append(cp)
            elif keep_labeled and cp.get("label", ""):
                to_keep.append(cp)
            else:
                to_prune.append(cp)

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would prune {len(to_prune)} checkpoints, keep {len(to_keep)}",
                data={
                    "to_prune": [cp["id"] for cp in to_prune],
                    "to_keep": [cp["id"] for cp in to_keep],
                },
            )

        # Actually delete
        pruned = []
        freed_bytes = 0
        for cp in to_prune:
            cp_dir = self._checkpoint_dir / cp["id"]
            if cp_dir.exists():
                try:
                    freed_bytes += sum(f.stat().st_size for f in cp_dir.rglob("*") if f.is_file())
                    shutil.rmtree(str(cp_dir))
                    pruned.append(cp["id"])
                except (IOError, OSError):
                    continue

        # Update index
        pruned_set = set(pruned)
        index["checkpoints"] = [cp for cp in index["checkpoints"] if cp["id"] not in pruned_set]
        _save_index(index)

        return SkillResult(
            success=True,
            message=f"Pruned {len(pruned)} checkpoints, freed {freed_bytes} bytes",
            data={
                "pruned": pruned,
                "freed_bytes": freed_bytes,
                "remaining_count": len(index["checkpoints"]),
            },
        )

    async def _auto_policy(self, params: Dict) -> SkillResult:
        """Configure automatic checkpoint triggers."""
        index = _load_index()
        policy = index.get("auto_policy", {"enabled": False, "triggers": [], "max_checkpoints": 20})

        updated = False
        if "enabled" in params:
            policy["enabled"] = bool(params["enabled"])
            updated = True
        if "triggers" in params:
            valid_triggers = {"pre_self_modify", "pre_deploy", "hourly", "daily", "on_error", "pre_restore"}
            triggers = params["triggers"] if isinstance(params["triggers"], list) else [params["triggers"]]
            invalid = [t for t in triggers if t not in valid_triggers]
            if invalid:
                return SkillResult(
                    success=False,
                    message=f"Invalid triggers: {invalid}. Valid: {sorted(valid_triggers)}",
                )
            policy["triggers"] = triggers
            updated = True
        if "max_checkpoints" in params:
            policy["max_checkpoints"] = max(1, int(params["max_checkpoints"]))
            updated = True

        index["auto_policy"] = policy
        if updated:
            _save_index(index)

        return SkillResult(
            success=True,
            message=f"Auto-checkpoint policy: enabled={policy['enabled']}, triggers={policy['triggers']}",
            data={"policy": policy},
        )

    async def should_auto_checkpoint(self, trigger: str) -> bool:
        """Check if an auto-checkpoint should be created for the given trigger."""
        index = _load_index()
        policy = index.get("auto_policy", {})
        if not policy.get("enabled", False):
            return False
        return trigger in policy.get("triggers", [])

    async def auto_save(self, trigger: str) -> Optional[SkillResult]:
        """Create an auto-checkpoint if the policy says so."""
        if await self.should_auto_checkpoint(trigger):
            return await self._save({"label": f"auto-{trigger}", "reason": f"auto_{trigger}"})
        return None
