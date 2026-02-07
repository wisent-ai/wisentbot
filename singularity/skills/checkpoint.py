#!/usr/bin/env python3
"""
Checkpoint Skill - Persistent agent state management.

Enables agents to save and restore their state across restarts.
This is foundational infrastructure that all four pillars need:
- Self-Improvement: track performance over time
- Revenue: persist financial records
- Replication: remember spawn history
- Goal Setting: maintain goals across sessions

State is saved to JSON files in the agent's data directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

# Default checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "checkpoints"


class CheckpointSkill(Skill):
    """
    Persistent state management for autonomous agents.

    Saves and restores agent state (balance, cycle, recent actions,
    custom data) to JSON files, enabling continuity across restarts.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._agent_name: Optional[str] = None
        self._checkpoint_dir: Path = CHECKPOINT_DIR
        self._get_state: Optional[Callable] = None
        self._set_state: Optional[Callable] = None
        self._auto_save_interval: int = 0  # 0 = disabled
        self._cycles_since_save: int = 0
        self._custom_data: Dict[str, Any] = {}

    def set_agent_hooks(
        self,
        agent_name: str,
        get_state: Callable,
        set_state: Callable,
        checkpoint_dir: Optional[str] = None,
    ):
        """Wire up agent state access.

        Args:
            agent_name: Agent identifier for checkpoint files
            get_state: Callable returning agent state dict
            set_state: Callable accepting state dict to restore
            checkpoint_dir: Optional custom checkpoint directory
        """
        self._agent_name = agent_name.lower().replace(" ", "_")
        self._get_state = get_state
        self._set_state = set_state
        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="checkpoint",
            name="Checkpoint",
            version="1.0.0",
            category="persistence",
            description="Save and restore agent state across restarts",
            actions=[
                SkillAction(
                    name="save",
                    description="Save current agent state to a checkpoint file",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Optional label for this checkpoint (default: auto-generated timestamp)",
                        },
                        "note": {
                            "type": "string",
                            "required": False,
                            "description": "Optional note about why this checkpoint was created",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="restore",
                    description="Restore agent state from a checkpoint file",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Checkpoint label to restore (default: latest)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all available checkpoints",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="delete",
                    description="Delete a checkpoint",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": True,
                            "description": "Checkpoint label to delete",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="diff",
                    description="Compare current state with a checkpoint",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Checkpoint label to compare against (default: latest)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_auto_save",
                    description="Configure automatic checkpointing every N cycles",
                    parameters={
                        "interval": {
                            "type": "integer",
                            "required": True,
                            "description": "Save every N cycles (0 to disable)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="store",
                    description="Store custom key-value data in the checkpoint",
                    parameters={
                        "key": {
                            "type": "string",
                            "required": True,
                            "description": "Key to store data under",
                        },
                        "value": {
                            "type": "string",
                            "required": True,
                            "description": "Value to store (will be JSON-parsed if valid JSON)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="retrieve",
                    description="Retrieve custom data from the latest checkpoint",
                    parameters={
                        "key": {
                            "type": "string",
                            "required": False,
                            "description": "Key to retrieve (omit for all custom data)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return self._get_state is not None and self._set_state is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._get_state or not self._set_state:
            return SkillResult(
                success=False,
                message="Checkpoint skill not wired up. Agent hooks required.",
            )

        handlers = {
            "save": self._save,
            "restore": self._restore,
            "list": self._list,
            "delete": self._delete,
            "diff": self._diff,
            "set_auto_save": self._set_auto_save,
            "store": self._store,
            "retrieve": self._retrieve,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _get_checkpoint_path(self, label: str) -> Path:
        """Get the file path for a checkpoint label."""
        safe_label = label.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._checkpoint_dir / self._agent_name / f"{safe_label}.json"

    def _generate_label(self) -> str:
        """Generate a timestamp-based label."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _list_checkpoints(self) -> List[Dict]:
        """List all checkpoints for this agent."""
        agent_dir = self._checkpoint_dir / self._agent_name
        if not agent_dir.exists():
            return []

        checkpoints = []
        for f in sorted(agent_dir.glob("*.json")):
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                checkpoints.append(
                    {
                        "label": f.stem,
                        "created_at": data.get("created_at", "unknown"),
                        "note": data.get("note", ""),
                        "balance": data.get("agent_state", {}).get("balance"),
                        "cycle": data.get("agent_state", {}).get("cycle"),
                        "file_size": f.stat().st_size,
                    }
                )
            except (json.JSONDecodeError, OSError):
                checkpoints.append(
                    {
                        "label": f.stem,
                        "created_at": "unknown",
                        "note": "corrupted",
                        "balance": None,
                        "cycle": None,
                        "file_size": f.stat().st_size if f.exists() else 0,
                    }
                )

        return checkpoints

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint file."""
        agent_dir = self._checkpoint_dir / self._agent_name
        if not agent_dir.exists():
            return None

        files = sorted(agent_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        return files[-1] if files else None

    # === Action handlers ===

    async def _save(self, params: Dict) -> SkillResult:
        """Save current agent state to a checkpoint."""
        label = params.get("label", "").strip() or self._generate_label()
        note = params.get("note", "").strip()

        try:
            state = self._get_state()

            checkpoint = {
                "agent_name": self._agent_name,
                "label": label,
                "note": note,
                "created_at": datetime.now().isoformat(),
                "agent_state": state,
                "custom_data": self._custom_data,
                "auto_save_interval": self._auto_save_interval,
            }

            path = self._get_checkpoint_path(label)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)

            return SkillResult(
                success=True,
                message=f"Checkpoint '{label}' saved",
                data={
                    "label": label,
                    "path": str(path),
                    "balance": state.get("balance"),
                    "cycle": state.get("cycle"),
                    "note": note,
                },
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Save failed: {e}")

    async def _restore(self, params: Dict) -> SkillResult:
        """Restore agent state from a checkpoint."""
        label = params.get("label", "").strip()

        try:
            if label:
                path = self._get_checkpoint_path(label)
            else:
                path = self._get_latest_checkpoint()
                if not path:
                    return SkillResult(
                        success=False, message="No checkpoints found"
                    )

            if not path.exists():
                return SkillResult(
                    success=False,
                    message=f"Checkpoint not found: {label or 'latest'}",
                )

            with open(path, "r") as f:
                checkpoint = json.load(f)

            agent_state = checkpoint.get("agent_state", {})
            self._set_state(agent_state)

            # Restore custom data
            self._custom_data = checkpoint.get("custom_data", {})
            self._auto_save_interval = checkpoint.get("auto_save_interval", 0)

            actual_label = checkpoint.get("label", path.stem)

            return SkillResult(
                success=True,
                message=f"Restored from checkpoint '{actual_label}'",
                data={
                    "label": actual_label,
                    "created_at": checkpoint.get("created_at"),
                    "note": checkpoint.get("note", ""),
                    "restored_balance": agent_state.get("balance"),
                    "restored_cycle": agent_state.get("cycle"),
                    "custom_data_keys": list(self._custom_data.keys()),
                },
            )
        except json.JSONDecodeError:
            return SkillResult(
                success=False, message="Checkpoint file is corrupted"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Restore failed: {e}")

    async def _list(self, params: Dict) -> SkillResult:
        """List all available checkpoints."""
        checkpoints = self._list_checkpoints()

        return SkillResult(
            success=True,
            message=f"Found {len(checkpoints)} checkpoint(s)",
            data={
                "checkpoints": checkpoints,
                "count": len(checkpoints),
                "auto_save_interval": self._auto_save_interval,
            },
        )

    async def _delete(self, params: Dict) -> SkillResult:
        """Delete a checkpoint."""
        label = params.get("label", "").strip()
        if not label:
            return SkillResult(success=False, message="Label required")

        path = self._get_checkpoint_path(label)
        if not path.exists():
            return SkillResult(
                success=False, message=f"Checkpoint not found: {label}"
            )

        try:
            path.unlink()
            return SkillResult(
                success=True,
                message=f"Deleted checkpoint '{label}'",
                data={"label": label},
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Delete failed: {e}")

    async def _diff(self, params: Dict) -> SkillResult:
        """Compare current state with a checkpoint."""
        label = params.get("label", "").strip()

        try:
            if label:
                path = self._get_checkpoint_path(label)
            else:
                path = self._get_latest_checkpoint()
                if not path:
                    return SkillResult(
                        success=False, message="No checkpoints found"
                    )

            if not path.exists():
                return SkillResult(
                    success=False,
                    message=f"Checkpoint not found: {label or 'latest'}",
                )

            with open(path, "r") as f:
                checkpoint = json.load(f)

            saved_state = checkpoint.get("agent_state", {})
            current_state = self._get_state()

            # Compute differences
            differences = {}
            all_keys = set(list(saved_state.keys()) + list(current_state.keys()))

            for key in all_keys:
                saved_val = saved_state.get(key)
                current_val = current_state.get(key)

                if saved_val != current_val:
                    # For numeric values, compute delta
                    if isinstance(saved_val, (int, float)) and isinstance(
                        current_val, (int, float)
                    ):
                        differences[key] = {
                            "saved": saved_val,
                            "current": current_val,
                            "delta": current_val - saved_val,
                        }
                    else:
                        # Truncate long values for readability
                        sv = str(saved_val)[:200] if saved_val is not None else None
                        cv = (
                            str(current_val)[:200]
                            if current_val is not None
                            else None
                        )
                        differences[key] = {"saved": sv, "current": cv}

            actual_label = checkpoint.get("label", path.stem)

            return SkillResult(
                success=True,
                message=f"Compared with checkpoint '{actual_label}': {len(differences)} difference(s)",
                data={
                    "checkpoint_label": actual_label,
                    "checkpoint_created": checkpoint.get("created_at"),
                    "differences": differences,
                    "diff_count": len(differences),
                },
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Diff failed: {e}")

    async def _set_auto_save(self, params: Dict) -> SkillResult:
        """Configure automatic checkpointing."""
        interval = params.get("interval", 0)

        if not isinstance(interval, int) or interval < 0:
            return SkillResult(
                success=False, message="Interval must be a non-negative integer"
            )

        self._auto_save_interval = interval
        self._cycles_since_save = 0

        if interval == 0:
            return SkillResult(
                success=True,
                message="Auto-save disabled",
                data={"auto_save_interval": 0},
            )

        return SkillResult(
            success=True,
            message=f"Auto-save enabled every {interval} cycle(s)",
            data={"auto_save_interval": interval},
        )

    async def _store(self, params: Dict) -> SkillResult:
        """Store custom key-value data."""
        key = params.get("key", "").strip()
        value = params.get("value", "")

        if not key:
            return SkillResult(success=False, message="Key required")

        # Try to JSON-parse the value
        parsed_value = value
        if isinstance(value, str):
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value

        self._custom_data[key] = parsed_value

        return SkillResult(
            success=True,
            message=f"Stored '{key}'",
            data={
                "key": key,
                "value_type": type(parsed_value).__name__,
                "hint": "Use checkpoint:save to persist this data",
            },
        )

    async def _retrieve(self, params: Dict) -> SkillResult:
        """Retrieve custom data."""
        key = params.get("key", "").strip()

        if key:
            if key not in self._custom_data:
                return SkillResult(
                    success=False,
                    message=f"Key not found: {key}",
                    data={"available_keys": list(self._custom_data.keys())},
                )
            return SkillResult(
                success=True,
                message=f"Retrieved '{key}'",
                data={"key": key, "value": self._custom_data[key]},
            )

        return SkillResult(
            success=True,
            message=f"{len(self._custom_data)} custom data entries",
            data={
                "custom_data": self._custom_data,
                "keys": list(self._custom_data.keys()),
            },
        )

    async def maybe_auto_save(self) -> Optional[SkillResult]:
        """Called each cycle to handle auto-save. Returns result if saved."""
        if self._auto_save_interval <= 0:
            return None

        self._cycles_since_save += 1
        if self._cycles_since_save >= self._auto_save_interval:
            self._cycles_since_save = 0
            return await self._save({"label": f"auto_{self._generate_label()}"})

        return None
