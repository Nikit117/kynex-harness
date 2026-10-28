"""Task memory: subtasks, unresolved fields, blocked tasks, verification needs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.schemas import TaskStatus, make_id


@dataclass
class SubTask:
    task_id: str
    description: str
    need_id: Optional[str] = None
    status: str = TaskStatus.OPEN.value
    attempts: int = 0
    max_attempts: int = 3
    result: str = ""
    blocked_reason: str = ""


@dataclass
class TaskLedger:
    tasks: Dict[str, SubTask] = field(default_factory=dict)

    def add(self, description: str, need_id: Optional[str] = None) -> SubTask:
        task = SubTask(
            task_id=make_id("task"), description=description, need_id=need_id
        )
        self.tasks[task.task_id] = task
        return task

    def open_tasks(self) -> List[SubTask]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.OPEN.value]

    def mark_done(self, task_id: str, result: str = "") -> None:
        task = self.tasks[task_id]
        task.status = TaskStatus.DONE.value
        task.result = result

    def mark_blocked(self, task_id: str, reason: str) -> None:
        task = self.tasks[task_id]
        task.status = TaskStatus.BLOCKED.value
        task.blocked_reason = reason

    def retry(self, task_id: str) -> bool:
        """Re-open a blocked task while attempts remain. Returns False if exhausted."""
        task = self.tasks[task_id]
        task.attempts += 1
        if task.attempts >= task.max_attempts:
            return False
        task.status = TaskStatus.OPEN.value
        task.blocked_reason = ""
        return True

    def unresolved(self) -> List[SubTask]:
        return [
            t
            for t in self.tasks.values()
            if t.status in (TaskStatus.OPEN.value, TaskStatus.BLOCKED.value)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "need_id": t.need_id,
                    "status": t.status,
                    "attempts": t.attempts,
                    "result": t.result,
                    "blocked_reason": t.blocked_reason,
                }
                for t in self.tasks.values()
            ]
        }
