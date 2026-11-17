"""Recipe 的轻量内存模型。

刻意不使用 pydantic / attrs：字段大量可选，且规则本身需要区分
「字段缺失」「字段为 None」「字段显式写 none」，用原始 dict + 点路径访问最不容易误判。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Recipe:
    id: str
    path: Path
    data: dict[str, Any]

    def has(self, dotted: str) -> bool:
        """字段是否存在（即使值为 None / False 也算存在）。"""
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def status(self) -> str:
        return str(self.get("status", ""))

    @property
    def coverage(self) -> str:
        return str(self.get("coverage", ""))

    @property
    def family(self) -> str:
        return str(self.get("method.family", ""))

    @property
    def modality(self) -> str:
        return str(self.get("model.modality", ""))

    @property
    def is_catalog_only(self) -> bool:
        return self.status == "cataloged"

    @property
    def is_vlm(self) -> bool:
        return self.modality == "vlm" or bool(self.get("dataset.has_images", False))

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "family": self.family,
            "trainer": self.get("method.trainer", "-"),
            "model": self.get("model.base", "-"),
            "dataset": self.get("dataset.name", "-"),
            "modality": self.modality or "-",
            "baseline": self.get("baseline.result", "-"),
            "trl": self.get("dependencies.trl", "-"),
        }


@dataclass(frozen=True)
class Finding:
    level: str
    rule: str
    severity: str  # error | warning | info
    recipe_id: str
    message: str
    hint: str = ""

    def render(self) -> str:
        icon = {"error": "x", "warning": "!", "info": "i"}[self.severity]
        line = f"  [{icon}] {self.level}/{self.rule} {self.message}"
        if self.hint:
            line += f"\n        -> {self.hint}"
        return line

    def as_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "rule": self.rule,
            "severity": self.severity,
            "recipe_id": self.recipe_id,
            "message": self.message,
            "hint": self.hint,
        }
