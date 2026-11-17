"""TOML 加载与仓库发现。

主格式为 TOML（Python 3.11+ 内置 tomllib）。
若检测到遗留的 recipe.yaml，doctor 会明确提示迁移，而不是静默忽略。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .model import Recipe


class LoadError(Exception):
    pass


def repo_root(start: Path | None = None) -> Path:
    """从任意位置向上找仓库根（以 registry/taxonomy.toml 为标志）。"""
    p = (start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "registry" / "taxonomy.toml").is_file():
            return candidate
    raise LoadError(
        "找不到仓库根（缺少 registry/taxonomy.toml）。请在仓库内运行，或设置 RECIPE_LAB_ROOT。"
    )


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise LoadError(f"文件不存在: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LoadError(f"TOML 解析失败: {path}: {exc}") from exc


def load_taxonomy(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    return read_toml(root / "registry" / "taxonomy.toml")


def load_one(path: Path, root: Path) -> Recipe:
    data = read_toml(path)
    rid = str(data.get("id", ""))
    if not rid:
        raise LoadError(f"{path}: 缺少 id")
    return Recipe(id=rid, path=path, data=data)


def load_recipes(root: Path | None = None, only: list[str] | None = None) -> list[Recipe]:
    """扫描 recipes/**/recipe.toml，按 id 稳定排序。"""
    root = root or repo_root()
    base = root / "recipes"
    if not base.is_dir():
        raise LoadError(f"缺少 recipes/ 目录: {base}")

    found: list[Recipe] = []
    for path in sorted(base.glob("**/recipe.toml")):
        recipe = load_one(path, root)
        if only and not any(recipe.id == x or recipe.id.startswith(x.rstrip("/") + "/") for x in only):
            continue
        found.append(recipe)

    seen: dict[str, Path] = {}
    for recipe in found:
        if recipe.id in seen:
            raise LoadError(f"recipe id 重复: {recipe.id} ({seen[recipe.id]} 与 {recipe.path})")
        seen[recipe.id] = recipe.path
    return sorted(found, key=lambda r: r.id)


def find_legacy_configs(root: Path | None = None) -> list[Path]:
    root = root or repo_root()
    return sorted((root / "recipes").glob("**/recipe.y*ml"))


def relative_recipe_id(path: Path, root: Path) -> str:
    """recipes/vlm/sft/foo/recipe.toml -> vlm/sft/foo"""
    return path.parent.relative_to(root / "recipes").as_posix()
