"""标准库实验身份：临时seed覆盖、执行代码内容快照、文件内容hash。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .model import Recipe

CODE_ROOT = Path(__file__).resolve().parents[2]


def effective_recipe(r: Recipe, train_seed: int | None = None) -> Recipe:
    data = copy.deepcopy(r.data)
    original_seed = r.get("train.seed")
    if train_seed is not None:
        if type(train_seed) is not int or not 0 <= train_seed < 2**32:
            raise ValueError("train_seed必须为[0, 2**32)整数，不能是bool")
        data.setdefault("train", {})["seed"] = train_seed
        # 未显式设eval seed时固定在原始seed，不随三次训练变化。
        data.setdefault("evaluation", {}).setdefault("seed", original_seed)
    return Recipe(r.id, r.path, data)


def code_snapshot(root: Path = CODE_ROOT) -> dict:
    files = []
    for directory in ("src/recipe_lab", "runner"):
        for path in sorted((root / directory).glob("*.py")):
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError(f"执行代码不能为symlink: {path}")
            raw = path.read_bytes()
            files.append({"path": path.relative_to(root).as_posix(), "content": raw.decode("utf-8"),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    if not files:
        raise ValueError("代码快照为空")
    return {"schema_version": 1, "kind": "execution-code-snapshot", "files": files}


def snapshot_bytes(snapshot: dict) -> bytes:
    return (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def code_hash(root: Path = CODE_ROOT) -> str:
    return hashlib.sha256(snapshot_bytes(code_snapshot(root))).hexdigest()


def file_record(base: Path, path: Path) -> dict:
    # 检查解析前路径，不能让 resolve() 隐藏 symlink 来源。
    if any(p.is_symlink() for p in (path, *path.parents, base, *base.parents)):
        raise ValueError("证据路径不能包含symlink")
    path, base = path.resolve(), base.resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("证据必须是集合目录内的普通文件")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(base).as_posix(), "sha256": digest.hexdigest()}


def file_hash(path: Path) -> str:
    return file_record(path.parent, path)["sha256"]


def output_plan(plan: dict, root: Path, output_root: str | None = None) -> dict:
    """统一模式输出隔离；只读，不创建目录，显式输出根必须绝对路径。"""
    result = copy.deepcopy(plan)
    if output_root is not None:
        directory = Path(output_root).expanduser()
        if not directory.is_absolute():
            raise ValueError("--output-root必须为绝对路径")
        directory = directory / plan["mode"]
    else:
        directory = Path(plan["run"]["output_dir"])
        if not directory.is_absolute():
            directory = root / directory
    result["run"]["output_dir"] = str(directory)
    if "output_dir" in result["run"].get("params", {}):
        result["run"]["params"]["output_dir"] = str(directory)
    name = "training.json" if plan["mode"] in ("dry-run", "smoke", "full") else "evaluation.json"
    result["evidence"] = str(directory / name)
    return result
