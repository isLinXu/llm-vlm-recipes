#!/usr/bin/env python3
"""冻结输入与捕获实际GPU环境分开：离线预演只读，失败不覆盖旧锁、不返回成功。

--apply 解析commit并写 resolved-inputs（不是frozen环境）。
--capture-environment 在已锁输入的GPU环境捕获完整pip freeze/Python/CUDA/driver。
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipe_lab.loader import load_recipes
from recipe_lab.toml_edit import quote, upsert_scalar
from recipe_lab.baseline_evidence import HEX40

LOCK_DIR = ROOT / "runner" / "locks"
COOKBOOK_REPO = "huggingface/cookbook"
TIMEOUT = 15


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "recipe-lab-lock/0.2"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def query(kind, repo_id):
    if not isinstance(repo_id, str) or "/" not in repo_id or " " in repo_id:
        return None, f"{kind} id非法：{repo_id!r}"
    url = f"https://huggingface.co/api/{'models' if kind == 'model' else 'datasets'}/{repo_id}/commits/main?limit=1"
    try:
        data = _get_json(url)
        item = data[0] if isinstance(data, list) and data else data
        sha = item.get("id") or item.get("commit", {}).get("id")
        return (sha if isinstance(sha, str) and HEX40.fullmatch(sha) else None), url
    except (urllib.error.URLError, OSError, ValueError, AttributeError, KeyError, IndexError) as error:
        return None, f"{url}: {error}"


def query_source_sha():
    url = f"https://api.github.com/repos/{COOKBOOK_REPO}/commits/main"
    try:
        sha = _get_json(url)["sha"]
        return (sha if isinstance(sha, str) and HEX40.fullmatch(sha) else None), url
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
        return None, f"{url}: {error}"


def lock_path(r):
    return LOCK_DIR / (r.id.replace("/", "__") + ".lock.toml")


def capture_environment(r):
    import importlib.metadata
    try:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("需要实际CUDA GPU，不能把本机CPU环境当成训练环境")
        for name, expected in (r.get("dependencies", {}) or {}).items():
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise ValueError(f"{name}声明{expected}，实际{actual}；先对齐或声明有依据的版本迁移")
        requirements = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True).splitlines()
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        runtime = {"python": sys.version.split()[0], "cuda": torch.version.cuda,
                   "driver": driver, "gpu": torch.cuda.get_device_name(0)}
        return requirements, runtime
    except (ImportError, importlib.metadata.PackageNotFoundError, subprocess.SubprocessError, OSError) as error:
        raise ValueError(f"不能捕获完整训练环境：{error}") from error


def render_lock(r, resolved, requirements=None, runtime=None):
    lines = ["# 真实查询输入与实际环境分开；resolved-inputs不代表可重放环境",
             "schema_version = 1", f"status = {quote('frozen' if runtime else 'resolved-inputs')}",
             f"created_at = {quote(datetime.now(timezone.utc).isoformat())}",
             f"recipe_id = {quote(r.id)}", f"experiment_id = {quote(r.get('baseline_contract.experiment_id', ''))}"]
    for key, value in {"model_id": r.get("model.base"), "dataset_id": r.get("dataset.name"),
                       "model_revision": resolved.get("model.revision"), "dataset_revision": resolved.get("dataset.revision"),
                       "source_sha": resolved.get("source_sha"), "environment_revision": resolved.get("environment.revision")}.items():
        lines.append(f"{key} = {quote(value or '')}")
    lines.append("requirements = [" + ", ".join(quote(s) for s in (requirements or [])) + "]")
    lines.extend(["", "[declared_dependencies]"])
    lines.extend(f"{key} = {quote(value)}" for key, value in sorted((r.get("dependencies", {}) or {}).items()))
    lines.extend(["", "[runtime]"])
    lines.extend(f"{key} = {quote(value)}" for key, value in (runtime or {}).items())
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)
    return text


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="锁定输入/捕获实际环境，不把模板当成freeze")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--recipe")
    target.add_argument("--all", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--capture-environment", action="store_true")
    args = parser.parse_args()
    recipes = sorted(load_recipes(ROOT), key=lambda r: r.id)
    targets = recipes if args.all else [r for r in recipes if r.id.startswith(args.recipe)]
    if not targets or (not args.all and len(targets) != 1):
        parser.error("--recipe必须唯一匹配；批量操作明确使用--all")
    failures = 0
    for r in targets:
        print(f"=== {r.id} ===")
        if not args.apply and not args.capture_environment:
            print("[只读预演] 模型/静态数据查询Hub commit；在线数据需已有环境revision与词表hash。未联网未写文件。")
            continue
        try:
            online = r.get("dataset.kind", "static") == "online"
            resolved = {key: r.get(key) for key in ("model.revision", "dataset.revision", "source_sha", "environment.revision")}
            if args.apply:
                # 已锁输入保持不动；重新查询main会让同一实验漂移并覆盖有效freeze。
                if not isinstance(resolved["model.revision"], str) or not HEX40.fullmatch(resolved["model.revision"]):
                    resolved["model.revision"], info = query("model", r.get("model.base"))
                    if not resolved["model.revision"]:
                        raise ValueError(info)
                if not online and (not isinstance(resolved["dataset.revision"], str) or not HEX40.fullmatch(resolved["dataset.revision"])):
                    resolved["dataset.revision"], info = query("dataset", r.get("dataset.name"))
                    if not resolved["dataset.revision"]:
                        raise ValueError(info)
                if not isinstance(resolved["source_sha"], str) or not HEX40.fullmatch(resolved["source_sha"]):
                    resolved["source_sha"], info = query_source_sha()
                    if not resolved["source_sha"]:
                        raise ValueError(info)
                from recipe_lab.baseline_evidence import frozen_environment
                if frozen_environment(r):
                    print("[ok] 已有输入与frozen环境锁一致；保持原版本与原文件，不重新查询main")
                    continue
            required = ("model.revision", "source_sha", "environment.revision" if online else "dataset.revision")
            if any(not isinstance(resolved[k], str) or not HEX40.fullmatch(resolved[k]) for k in required):
                raise ValueError("必要commit未锁：先锁定输入/在线环境，不写部分成功锁")
            requirements, runtime = capture_environment(r) if args.capture_environment else (None, None)
            lock = render_lock(r, resolved, requirements, runtime)
            if args.capture_environment:
                # 对安装依赖的精确格式做独立预检；拒绝editable和浮动URL。
                import re
                if not requirements or any(not (re.fullmatch(r"[\w.-]+==[^\s*<>~=]+", s) or re.fullmatch(r"[\w.-]+ @ git\+https://[^\s]+@[0-9a-fA-F]{40}", s)) for s in requirements):
                    raise ValueError("pip freeze含未冻结editable/本地路径/浮动URL，不能标为frozen")
            text = r.path.read_text(encoding="utf-8")
            for key in required:
                section, field = key.split(".") if "." in key else (None, key)
                text = upsert_scalar(text, section, field, quote(resolved[key]))
            tomllib.loads(text)
            # 保存失败则恢复原锁，避免recipe与lock分叉。
            path = lock_path(r)
            old_lock = path.read_text(encoding="utf-8") if path.exists() else None
            atomic_write(path, lock)
            try:
                atomic_write(r.path, text)
            except OSError:
                if old_lock is not None:
                    atomic_write(path, old_lock)
                else:
                    path.unlink(missing_ok=True)
                raise
            print("[ok] " + ("已捕获frozen运行环境" if runtime else "已解析输入；尚未冻结实际运行环境"))
        except (ValueError, OSError) as error:
            failures += 1
            print(f"[FAIL] {error}；查询/校验失败不改原recipe和lock", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
