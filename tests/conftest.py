from __future__ import annotations

import copy
from pathlib import Path

import pytest

from recipe_lab.loader import load_recipes, load_taxonomy, read_toml, repo_root
from recipe_lab.model import Recipe
from recipe_lab.rules import Context

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root(Path(__file__).resolve())


@pytest.fixture(scope="session")
def ctx(root: Path) -> Context:
    return Context(
        recipes=load_recipes(root),
        taxonomy=load_taxonomy(root),
        root=root,
        level=3,
    )


def valid_data() -> dict:
    return read_toml(FIXTURES / "valid_minimal.toml")


def bad_data() -> dict:
    return read_toml(FIXTURES / "bad_vlm_contract.toml")


def repro_data() -> dict:
    """最小合法 recipe + 一个自洽的 L0 [repro] 审计段。

    L0 只要求 provenance >= declared，其余维度全部 missing；
    每个非 pinned 维度都写 gap，因此 L5 应当零 error。
    """
    from recipe_lab.repro import DIMS

    data = valid_data()
    dims = {
        d: {
            "state": "declared" if d == "provenance" else "missing",
            "gap": f"测试用缺口：{d} 尚未锁定",
        }
        for d in DIMS
    }
    data["repro"] = {
        "level": "L0",
        "assessed_at": "2026-11-13",
        "assessed_by": "unit-test",
        "evidence_ref": "local-notebook:fine_tuning_vlm_trl.txt",
        "evidence": "",
        **dims,
    }
    return data


def baseline_data() -> dict:
    """最小合法 recipe + 自洽的 L0 [repro] + 自洽的 B0 [baseline_contract]。

    `make_recipe` 用的是虚拟路径，因此所有 kind=file 检查都失败，按 checks
    重算必然是 B0；只有声明 B0 才能通过 R603。[todo] 由 fixture 提供，
    R604 也随之通过。要测「真正达到 B1」请用真实仓库的 14 篇做集成测试。
    """
    data = repro_data()
    data["baseline_contract"] = {
        "state": "B0",
        "target_tier": "research",
        "experiment_id": "vlm-sft-valid-minimal-v1",
        "revision_policy": "runner/lock.py --apply@有网环境",
        "determinism": "deterministic",
        "python_cuda": "python>=3.11",
        "single_variable": "[train] 为唯一可变超参入口",
    }
    return data


def make_recipe(ctx: Context, data: dict) -> Recipe:
    rid = str(data["id"])
    return Recipe(
        id=rid,
        path=ctx.root / "recipes" / f"{rid}/recipe.toml",
        data=copy.deepcopy(data),
    )


def mutate(data: dict, dotted: str, value: object) -> dict:
    """按点路径改一个字段，返回新 dict。"""
    out = copy.deepcopy(data)
    cur = out
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value
    return out
