"""代码/锁身份及三个CLI的隔离测试；不加载训练框架、不产生GPU证据。"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from recipe_lab import math_task as mt
from recipe_lab import provenance as pv
from test_math_task import recipe, runner  # noqa: F401


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.0, "42"])
def test_invalid_effective_seed(recipe, seed):
    with pytest.raises(ValueError):
        pv.effective_recipe(recipe, seed)


def test_effective_seed_does_not_mutate_data_or_eval(recipe):
    recipe.data["evaluation"].pop("seed")
    original = copy.deepcopy(recipe.data)
    for seed in (0, 43, 2**32 - 1):
        effective = pv.effective_recipe(recipe, seed)
        assert effective.get("train.seed") == seed
        assert effective.get("train.data_seed") == 42
        assert effective.get("evaluation.seed") == 42
    assert recipe.data == original


def test_code_snapshot_canonical_and_changes_with_content(tmp_path):
    folder = tmp_path / "src" / "recipe_lab"
    folder.mkdir(parents=True)
    source = folder / "example.py"
    source.write_text("x = 1\n", encoding="utf-8")
    snapshot = pv.code_snapshot(tmp_path)
    assert snapshot["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert pv.code_hash(tmp_path) == hashlib.sha256(pv.snapshot_bytes(snapshot)).hexdigest()
    source.write_text("x = 2\n", encoding="utf-8")
    assert pv.code_hash(tmp_path) != hashlib.sha256(pv.snapshot_bytes(snapshot)).hexdigest()
    source.unlink()
    source.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        pv.code_snapshot(tmp_path)


def test_file_record_rejects_symlink_and_escape(tmp_path):
    file = tmp_path / "input"
    file.write_bytes(b"evidence")
    assert pv.file_hash(file) == hashlib.sha256(b"evidence").hexdigest()
    link = tmp_path / "link"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="symlink"):
        pv.file_hash(link)
    inside = tmp_path / "inside"
    inside.mkdir()
    with pytest.raises(ValueError):
        pv.file_record(inside, file)


def test_completed_evidence_requires_identity_and_rejects_drift(recipe):
    plan = {"mode": "full"}
    identity = mt.execution_identity(recipe)
    value = mt.completed_evidence(recipe, plan, "training", {}, {}, {})
    assert value["code_sha256"] == identity["code_sha256"]
    assert value["environment_lock_sha256"] == identity["environment_lock_sha256"]
    plan["execution_identity"] = identity
    (recipe.path.parent / recipe.get("execution.lockfile")).write_text("changed", encoding="utf-8")
    with pytest.raises(mt.TaskExecutionError, match="漂移"):
        mt.completed_evidence(recipe, plan, "training", {}, {}, {})
    (recipe.path.parent / recipe.get("execution.lockfile")).unlink()
    with pytest.raises(mt.TaskExecutionError, match="身份"):
        mt.completed_evidence(recipe, {"mode": "full"}, "training", {}, {}, {})


@pytest.mark.parametrize("name,mode", [("train", "full"), ("eval", "eval"), ("zero_shot", "zero-shot")])
def test_cli_effective_seed_output_root_and_real_dispatch(recipe, tmp_path, monkeypatch, name, mode):
    module = runner(name)
    monkeypatch.setattr(module, "find_recipe", lambda _: recipe)
    seen = []
    def capture(r, plan, *args):
        seen.append((r, plan))
        return {"kind": "training", "stage": "completed"} if name == "train" else tmp_path / "test-only.json"
    if name == "train":
        monkeypatch.setattr(module, "_build_and_train", capture)
        monkeypatch.setattr(module, "write_evidence", lambda *args: tmp_path / "test-only.json")
    else:
        monkeypatch.setattr(module, "run_real", capture)
    argv = [name + ".py", "--recipe", recipe.id, "--train-seed", "43", "--output-root", str(tmp_path / "new")]
    original = copy.deepcopy(recipe.data)
    monkeypatch.setattr(sys, "argv", argv + ["--full"])
    assert module.main() == 0
    effective, plan = seen[0]
    assert effective.get("train.seed") == 43
    assert effective.get("train.data_seed") == effective.get("evaluation.seed") == 42
    assert plan["run"]["output_dir"] == str(tmp_path / "new" / mode)
    assert recipe.data == original
    monkeypatch.setattr(sys, "argv", argv + ["--dry-run"])
    assert module.main() == 0 and len(seen) == 1
    monkeypatch.setattr(sys, "argv", [name + ".py", "--recipe", recipe.id, "--output-root", "relative"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert not (tmp_path / "new").exists()
