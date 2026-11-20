"""Agent 交接契约测试：事实源、生成物、命令门禁与可重建性。

负例用内存中的 registry 变体，不改动真实仓库；真实审计只读加载。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipe_lab import agent_contract as ac
from recipe_lab.loader import load_recipes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recipes():
    return load_recipes(ROOT)


@pytest.fixture(scope="module")
def real_audit(recipes):
    return ac.audit(ROOT, recipes)


def _mutate(fn):
    data = ac.load(ROOT)
    fn(data)
    return data


# -------------------------------------------------------------------------
# 真实契约正例
# -------------------------------------------------------------------------
def test_real_contract_is_clean(real_audit):
    assert real_audit.ok, real_audit.errors
    assert real_audit.data["contract_version"] == "1.0.0"


def test_facts_are_read_from_existing_audits(real_audit):
    f = real_audit.facts
    assert f["recipes"] == 14
    assert f["implemented_recipes"] == 9
    assert f["gpu_verified"] == 0
    assert f["eligible"] == 0
    assert f["post_training_directions"] == 5
    assert f["implemented_directions"] == 5
    assert f["gap_directions"] == 0


def test_sources_of_truth_are_present(real_audit):
    assert len(real_audit.data["sources_of_truth"]) == 10
    for item in real_audit.data["sources_of_truth"]:
        assert list(ROOT.glob(item["path"])), item["path"]


def test_generated_artifacts_have_matching_hashes(real_audit):
    assert len(real_audit.artifacts) == 5
    for item in real_audit.artifacts:
        p = ROOT / item["path"]
        assert p.is_file(), item["path"]
        assert item["bytes"] == p.stat().st_size
        assert item["sha256"] == ac._sha256(p)


def test_commands_include_pass_and_fail_closed(real_audit):
    commands = {c["id"]: c for c in real_audit.data["commands"]}
    assert commands["ci"]["expectation"] == "pass"
    assert commands["handoff-check"]["expectation"] == "pass"
    for cid in ("gate-b2", "gate-b3", "gate-l2", "gate-feasibility-b2",
                "gate-methodology", "gate-post-training"):
        assert commands[cid]["expectation"] == "fail_closed", cid


def test_guardrails_include_no_unasked_push(real_audit):
    guards = {g["id"]: g for g in real_audit.data["guardrails"]}
    assert "no-unasked-push" in guards
    assert "明确要求" in guards["no-unasked-push"]["statement"]


def test_agents_md_has_discoverable_markers(real_audit):
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for marker in ac.AGENT_MARKERS:
        assert marker in text


def test_render_json_is_deterministic(real_audit):
    a = ac.render_json(ROOT, real_audit)
    b = ac.render_json(ROOT, real_audit)
    assert a == b
    data = json.loads(a)
    assert data["schema"] == 1
    assert data["contract_version"] == "1.0.0"
    assert data["facts"]["gpu_verified"] == 0


def test_render_markdown_has_human_sections(real_audit):
    text = ac.render_markdown(ROOT, real_audit)
    for section in ("当前事实", "标准工作流", "事实源", "生成物与当前哈希",
                    "命令契约", "Agent Guardrails", "校验状态"):
        assert section in text


def test_handoff_artifacts_match_render(real_audit):
    md = (ROOT / "docs/AGENT-HANDOFF.md").read_text(encoding="utf-8")
    js = (ROOT / "docs/AGENT-HANDOFF.json").read_text(encoding="utf-8")
    assert md == ac.render_markdown(ROOT, real_audit)
    assert js == ac.render_json(ROOT, real_audit)


def test_generated_artifacts_are_reproducible(recipes):
    checked = ac.verify_regeneration(ROOT, ac.audit(ROOT, recipes))
    assert checked.ok, checked.errors
    assert checked.regeneration_checked is True


# -------------------------------------------------------------------------
# 负例：篡改内存契约，验证失败关闭
# -------------------------------------------------------------------------
def test_missing_source_of_truth_rejected(recipes):
    data = _mutate(lambda d: d["sources_of_truth"][0].update(path="registry/not-there.toml"))
    a = ac.audit(ROOT, recipes, data=data)
    assert not a.ok
    assert any("路径无匹配" in e for e in a.errors)


def test_missing_generated_artifact_rejected(recipes):
    data = _mutate(lambda d: d["generated_artifacts"][0].update(path="docs/NOT-THERE.md"))
    a = ac.audit(ROOT, recipes, data=data)
    assert not a.ok
    assert any("生成物缺失" in e for e in a.errors)


def test_generated_argv_without_out_rejected(recipes):
    def evil(d):
        d["generated_artifacts"][0]["argv"] = ["matrix", "--level", "l3"]
    a = ac.audit(ROOT, recipes, data=_mutate(evil))
    assert not a.ok
    assert any("{out}" in e for e in a.errors)


def test_duplicate_command_id_rejected(recipes):
    def evil(d):
        d["commands"][1]["id"] = d["commands"][0]["id"]
    a = ac.audit(ROOT, recipes, data=_mutate(evil))
    assert not a.ok
    assert any("id 缺失或重复" in e for e in a.errors)


def test_invalid_expectation_rejected(recipes):
    data = _mutate(lambda d: d["commands"][0].update(expectation="maybe"))
    a = ac.audit(ROOT, recipes, data=data)
    assert not a.ok
    assert any("非法 expectation" in e for e in a.errors)


def test_all_pass_contract_rejected(recipes):
    def evil(d):
        for c in d["commands"]:
            c["expectation"] = "pass"
    a = ac.audit(ROOT, recipes, data=_mutate(evil))
    assert not a.ok
    assert any("fail_closed" in e for e in a.errors)


def test_forbidden_guardrail_claim_rejected(recipes):
    def evil(d):
        d["guardrails"][0]["statement"] = "本仓已在本仓 GPU 验证全部训练。"
    a = ac.audit(ROOT, recipes, data=_mutate(evil))
    assert not a.ok
    assert any("禁止宣称" in e for e in a.errors)


# -------------------------------------------------------------------------
# CLI 行为
# -------------------------------------------------------------------------
def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["RECIPE_LAB_ROOT"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-S", "-m", "recipe_lab", "handoff", *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def test_cli_check_passes():
    p = _cli("--check")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "生成物可重建性：已校验" in p.stdout


def test_cli_json_shape():
    p = _cli("--json")
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data["ok"] is True
    assert data["facts"]["recipes"] == 14
    assert len(data["generated_artifacts"]) == 5


def test_cli_report_is_idempotent():
    before = (ROOT / "docs/AGENT-HANDOFF.md").read_text(encoding="utf-8")
    p = _cli("--report")
    assert p.returncode == 0, p.stderr
    after = (ROOT / "docs/AGENT-HANDOFF.md").read_text(encoding="utf-8")
    assert before == after
