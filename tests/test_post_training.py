"""MLLM 后训练方向层审计测试：来源边界、五方向闭合、14 篇映射与门禁。

负例全部在 tmp_path 复制的最小 registry 上做，不改动真实仓库；
真实 14 篇 Recipe 只读加载，篡改用文本定点替换。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from recipe_lab import post_training as pm
from recipe_lab.loader import load_recipes

ROOT = Path(__file__).resolve().parents[1]
IDS = {r.id for r in load_recipes(ROOT)}
BLOCKED = "vlm/sft/qwen2-vl-7b-chartqa"
CONTAMINATED = "vlm/sft/smolvlm-chartqa"
PREFERENCE = "preference_calibration"


@pytest.fixture(scope="module")
def recipes():
    return load_recipes(ROOT)


@pytest.fixture(scope="module")
def real_audit(recipes):
    return pm.audit(ROOT, recipes)


def _orig() -> str:
    return (ROOT / pm.REGISTRY).read_text(encoding="utf-8")


def _sandbox(tmp_path, *, registry=None, capabilities=None, feasibility=None):
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    for name, override in (("mllm_post_training.toml", registry),
                           ("capabilities.toml", capabilities),
                           ("feasibility.toml", feasibility)):
        source = ROOT / "registry" / name
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        (tmp_path / "registry" / name).write_text(
            text if override is None else override, encoding="utf-8")
    return tmp_path


def _replace_in_section(text: str, header: str, key: str, new_line: str) -> str:
    head, sep, tail = text.partition(header)
    assert sep, header
    body, sep2, rest = tail.partition("\n[")
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(key + " "):
            lines[i] = new_line
            break
    else:
        raise AssertionError(f"{key} not found in {header}")
    rebuilt = head + header + "\n" + "\n".join(lines)
    return rebuilt + sep2 + rest if sep2 else rebuilt + "\n"


# -------------------------------------------------------------------------
# 真实登记正例
# -------------------------------------------------------------------------
def test_real_registry_is_clean(real_audit):
    assert real_audit.ok, real_audit.errors
    assert real_audit.counts["sources"] == 3
    assert real_audit.counts["directions"] == 5
    assert real_audit.counts["recipes"] == 14
    assert real_audit.counts["implemented_recipes"] == 6


def test_covers_all_recipes(real_audit, recipes):
    assert set(real_audit.rows) == {r.id for r in recipes}


def test_five_directions_all_have_primary_mapping(real_audit):
    for did, row in real_audit.coverage.items():
        assert row["primary_count"] >= 1, did
        assert row["mapped_count"] >= 1, did


def test_source_boundaries_are_explicit(real_audit):
    assert real_audit.sources["mllm_post_training_survey"]["verification"] == "abstract_and_index"
    assert "正文页被 Akamai 防护" in real_audit.sources["mllm_post_training_survey"]["boundary"]
    assert real_audit.sources["llm_post_training_survey"]["used_for"] == ["cross_check"]
    assert real_audit.sources["awesome_mllm_tuning"]["used_for"] == ["method_index"]


def test_preference_calibration_is_the_real_gap(real_audit):
    assert real_audit.coverage[PREFERENCE]["gap"] is True
    assert real_audit.coverage[PREFERENCE]["implemented_count"] == 0
    assert real_audit.counts["implemented_directions"] == 4
    assert real_audit.counts["gap_directions"] == 1


def test_blocked_and_contaminated_are_not_valid_coverage(real_audit):
    assert real_audit.rows[BLOCKED]["valid_coverage_evidence"] is False
    assert real_audit.rows[CONTAMINATED]["valid_coverage_evidence"] is False
    assert real_audit.rows[BLOCKED]["feasibility_verdict"] == "blocked_zero_train"
    assert real_audit.rows[CONTAMINATED]["feasibility_verdict"] == "contaminated_pretraining"


def test_implementation_state_is_recomputed_from_capabilities(real_audit):
    implemented = {rid for rid, row in real_audit.rows.items()
                   if row["implementation_state"] == "five_dimension_implemented"}
    assert len(implemented) == 6
    assert "llm/grpo/qwen2-0-5b-numinamath" in implemented
    assert BLOCKED not in implemented


def test_report_contains_key_sections(real_audit, recipes):
    text = pm.render(ROOT, recipes)
    assert "MLLM 后训练方向覆盖审计" in text
    assert "来源边界" in text
    assert "五方向与覆盖" in text
    assert "逐篇映射" in text
    assert "[GAP] `preference_calibration`" in text


# -------------------------------------------------------------------------
# 负例：篡改 registry，验证失败关闭
# -------------------------------------------------------------------------
def test_invalid_verification_enum_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), "[sources.mllm_post_training_survey]", "verification",
        'verification = "made_up"')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("verification" in e for e in a.errors)


def test_non_survey_source_cannot_be_direction_primary(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), "[directions.instruction_following]", "primary_source",
        'primary_source = "awesome_mllm_tuning"')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("mllm_post_training_survey" in e for e in a.errors)


def test_llm_survey_cannot_claim_directions(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), "[sources.llm_post_training_survey]", "used_for",
        'used_for = ["directions"]')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("只能作 cross_check" in e for e in a.errors)


def test_awesome_index_cannot_claim_directions(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), "[sources.awesome_mllm_tuning]", "used_for",
        'used_for = ["directions"]')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("只能作 method_index" in e for e in a.errors)


def test_missing_direction_rejected(tmp_path, recipes):
    header = "[directions.scalable_learning]"
    head, sep, tail = _orig().partition(header)
    assert sep
    _body, next_sec, rest = tail.partition("\n[recipes.")
    assert next_sec
    text = head + next_sec + rest
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("未登记方向" in e for e in a.errors)


def test_wrong_direction_count_rejected(tmp_path, recipes):
    text = _orig().replace("expected_direction_count = 5", "expected_direction_count = 4", 1)
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("expected_direction_count" in e for e in a.errors)


def test_wrong_recipe_count_rejected(tmp_path, recipes):
    text = _orig().replace("expected_recipe_count = 14", "expected_recipe_count = 13", 1)
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("expected_recipe_count" in e for e in a.errors)


def test_missing_recipe_coverage_rejected(tmp_path, recipes):
    header = '[recipes."vlm/sft/paligemma2-refcoco-grounding"]'
    head, sep, tail = _orig().partition(header)
    assert sep
    _body, next_sec, rest = tail.partition("\n[recipes.")
    assert next_sec
    text = head + next_sec + rest
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("未精确覆盖 14 篇" in e for e in a.errors)


def test_invalid_direction_id_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', "primary",
        'primary = ["reasoning_enhancement", "made_up"]')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("made_up" in e for e in a.errors)


def test_empty_primary_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', "primary",
        "primary = []")
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("primary 必须是非空字符串列表" in e for e in a.errors)


def test_primary_secondary_overlap_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', "secondary",
        'secondary = ["reasoning_enhancement"]')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("primary/secondary 重叠" in e for e in a.errors)


def test_forbidden_claim_phrase_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), "[directions.reasoning_enhancement]", "boundary",
        'boundary = "本仓已在本仓 GPU 验证 GRPO 收益。"')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("禁止宣称" in e for e in a.errors)


def test_direction_without_primary_recipe_rejected(tmp_path, recipes):
    text = _orig()
    for rid in ("llm/grpo/qwen2-0-5b-vllm-colocate",):
        header = f'[recipes."{rid}"]'
        text = _replace_in_section(text, header, "primary",
                                   'primary = ["instruction_following"]')
    a = pm.audit(_sandbox(tmp_path, registry=text), recipes)
    assert not a.ok
    assert any("scalable_learning" in e and "主方向" in e for e in a.errors)


# -------------------------------------------------------------------------
# 交叉层漂移
# -------------------------------------------------------------------------
def test_capability_drift_changes_implementation_state(tmp_path, recipes):
    cap = (ROOT / pm.CAPABILITIES).read_text(encoding="utf-8")
    header = '[recipes."llm/grpo/qwen2-0-5b-numinamath"]'
    head, sep, tail = cap.partition(header)
    assert sep
    body, sep2, rest = tail.partition("\n[recipes.")
    new_body = body.replace('status = "implemented"', 'status = "missing"', 1)
    assert new_body != body
    cap = head + header + new_body + sep2 + rest
    a = pm.audit(_sandbox(tmp_path, capabilities=cap), recipes)
    assert a.ok
    assert a.rows["llm/grpo/qwen2-0-5b-numinamath"]["implementation_state"] == "not_implemented"
    assert a.counts["implemented_recipes"] == 5


def test_feasibility_drift_changes_coverage_validity(tmp_path, recipes):
    feas = (ROOT / pm.FEASIBILITY).read_text(encoding="utf-8")
    rid = 'vlm/sft/qwen2-vl-7b-chartqa'
    head, sep, tail = feas.partition(f'[recipes."{rid}"]')
    assert sep
    body, sep2, rest = tail.partition("\n[recipes.")
    new_body = body.replace('verdict = "blocked_zero_train"', 'verdict = "pending_freeze"', 1)
    assert new_body != body
    feas = head + f'[recipes."{rid}"]' + new_body + sep2 + rest
    a = pm.audit(_sandbox(tmp_path, feasibility=feas), recipes)
    assert a.ok
    assert a.rows[rid]["valid_coverage_evidence"] is True


# -------------------------------------------------------------------------
# CLI 行为
# -------------------------------------------------------------------------
def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["RECIPE_LAB_ROOT"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-S", "-m", "recipe_lab", *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def test_cli_check_passes():
    p = _cli("post-training", "--check")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "登记完整性：通过" in p.stdout


def test_cli_json_shape():
    p = _cli("post-training", "--json")
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data["ok"] is True
    assert data["counts"]["directions"] == 5
    assert set(data["coverage"]) >= {PREFERENCE, "reasoning_enhancement"}


def test_cli_min_fivedim_target_fails_closed():
    p = _cli("post-training", "--check", "--min-fivedim-directions", "5")
    assert p.returncode == 1
    assert "preference_calibration" in p.stdout


def test_cli_report_written(tmp_path):
    out = tmp_path / "POST-TRAINING.md"
    p = _cli("post-training", "--report", str(out))
    assert p.returncode == 0, p.stderr
    text = out.read_text(encoding="utf-8")
    assert "MLLM 后训练方向覆盖审计" in text
    assert "preference_calibration" in text
