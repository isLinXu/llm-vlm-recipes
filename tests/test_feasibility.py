"""baseline 可行性判定：登记完整性、一手事实交叉校验与 B2/B3 晋级门禁。

所有负例都在 tmp_path 复制的最小 registry 上做，不改动真实仓库；
真实 14 篇 Recipe 只读加载，篡改时用 copy.deepcopy 构造内存变体。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from recipe_lab import feasibility as fz
from recipe_lab.loader import load_recipes
from recipe_lab.model import Recipe

ROOT = Path(__file__).resolve().parents[1]
IDS = {r.id for r in load_recipes(ROOT)}
CHARTQA_BLOCKED = "vlm/sft/qwen2-vl-7b-chartqa"
SMOLVLM_CONTAMINATED = "vlm/sft/smolvlm-chartqa"
GRANITE = "vlm/sft/granite-vision-line-comparison"
PALIGEMMA = "vlm/sft/paligemma2-refcoco-grounding"
# 真实登记中仍五维全 missing 且 verdict=not_implemented 的篇目，供负例夹具使用。
NOT_IMPLEMENTED = "llm/grpo/qwen2-0-5b-vllm-colocate"


@pytest.fixture(scope="module")
def recipes():
    return load_recipes(ROOT)


@pytest.fixture(scope="module")
def real_audit(recipes):
    return fz.audit(ROOT, recipes)


def _sandbox(tmp_path, *, registry=None, caps=None, recipes=None):
    """把真实 registry 复制到 tmp_path；可对 TOML 文本做定点篡改。"""
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    for name, override in ((fz.REGISTRY.split("/")[-1], registry),
                           (fz.CAPABILITIES.split("/")[-1], caps)):
        source = ROOT / "registry" / name
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        (tmp_path / "registry" / name).write_text(
            text if override is None else override, encoding="utf-8")
    return tmp_path, recipes if recipes is not None else load_recipes(ROOT)


def _replace_verdict(text, rid, value):
    """只改指定 recipe 段的 verdict 行。"""
    marker = f'[recipes."{rid}"]'
    head, _, tail = text.partition(marker)
    assert tail, marker
    body, sep, rest = tail.partition('\n[recipes."')
    assert body.startswith("\nverdict = ")
    body = body.replace(body.splitlines()[1], f"verdict = {json.dumps(value)}", 1)
    return head + marker + body + sep + rest


def _set_field(text, rid, field, value):
    marker = f'[recipes."{rid}"]'
    head, _, tail = text.partition(marker)
    assert tail, marker
    body, sep, rest = tail.partition('\n[recipes."')
    lines = body.splitlines()
    out, replaced = [], False
    for line in lines:
        if line.startswith(field + " ="):
            out.append(f"{field} = {value}")
            replaced = True
        else:
            out.append(line)
    assert replaced, field
    return head + marker + "\n" + "\n".join(out) + sep + rest


# ---------------------------------------------------------------- 正向：真实登记

def test_real_registry_is_complete_and_no_recipe_is_eligible(real_audit):
    assert real_audit.ok, real_audit.errors
    assert set(real_audit.recipes) == IDS and len(IDS) == 14
    assert real_audit.counts["total"] == 14
    assert real_audit.counts["eligible"] == 0
    assert real_audit.counts["pending_freeze"] == 8
    assert real_audit.counts["not_implemented"] == 4
    # ADR-022 / ADR-023 补齐发布级两维后，partial_implementation 归零；
    # 枚举与交叉校验仍保留，防止未来出现「只做三维」的篇目被误判。
    assert real_audit.counts["partial_implementation"] == 0
    assert real_audit.counts["blocked_zero_train"] == 1
    assert real_audit.counts["contaminated_pretraining"] == 1
    assert len(real_audit.promotion_blocking) == 14
    assert set(real_audit.needs_override) == {CHARTQA_BLOCKED, SMOLVLM_CONTAMINATED}


@pytest.mark.parametrize("rid,verdict", [
    ("llm/grpo/qwen2-0-5b-numinamath", "pending_freeze"),
    ("llm/sft/llama3-8b-xlam-function-calling", "pending_freeze"),
    ("llm/sft/rapidfire-multiconfig", "pending_freeze"),
    ("llm/sft/starcoder1b-fim", "pending_freeze"),
    ("vlm/dpo/smolvlm-rlaif-v", "pending_freeze"),
    ("vlm/mpo/qwen2-5-vl-3b-mpo", "pending_freeze"),
    (CHARTQA_BLOCKED, "blocked_zero_train"),
    (SMOLVLM_CONTAMINATED, "contaminated_pretraining"),
    (GRANITE, "pending_freeze"),
    (PALIGEMMA, "pending_freeze"),
])
def test_verdicts_are_bound_to_primary_facts(real_audit, rid, verdict):
    row = real_audit.rows[rid]
    assert row["verdict"] == verdict
    assert row["reason"].strip() and row["primary_source"].strip()
    assert row["blockers"] and row["requires"]


def test_granite_primary_facts_are_recorded(real_audit):
    reason = real_audit.rows[GRANITE]["reason"]
    for fact in ("50/50", "assistant-only", "DoRA", "skip_prepare_dataset", "ChartQA"):
        assert fact in reason


def test_paligemma_primary_facts_are_recorded(real_audit):
    """ADR-023：PaliGemma 改判必须绑定一手页面事实，不能只靠五维代码齐备。"""
    reason = real_audit.rows[PALIGEMMA]["reason"]
    for fact in ("SFTTrainer", "coco_to_xyxy", "<loc####>", "1024", "train[20]",
                 "val[13]", "do_sample=True", "gated", "mAP/AP"):
        assert fact in reason


def test_promotion_problems_name_the_two_principle_blocks():
    problems = fz.promotion_problems(ROOT, load_recipes(ROOT))
    assert len(problems) == 14
    assert sum("[需改写目标并重判" in p for p in problems) == 2
    assert any(CHARTQA_BLOCKED in p and "零训练基线" in p for p in problems)
    assert any(SMOLVLM_CONTAMINATED in p and "预训练" in p for p in problems)


def test_render_is_deterministic_and_contains_key_sections(recipes):
    first = fz.render(ROOT, recipes)
    assert first == fz.render(ROOT, recipes)
    for text in ("逐篇 baseline 可行性判定", "eligible 0/14", "blocked_zero_train",
                 "contaminated_pretraining", "wave3 视觉任务的实装排序",
                 "fine_tuning_granite_vision_sft_trl", "image_token_id",
                 "fine_tuning_vlm_object_detection_grounding", "bbox_iou"):
        assert text in first
    assert "[FAIL]" not in first


# ---------------------------------------------------------------- 负例：篡改登记

def test_unknown_verdict_is_rejected(tmp_path, recipes):
    text = _replace_verdict((ROOT / fz.REGISTRY).read_text(encoding="utf-8"),
                            GRANITE, "eligible_without_evidence")
    root, rows = _sandbox(tmp_path, registry=text, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok and any("非法 verdict" in e for e in checked.errors)


def test_missing_blockers_is_rejected(tmp_path, recipes):
    text = _set_field((ROOT / fz.REGISTRY).read_text(encoding="utf-8"),
                      GRANITE, "blockers", "[]")
    root, rows = _sandbox(tmp_path, registry=text, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok and any("blockers 必须是非空" in e for e in checked.errors)


def test_self_declared_override_is_rejected(tmp_path, recipes):
    text = _set_field((ROOT / fz.REGISTRY).read_text(encoding="utf-8"),
                      CHARTQA_BLOCKED, "reason", '"x"') + "\n"
    # 在 blocked 篇尾部追加 override 字段（同段内，位于下一个 [recipes.] 之前）
    marker = f'[recipes."{CHARTQA_BLOCKED}"]'
    head, _, tail = text.partition(marker)
    body, sep, rest = tail.partition('\n[recipes."')
    text = head + marker + body + "\noverride = true" + sep + rest
    root, rows = _sandbox(tmp_path, registry=text, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok and any("自设 override" in e for e in checked.errors)


def test_missing_recipe_or_wrong_expected_count_is_rejected(tmp_path, recipes):
    text = (ROOT / fz.REGISTRY).read_text(encoding="utf-8")
    text = text.replace("expected_recipe_count = 14", "expected_recipe_count = 13", 1)
    root, rows = _sandbox(tmp_path, registry=text, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok
    assert any("expected_recipe_count" in e for e in checked.errors)

    # 删掉一篇登记项 → 覆盖校验必须报 missing
    marker = f'\n[recipes."{GRANITE}"]'
    start = text.index(marker)
    nxt = text.find('\n[recipes."', start + 1)
    removed = text[:start] + (text[nxt:] if nxt != -1 else "")
    root2, rows2 = _sandbox(tmp_path / "second", registry=removed, recipes=recipes)
    checked2 = fz.audit(root2, rows2)
    assert not checked2.ok and any("未精确覆盖" in e for e in checked2.errors)


def test_incomplete_criteria_or_policy_is_rejected(tmp_path, recipes):
    text = (ROOT / fz.REGISTRY).read_text(encoding="utf-8")
    text = text.replace('eligible = "', 'eligible_removed = "', 1)
    root, rows = _sandbox(tmp_path, registry=text, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok
    assert any("criteria" in e for e in checked.errors)


def test_blocked_zero_train_must_match_decision_gate(recipes):
    data = copy.deepcopy(next(r for r in recipes if r.id == CHARTQA_BLOCKED).data)
    data["decision"]["gate"] = "open"
    changed = Recipe(CHARTQA_BLOCKED, ROOT / "registry" / "dummy.toml", data)
    rows = [changed if r.id == CHARTQA_BLOCKED else r for r in recipes]
    checked = fz.audit(ROOT, rows)
    assert not checked.ok
    assert any("decision.gate=blocked" in e for e in checked.errors)


def test_contaminated_verdict_requires_primary_contamination_note(recipes):
    data = copy.deepcopy(next(r for r in recipes if r.id == SMOLVLM_CONTAMINATED).data)
    data["upstream_deviations"]["items"] = ["推理保留 system message"]
    changed = Recipe(SMOLVLM_CONTAMINATED, ROOT / "registry" / "dummy.toml", data)
    rows = [changed if r.id == SMOLVLM_CONTAMINATED else r for r in recipes]
    checked = fz.audit(ROOT, rows)
    assert not checked.ok
    assert any("预训练污染记录" in e for e in checked.errors)


def test_pending_freeze_requires_all_five_dimensions_implemented(tmp_path, recipes):
    caps = (ROOT / fz.CAPABILITIES).read_text(encoding="utf-8")
    target = "llm/grpo/qwen2-0-5b-numinamath"
    marker = f'[recipes."{target}"]'
    head, _, tail = caps.partition(marker)
    body, sep, rest = tail.partition('\n[recipes."')
    changed = body.replace('data = { status = "implemented"',
                           'data = { status = "missing"', 1)
    root, rows = _sandbox(tmp_path, caps=head + marker + changed + sep + rest, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok
    assert any("pending_freeze 要求" in e for e in checked.errors)


def test_not_implemented_rejects_claimed_implementation(tmp_path, recipes):
    # Granite / PaliGemma 五维已齐备并改判 pending_freeze；这里必须挑一篇真正五维全
    # missing 且 verdict 仍为 not_implemented 的 recipe，才能验证该交叉校验不被绕过。
    caps = (ROOT / fz.CAPABILITIES).read_text(encoding="utf-8")
    marker = f'[recipes."{NOT_IMPLEMENTED}"]'
    head, _, tail = caps.partition(marker)
    body, sep, rest = tail.partition('\n[recipes."')
    changed = body.replace('data = { status = "missing"',
                           'data = { status = "implemented"', 1)
    assert changed != body, "夹具必须真的改动了 not_implemented 的能力登记"
    root, rows = _sandbox(tmp_path, caps=head + marker + changed + sep + rest, recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok
    assert any("not_implemented 要求" in e for e in checked.errors)


def test_partial_implementation_requires_exact_three_dimensions(tmp_path, recipes):
    """恰三维实装 + 发布级两维缺失才允许 partial_implementation；少/多一维失败关闭。

    ADR-022 补齐 Granite 五维后真实登记已无篇目使用该 verdict，因此在隔离 sandbox
    中重建该状态：先做正例证明规则仍生效，再逐个篡改能力登记验证失败关闭。
    不改动真实仓库，也不放宽「五维齐备才能 pending_freeze」的交叉校验。
    """
    caps = (ROOT / fz.CAPABILITIES).read_text(encoding="utf-8")
    registry = (ROOT / fz.REGISTRY).read_text(encoding="utf-8")
    marker = f'[recipes."{GRANITE}"]'
    head, _, tail = caps.partition(marker)
    body, sep, rest = tail.partition('\n[recipes."')
    three_dim = (body.replace('comparison = { status = "implemented"',
                              'comparison = { status = "missing"', 1)
                 .replace('export = { status = "implemented"',
                          'export = { status = "missing"', 1))
    assert three_dim != body, "夹具必须真的把发布级两维改回 missing"
    partial_registry = _replace_verdict(registry, GRANITE, "partial_implementation")

    root, rows = _sandbox(tmp_path / "valid-three-dim",
                          caps=head + marker + three_dim + sep + rest,
                          registry=partial_registry, recipes=recipes)
    assert fz.audit(root, rows).ok, "恰三维实装应通过 partial_implementation 交叉校验"

    for label, old, new in (
        ("missing-train", 'train = { status = "implemented"', 'train = { status = "missing"'),
        ("claimed-comparison", 'comparison = { status = "missing"',
         'comparison = { status = "implemented"'),
    ):
        changed = three_dim.replace(old, new, 1)
        assert changed != three_dim, f"{label} 夹具必须真的改动能力登记"
        root, rows = _sandbox(tmp_path / label, caps=head + marker + changed + sep + rest,
                              registry=partial_registry, recipes=recipes)
        checked = fz.audit(root, rows)
        assert not checked.ok, label
        assert any("partial_implementation 要求" in e for e in checked.errors), label


def test_missing_registry_fails_closed(tmp_path, recipes):
    root, rows = _sandbox(tmp_path, registry="", recipes=recipes)
    checked = fz.audit(root, rows)
    assert not checked.ok
    assert any("feasibility.toml" in e for e in checked.errors)
    with pytest.raises(fz.FeasibilityError):
        fz.load(root)


# ---------------------------------------------------------------- CLI / 门禁

def _cli(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "recipe_lab", "feasibility", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, check=False)


def test_cli_check_passes_but_min_state_fails_closed():
    ok = _cli("--check")
    assert ok.returncode == 0, ok.stderr
    assert "登记完整性：通过" in ok.stdout and "0/14" in ok.stdout
    assert "--min-state" in ok.stdout
    promoted = _cli("--check", "--min-state", "B2")
    assert promoted.returncode == 1
    assert promoted.stdout.count("[FAIL]") == 14
    assert "需改写目标并重判" in promoted.stdout


def test_cli_json_is_machine_readable():
    result = _cli("--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["counts"]["eligible"] == 0
    assert set(payload["rows"]) == IDS
    assert len(payload["promotion_blocking"]) == 14


def test_cli_report_is_written_only_when_requested(tmp_path):
    target = tmp_path / "feasibility.md"
    result = _cli("--report", str(target))
    assert result.returncode == 0
    content = target.read_text(encoding="utf-8")
    assert "逐篇 baseline 可行性判定" in content and "eligible 0/14" in content
    assert content == fz.render(ROOT, load_recipes(ROOT))


def test_baseline_min_state_includes_feasibility_gate():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "recipe_lab", "baseline", "--min-state", "B2"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "[可行性]" in result.stdout
    assert CHARTQA_BLOCKED in result.stdout and SMOLVLM_CONTAMINATED in result.stdout


def test_registry_toml_is_parseable_and_policy_is_self_consistent():
    data = tomllib.loads((ROOT / fz.REGISTRY).read_text(encoding="utf-8"))
    assert data["schema"] == 1
    assert set(data["criteria"]) == set(fz.VERDICTS)
    assert set(data["verdict_policy"]["needs_override"]) <= set(
        data["verdict_policy"]["promotion_blocking"])
