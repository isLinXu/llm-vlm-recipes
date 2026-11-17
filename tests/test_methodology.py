"""方法论依据层审计：结构完整性、一手字段、引用闭合、与可行性层交叉校验。

所有负例都在 tmp_path 复制的最小 registry 上做，不改动真实仓库；
真实 14 篇 Recipe 只读加载，篡改时用文本定点替换或 deepcopy 内存变体。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from recipe_lab import methodology as mm
from recipe_lab.loader import load_recipes
from recipe_lab.model import Recipe

ROOT = Path(__file__).resolve().parents[1]
IDS = {r.id for r in load_recipes(ROOT)}
BLOCKED = "vlm/sft/qwen2-vl-7b-chartqa"
CONTAMINATED = "vlm/sft/smolvlm-chartqa"
PALIGEMMA = "vlm/sft/paligemma2-refcoco-grounding"


@pytest.fixture(scope="module")
def recipes():
    return load_recipes(ROOT)


@pytest.fixture(scope="module")
def real_audit(recipes):
    return mm.audit(ROOT, recipes)


def _sandbox(tmp_path, *, methodology=None, feasibility=None):
    """把真实 registry 复制到 tmp_path；可对 TOML 文本做定点篡改。"""
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    for name, override in (("methodology.toml", methodology),
                           ("feasibility.toml", feasibility)):
        source = ROOT / "registry" / name
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        (tmp_path / "registry" / name).write_text(
            text if override is None else override, encoding="utf-8")
    return tmp_path


def _orig() -> str:
    return (ROOT / mm.REGISTRY).read_text(encoding="utf-8")


def _replace_in_section(text: str, header: str, key: str, new_line: str) -> str:
    """只改指定 TOML 表内某个 key 的行，保留其余原文（含换行）。"""
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


def _with_recipe_methodology(recipes, rid, *, drop=False, **changes):
    """只在内存里制造一份 recipe.toml [methodology] 变体，不写真实文件。"""
    out = []
    for r in recipes:
        if r.id != rid:
            out.append(r)
            continue
        data = copy.deepcopy(r.data)
        if drop:
            data.pop("methodology", None)
        else:
            for key, value in changes.items():
                if value is None:
                    data["methodology"].pop(key, None)
                else:
                    data["methodology"][key] = value
        out.append(replace(r, data=data))
    return out


# -------------------------------------------------------------------------
# 真实登记的正例
# -------------------------------------------------------------------------
def test_real_registry_is_clean(real_audit):
    assert real_audit.ok, real_audit.errors
    assert real_audit.counts["total"] == 14
    assert real_audit.counts["recipes"] == 14
    assert real_audit.counts["papers"] >= 20
    assert real_audit.counts["verified_papers"] >= 18
    assert real_audit.counts["corrections"] >= 5


def test_covers_all_recipes(real_audit, recipes):
    assert set(real_audit.rows) == {r.id for r in recipes}


def test_every_recipe_has_methodology_backfill(recipes):
    for r in recipes:
        sec = r.get("methodology")
        assert isinstance(sec, dict), r.id
        assert sec.get("registry") == mm.REGISTRY, r.id
        for key in mm.RECIPE_FIELDS:
            assert key in sec, (r.id, key)


def test_recipe_backfill_matches_registry(recipes):
    registry = tomllib.loads(_orig())["recipes"]
    for r in recipes:
        sec = r.get("methodology", {})
        row = registry[r.id]
        assert sec["principles"] == row["principles"], r.id
        assert sec["primary_papers"] == row["primary_papers"], r.id
        assert sec["supplement_papers"] == row.get("supplement_papers", []), r.id
        assert sec["stance"] == row["stance"], r.id


def test_every_paper_has_boundary(real_audit):
    for pid, paper in real_audit.papers.items():
        assert isinstance(paper.get("boundary"), str) and paper["boundary"].strip(), pid


def test_verified_papers_have_arxiv_id_and_claim(real_audit):
    for pid, paper in real_audit.papers.items():
        if paper.get("verified") is True:
            assert mm.ARXIV_ID.match(paper["arxiv_id"]), pid
            assert mm.YEAR.match(paper["year"]), pid
            assert paper.get("core_claim", "").strip(), pid
            assert paper.get("url", "").startswith("https://"), pid


def test_unverified_papers_are_not_used_as_supports(real_audit):
    unverified = {p for p, v in real_audit.papers.items() if v.get("verified") is not True}
    unverified.add("")
    for cid, claim in real_audit.principles.items():
        if claim.get("support") == "paper_supports":
            assert claim["paper"] not in unverified, cid


def test_corrections_record_wrong_ids(tmp_path, recipes):
    a = mm.audit(ROOT, recipes)
    # 一手核查发现的错误编号必须以 was 形式留档。
    was = {c["was"] for c in tomllib.loads(_orig()).get("corrections", [])}
    for wrong in ("2204.14108", "2404.14367", "1604.07399", "2109.07980"):
        assert wrong in was
    assert a.counts["corrections"] == len(was)


def test_blocked_and_contaminated_keep_required_principles(real_audit):
    rows = real_audit.rows
    assert "chartqa_zero_train_gate" in rows[BLOCKED]["principles"]
    assert "chartqa_contamination" in rows[CONTAMINATED]["principles"]
    assert "chartqa_zero_train_gate" not in rows[CONTAMINATED]["principles"]


def test_no_forbidden_claim_in_real_registry(real_audit):
    for e in real_audit.errors:
        assert "禁止宣称" not in e


def test_report_render_contains_key_sections(real_audit, recipes):
    text = mm.render(ROOT, recipes)
    assert "方法论依据层" in text
    assert "一手核查" in text
    assert "未核实来源" in text
    assert BLOCKED in text


# -------------------------------------------------------------------------
# 负例：篡改 registry，验证失败关闭
# -------------------------------------------------------------------------
def test_arxiv_id_format_enforced(tmp_path, recipes):
    text = _orig().replace('arxiv_id = "2402.03300"', 'arxiv_id = "not-an-id"', 1)
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("arxiv_id" in e for e in a.errors)


def test_verified_paper_without_core_claim_rejected(tmp_path, recipes):
    marker = '[papers.deepseekmath_grpo]'
    text = _replace_in_section(_orig(), marker, "core_claim", 'core_claim = ""')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("core_claim" in e for e in a.errors)


def test_verified_paper_without_boundary_rejected(tmp_path, recipes):
    text = _replace_in_section(_orig(), '[papers.deepseekmath_grpo]', 'boundary', 'boundary = ""')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("boundary" in e for e in a.errors)


def test_unverified_paper_cannot_be_primary(tmp_path, recipes):
    # 把未核实的 mpl_mpo 塞进 MPO 篇的 primary_papers。
    text = _replace_in_section(
        _orig(), '[recipes."vlm/mpo/qwen2-5-vl-3b-mpo"]', 'primary_papers',
        'primary_papers = ["dpo", "underspecification", "qlora", "mpl_mpo"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("未核实论文" in e for e in a.errors)


def test_unverified_paper_cannot_support_a_claim(tmp_path, recipes):
    # 让一条 paper_supports 主张指向未核实论文。
    text = _replace_in_section(
        _orig(), '[principles.dpo_closed_form]', 'paper', 'paper = "mpl_mpo"')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("不得作为 paper_supports" in e for e in a.errors)


def test_forbidden_claim_phrase_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[principles.grpo_critic_free]', 'claim',
        'claim = "本仓已在本仓 GPU 验证 GRPO 收益。"')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("禁止宣称" in e for e in a.errors)


def test_dangling_principle_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', 'principles',
        'principles = ["grpo_critic_free", "not_exist"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("not_exist" in e for e in a.errors)


def test_principle_paper_not_declared_in_recipe_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', 'primary_papers',
        'primary_papers = ["deepseekmath_grpo"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("未列入 primary/supplement" in e for e in a.errors)


def test_correction_with_same_was_now_rejected(tmp_path, recipes):
    text = _orig().replace(
        'was = "2204.14108"\nnow = "2207.14255"',
        'was = "2204.14108"\nnow = "2204.14108"', 1)
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("was 与 now 不得相同" in e for e in a.errors)


def test_blocked_recipe_losing_required_principle_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."vlm/sft/qwen2-vl-7b-chartqa"]', 'principles',
        'principles = ["qualitative_only_gap"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("chartqa_zero_train_gate" in e for e in a.errors)


def test_contaminated_recipe_losing_required_principle_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."vlm/sft/smolvlm-chartqa"]', 'principles',
        'principles = ["dedup_reduces_leakage", "qualitative_only_gap"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("chartqa_contamination" in e for e in a.errors)


def test_missing_recipe_coverage_rejected(tmp_path, recipes):
    header = '[recipes."' + PALIGEMMA + '"]'
    head, sep, tail = _orig().partition(header)
    assert sep
    # 丢掉该篇整段（直到下一个 [recipes. 表），制造覆盖缺口。
    _body, next_sec, rest = tail.partition("\n[recipes.")
    assert next_sec
    text = head + next_sec + rest
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("未精确覆盖" in e for e in a.errors)


def test_wrong_expected_count_rejected(tmp_path, recipes):
    text = _orig().replace("expected_recipe_count = 14", "expected_recipe_count = 13", 1)
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("expected_recipe_count" in e for e in a.errors)


def test_invalid_support_enum_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[principles.grpo_critic_free]', 'support', 'support = "made_up"')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("非法 support" in e for e in a.errors)


def test_orphan_principle_rejected(tmp_path, recipes):
    # 追加一条无人引用的主张。
    text = _orig() + '''\n[principles.zombie]\npaper = "lora"\nsupport = "paper_supports"\nscope = "design_guidance"\nclaim = "x"\nboundary = "y"\n'''
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("未被任何 recipe 引用" in e for e in a.errors)


# -------------------------------------------------------------------------
# 负例：配方本体回填漂移，验证失败关闭
# -------------------------------------------------------------------------
def test_recipe_without_methodology_section_rejected(recipes):
    mutated = _with_recipe_methodology(recipes, PALIGEMMA, drop=True)
    a = mm.audit(ROOT, mutated)
    assert not a.ok
    assert any("缺少 [methodology]" in e for e in a.errors)


def test_recipe_wrong_methodology_registry_rejected(recipes):
    mutated = _with_recipe_methodology(
        recipes, PALIGEMMA, registry="registry/other.toml")
    a = mm.audit(ROOT, mutated)
    assert not a.ok
    assert any("[methodology].registry 必须指向" in e for e in a.errors)


def test_recipe_missing_methodology_field_rejected(recipes):
    mutated = _with_recipe_methodology(
        recipes, PALIGEMMA, supplement_papers=None)
    a = mm.audit(ROOT, mutated)
    assert not a.ok
    assert any("缺少 [methodology].supplement_papers" in e for e in a.errors)


def test_recipe_methodology_field_drift_rejected(recipes):
    mutated = _with_recipe_methodology(
        recipes, PALIGEMMA, principles=["qualitative_only_gap"])
    a = mm.audit(ROOT, mutated)
    assert not a.ok
    assert any("[methodology].principles 与" in e for e in a.errors)


def test_registry_drift_from_recipe_rejected(tmp_path, recipes):
    text = _replace_in_section(
        _orig(), '[recipes."llm/grpo/qwen2-0-5b-numinamath"]', 'principles',
        'principles = ["grpo_critic_free"]')
    a = mm.audit(_sandbox(tmp_path, methodology=text), recipes)
    assert not a.ok
    assert any("[methodology].principles 与" in e for e in a.errors)


# -------------------------------------------------------------------------
# CLI 行为
# -------------------------------------------------------------------------
def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env.pop("RECIPE_LAB_ROOT", None)
    return subprocess.run(
        [sys.executable, "-S", "-m", "recipe_lab", *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def test_cli_check_passes():
    p = _cli("methodology", "--check")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "登记完整性：通过" in p.stdout


def test_cli_json_shape():
    p = _cli("methodology", "--json")
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data["ok"] is True
    assert data["counts"]["recipes"] == 14
    assert "papers" in data and "principles" in data


def test_cli_min_verified_fails_closed():
    p = _cli("methodology", "--check", "--min-verified-papers", "99")
    assert p.returncode == 1
    assert "低于要求" in p.stdout


def test_cli_report_written(tmp_path):
    out = tmp_path / "METHODOLOGY.md"
    p = _cli("methodology", "--report", str(out))
    assert p.returncode == 0, p.stderr
    assert "方法论依据层" in out.read_text(encoding="utf-8")
