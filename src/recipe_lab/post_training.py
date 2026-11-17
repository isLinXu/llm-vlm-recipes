"""MLLM 后训练方向层审计：按「行为/能力塑造」的五方向映射 14 篇 recipe。

与前四层互不替代，也不解锁 GPU 晋级：
  * capabilities.py：能不能跑起来；
  * baseline.py / baseline_evidence.py：有没有真实跑过；
  * feasibility.py：实验目标是否成立；
  * methodology.py：设计依据哪篇论文的一手结论；
  * 本模块：训练工作覆盖 MLLM 后训练的哪个能力方向，缺口在哪。

关键约束：
  - 五方向原始出处只能是 mllm_post_training_survey（摘要 + 索引级核实）；
  - LLM PoLM 综述只能作 cross_check，Awesome repo 只能作 method_index；
  - abstract_and_index / readme 级来源不得支撑方法细节、数值或本仓实测；
  - 14 篇精确覆盖，每篇有 primary 方向与 rationale；
  - primary/secondary 必须是合法方向且互不重叠；
  - 每个方向至少有一篇 primary 映射；
  - implementation state 从 capabilities.toml 重算，不采信本登记自称；
  - blocked_zero_train / contaminated_pretraining 篇在报告中显式标为无效覆盖证据。

零第三方依赖；只读 registry，不写运行证据、不认证 GPU 执行。
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import Recipe

REGISTRY = "registry/mllm_post_training.toml"
CAPABILITIES = "registry/capabilities.toml"
FEASIBILITY = "registry/feasibility.toml"
DIMS = ("data", "train", "eval", "comparison", "export")
INVALID_VERDICTS = {"blocked_zero_train", "contaminated_pretraining"}


class PostTrainingError(ValueError):
    pass


@dataclass
class PostTrainingAudit:
    ok: bool
    errors: list[str] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
    directions: dict[str, dict] = field(default_factory=dict)
    rows: dict[str, dict] = field(default_factory=dict)
    coverage: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    recipes: list[str] = field(default_factory=list)


def _read(root: Path, rel: str) -> dict[str, Any]:
    path = Path(root) / rel
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PostTrainingError(f"无法加载 {rel}: {exc}") from exc


def load(root: Path) -> dict[str, Any]:
    data = _read(root, REGISTRY)
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise PostTrainingError("mllm_post_training.toml 必须是 schema=1 的登记")
    return data


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _all_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty_str(v) for v in value)


def _five_dim_implemented(cap_row: dict) -> bool:
    return all(cap_row.get(dim, {}).get("status") == "implemented" for dim in DIMS)


def audit(root: Path, recipes: Iterable[Recipe]) -> PostTrainingAudit:
    """只读审计：来源边界 + 五方向闭合 + 14 篇映射 + 能力/可行性交叉。"""
    recipes = list(recipes)
    ids = [r.id for r in recipes]
    result = PostTrainingAudit(ok=True, recipes=ids)
    try:
        data = load(root)
        cap = _read(root, CAPABILITIES).get("recipes", {})
        feas = _read(root, FEASIBILITY).get("recipes", {})
    except PostTrainingError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    integrity = data.get("integrity", {})
    allowed_verification = set(_list(integrity.get("allowed_verification")))
    allowed_supports = set(_list(integrity.get("allowed_supports")))
    allowed_used_for = set(_list(integrity.get("allowed_used_for")))
    forbidden = _list(integrity.get("forbidden_claim_phrases"))
    if not allowed_verification or not allowed_supports or not allowed_used_for:
        result.errors.append("integrity 的 allowed_* 枚举不得为空")
    if not _all_strings(forbidden):
        result.errors.append("integrity.forbidden_claim_phrases 必须是非空字符串列表")

    def _scan(text: str, where: str) -> None:
        for phrase in forbidden:
            if phrase and phrase in text:
                result.errors.append(f"{where}: 出现禁止宣称 {phrase!r}（方向层不等于本仓实测）")

    # ---- sources ----
    sources = data.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        result.errors.append("sources 不得为空")
    for sid, source in sources.items():
        if not isinstance(source, dict):
            result.errors.append(f"source {sid} 必须是表")
            continue
        result.sources[sid] = dict(source)
        for key in ("kind", "title", "year", "verification", "core_claim", "boundary"):
            value = source.get(key)
            if key == "year":
                if not isinstance(value, (int, str)) or not str(value).strip():
                    result.errors.append(f"source {sid}: {key} 必须是非空值")
            elif not _nonempty_str(value):
                result.errors.append(f"source {sid}: {key} 必须是非空值")
        if source.get("verification") not in allowed_verification:
            result.errors.append(
                f"source {sid}: 非法 verification {source.get('verification')!r}")
        used_for = _list(source.get("used_for"))
        if not used_for or not set(used_for) <= allowed_used_for:
            result.errors.append(f"source {sid}: used_for 非法或为空 {used_for!r}")
        for key in ("core_claim", "boundary"):
            if _nonempty_str(source.get(key)):
                _scan(source[key], f"source {sid}.{key}")

    # 五方向的原始出处必须是摘要+索引级 MLLM 综述；其它来源不得越权。
    for sid, source in sources.items():
        used_for = set(_list(source.get("used_for")))
        if "directions" in used_for and source.get("verification") not in {
            "abstract", "abstract_and_index", "full_text"
        }:
            result.errors.append(f"source {sid}: readme 级索引不得作方向原始出处")
        if sid == "mllm_post_training_survey" and source.get("verification") != "abstract_and_index":
            result.errors.append("mllm_post_training_survey 必须保持 abstract_and_index 边界")
        if sid == "llm_post_training_survey" and used_for - {"cross_check"}:
            result.errors.append("llm_post_training_survey 只能作 cross_check")
        if sid == "awesome_mllm_tuning" and used_for - {"method_index"}:
            result.errors.append("awesome_mllm_tuning 只能作 method_index")

    # ---- directions ----
    directions = data.get("directions", {})
    expected_direction_count = data.get("expected_direction_count", 5)
    if type(expected_direction_count) is not int or expected_direction_count != len(directions):
        result.errors.append(
            f"expected_direction_count 必须等于实际方向数 {len(directions)}")
    for did, direction in directions.items():
        if not isinstance(direction, dict):
            result.errors.append(f"direction {did} 必须是表")
            continue
        result.directions[did] = dict(direction)
        for key in ("label", "definition", "primary_source", "cross_check_source",
                    "support", "boundary"):
            if not _nonempty_str(direction.get(key)):
                result.errors.append(f"direction {did}: {key} 必须是非空字符串")
        if direction.get("support") not in allowed_supports:
            result.errors.append(f"direction {did}: 非法 support {direction.get('support')!r}")
        primary_source = direction.get("primary_source")
        if primary_source not in sources:
            result.errors.append(f"direction {did}: primary_source {primary_source!r} 未登记")
        elif "directions" not in _list(sources[primary_source].get("used_for")):
            result.errors.append(f"direction {did}: primary_source {primary_source!r} 未授权作方向出处")
        if primary_source != "mllm_post_training_survey":
            result.errors.append(
                f"direction {did}: 五方向原始出处必须是 mllm_post_training_survey")
        cross_source = direction.get("cross_check_source")
        if cross_source not in sources:
            result.errors.append(f"direction {did}: cross_check_source {cross_source!r} 未登记")
        elif "cross_check" not in _list(sources[cross_source].get("used_for")):
            result.errors.append(f"direction {did}: cross_check_source {cross_source!r} 未授权交叉印证")
        for key in ("definition", "boundary"):
            if _nonempty_str(direction.get(key)):
                _scan(direction[key], f"direction {did}.{key}")

    # ---- recipe mapping ----
    declared = data.get("recipes", {})
    expected = data.get("expected_recipe_count")
    if type(expected) is not int or expected != len(recipes):
        result.errors.append(f"expected_recipe_count 必须等于实际 recipe 数 {len(recipes)}")
    missing = sorted(set(ids) - set(declared))
    extra = sorted(set(declared) - set(ids))
    if missing or extra:
        result.errors.append(f"方向映射未精确覆盖 14 篇：missing={missing} extra={extra}")

    for rid in sorted(set(ids) & set(declared)):
        row = declared[rid]
        if not isinstance(row, dict):
            result.errors.append(f"{rid}: 映射项必须是表")
            continue
        primary = _list(row.get("primary"))
        secondary = _list(row.get("secondary", []))
        rationale = row.get("rationale")
        if not _all_strings(primary):
            result.errors.append(f"{rid}: primary 必须是非空字符串列表")
            continue
        if not all(_nonempty_str(v) for v in secondary):
            result.errors.append(f"{rid}: secondary 必须是字符串列表")
            continue
        for did in primary + secondary:
            if did not in directions:
                result.errors.append(f"{rid}: 引用未登记方向 {did!r}")
        overlap = set(primary) & set(secondary)
        if overlap:
            result.errors.append(f"{rid}: primary/secondary 重叠 {sorted(overlap)}")
        if not _nonempty_str(rationale):
            result.errors.append(f"{rid}: rationale 必须是非空说明")
        else:
            _scan(rationale, f"{rid}.rationale")

        implemented = _five_dim_implemented(cap.get(rid, {}))
        verdict = feas.get(rid, {}).get("verdict", "unknown")
        result.rows[rid] = {
            "primary": list(primary),
            "secondary": list(secondary),
            "rationale": rationale,
            "implementation_state": "five_dimension_implemented" if implemented else "not_implemented",
            "feasibility_verdict": verdict,
            "valid_coverage_evidence": verdict not in INVALID_VERDICTS,
        }

    # ---- coverage summary ----
    for did in directions:
        primary_rows = [rid for rid, row in result.rows.items() if did in row.get("primary", [])]
        all_rows = [rid for rid, row in result.rows.items()
                    if did in row.get("primary", []) or did in row.get("secondary", [])]
        implemented_rows = [rid for rid in all_rows
                            if result.rows[rid]["implementation_state"] == "five_dimension_implemented"
                            and result.rows[rid]["valid_coverage_evidence"]]
        if not primary_rows:
            result.errors.append(f"direction {did}: 没有任何 recipe 以它为主方向")
        if not all_rows:
            result.errors.append(f"direction {did}: 没有任何 recipe 映射到该方向")
        result.coverage[did] = {
            "primary_count": len(primary_rows),
            "mapped_count": len(all_rows),
            "implemented_count": len(implemented_rows),
            "implemented_rows": sorted(implemented_rows),
            "gap": not implemented_rows,
        }

    implemented_recipes = sum(
        1 for row in result.rows.values()
        if row.get("implementation_state") == "five_dimension_implemented")
    implemented_directions = sum(
        1 for item in result.coverage.values() if not item.get("gap"))
    result.counts = {
        "sources": len(result.sources),
        "directions": len(result.directions),
        "recipes": len(result.rows),
        "total": len(ids),
        "implemented_recipes": implemented_recipes,
        "implemented_directions": implemented_directions,
        "gap_directions": len(result.coverage) - implemented_directions,
    }
    result.ok = not result.errors
    return result


def promotion_problems(checked: PostTrainingAudit, min_directions: int) -> list[str]:
    if min_directions <= 0:
        return []
    actual = checked.counts.get("implemented_directions", 0)
    if actual >= min_directions:
        return []
    gaps = sorted(did for did, row in checked.coverage.items() if row.get("gap"))
    return [f"仅 {actual}/{len(checked.coverage)} 个方向有五维实现，低于要求 {min_directions}；缺口={gaps}"]


def render(root: Path, recipes: Iterable[Recipe]) -> str:
    """生成可入库的 Markdown 报告。"""
    checked = audit(root, recipes)
    c = checked.counts
    lines = [
        "# MLLM 后训练方向覆盖审计", "",
        "> 由 `recipe-lab post-training` 生成；单一事实源是 `registry/mllm_post_training.toml`。"
        "本文件每次 `make report` 重建，请勿手工编辑。", "",
        "## 为什么需要这一层", "",
        "前四层回答的是能不能跑、有没有跑过、值不值得当 baseline、设计依据哪篇论文；"
        "本层回答**训练工作覆盖 MLLM 后训练的哪个能力方向、缺口在哪**。", "",
        "五方向来自 *A Survey on Post-Training of Multimodal Large Language Models* 的摘要级框架："
        "指令遵循、偏好校准、推理增强、领域自适应、可扩展与高效学习。"
        "arXiv 2503.06072 是 LLM 后训练五范式的跨域交叉印证，Awesome-MLLM-Tuning 是不同轴的"
        "方法索引；两者都不是五方向原始出处。", "",
        "来源边界：preprints 正文页被防护，仅标题/作者/日期/DOI/摘要经索引交叉核实；"
        "本层不支撑具体方法细节、benchmark 数值或本仓 GPU 实测。"
        "任何一层通过都不解锁 B2/B3，eligible 与 GPU verified 仍独立判定。", "",
        f"当前覆盖 **{c.get('recipes', 0)}/{c.get('total', 0)} 篇**、"
        f"**{c.get('directions', 0)} 个方向**；"
        f"五维已实现 recipe {c.get('implemented_recipes', 0)} 篇，"
        f"有五维实现支撑的方向 {c.get('implemented_directions', 0)}/"
        f"{c.get('directions', 0)}，缺口 {c.get('gap_directions', 0)} 个。", "",
        "## 来源边界", "",
        "| id | 类型 | 核实级别 | 用途 | 边界 |", "| --- | --- | --- | --- | --- |",
    ]
    for sid in sorted(checked.sources):
        s = checked.sources[sid]
        boundary = str(s.get("boundary", "")).replace("|", "\\|")
        lines.append(
            f"| `{sid}` | {s.get('kind', '-')} | {s.get('verification', '-')} | "
            f"{', '.join(_list(s.get('used_for')))} | {boundary} |")
    lines.extend(["", "## 五方向与覆盖", "",
                  "| direction | 定义 | 主方向篇数 | 映射篇数 | 五维实现篇数 | 结论 |",
                  "| --- | --- | ---: | ---: | ---: | --- |"])
    for did in sorted(checked.directions):
        d = checked.directions[did]
        cov = checked.coverage.get(did, {})
        definition = str(d.get("definition", "")).replace("|", "\\|")
        status = "有缺口" if cov.get("gap") else "已有五维实现支撑"
        lines.append(
            f"| `{did}`（{d.get('label', '')}） | {definition} | "
            f"{cov.get('primary_count', 0)} | {cov.get('mapped_count', 0)} | "
            f"{cov.get('implemented_count', 0)} | {status} |")
    lines.extend(["", "## 逐篇映射", "",
                  "| recipe | 主方向 | 次方向 | 实现状态 | 可行性 | 有效覆盖证据 | 理由 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for rid in sorted(checked.rows):
        row = checked.rows[rid]
        rationale = str(row.get("rationale", "")).replace("|", "\\|")
        lines.append(
            f"| `{rid}` | {', '.join('`'+x+'`' for x in row.get('primary', []))} | "
            f"{', '.join('`'+x+'`' for x in row.get('secondary', [])) or '-'} | "
            f"{row.get('implementation_state', '-')} | {row.get('feasibility_verdict', '-')} | "
            f"{'是' if row.get('valid_coverage_evidence') else '否'} | {rationale} |")
    lines.extend(["", "## 边界与缺口", ""])
    for did in sorted(checked.directions):
        cov = checked.coverage.get(did, {})
        if cov.get("gap"):
            lines.append(
                f"- [GAP] `{did}`：没有五维实现且可作为有效覆盖证据的 recipe；"
                "登记中的映射只说明目标方向，不代表该方向已有 baseline。")
    lines.extend([
        "- 五维实现只表示真实任务代码 + 离线测试；GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 均未变。",
        "- `blocked_zero_train` 与 `contaminated_pretraining` 篇显式标为无效覆盖证据，不能用数量掩盖原理性阻塞。",
        "- 方向层不替代 capabilities / baseline / feasibility / methodology 任一层，也不给论文分数或本仓 expected/tolerance 背书。", "",
    ])
    if checked.errors:
        lines.extend(["### 登记完整性错误", ""] + [f"- [FAIL] {e}" for e in checked.errors] + [""])
    return "\n".join(lines)
