"""逐篇判断 recipe **在原理上**能否作为 research / business training baseline。

与另外两套审计的分工（互不替代）：
  * capabilities.py：代码 + offline test 是否实装（能不能跑起来）；
  * baseline.py / baseline_evidence.py：证据与就绪度（有没有真实跑过、够不够）；
  * 本模块：实验目标是否成立、数据是否被污染、决策门是否已阻断（值不值得当 baseline）。

存在一类用代码和 GPU 都补不回来的阻塞：上游自承零训练基线即达标、或基座已在
评估集上训练过。这两类不能被「五维 implemented + completed evidence」冲掉，
因此 verdict 参与 B2/B3 晋级门禁，且只能通过改写实验目标 / 换数据后重新核一手
材料来解除，不接受追加 evidence JSON 或抬高 B 状态绕过。

零第三方依赖；只读 registry，不写运行证据、不推断上游现状。
"""
from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import Recipe

REGISTRY = "registry/feasibility.toml"
CAPABILITIES = "registry/capabilities.toml"
VERDICTS = (
    "eligible",
    "blocked_zero_train",
    "contaminated_pretraining",
    "not_implemented",
    "partial_implementation",
    "pending_freeze",
)
DIMENSIONS = ("data", "train", "eval", "comparison", "export")


class FeasibilityError(ValueError):
    pass


@dataclass
class FeasibilityAudit:
    ok: bool
    errors: list[str] = field(default_factory=list)
    rows: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    promotion_blocking: list[str] = field(default_factory=list)
    needs_override: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)


def load(root: Path) -> dict[str, Any]:
    path = Path(root) / REGISTRY
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise FeasibilityError(f"无法加载 baseline 可行性登记 {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise FeasibilityError("feasibility.toml 必须是 schema=1 的登记")
    return data


def _capabilities(root: Path) -> dict[str, dict]:
    try:
        with (Path(root) / CAPABILITIES).open("rb") as handle:
            return tomllib.load(handle).get("recipes", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_list(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(_nonempty_str(item) for item in value))


def audit(root: Path, recipes: Iterable[Recipe]) -> FeasibilityAudit:
    """只读审计：登记完整性 + 与配方一手事实 / 能力登记的一致性。"""
    recipes = list(recipes)
    ids = [r.id for r in recipes]
    result = FeasibilityAudit(ok=True, recipes=ids)
    try:
        data = load(root)
    except FeasibilityError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    criteria = data.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != set(VERDICTS):
        result.errors.append("criteria 必须且只能定义 5 个合法 verdict")
    policy = data.get("verdict_policy", {})
    if not isinstance(policy, dict) or not _nonempty_list(policy.get("promotion_blocking")):
        result.errors.append("verdict_policy.promotion_blocking 缺失")
    else:
        unknown = set(policy["promotion_blocking"]) - set(VERDICTS)
        if unknown:
            result.errors.append(f"promotion_blocking 含未知 verdict: {sorted(unknown)}")
    if not _nonempty_list(policy.get("needs_override")):
        result.errors.append("verdict_policy.needs_override 缺失")
    elif not set(policy["needs_override"]) <= {"blocked_zero_train", "contaminated_pretraining"}:
        result.errors.append("只有两类原理性阻塞允许进入 needs_override")

    declared = data.get("recipes", {})
    expected = data.get("expected_recipe_count")
    if type(expected) is not int or expected != len(recipes):
        result.errors.append(f"expected_recipe_count 必须等于实际 recipe 数 {len(recipes)}")
    missing = sorted(set(ids) - set(declared))
    extra = sorted(set(declared) - set(ids))
    if missing or extra:
        result.errors.append(f"可行性登记未精确覆盖 14 篇：missing={missing} extra={extra}")

    caps = _capabilities(root)
    by_id = {r.id: r for r in recipes}
    verdict_counts: Counter[str] = Counter()
    for rid in sorted(set(ids) & set(declared)):
        row = declared[rid]
        r = by_id[rid]
        if not isinstance(row, dict):
            result.errors.append(f"{rid}: 登记项必须是表")
            continue
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            result.errors.append(f"{rid}: 非法 verdict {verdict!r}")
            continue
        verdict_counts[verdict] += 1
        for key in ("reason", "primary_source"):
            if not _nonempty_str(row.get(key)):
                result.errors.append(f"{rid}: {key} 必须是非空一手结论/来源")
        for key in ("blockers", "requires"):
            if not _nonempty_list(row.get(key)):
                result.errors.append(f"{rid}: {key} 必须是非空字符串列表")
        if row.get("override"):
            result.errors.append(f"{rid}: 不允许在登记里自设 override 绕过原理性阻塞")

        # 与配方自身的一手结论交叉校验，防止判定与事实脱节。
        if verdict == "blocked_zero_train":
            if r.get("decision.gate") != "blocked" or r.get("baseline.result") != "passes":
                result.errors.append(
                    f"{rid}: blocked_zero_train 必须对应 decision.gate=blocked 且 baseline.result=passes")
        if verdict == "contaminated_pretraining":
            text = " ".join(str(item) for item in
                            (r.get("upstream_deviations.items", []) or []))
            if "ChartQA" not in text or "污染" not in text and "预训练" not in text:
                result.errors.append(
                    f"{rid}: contaminated_pretraining 必须在配方 upstream_deviations 中有预训练污染记录")

        # 与能力登记交叉校验：pending_freeze 必须是五维已实装；not_implemented 必须全维缺失。
        cells = caps.get(rid, {})
        statuses = [cells.get(dim, {}).get("status") if isinstance(cells.get(dim), dict) else None
                    for dim in DIMENSIONS]
        if verdict == "pending_freeze" and any(s != "implemented" for s in statuses):
            result.errors.append(f"{rid}: pending_freeze 要求 capabilities 五维均 implemented，实际 {statuses}")
        if verdict == "not_implemented" and any(s != "missing" for s in statuses):
            result.errors.append(f"{rid}: not_implemented 要求 capabilities 五维均 missing，实际 {statuses}")
        if verdict == "partial_implementation":
            # 三维已实装、发布级两维未实装：既不能算「完全未实现」，也不能冒充
            # pending_freeze（后者要求五维齐备），更不能因有代码就晋级 eligible。
            expected = ["implemented", "implemented", "implemented", "missing", "missing"]
            if statuses != expected:
                result.errors.append(
                    f"{rid}: partial_implementation 要求 data/train/eval 均 implemented 且 "
                    f"comparison/export 均 missing，实际 {statuses}")

        result.rows[rid] = dict(row)
        if verdict in (policy.get("promotion_blocking") or []):
            result.promotion_blocking.append(rid)
        if verdict in (policy.get("needs_override") or []):
            result.needs_override.append(rid)

    result.counts = {v: verdict_counts.get(v, 0) for v in VERDICTS}
    result.counts["total"] = len(ids)
    result.ok = not result.errors
    return result


def promotion_problems(root: Path, recipes: Iterable[Recipe]) -> list[str]:
    """B2/B3 晋级门禁：返回逐篇原理性阻塞，供 `baseline --min-state` 使用。"""
    checked = audit(root, recipes)
    problems = list(checked.errors)
    for rid in sorted(checked.promotion_blocking):
        row = checked.rows.get(rid, {})
        verdict = row.get("verdict", "?")
        first = (row.get("blockers") or ["（未登记阻塞项）"])[0]
        tag = " [需改写目标并重判，不接受加证据绕过]" if rid in checked.needs_override else ""
        problems.append(f"{rid}: baseline 可行性判定 {verdict}；首要阻塞：{first}{tag}")
    return problems


def render(root: Path, recipes: Iterable[Recipe]) -> str:
    """生成可插入 BASELINE-READINESS.md 的 Markdown 章节。"""
    recipes = list(recipes)
    checked = audit(root, recipes)
    eligible = checked.counts.get("eligible", 0)
    lines = [
        "## 逐篇 baseline 可行性判定（原理层）", "",
        "> 由 `recipe-lab feasibility` 生成。代码实装、GPU 证据、**实验目标是否成立**是三件不同的事；"
        "本节只判定第三件，不因五维 implemented 或 completed JSON 而放行。", "",
        f"**当前 {eligible} 篇可直接作为 research / business baseline：eligible {eligible}/{len(recipes)}；"
        f"pending_freeze {checked.counts.get('pending_freeze', 0)}、"
        f"partial_implementation {checked.counts.get('partial_implementation', 0)}、"
        f"not_implemented {checked.counts.get('not_implemented', 0)}、"
        f"blocked_zero_train {checked.counts.get('blocked_zero_train', 0)}、"
        f"contaminated_pretraining {checked.counts.get('contaminated_pretraining', 0)}。**", "",
        "两类原理性阻塞（`blocked_zero_train` / `contaminated_pretraining`）即使补齐代码与 CUDA 实跑证据也不晋级："
        "前者上游自承零训练基线即可达标，后者基座本就在评估集上训练过。必须通过改写实验目标、换未污染数据、"
        "重核一手材料并显式改写 verdict 解除，不能用新增 evidence 或抬高 B 状态绕过。", "",
        "| verdict | 含义 |", "| --- | --- |",
    ]
    try:
        criteria = load(root).get("criteria", {})
    except FeasibilityError:
        criteria = {}
    for verdict in VERDICTS:
        lines.append(f"| `{verdict}` | {criteria.get(verdict, '')} |")
    lines.extend(["", "| recipe | verdict | 一手结论 | 首要阻塞 | 转化所需 |",
                  "| --- | --- | --- | --- | --- |"])
    for rid in sorted(checked.rows):
        row = checked.rows[rid]
        blockers = row.get("blockers") or []
        requires = row.get("requires") or []
        lines.append(
            f"| `{rid}` | `{row.get('verdict')}` | {row.get('reason', '')} | "
            f"{blockers[0] if blockers else '（缺）'} | {'; '.join(requires)} |")
    lines.extend(["", "### wave3 视觉任务的实装排序", "",
                  "本轮回一手页面核实 `fine_tuning_granite_vision_sft_trl`：Geoperception 的 "
                  "`predicate == \"LineComparison\"` 子集只做 `train_test_split(test_size=0.5, seed=42)`（无 validation）；"
                  "自定义 collator 仅按 `<|assistant|>` 位置隐式 mask，没有显式 `image_token_id` 屏蔽；"
                  "无 model/dataset/processor `revision`、无训练 seed、无 `compute_metrics`；"
                  "评估只有 `test_idx = 20` 的单样本定性生成，正文写 ChartQA 但代码从未加载 ChartQA；"
                  "文字称 merge adapters，代码只有 `PeftModel.from_pretrained`，无 `merge_and_unload()`；"
                  "LoraConfig 用动态表达式筛 `language_model`/`_proj` 且 `use_dora=True`（标题却写 LoRA）。"
                  "这些是本仓 `granite-linecomparison-v1` 自建协议必须显式纠正的一手事实。", "",
                  "按此排序，wave3 两篇视觉任务现已各补齐五维：`vlm/sft/granite-vision-line-comparison`"
                  "（图像内容 hash 三段隔离、显式 image token + assistant-only 双层 mask、冻结 processor revision、"
                  "逐题 held-out 重算）与 `vlm/sft/paligemma2-refcoco-grounding`（COCO xywh→xyxy 统一、"
                  "1024 桶 `<loc####>` round-trip/IoU、单框 completion-only 协议）。wave4 的 "
                  "`vlm/dpo/smolvlm-rlaif-v` 与 `vlm/mpo/qwen2-5-vl-3b-mpo` 亦已补齐五维并精确注册 "
                  "task_router（图像偏好 hash 隔离、显式 beta/三损失、冻结 ref、隐式奖励 pairwise），"
                  "但两篇基座不同，`cross_backbone_attribution_valid=false`。以上改判 pending_freeze "
                  "**只是代码齐备，不是晋级**：仍无冻结 revision / CUDA 三 seed 实跑 / 预声明容差 / "
                  "真实 HF offline 复载 / 许可与业务验收。两篇 ChartQA 保持原理性阻塞，不计入「适配进度」；"
                  "其余 4 篇仍为 not_implemented。", ""])
    if checked.errors:
        lines.extend(["### 登记完整性错误", ""] + [f"- [FAIL] {e}" for e in checked.errors] + [""])
    return "\n".join(lines)
