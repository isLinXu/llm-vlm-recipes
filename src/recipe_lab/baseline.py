"""声明就绪度与实际能力、实跑证据分别核查；静态检查不认证GPU执行真实性。"""
from __future__ import annotations
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .model import Recipe

STATE_ORDER = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}
STATES = ["B3", "B2", "B1"]
GROUPS = ["G1", "G2", "G3", "G4"]
STATE_LABEL = {"B0": "not-ready", "B1": "declared", "B2": "research-ready", "B3": "business-ready"}


def load_contract(root: Path | None = None) -> dict[str, Any]:
    from .loader import repo_root
    with ((root or repo_root()) / "registry" / "baseline.toml").open("rb") as handle:
        return tomllib.load(handle)


def check_passed(r: Recipe, check: dict) -> bool:
    from .baseline_evidence import (HEX40, meaningful, finite_number, dataset_pinned,
                                    frozen_environment, suite_problems, zero_train_measured)
    kind, field = check["kind"], check["field"]
    value = r.get(field)
    if kind == "commit":
        return isinstance(value, str) and bool(HEX40.fullmatch(value))
    if kind == "dataset_pin":
        return dataset_pinned(r)
    if kind == "frozen_environment":
        return frozen_environment(r)
    if kind == "baseline_suite":
        return not suite_problems(r)
    if kind == "zero_train_measurement":
        return zero_train_measured(r)
    if kind == "numbers":
        metrics = r.get("evaluation.metrics", [])
        return isinstance(value, dict) and bool(metrics) and set(metrics) <= set(value) and all(
            finite_number(value[m]) and (value[m] >= 0 if field.endswith("tolerance") else True) for m in metrics)
    if kind == "export_reload":
        if not (isinstance(value, str) and value and r.get(check["confirmation_field"]) is True
                and r.get("baseline_contract.rollback_verified") is True
                and meaningful(r.get("baseline_contract.rollback_evidence"))):
            return False
        try:
            from .task_router import load
            # 按任务分派内容复核；未知任务或未实现的 export 适配器一律失败关闭。
            return load(r, "export").validate_export_evidence(r, r.path.parent / value)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, ImportError):
            return False
    if kind == "integer":
        return type(value) is int and value >= 0
    if kind == "confirmed":
        return meaningful(value) and r.get(check["confirmation_field"]) is True
    if kind == "field":
        return meaningful(value)
    if kind == "file":
        return isinstance(value, str) and bool(value) and (r.path.parent / value).is_file()
    if kind == "equals":
        return type(value) is type(check["expect"]) and value == check["expect"]
    if kind == "not_in":
        return meaningful(value) and value not in check.get("expect", [])
    raise ValueError(f"未知 check kind: {kind}")


def compute_state(passed: dict[str, bool], contract: dict | None = None) -> str:
    contract = contract or load_contract()
    for state in STATES:
        needed = [c for c in contract["checks"] if STATE_ORDER[c["min_state"]] <= STATE_ORDER[state]]
        if all(passed.get(c["id"], False) for c in needed):
            return state
    return "B0"


@dataclass
class BaselineAudit:
    recipe_id: str
    claimed: str
    computed: str
    target_tier: str
    passed: dict[str, bool]
    gaps: list[dict]
    groups: dict[str, tuple[int, int]]
    state_ok: bool

    @property
    def n_passed(self):
        return sum(self.passed.values())
    @property
    def n_total(self):
        return len(self.passed)
    @property
    def blocking(self):
        target = min(STATE_ORDER[self.computed] + 1, 3)
        return [c["id"] for c in self.gaps if STATE_ORDER[c["min_state"]] <= target]


def audit_one(r: Recipe, contract: dict | None = None) -> BaselineAudit:
    contract = contract or load_contract()
    passed = {c["id"]: check_passed(r, c) for c in contract["checks"]}
    groups = {g: (sum(passed[c["id"]] for c in contract["checks"] if c["group"] == g),
                  sum(c["group"] == g for c in contract["checks"])) for g in GROUPS}
    computed = compute_state(passed, contract)
    claimed = str(r.get("baseline_contract.state", "") or "")
    return BaselineAudit(r.id, claimed, computed, str(r.get("baseline_contract.target_tier", "research")),
                         passed, [c for c in contract["checks"] if not passed[c["id"]]], groups, claimed == computed)


def summarize(audits):
    counts = Counter(a.computed for a in audits)
    return {s: counts.get(s, 0) for s in STATE_ORDER}


def group_totals(audits):
    return {g: (sum(a.groups[g][0] for a in audits), sum(a.groups[g][1] for a in audits)) for g in GROUPS}


def check_stats(audits, contract):
    return sorted([(c, sum(a.passed[c["id"]] for a in audits)) for c in contract["checks"]],
                  key=lambda item: (item[1], item[0]["min_state"], item[0]["id"]))


def render_text(audits):
    dist = summarize(audits)
    lines = [f"Baseline Readiness · {len(audits)}篇", f"状态分布：{dist}", "B1只表示声明，不表示任务实现或GPU通过。"]
    for a in sorted(audits, key=lambda x: x.recipe_id):
        lines.append(f"{a.recipe_id}: {a.computed} / 目标{a.target_tier} / {a.n_passed}/{a.n_total}项" + (" [状态不一致]" if not a.state_ok else ""))
    lines.append(f"research-ready B2+：{dist['B2'] + dist['B3']}/{len(audits)}")
    return "\n".join(lines)


def render_report(audits, contract, *, dry_run_note=""):
    dist = summarize(audits)
    lines = ["# Baseline Readiness 与系统能力审计", "",
             f"> {contract['assessed_at']}；由 `recipe-lab baseline --report` 生成。禁止手工修改派生报告。", "",
             "## 当前结论", "",
             f"**{len(audits)}篇：B0 {dist['B0']} / B1 {dist['B1']} / B2 {dist['B2']} / B3 {dist['B3']}。**",
             f"**可作 research baseline 的B2+：{dist['B2'] + dist['B3']}/{len(audits)}；业务B3：{dist['B3']}/{len(audits)}。**", "",
             "B1仅证明执行声明齐备，不能证明任务实现、真实split、环境兼容或训练跑通。",
             "此前把14篇B1作为进展主指标不够准确：必须同时查看下方五维实际能力矩阵。",
             "复现上游那次实验与维护自建baseline不同；后者仍必须冻结自有代码、数据、环境并实测。", "",
             "## 状态与证据门槛", "", "| 状态 | 含义 |", "| --- | --- |"]
    for s in STATE_ORDER:
        lines.append(f"| {s} | {contract['states_meta'][s]['description']} |")
    lines.extend(["", "## 逐篇声明就绪度", "", "| recipe | 状态 | 目标 | 下一档阻塞 |", "| --- | --- | --- | --- |"])
    for a in sorted(audits, key=lambda x: x.recipe_id):
        lines.append(f"| `{a.recipe_id}` | {a.computed} | {a.target_tier} | {', '.join(a.blocking) or '-'} |")
    lines.extend(["", "## 全量检查项", "", "| ID | 最低状态 | 检查 | 通过/总数 |", "| --- | --- | --- | --- |"])
    for c, count in check_stats(audits, contract):
        lines.append(f"| {c['id']} | {c['min_state']} | {c['label']} | {count}/{len(audits)} |")
    lines.extend(["", "## 本轮实装修正", "",
                  "- NuminaMath math-grpo-v1已补固定三段split、gold验证、跨split去重、统一prompt、reward、真实GRPO装配、基座与checkpoint评估以及逐样本证据；只在离线纯函数/mock测试中验证，未下载实际数据、未完成GPU运行。",
                  "- eval缺checkpoint拒绝，不再加载base冒充训练效果；zero-shot仅加载base。smoke/full隔离输出，预算越界和未知参数拒绝。",
                  "- C201/C202拒绝main与非commit字符串；C304/C305拒绝占位文本、NaN及负容差；C306要求基座实测文件而非上游一句passes。",
                  "- C212要求完整实际环境锁；C307复用生产聚合器强校验至少三个独立训练seed、代码/锁/有效配置、严格1500条manifest与gold hash、checkpoint及预测/统计/均值容差；C405自动核导出复载包，人工回滚另行确认。", 
                  "- 修正合法B2被错误要求B3许可/owner的规则；测试不再要求所有revision永远为空、所有配方永远B1。",
                  "- 在线环境使用environment revision、词表内容hash与对局seed；不再要求不存在的Hub dataset revision。", "",
                  "## 复查命令", "", "```bash", "make ci", "PYTHONPATH=src python3 -m recipe_lab capabilities --check",
                  "python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run",
                  "python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --apply",
                  "python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --capture-environment",
                  "python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --full",
                  "python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --smoke",
                  "python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --full",
                  "python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --full --checkpoint <真实训练产物目录>", "```", "",
                  "后三类命令仅供具备冻结依赖、实际数据与CUDA GPU的环境运行；本轮未执行。",
                  "数学gold解析或split去重失败会停止，而不是放宽门槛给满分。", ""])
    if dry_run_note:
        lines.extend(["### 计划预演（不是训练日志）", "", "```text", dry_run_note, "```", ""])
    lines.extend(["## 本机无法确认的部分与剩余工作", "",
                  "- 当前没有任何实际GPU训练/指标证据；先前Hub API请求失败不等于所有网络通道永久不可用。未编造任何revision或分数。",
                  "- execution.adapter_status仅记录不替代代码；数学math-grpo-v1、StarCoder code-fim-v1、xLAM xlam-fc-v1、RapidFire rapidfire-sft-v1、Granite granite-linecomparison-v1、PaliGemma paligemma-grounding-v1、SmolVLM DPO smolvlm-dpo-v1与Qwen2.5-VL MPO qwen25vl-mpo-v1八篇五维均有代码与隔离离线测试（偏好篇见 tests/test_preference.py）。八篇均按精确recipe_id注册task_router，发布级dry-run分派到各自专用协议，--full仍失败关闭；C306/C307/C405按任务分派，互不借用协议。DPO/MPO共享偏好split算法但backbone不同，cross_backbone_attribution_valid=false，不得跨篇归因算法。真实recipe未预声明export_tolerance时--full导出失败关闭。未知或冲突任务失败关闭；GPU、真实三seed、真实GPU复载、许可与业务验收未通过，其余6篇中advanced-reward的GSM8K gsm8k-reward-v1仅data/train/eval三维离线实装（partial_implementation，未注册task_router，发布级套件unavailable），另3篇任务逻辑拒绝执行、2篇原理性阻塞。", 
                  "- 环境捕获、框架API兼容与实际模型加载未在GPU实测；离线测试不证明训练收敛或性能。",
                  "- 八篇的独立训练seed编排/严格聚合、发布级逐题内容重算与导出新进程offline复载均已实装并有隔离离线测试，按task_router精确分派、互不借用协议。真实recipe未预声明export_tolerance时--full导出失败关闭；八篇GPU多seed/复载验收、可安装镜像锁、冻结model/dataset/source/processor revision、业务安全/成本回归未完成。RapidFire顺序执行4配置不得外推并发编排收益；Granite的exact_match只衡量文本一致性；PaliGemma的单框IoU不等于mAP/AP；DPO/MPO的隐式奖励pairwise不等于人类偏好一致率，且不得跨backbone归因算法。", 
                  "- B3若使用人工confirmed字段，只证明维护者记录了复核，不能自动认证许可授权或业务SLO；后续须接真实验收流程。",
                  "- 静态内容与hash检查证明证据内部一致，不是GPU执行真实性证明；需要独立运行流程与人工复核。",
                  "- 晋级条件必须是代码与证据齐备，不是增加空recipe、字段或测试数。完整逐篇缺口与实施依赖如下。", ""])
    return "\n".join(lines)
