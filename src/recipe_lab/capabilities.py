"""独立能力登记、fail-closed 审计及可插入 baseline.render_report 的 Markdown。"""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model import Recipe

DIMENSIONS = ("data", "train", "eval", "comparison", "export")
STATES = ("missing", "implemented", "unverified", "verified")
EVIDENCE_KINDS = {
    "data": "data", "train": "training", "eval": "evaluation",
    "comparison": "baseline-suite", "export": "export-reload",
}
SCOPE_STATES = ("in_scope", "backlog", "excluded")


class CapabilityError(ValueError):
    pass


def load(root: Path) -> dict[str, Any]:
    """只读当前 root 的登记；缺文件时报错，不从配方状态推断能力。"""
    path = Path(root) / "registry" / "capabilities.toml"
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CapabilityError(f"无法加载能力登记 {path}: {exc}") from exc


def _file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityError("证据路径必须为非空字符串")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CapabilityError(f"证据路径必须在仓库内且为相对路径: {value}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise CapabilityError(f"证据文件不存在或越出仓库: {value}")
    return path


def _paths(root: Path, cell: dict, field: str) -> list[Path]:
    values = cell.get(field)
    if not isinstance(values, list) or not values:
        raise CapabilityError(f"{field} 必须为非空文件路径列表")
    return [_file(root, value) for value in values]


def _completed(root: Path, path: Path, recipe_id: str, dimension: str) -> dict:
    if path.suffix != ".json":
        raise CapabilityError("verified 必须提供真实 evidence JSON 文件，不接受说明字符串")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CapabilityError(f"无效 evidence JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise CapabilityError("evidence JSON 必须为对象，不能只是字符串或数组")
    if payload.get("status" if dimension == "comparison" else "stage") != "completed":
        raise CapabilityError("evidence stage/status 必须为 completed")
    if payload.get("kind") != EVIDENCE_KINDS[dimension]:
        raise CapabilityError(f"evidence kind 必须为 {EVIDENCE_KINDS[dimension]}")
    if payload.get("recipe_id") != recipe_id:
        raise CapabilityError("evidence recipe_id 与登记不一致")
    if payload.get("mode") == "dry-run" or any(
        payload.get(key) is True for key in ("dry_run", "simulated", "mock")
    ):
        raise CapabilityError("dry-run/mock evidence 不得晋级 verified")
    if dimension in ("train", "eval"):
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or any(
            not isinstance(runtime.get(key), str) or not runtime[key].strip()
            or any(marker in runtime[key].lower() for marker in ("mock", "fake", "simulat"))
            for key in ("gpu", "cuda")
        ):
            raise CapabilityError("train/eval verified 缺少真实 GPU/CUDA runtime 证据")
        if dimension == "train" and payload.get("mode") != "full":
            raise CapabilityError("smoke 不是 full training 验证证据")
        if dimension == "eval":
            if payload.get("model_role") != "trained_checkpoint":
                raise CapabilityError("base/zero-shot/fallback 不得冒充训练后 eval verified")
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or not metrics or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in metrics.values()
            ):
                raise CapabilityError("evaluation 缺少有限数值的真实 metrics")
    return payload


def _check_cell(root: Path, recipe_id: str, dimension: str, cell: Any) -> dict:
    if not isinstance(cell, dict) or cell.get("status") not in STATES:
        raise CapabilityError(f"status 必须为 {'/'.join(STATES)}")
    if not isinstance(cell.get("note"), str) or not cell["note"].strip():
        raise CapabilityError("每个维度必须说明实现边界或具体缺失项 note")
    if cell["status"] in ("implemented", "verified"):
        implementations = _paths(root, cell, "implementation")
        tests = _paths(root, cell, "tests")
        if any(path.suffix != ".py" for path in implementations) or any(
            path.suffix != ".py" or not path.name.startswith("test_") for path in tests
        ):
            raise CapabilityError("implemented 必须引用 Python 实装与 test_*.py offline test 文件")
    if cell["status"] == "verified":
        for path in _paths(root, cell, "evidence"):
            _completed(root, path, recipe_id, dimension)
    return dict(cell)


def _metadata_errors(contract: dict, families: dict) -> list[str]:
    errors = []
    if contract.get("schema") != 1:
        errors.append("capabilities schema 必须为 1")
    if contract.get("dimensions") != list(DIMENSIONS):
        errors.append("登记 dimensions 必须精确包含 data/train/eval/comparison/export")
    if contract.get("evidence_kinds") != EVIDENCE_KINDS:
        errors.append("evidence_kinds 不得改变维度的 completed 证据类型")
    if not isinstance(contract.get("states"), dict) or set(contract["states"]) != set(STATES):
        errors.append("states 必须精确声明四种状态")
    scope = contract.get("scope", {})
    if not isinstance(scope, dict) or not scope:
        errors.append("缺少 scope 模块覆盖范围表")
    else:
        for name, item in scope.items():
            if not isinstance(item, dict):
                errors.append(f"scope {name} 必须为表")
                continue
            if item.get("status") not in SCOPE_STATES or not item.get("why") or not item.get("label"):
                errors.append(f"scope {name} 缺合法 status/label/why")
            members = item.get("families")
            if not isinstance(members, list) or any(
                not isinstance(family, str) or family not in families for family in members
            ):
                errors.append(f"scope {name} 引用未登记 family，未 taxon 模块应独立说明")
    waves = contract.get("waves")
    if not isinstance(waves, list) or len(waves) != 7:
        errors.append("实施队列必须包含 wave0 至 wave6")
        return errors
    seen = set()
    for index, item in enumerate(waves):
        if not isinstance(item, dict):
            errors.append(f"wave{index} 必须为表")
            continue
        wave = item.get("id")
        dependencies = item.get("dependencies")
        if wave != f"wave{index}" or not isinstance(dependencies, list) or any(
            not isinstance(dep, str) or dep not in seen for dep in dependencies
        ) or (index > 0 and not dependencies):
            errors.append(f"wave{index} 顺序/依赖无效，必须显式依赖已列出的前置波次")
        if any(not isinstance(item.get(key), str) or not item[key].strip()
               for key in ("label", "prerequisites", "work", "acceptance")):
            errors.append(f"wave{index} 缺 label/prerequisites/work/acceptance")
        if isinstance(wave, str):
            seen.add(wave)
    return errors


@dataclass
class CapabilityAudit:
    contract: dict[str, Any]
    rows: dict[str, dict[str, dict]]
    errors: list[str]
    registered_recipes: int
    capability_entries: int
    missing_families: list[str]
    families: dict[str, dict]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        return {dimension: {state: sum(
            row[dimension]["status"] == state for row in self.rows.values()
        ) for state in STATES} for dimension in DIMENSIONS}

    @property
    def totals(self) -> dict[str, int]:
        return {state: sum(counts[state] for counts in self.counts.values()) for state in STATES}

    @property
    def implemented_recipes(self) -> int:
        return sum(any(cell["status"] in ("implemented", "verified") for cell in row.values())
                   for row in self.rows.values())

    @property
    def gpu_verified_recipes(self) -> int:
        return sum(all(row[dim]["status"] == "verified" for dim in ("train", "eval"))
                   for row in self.rows.values())

    @property
    def untaxonomized_modules(self) -> list[str]:
        scope = self.contract.get("scope", {})
        if not isinstance(scope, dict):
            return []
        return sorted(name for name, item in scope.items() if isinstance(item, dict)
                      and item.get("taxonomy") == "not_registered")


def audit(root: Path, recipes: Iterable[Recipe]) -> CapabilityAudit:
    """精确 ID 覆盖审计；任何无效维度降为 missing，绝不从 B1/adapter_status 推断。"""
    root = Path(root)
    recipes = list(recipes)
    errors = []
    try:
        contract = load(root)
    except CapabilityError as exc:
        contract = {}
        errors.append(str(exc))
    try:
        with (root / "registry" / "taxonomy.toml").open("rb") as handle:
            families = tomllib.load(handle).get("families", {})
        if not isinstance(families, dict):
            raise ValueError("families 必须为表")
    except (OSError, ValueError) as exc:
        families = {}
        errors.append(f"无法加载当前 root 的 taxonomy: {exc}")
    if contract:
        errors.extend(_metadata_errors(contract, families))
    ids = [recipe.id for recipe in recipes]
    if len(set(ids)) != len(ids):
        errors.append("输入 recipes 有重复 id")
    if contract and (type(contract.get("expected_recipe_count")) is not int
                     or contract["expected_recipe_count"] != len(ids)):
        errors.append("expected_recipe_count 与完整 recipes 数量不一致")
    entries = contract.get("recipes", {})
    if not isinstance(entries, dict):
        errors.append("recipes 能力登记必须为表")
        entries = {}
    for recipe_id in sorted(set(ids) - set(entries)):
        errors.append(f"缺少 recipe 能力登记: {recipe_id}")
    for recipe_id in sorted(set(entries) - set(ids)):
        errors.append(f"orphan 能力登记: {recipe_id}")
    rows = {}
    for recipe in sorted(recipes, key=lambda item: item.id):
        if recipe.family not in families:
            errors.append(f"{recipe.id}: 未登记 method.family {recipe.family!r}")
        entry = entries.get(recipe.id, {})
        if not isinstance(entry, dict):
            errors.append(f"{recipe.id}: 能力登记必须为表")
            entry = {}
        unknown = set(entry) - set(DIMENSIONS) - {"wave"}
        if unknown:
            errors.append(f"{recipe.id}: 非法 dimension/字段 {sorted(unknown)}")
        if entry.get("wave") not in {f"wave{i}" for i in range(7)}:
            errors.append(f"{recipe.id}: 缺合法 wave")
        rows[recipe.id] = {}
        for dimension in DIMENSIONS:
            try:
                if dimension not in entry:
                    raise CapabilityError("缺少 dimension")
                cell = _check_cell(root, recipe.id, dimension, entry[dimension])
            except CapabilityError as exc:
                errors.append(f"{recipe.id}/{dimension}: {exc}")
                cell = {"status": "missing", "note": f"登记无效或缺失（fail-closed）：{exc}"}
            rows[recipe.id][dimension] = cell
    used = {recipe.family for recipe in recipes}
    return CapabilityAudit(contract, rows, errors, len(set(ids)), len(set(ids) & set(entries)),
                           sorted(set(families) - used), families)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render(root: Path, recipes: Iterable[Recipe]) -> str:
    """返回不超过 199 行的独立章节；只读，不自动晋级、不落盘、不运行训练。"""
    result = audit(root, recipes)
    contract = result.contract
    totals = result.totals
    lines = [
        "## 实际能力与全生态范围（独立于 Baseline Readiness）", "",
        f"> 本地时点：{_md(contract.get('assessed_at', '未登记'))}；不联网，不修改 taxonomy。",
        f"- recipe 已登记：{result.registered_recipes}；能力条目匹配：{result.capability_entries}；"
        f"至少一维代码已实现：{result.implemented_recipes} 篇；train+eval GPU verified：{result.gpu_verified_recipes} 篇。",
        f"- 五维实际计数：missing {totals['missing']} / implemented {totals['implemented']} / "
        f"unverified {totals['unverified']} / verified {totals['verified']}。",
        "- implemented = 真实任务代码 + offline test 文件/覆盖；不代表实际数据下载、依赖兼容、GPU 通过或收敛。",
        "- missing = 任务逻辑缺失；unverified = 证据不足；verified = impl/test 齐备且真实 completed JSON kind 相符。",
        "- 配方、family、B1、lock 模板与 dry-run 只是声明；不等于实现、GPU 验证、research-ready 或 business-ready。",
        "- GPU verified 单次运行仍不代表 3 training seeds 的可比基线；本节绝不自动更新 B2/B3。", "",
    ]
    if result.errors:
        lines += [f"**能力审计失败（{len(result.errors)} 项）：禁止依据该登记晋级；无效维度已按 missing 关闭。**"]
        lines.extend(f"- {_md(error)}" for error in result.errors[:8])
        if len(result.errors) > 8:
            lines.append("- 其余错误请读取 `audit(root, recipes).errors`。")
        lines.append("")
    else:
        lines += ["能力登记结构/文件审计通过；不是 GPU 测试结果，也不是 offline test 实时执行凭据。", ""]
    lines += ["### 五维能力计数", "", "| 维度 | missing | implemented | unverified | verified |",
              "| --- | --- | --- | --- | --- |"]
    for dimension, counts in result.counts.items():
        lines.append(f"| {dimension} | " + " | ".join(str(counts[state]) for state in STATES) + " |")
    lines += ["", "### 逐篇能力矩阵", "", "| recipe | data | train | eval | comparison | export |",
              "| --- | --- | --- | --- | --- | --- |"]
    for recipe_id, row in result.rows.items():
        lines.append(f"| `{_md(recipe_id)}` | " + " | ".join(row[d]["status"] for d in DIMENSIONS) + " |")
    lines += ["", "### 每篇具体缺失与实现边界", ""]
    for recipe_id, row in result.rows.items():
        notes = []
        for dimension, cell in row.items():
            files = [path for field in ("implementation", "tests")
                     if isinstance(cell.get(field), list) for path in cell[field]
                     if isinstance(path, str)]
            evidence = "；仓内证据 " + ", ".join(f"`{_md(p)}`" for p in files) if files else ""
            notes.append(f"**{dimension} [{cell['status']}]** {_md(cell['note'])}{evidence}")
        lines.append(f"- `{_md(recipe_id)}`：" + "；".join(notes))
    lines += ["", "### 方法缺口：taxonomy 已登记但没有 recipe", ""]
    if result.missing_families:
        for family in result.missing_families:
            meta = result.families[family]
            lines.append(f"- `{_md(family)}`：trainer `{_md(meta.get('trainer', '-'))}`，"
                         f"tier `{_md(meta.get('tier', '-'))}`；配方数 0，不能计入能力覆盖。")
    else:
        lines.append("- 未发现已登记但无 recipe 的 family；taxonomy 读失败时不可据此声称全覆盖。")
    lines += [
        "- 上述列表由当前 root 的 families − recipes.method.family 自动计算；不是按实现数计算。",
        "- 未纳入 taxonomy 的独立模块：" + ("、".join(f"`{name}`" for name in result.untaxonomized_modules) or "未登记") + "。",
        "- 数据构建/eval/发布是跨方法管线，不是空缺 trainer family；已有视觉 DPO/MPO 也不等于文本偏好覆盖。",
        "", "### Scope 模块覆盖范围", "", "| 模块 | 范围状态 | taxonomy 边界 | why |", "| --- | --- | --- | --- |",
    ]
    scope = contract.get("scope", {})
    if isinstance(scope, dict):
        for name, item in scope.items():
            if not isinstance(item, dict):
                continue
            members = item.get("families", [])
            boundary = ("未纳入 taxonomy" if item.get("taxonomy") == "not_registered"
                        else ", ".join(_md(member) for member in members)
                        if isinstance(members, list) and members else "跨方法管线")
            lines.append(f"| {_md(item.get('label', name))} | {_md(item.get('status', '未登记'))} | "
                         f"{_md(boundary)} | {_md(item.get('why', '未登记'))} |")
    lines += ["", "### 实施顺序：依赖、前置条件与验收", "",
              "| 波次 | dependencies | 前置条件 | 工作 | 验收 |", "| --- | --- | --- | --- | --- |"]
    waves = contract.get("waves", [])
    if isinstance(waves, list):
        for wave in waves:
            if not isinstance(wave, dict):
                continue
            dependencies = wave.get("dependencies")
            dependency_text = (", ".join(_md(dep) for dep in dependencies)
                               if isinstance(dependencies, list) else "无效依赖")
            lines.append(f"| {_md(wave.get('id', '?'))} {_md(wave.get('label', ''))} | "
                         f"{dependency_text or '无'} | "
                         f"{_md(wave.get('prerequisites', ''))} | {_md(wave.get('work', ''))} | "
                         f"{_md(wave.get('acceptance', ''))} |")
    lines += [
        "", "### Pipeline 边界与验收条款", "",
        "**数据构建/冻结 → zero train（zero-shot）→ smoke → full（至少 3 train seeds）→ held-out → aggregate → export reload → business**。",
        "- 数据/冻结：许可与来源可追踪；model/dataset/processor/依赖真实 revision 锁定；schema、gold 剥离、去重、train/val/test manifest 与 hash 齐全。",
        "- zero train：只加载 base，不是训练证据；与训练后模型共享 prompt、test IDs/hash、解码参数、预算和指标协议；若零训练已满足目标则记录不训练决策。",
        "- smoke：少量真实训练验证 loss/reward/梯度、资源、保存/加载与失败路径；不代表 full，不产出研究结论。",
        "- full：至少 3 个不同 training seeds 各自从同一冻结 base 独立训练，独立输出/日志/checkpoint；固定 data split/data seed；重复 greedy decode 不能代替训练方差。",
        "- held-out：只用 val 选超参/checkpoint，test 不参与选型；各训练 seed 对应自己的 trained checkpoint；缺 checkpoint 必须报错，不回退 base；逐样本预测与指标可追踪。",
        "- aggregate：校验训练/评估 JSON 的 recipe、kind、stage、seed、revision、config/test hash 与 checkpoint 身份；同协议 zero-shot 对照、均值/标准差/置信区间及预先声明的容差。",
        "- export reload：模型/adapter/base、tokenizer/processor、模板、依赖、manifest 与许可打包；全新进程离线复载，检查 artifact hash 和指标容差；save_model 单独不算交付。",
        "- business：research 门禁通过后再验成本、吞吐/显存、SLO、任务质量、安全/许可、owner、监控漂移和回滚；技术 verified 不自动成为业务验收。",
        "- 证据门禁：implemented/verified 必须引用仓内 implementation 与 test 文件；verified 必须读 JSON 对象，stage=completed 且 kind 匹配，train/eval 必须真实 GPU/CUDA runtime；mock/dry-run/smoke/base 不得冒充 full/eval。",
        "- 审计不是密码学执行证明：不能仅靠人工填写 JSON 认证 GPU；保留运行日志/artifact 并由独立流程复查，禁止用测试夹具作为真实实跑证据。",
        "- 数学样板的当前有效切分以 math_task.py 为准（0:500 / 500:1000 / 1000:1500）；旧 recipe 中的声明或通用 fallback 不覆盖此事实。",
        "- 本节只读报告，可由主线程直接拼接 `capabilities.render(root, recipes)`；不实现聚合器、export 工具或远端发布，不估工期。",
    ]
    if len(lines) >= 200:
        raise CapabilityError("能力报告超过 199 行，需收敛范围后再插入主报告")
    return "\n".join(lines) + "\n"
