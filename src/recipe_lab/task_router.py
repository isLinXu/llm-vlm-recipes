"""发布级管线的任务路由；标准库、只读、未知任务失败关闭。

runner/suite.py、runner/export.py 与 baseline 门禁此前都把数学样板当成唯一实现。
这里只做精确 ID / task_id 分派，不推断能力、不降级为通用 JSON 校验。
"""
from __future__ import annotations

from .math_task import TaskExecutionError

MATH = "math-grpo-v1"
FIM = "code-fim-v1"
XLAM = "xlam-fc-v1"
RAPIDFIRE = "rapidfire-sft-v1"
GRANITE = "granite-linecomparison-v1"
PALIGEMMA = "paligemma-grounding-v1"
DPO = "smolvlm-dpo-v1"
MPO = "qwen25vl-mpo-v1"
GSM8K = "gsm8k-reward-v1"


def resolve(r) -> str | None:
    """返回精确 task_id；声明与 recipe id 冲突时拒绝。"""
    from . import dpo_task as dt
    from . import fim_task as ft
    from . import granite_task as gt
    from . import math_task as mt
    from . import mpo_task as mpt
    from . import paligemma_task as pt
    from . import rapidfire_task as rf
    from . import gsm8k_task as gt8k
    from . import xlam_task as xt
    by_id = {mt.RECIPE_ID: MATH, ft.RECIPE_ID: FIM, xt.RECIPE_ID: XLAM,
             rf.RECIPE_ID: RAPIDFIRE, gt.RECIPE_ID: GRANITE,
             pt.RECIPE_ID: PALIGEMMA, dt.RECIPE_ID: DPO, mpt.RECIPE_ID: MPO,
             gt8k.RECIPE_ID: GSM8K}.get(r.id)
    values = {value for value in (r.get("execution.task_id"), by_id) if value is not None}
    if len(values) > 1:
        raise TaskExecutionError(f"任务路由冲突：recipe={r.id}, candidates={sorted(values)}")
    if not values:
        return None
    if by_id is None:
        raise TaskExecutionError(
            f"recipe={r.id} 声明了 execution.task_id，但没有该 recipe 的发布级管线实现；拒绝借用其他任务协议")
    task_id = next(iter(values))
    if task_id not in (MATH, FIM, XLAM, RAPIDFIRE, GRANITE, PALIGEMMA, DPO, MPO, GSM8K):
        raise TaskExecutionError(f"未知 execution.task_id={task_id!r}；拒绝通用发布级验收")
    return task_id


def _module(task_id, kind):
    if task_id == MATH:
        return {"task": "math_task", "suite": "suite_task", "export": "export_task"}[kind]
    if task_id == FIM:
        return {"task": "fim_task", "suite": "fim_suite", "export": "fim_export"}[kind]
    if task_id == XLAM:
        # xLAM 用严格 JSON / tool schema 的独立套件与 adapter-only 导出协议；
        # 不复用数学 1500 题或 FIM PSM 的结构，也不用通用 JSON 放行。
        return {"task": "xlam_task", "suite": "xlam_suite", "export": "xlam_export"}[kind]
    if task_id == RAPIDFIRE:
        # RapidFire 是 4 配置 × 独立 seed 网格；validation 选配置、test 单次验收，
        # 导出只针对选中配置的 fp16 LoRA adapter，不借用上面三种内容协议。
        return {"task": "rapidfire_task", "suite": "rapidfire_suite",
                "export": "rapidfire_export"}[kind]
    if task_id == GRANITE:
        # Granite 是图像 hash 隔离的 VLM：逐题从 eval_manifest 的 question/answer
        # 重建 gold 并重算 exact_match，导出 DoRA/4bit adapter + processor identity；
        # 不复用数学 1500 题、FIM PSM、xLAM 严格 JSON 或 RapidFire ROUGE-L 协议。
        return {"task": "granite_task", "suite": "granite_suite",
                "export": "granite_export"}[kind]
    if task_id == PALIGEMMA:
        # PaliGemma 是单框 grounding：逐题从 eval_manifest 的 raw COCO bbox +
        # 冻结宽高重建 <loc####> gold，并重算 bbox IoU；导出普通 LoRA adapter +
        # 统一 prompt/loc 协议；不复用 Granite 的 exact_match 或上面四种协议。
        return {"task": "paligemma_task", "suite": "paligemma_suite",
                "export": "paligemma_export"}[kind]
    if task_id == DPO:
        # SmolVLM DPO 是偏好对图像 hash 隔离 + beta/ref 冻结 + 双层 completion-only
        # label mask + 逐题隐式奖励 pairwise 重算；adapter-only 导出不打包 ref 权重；
        # 与 MPO 篇基座不同，不复用任何上面任务的协议或身份。
        return {"task": "dpo_task", "suite": "dpo_suite",
                "export": "dpo_export"}[kind]
    if task_id == MPO:
        # Qwen2.5-VL MPO 与 DPO 篇共享偏好 split，但叠加 sigmoid/bco_pair/sft
        # 三损失消融 × 独立 seed，并如实登记 bf16=False 与量化 compute dtype 矛盾。
        # 三损失消融只在同一 backbone 内，不把 DPO 篇当算法消融。
        return {"task": "mpo_task", "suite": "mpo_suite",
                "export": "mpo_export"}[kind]
    if task_id == GSM8K:
        # GSM8K 四 reward 是全参数 GRPO：drop-one 消融 × 独立 seed 编排，
        # 独立严格 Decimal verifier 重算 pass@1 / format_compliance /
        # reward_hacking_rate；导出整模型 safetensors（非 adapter）。
        return {"task": "gsm8k_task", "suite": "gsm8k_suite",
                "export": "gsm8k_export"}[kind]
    raise TaskExecutionError(f"没有实现 {task_id!r} 的 {kind} 管线；不得用数学协议替代")


def load(r, kind: str):
    """按 recipe 载入 task / suite / export 模块；未知任务抛 TaskExecutionError。"""
    import importlib
    return importlib.import_module("." + _module(resolve(r), kind), __package__)


def from_bundle(bundle, kind: str):
    """worker 从包内 recipe 快照推断任务，不接受命令行自称的任务名。"""
    import json
    from pathlib import Path
    from .model import Recipe
    bundle = Path(bundle)
    snapshot = json.loads((bundle / "recipe-snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("id"), str):
        raise TaskExecutionError("bundle 缺少有效 recipe-snapshot.json")
    r = Recipe(snapshot["id"], bundle / "recipe.toml", snapshot)
    return load(r, kind), r
