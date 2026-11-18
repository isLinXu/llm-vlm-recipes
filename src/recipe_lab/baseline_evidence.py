"""证据内容验证，不把文件存在/声明文本等同于真实基线。

本模块只能验证内部一致性，不能证明提交的日志确实来自 GPU，也不能替代人工复核。
B2 使用 baseline-suite（独立训练 seeds 的基座/训练模型同协议对照），
单次 training.json/evaluation.json 不足以晋级。所有函数标准库、只读、失败关闭。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tomllib
from pathlib import Path

HEX40 = re.compile(r"[0-9a-fA-F]{40}")
HEX64 = re.compile(r"[0-9a-fA-F]{64}")
PLACEHOLDERS = {"", "unknown", "unmeasured", "missing", "pending", "todo", "main", "latest", "(未锁定)", "待确认", "待实跑"}


def nonempty(value):
    return value is not None and value != "" and value != [] and value != {}


def meaningful(value):
    return nonempty(value) and (not isinstance(value, str) or value.strip().lower() not in PLACEHOLDERS)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _task(r):
    """按精确 task_id 分派发布级证据重算器；未知/冲突任务返回 None 以失败关闭。"""
    try:
        from .task_router import resolve
        return resolve(r)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError):
        return None


def fingerprint(r):
    # 排除可变的审核状态、首次实测后回填的 expected/tolerance 与成本字段。
    fields = ["method", "model", "dataset", "environment", "dependencies", "train", "peft", "quantization", "rewards", "budget", "source_sha"]
    payload = {key: r.get(key) for key in fields}
    if r.get("fim") is not None:
        payload["fim"] = r.get("fim")
    # 任务专属数据/输出协议必须进入内容身份：否则改了 PSM 或严格 JSON 协议，
    # 旧 checkpoint 仍能凭相同 fingerprint 冒充当前实验产物。
    if r.get("xlam") is not None:
        payload["xlam"] = r.get("xlam")
        payload["upstream"] = r.get("upstream")
    # RapidFire 是 4 配置网格：协议段、完整网格与当前 config_id 都必须进入内容身份，
    # 否则改 lr/rank 后旧 checkpoint 仍能凭同一 fingerprint 冒充当前配置产物。
    if r.get("rapidfire") is not None:
        payload["rapidfire"] = r.get("rapidfire")
        payload["configs"] = r.get("configs")
        payload["config_id"] = r.get("execution.config_id")
    # Granite 自建 VLM 协议把图像 hash 划分、双层 label mask、processor identity 与
    # 逐题 exact_match 定义写在 [granite] / [upstream] / [modality_contract]；
    # 不绑定这些段，改了 mask 或图像输入协议后旧 checkpoint 仍能冒充当前实验产物。
    if r.get("granite") is not None:
        payload["granite"] = r.get("granite")
        payload["upstream"] = r.get("upstream")
        payload["modality_contract"] = r.get("modality_contract")
    # PaliGemma 自建协议把 coco_xywh→xyxy→1024 桶 loc round-trip、统一 prompt
    # 模板、单框 IoU 指标与 processor 合同写在 [paligemma] / [upstream] /
    # [modality_contract]；不绑定这些段，改了 bbox/prompt 协议后旧 checkpoint
    # 仍能凭同一 fingerprint 冒充当前实验产物。
    if r.get("paligemma") is not None:
        payload["paligemma"] = r.get("paligemma")
        payload["upstream"] = r.get("upstream")
        payload["modality_contract"] = r.get("modality_contract")
    # wave4 自建偏好协议把 beta/ref 或三损失/消融、图像 hash 划分、双层 label mask 与
    # 隐式奖励 pairwise 定义写在 [preference] / [upstream] / [modality_contract]；
    # MPO 还把当前 config_id 写进身份。不绑定这些段，改 beta 或消融权重后旧
    # checkpoint 仍能凭同一 fingerprint 冒充当前实验产物。
    if r.get("preference") is not None:
        payload["preference"] = r.get("preference")
        payload["upstream"] = r.get("upstream")
        payload["modality_contract"] = r.get("modality_contract")
        if r.get("loss") is not None:
            payload["loss"] = r.get("loss")
        if r.get("execution.config_id") is not None:
            payload["config_id"] = r.get("execution.config_id")
    payload["task_id"] = r.get("execution.task_id")
    payload["evaluation"] = {key: r.get("evaluation." + key) for key in ("metrics", "split", "seed")}
    if any(r.get(name) is not None for name in ("fim", "xlam", "rapidfire", "granite", "paligemma",
                                                "preference")):
        payload["evaluation"]["max_new_tokens"] = r.get("evaluation.max_new_tokens")
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(r, field):
    try:
        path = (r.path.parent / str(r.get(field))).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        return (value, path) if isinstance(value, dict) else (None, path)
    except (OSError, ValueError, TypeError):
        return None, None


def hashed_file(base: Path, record):
    if not isinstance(record, dict) or not HEX64.fullmatch(str(record.get("sha256", ""))):
        return False
    try:
        path = (base / str(record["path"])).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == record["sha256"]
    except (OSError, KeyError, TypeError):
        return False


def frozen_environment(r):
    try:
        path = (r.path.parent / str(r.get("execution.lockfile"))).resolve()
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = lock.get("runtime", {})
        requirements = lock.get("requirements", [])
        if not isinstance(runtime, dict) or not isinstance(requirements, list) or not requirements:
            return False
        installed = {}
        for requirement in requirements:
            if not isinstance(requirement, str):
                return False
            exact = re.fullmatch(r"([\w.-]+)==([^\s*<>~=]+)", requirement)
            vcs = re.fullmatch(r"([\w.-]+) @ git\+https://[^\s]+@[0-9a-fA-F]{40}", requirement)
            if not (exact and meaningful(exact[2]) or vcs):
                return False
            name = re.sub(r"[-_.]+", "-", (exact or vcs)[1]).lower()
            if name in installed:
                return False
            installed[name] = exact[2] if exact else requirement
        declared = r.get("dependencies", {}) or {}
        # 冻结依赖集合必须与任务管线一致：xLAM / Granite 用 Trainer + safetensors，
        # 不依赖 trl；若沿用数学的 trl 要求，真实 frozen lock 会永远被判为未冻结。
        required = {
            "code-fim-v1": {"torch", "transformers", "datasets", "peft", "bitsandbytes"},
            "xlam-fc-v1": {"torch", "transformers", "datasets", "peft", "bitsandbytes", "safetensors"},
            "rapidfire-sft-v1": {"torch", "transformers", "datasets", "peft", "safetensors"},
            "granite-linecomparison-v1": {"torch", "transformers", "datasets", "peft",
                                          "bitsandbytes", "safetensors"},
            # PaliGemma 是普通 Trainer + 普通 LoRA（无 4bit/DoRA、无在线增强），
            # 不依赖 trl / bitsandbytes；要求它们会让真实 frozen lock 永远失配。
            "paligemma-grounding-v1": {"torch", "transformers", "datasets", "peft",
                                        "safetensors"},
            # DPO / MPO 都是普通 Trainer + 4bit/DoRA + AutoProcessor + safetensors 严格复载，
            # 不依赖 trl（自建 loss 不依赖未固定 TRL 源码）。
            "smolvlm-dpo-v1": {"torch", "transformers", "datasets", "peft",
                               "bitsandbytes", "safetensors"},
            "qwen25vl-mpo-v1": {"torch", "transformers", "datasets", "peft",
                                "bitsandbytes", "safetensors"},
        }.get(r.get("execution.task_id"))
        if required is None:
            # 未声明发布级 task_id 的任务按协议段判定：RapidFire 是 Trainer + fp16 LoRA，
            # Granite 是 Trainer + 4bit/DoRA + AutoProcessor，PaliGemma 是 Trainer +
            # 普通 LoRA，三者都不依赖 trl；未知 recipe 仍按数学默认集合失败关闭。
            if r.get("rapidfire") is not None or r.get("paligemma") is not None:
                required = {"torch", "transformers", "datasets", "peft", "safetensors"}
            elif r.get("granite") is not None or r.get("preference") is not None:
                required = {"torch", "transformers", "datasets", "peft",
                            "bitsandbytes", "safetensors"}
            else:
                required = {"torch", "transformers", "datasets", "trl"}
        if not required <= set(installed):
            return False
        if any(installed.get(re.sub(r"[-_.]+", "-", name).lower()) != version for name, version in declared.items()):
            return False
        return (type(lock.get("schema_version")) is int and lock["schema_version"] == 1 and lock.get("status") == "frozen"
                and lock.get("recipe_id") == r.id and lock.get("model_id") == r.get("model.base")
                and bool(HEX40.fullmatch(str(r.get("model.revision", ""))))
                and lock.get("model_revision") == r.get("model.revision")
                and bool(HEX40.fullmatch(str(r.get("source_sha", ""))))
                and lock.get("source_sha") == r.get("source_sha") and dataset_pinned(r)
                and (lock.get("dataset_id") == r.get("dataset.name") and lock.get("dataset_revision") == r.get("dataset.revision") if r.get("dataset.kind", "static") == "static" else lock.get("environment_revision") == r.get("environment.revision"))
                and all(meaningful(runtime.get(key)) for key in ("python", "cuda", "driver", "gpu")))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


def dataset_pinned(r):
    if r.get("dataset.kind", "static") == "online":
        return (bool(HEX40.fullmatch(str(r.get("environment.revision", ""))))
                and bool(HEX64.fullmatch(str(r.get("environment.corpus_sha256", ""))))
                and type(r.get("environment.seed")) is int)
    return bool(HEX40.fullmatch(str(r.get("dataset.revision", ""))))


def linked_json(base, record):
    if not hashed_file(base, record):
        return None
    try:
        return json.loads((base / record["path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def valid_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        return False
    from .math_task import sha256_json
    if manifest.get("sha256") != sha256_json({k: v for k, v in manifest.items() if k != "sha256"}):
        return False
    owners = {}
    for name in ("train", "validation", "test"):
        split = manifest["splits"].get(name, {})
        if not isinstance(split, dict):
            return False
        samples = split.get("samples")
        if not isinstance(samples, list) or not samples or split.get("count") != len(samples):
            return False
        if split.get("sha256") != sha256_json(samples):
            return False
        ids = set()
        for sample in samples:
            if not isinstance(sample, dict) or not meaningful(sample.get("sample_id")) or sample["sample_id"] in ids:
                return False
            ids.add(sample["sample_id"])
            for field in ("sample_id", "content_sha256", "problem_sha256"):
                key = sample.get(field)
                if field.endswith("sha256") and not HEX64.fullmatch(str(key)):
                    return False
                identity = (field, key)
                if identity in owners and owners[identity] != name:
                    return False
                owners[identity] = name
    return True


def runtime_consistent(r, evidence):
    runtime = evidence.get("runtime", {})
    if not isinstance(runtime, dict) or any(not meaningful(runtime.get(k)) for k in ("python", "cuda", "gpu")):
        return False
    if any(marker in str(runtime[k]).lower() for k in ("gpu", "cuda") for marker in ("mock", "fake", "simulat")):
        return False
    try:
        lock = tomllib.loads((r.path.parent / str(r.get("execution.lockfile"))).read_text(encoding="utf-8"))
        target = lock["runtime"]
        return (str(runtime["python"]).split()[0] == target["python"]
                and runtime["cuda"] == target["cuda"] and runtime["gpu"] == target["gpu"])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def evaluation_problems(r, data, path, *, role, manifest=None):
    task = _task(r)
    if task in ("code-fim-v1", "xlam-fc-v1", "rapidfire-sft-v1", "granite-linecomparison-v1",
                            "paligemma-grounding-v1", "smolvlm-dpo-v1", "qwen25vl-mpo-v1"):
        # FIM 用 PSM 输入/gold 重建器，xLAM 用严格 JSON prompt/tools/gold 重建器，
        # RapidFire 用 completion-only 5 列 raw / 无 stemmer ROUGE-L 重建器；
        # DPO/MPO 用偏好对 raw + 隐式奖励 pairwise 重建器；都不套用数学协议，也不互相借用。
        from .task_router import load
        return load(r, "suite").evaluation_problems(r, data, path, role=role, manifest=manifest)
    if task != "math-grpo-v1":
        return ["尚未实现该任务的发布级来源/gold重算器；执行评估不等于B2证据验收"]
    from .math_task import sha256_json
    if not isinstance(data, dict):
        return ["评估证据不是JSON对象"]
    if (data.get("kind") != "evaluation" or data.get("stage") != "completed"
            or data.get("model_role") != role or data.get("recipe_id") != r.id
            or data.get("model_revision") != r.get("model.revision")
            or data.get("dataset_revision") != r.get("dataset.revision")
            or any(data.get(key) is True for key in ("mock", "simulated", "dry_run"))):
        return ["评估来源、角色或输入不一致"]
    if not runtime_consistent(r, data):
        return ["评估运行环境与frozen lock不一致或缺失"]
    config = data.get("config")
    if not isinstance(config, dict) or data.get("config_sha256") != sha256_json(config):
        return ["评估配置内容hash不一致"]
    actual_manifest = data.get("split_manifest")
    if not valid_manifest(actual_manifest) or (manifest is not None and actual_manifest != manifest):
        return ["held-out manifest无效或与套件不一致"]
    predictions = linked_json(path.parent, data.get("predictions_file"))
    count = data.get("sample_count")
    if type(count) is not int or count <= 0 or not isinstance(predictions, list) or len(predictions) != count:
        return ["逐样本预测缺失或数量不符"]
    test = actual_manifest["splits"]["test"]["samples"]
    if len(test) != count or any(not isinstance(p, dict) or p.get("sample_id") != s["sample_id"]
                               or p.get("content_sha256") != s["content_sha256"] for p, s in zip(predictions, test)):
        return ["预测与held-out样本身份不一致"]
    metrics = data.get("metrics")
    names = r.get("evaluation.metrics", [])
    if not isinstance(metrics, dict) or not names or any(not finite_number(metrics.get(name)) for name in names):
        return ["指标缺失或不是有限数值"]
    # 数学样板可从逐样本binary分数独立重算；其他任务必须另补任务专用重算器。
    from .math_task import resolve_task_id
    if resolve_task_id(r) == "math-grpo-v1":
        for metric, field in (("accuracy", "accuracy"), ("pass@1", "accuracy"), ("format_compliance", "format_compliance")):
            if any(type(p.get(field)) is not int or p[field] not in (0, 1) for p in predictions):
                return ["数学逐样本分数非法"]
            if metric in metrics and not math.isclose(metrics[metric], sum(p[field] for p in predictions) / count, abs_tol=1e-12):
                return ["数学指标与逐样本重算不一致"]
    return []


def suite_problems(r):
    try:
        return _suite_problems(r)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return ["证据结构非法；失败关闭，不允许晋级"]


def _suite_problems(r):
    task = _task(r)
    if task in ("code-fim-v1", "xlam-fc-v1", "rapidfire-sft-v1", "granite-linecomparison-v1",
                            "paligemma-grounding-v1", "smolvlm-dpo-v1", "qwen25vl-mpo-v1"):
        from .task_router import load
        return load(r, "suite").suite_problems(r)
    if task != "math-grpo-v1":
        return ["尚未实现该任务的独立训练seed套件与强校验；拒绝通用JSON晋级"]
    data, path = read_json(r, "execution.evidence")
    if data is None:
        return ["缺少可读的 baseline-suite JSON"]
    problems = []
    def need(ok, text):
        if not ok:
            problems.append(text)
    need(data.get("schema_version") == 1 and data.get("kind") == "baseline-suite" and data.get("status") == "completed", "不是已完成的 baseline-suite v1")
    need(data.get("recipe_id") == r.id and data.get("fingerprint") == fingerprint(r), "证据与当前实验配置不一致")
    need(bool(HEX64.fullmatch(str(data.get("code_sha256", "")))) and bool(HEX64.fullmatch(str(data.get("split_sha256", "")))), "缺少代码/数据划分内容指纹")
    need(data.get("model_revision") == r.get("model.revision") and data.get("dataset_revision") == r.get("dataset.revision"), "模型或数据 revision 不一致")
    need(frozen_environment(r), "没有冻结的完整运行环境")
    need(hashed_file(path.parent, data.get("environment_lock")), "环境锁证据文件校验失败")
    if hashed_file(path.parent, data.get("environment_lock")):
        actual_lock = (r.path.parent / str(r.get("execution.lockfile"))).read_bytes()
        need((path.parent / data["environment_lock"]["path"]).read_bytes() == actual_lock, "套件环境锁与当前frozen lock不一致")
    snapshot = data.get("code_snapshot", {})
    need(hashed_file(path.parent, snapshot) and snapshot.get("sha256") == data.get("code_sha256"), "代码快照文件缺失或hash不一致")
    manifest = linked_json(path.parent, data.get("split_manifest"))
    need(valid_manifest(manifest) and manifest.get("sha256") == data.get("split_sha256"), "划分 manifest 内容/身份/去重校验失败")
    metric_names = r.get("evaluation.metrics", [])
    need(isinstance(metric_names, list) and bool(metric_names) and all(isinstance(m, str) for m in metric_names), "指标列表必须非空")
    expected, tolerance = r.get("evaluation.expected"), r.get("evaluation.tolerance")
    need(isinstance(expected, dict) and set(metric_names) <= set(expected), "期望指标必须为覆盖全部 metrics 的数值表")
    need(isinstance(tolerance, dict) and set(metric_names) <= set(tolerance), "容差必须为覆盖全部 metrics 的非负数值表")
    runs = data.get("runs")
    if not isinstance(runs, list) or len(runs) < 3:
        return problems + ["至少需要三个独立训练 seed 的运行，而不是重复 greedy 解码"]
    seeds = [run.get("train_seed") for run in runs if isinstance(run, dict)]
    need(len(seeds) == len(runs) and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds), "训练 seed 必须是三个以上互异整数")
    for run in runs:
        if not isinstance(run, dict):
            problems.append("run 结构非法")
            continue
        need(run.get("kind") == "paired-evaluation" and run.get("training_completed") is True, "每轮需要真实训练后的配对评估")
        need(type(run.get("sample_count")) is int and run["sample_count"] > 0, "评测样本数必须为正整数")
        need(run.get("split_sha256") == data.get("split_sha256"), "基座/训练模型必须共用同一 held-out 划分")
        training = linked_json(path.parent, run.get("training_evidence"))
        need(isinstance(training, dict) and training.get("kind") == "training"
             and training.get("stage") == "completed" and training.get("mode") == "full"
             and training.get("recipe_id") == r.id and training.get("seed") == run.get("train_seed")
             and training.get("data_seed") == r.get("train.data_seed")
             and training.get("model_revision") == r.get("model.revision")
             and training.get("dataset_revision") == r.get("dataset.revision")
             and training.get("split_manifest") == manifest
             and not any(training.get(k) is True for k in ("mock", "simulated", "dry_run")), "训练证据内容与运行身份不一致")
        if isinstance(training, dict):
            need(runtime_consistent(r, training), "训练运行环境与frozen lock不一致或缺失")
            from .math_task import sha256_json
            config = training.get("config")
            max_steps = r.get("train.max_steps", r.get("budget.full_steps"))
            need(isinstance(config, dict) and training.get("config_sha256") == sha256_json(config)
                 and type(training.get("steps")) is int and training["steps"] == max_steps
                 and training.get("max_steps") == max_steps, "训练未完成full预算或配置hash无效")
            if isinstance(config, dict):
                need(all(config.get(k) == (run.get("train_seed") if k == "seed" else v)
                         for k, v in (r.get("train", {}) or {}).items()), "训练超参不一致，不能混合不同实验求均值")
        base_eval = linked_json(path.parent, run.get("base_evaluation"))
        trained_eval = linked_json(path.parent, run.get("trained_evaluation"))
        for side, evaluation, record, role in (
                ("base_metrics", base_eval, run.get("base_evaluation"), "base"),
                ("trained_metrics", trained_eval, run.get("trained_evaluation"), "trained_checkpoint")):
            if not isinstance(evaluation, dict) or not isinstance(record, dict):
                problems.append("每轮必须关联真实基座与训练模型评估JSON，不能只写指标摘要")
                continue
            ep = path.parent / record["path"]
            problems.extend(evaluation_problems(r, evaluation, ep, role=role, manifest=manifest))
            need(evaluation.get("metrics") == run.get(side) and evaluation.get("sample_count") == run.get("sample_count"), "套件摘要与关联评估指标或样本数不一致")
        if isinstance(base_eval, dict) and isinstance(trained_eval, dict):
            need(base_eval.get("config", {}).get("generation") == trained_eval.get("config", {}).get("generation")
                 and base_eval.get("config", {}).get("system_prompt") == trained_eval.get("config", {}).get("system_prompt"), "基座与训练模型解码/prompt协议不同")
            need(trained_eval.get("seed") == run.get("train_seed"), "checkpoint评估与训练seed不一致")
            if isinstance(training, dict):
                need(training.get("model_artifact", {}).get("sha256") == trained_eval.get("model_artifact", {}).get("sha256")
                     and bool(HEX64.fullmatch(str(training.get("model_artifact", {}).get("sha256", "")))), "训练与评估checkpoint身份不一致")
        need(hashed_file(path.parent, run.get("predictions")), "每轮预测证据文件必须可校验")
        for side in ("base_metrics", "trained_metrics"):
            scores = run.get(side)
            need(isinstance(scores, dict) and set(metric_names) <= set(scores) and all(finite_number(scores.get(m)) for m in metric_names), "每轮基座与训练模型指标都须为有限数值")
    from .math_task import resolve_task_id
    if resolve_task_id(r) == "math-grpo-v1":
        # 与生产聚合器共用强校验，不能绕过聚合入口直接提交宽松JSON晋级。
        from .suite_task import validate_suite_payload
        try:
            validate_suite_payload(r, data, path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            problems.append(f"数学套件内容拒绝：{exc}")
    if problems:
        return problems
    for name in metric_names:
        values = [run["trained_metrics"][name] for run in runs]
        need(finite_number(expected[name]) and finite_number(tolerance[name]) and tolerance[name] >= 0, "期望值/容差必须为有限数值，容差非负")
        if not problems:
            need(abs(statistics.mean(values) - expected[name]) <= tolerance[name], f"{name} 的重跑均值超出验收容差")
    return problems


def zero_train_measured(r):
    try:
        # 只有实现发布级内容重算器的任务才有资格被计为「已测量零训练基线」；
        # 其余任务不得靠通用 JSON 走到 return True。
        if _task(r) not in ("math-grpo-v1", "code-fim-v1", "xlam-fc-v1",
                            "rapidfire-sft-v1", "granite-linecomparison-v1",
                            "paligemma-grounding-v1", "smolvlm-dpo-v1", "qwen25vl-mpo-v1"):
            return False
        data, path = read_json(r, "baseline.measurement")
        valid = (data is not None and type(data.get("schema_version")) is int and data["schema_version"] == 1
                 and data.get("fingerprint") == fingerprint(r)
                 and not evaluation_problems(r, data, path, role="base"))
        if not valid:
            return False
        if _task(r) in ("code-fim-v1", "xlam-fc-v1", "rapidfire-sft-v1", "granite-linecomparison-v1",
                            "paligemma-grounding-v1", "smolvlm-dpo-v1", "qwen25vl-mpo-v1"):
            from .task_router import load
            load(r, "suite").zero_train_validation(r, data, path)
            return True
        from .math_task import resolve_task_id, execution_identity, MODEL_ID
        if resolve_task_id(r) == "math-grpo-v1":
            from .suite_task import _manifest, _identity, _evaluation_content
            _manifest(r, data["split_manifest"])
            identity = execution_identity(r)
            _identity(r, data, (identity["code_sha256"], identity["environment_lock_sha256"]))
            if data.get("mode") != "zero-shot" or data.get("model_artifact") != {
                    "id": MODEL_ID, "revision": r.get("model.revision"), "role": "base"}:
                return False
            _evaluation_content(r, path.parent.resolve(), path, data, data["split_manifest"])
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        return False
