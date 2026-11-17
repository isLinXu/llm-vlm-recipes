"""受控数学任务；模块导入仅依赖标准库，框架与 verifier 在执行时加载。"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TASK_ID = "math-grpo-v1"
RECIPE_ID = "llm/grpo/qwen2-0-5b-numinamath"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_ID = "AI-MO/NuminaMath-TIR"
TRL_VERSION = "0.14.0"
SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<think>your reasoning</think>\n<answer>your final answer</answer>"
)
SPLIT_RANGES = {"train": (0, 500), "validation": (500, 1000), "test": (1000, 1500)}
TRAIN_KEYS = {
    "learning_rate", "gradient_accumulation_steps", "num_train_epochs",
    "max_completion_length", "num_generations", "remove_unused_columns",
    "seed", "data_seed", "per_device_train_batch_size", "max_steps",
}
FORMAT_RE = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)


class TaskExecutionError(RuntimeError):
    pass


class DatasetValidationError(ValueError):
    pass


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID:
        raise TaskExecutionError(
            f"尚未实现该执行路由：recipe={r.id}, task_id={resolved!r}；"
            f"仅支持 {RECIPE_ID} / {TASK_ID}，其他 recipe 请使用 --dry-run。"
        )
    required = {
        "method.family": "grpo", "method.trainer": "GRPOTrainer",
        "model.base": MODEL_ID, "dataset.name": DATASET_ID,
        "dependencies.trl": TRL_VERSION,
        "rewards.functions": ["format_reward", "accuracy_reward"],
    }
    for key, value in required.items():
        if r.get(key) != value:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={value!r}，不能替换为 {r.get(key)!r}")
    if r.is_vlm or r.get("peft") or r.get("quantization"):
        raise TaskExecutionError("该路由仅实现纯文本全参数 GRPO，不支持 VLM / PEFT / 量化")
    unknown = sorted(set(r.get("train", {}) or {}) - TRAIN_KEYS)
    if unknown:
        raise TaskExecutionError(f"[train] 含未支持参数 {unknown}；拒绝静默忽略")
    if r.get("train.remove_unused_columns") is not False:
        raise TaskExecutionError("train.remove_unused_columns 必须为 false，保留 reward 所需 solution")
    return resolved


def split_spec() -> dict:
    return {
        "source_split": "train", "selection": "连续原始索引，不 shuffle",
        "ranges": {name: {"start": lo, "stop": hi, "count": hi - lo}
                   for name, (lo, hi) in SPLIT_RANGES.items()},
        "overlap_policy": "索引、内容与问题在各 split 之间不得重叠；重复则拒绝执行",
    }


def mode_output_dir(plan, mode: str) -> Path:
    output = Path(plan["run"]["output_dir"])
    if output.name in ("dry-run", "smoke", "full", "eval", "zero-shot"):
        output = output.parent
    return output / mode


def task_plan(r, plan: dict, mode: str, task_id=None) -> dict:
    result = dict(plan)
    resolved = resolve_task_id(r, plan, task_id)
    result["task_id"] = resolved
    try:
        require_supported(r, plan, task_id)
    except TaskExecutionError as exc:
        result["execution_supported"] = False
        result["unsupported_reason"] = str(exc)
    else:
        result["execution_supported"] = True
    if result["execution_supported"]:
        result["dataset"] = {**plan["dataset"], "split": "train[1000:1500] (held-out test)",
                             "split_plan": split_spec()}
        result["run"] = {**plan["run"], "output_dir": str(mode_output_dir(plan, mode))}
        if mode in ("dry-run", "smoke", "full"):
            budget_mode = "smoke" if mode == "dry-run" else mode
            result["run"]["steps"] = r.get("train.max_steps", r.get(f"budget.{budget_mode}_steps"))
            result["run"]["params"] = {
                **plan["run"]["params"], "max_steps": result["run"]["steps"],
                "per_device_train_batch_size": r.get("train.per_device_train_batch_size", 4),
            }
        result["warnings"] = [w for w in plan.get("warnings", []) if "evaluation.expected" not in w]
        result["prompt"] = {"system": SYSTEM_PROMPT}
        result["evaluation"] = {
            "metrics": r.get("evaluation.metrics", ["accuracy", "format_compliance"]), "held_out": True,
            "seed": r.get("evaluation.seed", r.get("train.seed")), "runs": 1, "do_sample": False,
            "protocol": "single_checkpoint_evaluation",
            "training_seed_variance_measured": False,
            "baseline_promotion": "none",
        }
    return result


def make_prompt(problem: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}]


def completion_text(completion) -> str | None:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        if all(isinstance(m, dict) and m.get("role") == "assistant"
               and isinstance(m.get("content"), str) for m in completion):
            return "".join(m["content"] for m in completion)
    return None


def answer_text(completion) -> str | None:
    text = completion_text(completion)
    if not text:
        return None
    match = FORMAT_RE.fullmatch(text.strip())
    if not match or not all(part.strip() for part in match.groups()):
        return None
    if any(tag in part for part in match.groups()
           for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        return None
    return match.group(2).strip()


def format_reward(completions, **kwargs) -> list[float]:
    return [float(answer_text(c) is not None) for c in completions]


def math_backend():
    try:
        module = importlib.import_module("math_verify")
        return module.parse, module.verify
    except (ImportError, AttributeError) as exc:
        raise TaskExecutionError(
            "缺少可用 math_verify：请按 recipe 的 lock 安装 math_verify==0.3.3；"
            "--dry-run 无需此依赖。"
        ) from exc


def parse_gold(solution, parse):
    if not isinstance(solution, str) or not solution.strip():
        raise DatasetValidationError("gold solution 必须为非空字符串")
    try:
        parsed = parse(solution)
        if not parsed:
            raise ValueError("解析为空")
    except Exception as exc:
        raise DatasetValidationError(f"gold solution 无法解析：{exc}") from exc
    return parsed


def score_completion(completion, gold, parse, verify) -> dict:
    answer = answer_text(completion)
    score = {"format_compliance": int(answer is not None), "accuracy": 0,
             "answer": answer, "parse_status": "invalid_format"}
    if answer is None:
        return score
    try:
        parsed = parse(answer)
        if not parsed:
            score["parse_status"] = "empty_prediction_parse"
            return score
        score["accuracy"] = int(bool(verify(gold, parsed)))
        score["parse_status"] = "verified"
    except Exception as exc:
        score["parse_status"] = "prediction_or_verification_error"
        score["error_type"] = type(exc).__name__
    return score


def accuracy_reward(completions, solution, *, parse=None, verify=None, **kwargs) -> list[float]:
    if parse is None or verify is None:
        parse, verify = math_backend()
    if len(completions) != len(solution):
        raise DatasetValidationError("completions 与 solution 数量不一致")
    golds = [parse_gold(s, parse) for s in solution]
    return [float(score_completion(c, g, parse, verify)["accuracy"])
            for c, g in zip(completions, golds)]


def prepare_splits(rows, *, parse=None, verify=None) -> tuple[dict, dict]:
    if parse is None:
        parse, _ = math_backend()
    selected = list(itertools.islice(iter(rows), 1500))
    if len(selected) != 1500:
        raise DatasetValidationError(f"需要 train 前 1500 条，实际仅 {len(selected)} 条")
    splits, entries = {}, {}
    content_owners, problem_owners = {}, {}
    for name, (lo, hi) in SPLIT_RANGES.items():
        splits[name], entries[name] = [], []
        for index in range(lo, hi):
            raw = selected[index]
            if not isinstance(raw, dict):
                raise DatasetValidationError(f"train:{index} 不是结构化对象")
            problem, solution = raw.get("problem"), raw.get("solution")
            if not isinstance(problem, str) or not problem.strip():
                raise DatasetValidationError(f"train:{index} 的 problem 为空或类型错误")
            try:
                parse_gold(solution, parse)
                content_hash = sha256_json(raw)
            except (DatasetValidationError, TypeError, ValueError) as exc:
                raise DatasetValidationError(f"train:{index} 数据无效：{exc}") from exc
            problem_hash = sha256_json(problem)
            for owners, key in ((content_owners, content_hash), (problem_owners, problem_hash)):
                if key in owners and owners[key] != name:
                    raise DatasetValidationError(f"train:{index} 与 {owners[key]} 内容或问题重叠")
                owners[key] = name
            sample_id = f"train:{index}"
            entry = {"sample_id": sample_id, "source_index": index, "content_sha256": content_hash,
                     "problem_sha256": problem_hash, "solution_sha256": sha256_json(solution)}
            entries[name].append(entry)
            splits[name].append({**entry, "prompt": make_prompt(problem), "solution": solution})
    manifest = {
        "task_id": TASK_ID, "dataset_id": DATASET_ID, **split_spec(),
        "splits": {name: {"count": len(items), "samples": items,
                          "sha256": sha256_json(items)} for name, items in entries.items()},
    }
    manifest["sha256"] = sha256_json(manifest)
    return splits, manifest


def require_revision(r, key: str) -> str:
    revision = r.get(key)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise TaskExecutionError(f"{key} 必须锁定为真实 commit SHA；请先在授权环境运行 runner/lock.py --apply")
    return revision


def fixed_seeds(r) -> tuple[int, int]:
    seeds = (r.get("train.seed"), r.get("train.data_seed"))
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise TaskExecutionError("train.seed / train.data_seed 必须为固定的非负整数")
    evaluation_seed = r.get("evaluation.seed")
    if evaluation_seed is not None and (type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**32):
        raise TaskExecutionError("evaluation.seed 必须为固定非负整数，与训练seed独立控制")
    return seeds


def config_kwargs(r, plan: dict, output_dir: Path) -> dict:
    require_supported(r, plan)
    seed, data_seed = fixed_seeds(r)
    params = dict(r.get("train", {}))
    mode = plan["mode"]
    budget = r.get(f"budget.{mode}_steps")
    max_steps = params.get("max_steps", budget)
    if type(budget) is not int or budget <= 0 or type(max_steps) is not int or not 0 < max_steps <= budget:
        raise TaskExecutionError(f"{mode} 预算必须覆盖 max_steps：budget={budget}, max_steps={max_steps}")
    batch = params.get("per_device_train_batch_size", 4)
    generations = params.get("num_generations")
    if type(generations) is not int or generations < 2 or type(batch) is not int or batch < generations or batch % generations:
        raise TaskExecutionError("单 GPU GRPO batch 必须是 num_generations（至少 2）的正整数倍")
    for key in ("gradient_accumulation_steps", "max_completion_length"):
        if type(params.get(key)) is not int or params[key] <= 0:
            raise TaskExecutionError(f"train.{key} 必须为正整数")
    params.update(seed=seed, data_seed=data_seed, max_steps=max_steps,
                  per_device_train_batch_size=batch, output_dir=str(output_dir), report_to="none")
    return params


def check_signature(target, kwargs: dict, label: str) -> None:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise TaskExecutionError(f"无法核验 {label} API signature") from exc
    unsupported = [key for key in kwargs if key not in signature.parameters
                   or signature.parameters[key].kind not in (
                       inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    if unsupported:
        raise TaskExecutionError(f"{label} API 不显式支持参数 {unsupported}；拒绝依赖 **kwargs 静默忽略")
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise TaskExecutionError(f"{label} API 参数不兼容：{exc}") from exc


def load_runtime(*, training=False) -> dict:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise TaskExecutionError("缺少 torch CUDA build；请按 lock 安装依赖，或使用 --dry-run") from exc
    if not torch.cuda.is_available():
        raise TaskExecutionError("没有可用 CUDA GPU；拒绝执行或下载模型，请使用 --dry-run")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TaskExecutionError("受控样板仅支持单进程单 GPU，不支持分布式运行")
    dependencies = {"torch": torch}
    names = ["transformers", "datasets"] + (["trl"] if training else [])
    for name in names:
        try:
            dependencies[name] = importlib.import_module(name)
        except ImportError as exc:
            raise TaskExecutionError(f"缺少 {name} 或其依赖：{exc}；请按 recipe lock 安装环境") from exc
    if training and getattr(dependencies["trl"], "__version__", None) != TRL_VERSION:
        raise TaskExecutionError(f"该路由要求 trl=={TRL_VERSION}，实际为 {getattr(dependencies['trl'], '__version__', None)}")
    parse, verify = math_backend()
    dependencies.update(parse=parse, verify=verify)
    return dependencies


def require_frozen_runtime(r, dependencies) -> None:
    """执行前校验实际安装环境；模板/只解析了输入的锁不可用于实跑。"""
    import importlib.metadata
    import subprocess
    import tomllib
    from .baseline_evidence import frozen_environment
    if not frozen_environment(r):
        raise TaskExecutionError("环境尚未冻结或与输入不一致：先运行 lock.py --capture-environment；模板不能用于实跑")
    lock = tomllib.loads((r.path.parent / r.get("execution.lockfile")).read_text(encoding="utf-8"))
    runtime = lock["runtime"]
    torch = dependencies["torch"]
    if (runtime["python"] != sys.version.split()[0] or runtime["cuda"] != torch.version.cuda
            or runtime["gpu"] != torch.cuda.get_device_name(0)):
        raise TaskExecutionError("实际 Python/CUDA/GPU 与 frozen lock 不一致；不可静默迁移运行环境")
    try:
        for name, expected in (r.get("dependencies", {}) or {}).items():
            if importlib.metadata.version(name) != expected:
                raise TaskExecutionError(f"依赖 {name} 与 recipe 声明版本不一致")
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True).splitlines()
    except (importlib.metadata.PackageNotFoundError, OSError, subprocess.SubprocessError) as exc:
        raise TaskExecutionError(f"无法核对完整运行环境：{exc}") from exc
    if driver != runtime["driver"] or sorted(installed) != sorted(lock["requirements"]):
        raise TaskExecutionError("实际驱动或完整 pip freeze 与 frozen lock 不一致")


def apply_determinism(r, dependencies, *, evaluation=False) -> None:
    seed, _ = fixed_seeds(r)
    if evaluation:
        seed = r.get("evaluation.seed", seed)
    torch = dependencies["torch"]
    dependencies["transformers"].set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    dataset = dependencies["datasets"].load_dataset(DATASET_ID, revision=revision, split="train")
    splits, manifest = prepare_splits(dataset, parse=dependencies["parse"])
    manifest["dataset_revision"] = revision
    manifest.pop("sha256")
    manifest["sha256"] = sha256_json(manifest)
    return splits, manifest


def artifact_manifest(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and (path.suffix in (".json", ".safetensors", ".bin", ".model", ".txt")
                              and path.name not in ("training.json", "evaluation.json", "manifest.json")
                              and "predictions" not in path.name):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return {"path": str(directory.resolve()), "files": files, "sha256": sha256_json(files)}


def checkpoint_path(plan, checkpoint=None, *, recipe=None, require_evidence=True) -> Path:
    path = Path(checkpoint).expanduser().resolve() if checkpoint else mode_output_dir(plan, "full").resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise TaskExecutionError(f"没有训练 checkpoint：{path}；请显式 --checkpoint 本地模型目录，或先完成 --full 训练")
    if not (path / "tokenizer_config.json").is_file() or not any(
        p.is_file() for pattern in ("*.safetensors", "pytorch_model*.bin") for p in path.glob(pattern)
    ):
        raise TaskExecutionError(f"checkpoint 缺少完整模型权重或 tokenizer：{path}；不会回退到 base")
    if require_evidence:
        try:
            evidence = json.loads((path / "training.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TaskExecutionError("checkpoint 缺少已完成 training.json；显式路径也不能绕过训练来源校验") from exc
        if not (isinstance(evidence, dict) and evidence.get("kind") == "training"
                and evidence.get("stage") == "completed" and evidence.get("task_id") == TASK_ID
                and evidence.get("recipe_id") == RECIPE_ID
                and evidence.get("mode") in ("smoke", "full")
                and (checkpoint is not None or evidence.get("mode") == "full")):
            raise TaskExecutionError("checkpoint 没有对应的已完成训练证据")
        if evidence.get("model_artifact", {}).get("sha256") != artifact_manifest(path)["sha256"]:
            raise TaskExecutionError("checkpoint 内容与训练证据不一致")
        if recipe is not None:
            from .baseline_evidence import fingerprint
            if (evidence.get("fingerprint") != fingerprint(recipe)
                    or evidence.get("model_revision") != recipe.get("model.revision")
                    or evidence.get("dataset_revision") != recipe.get("dataset.revision")):
                raise TaskExecutionError("checkpoint 训练配置或 revision 与当前 recipe 不一致")
    return path


def runtime_evidence(dependencies, started: float) -> dict:
    torch = dependencies["torch"]
    versions = {name: getattr(module, "__version__", None)
                for name, module in dependencies.items() if name not in ("parse", "verify")}
    from importlib.metadata import PackageNotFoundError, version
    try:
        versions["math_verify"] = version("math_verify")
    except PackageNotFoundError:
        versions["math_verify"] = None
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "world_size": 1,
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def execution_identity(r) -> dict:
    """真实代码快照与锁字节身份；不接受缺失锁或 symlink。"""
    from .provenance import code_hash, file_hash
    try:
        lock = r.path.parent / r.get("execution.lockfile")
        return {"code_sha256": code_hash(), "environment_lock_sha256": file_hash(lock)}
    except (OSError, TypeError, ValueError) as exc:
        raise TaskExecutionError(f"无法冻结执行代码/环境锁身份：{exc}") from exc


def completed_evidence(r, plan, kind, config, manifest, runtime) -> dict:
    seed, data_seed = fixed_seeds(r)
    from .baseline_evidence import fingerprint
    identity = execution_identity(r)
    if plan.get("execution_identity", identity) != identity:
        raise TaskExecutionError("执行期间代码或环境锁发生漂移；不写completed证据")
    return {
        **identity, "fingerprint": fingerprint(r),
        "schema_version": 1, "kind": kind, "stage": "completed", "task_id": TASK_ID,
        "recipe_id": r.id, "experiment_id": plan.get("experiment_id", ""), "mode": plan["mode"],
        "completed_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "data_seed": data_seed,
        "model_id": MODEL_ID, "model_revision": require_revision(r, "model.revision"),
        "dataset_id": DATASET_ID, "dataset_revision": require_revision(r, "dataset.revision"),
        "recipe_sha256": sha256_json(r.data), "config": config, "config_sha256": sha256_json(config),
        "split_manifest": manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
    }


def save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def run_evaluation(r, plan, *, checkpoint=None, zero_shot=False) -> Path:
    require_supported(r, plan)
    revision = require_revision(r, "model.revision")
    require_revision(r, "dataset.revision")
    fixed_seeds(r)
    if zero_shot and checkpoint is not None:
        raise TaskExecutionError("zero-shot 只能加载 base，不接受 checkpoint")
    source = MODEL_ID if zero_shot else checkpoint_path(plan, checkpoint, recipe=r)
    output = mode_output_dir(plan, "zero-shot" if zero_shot else "eval").resolve()
    if not zero_shot and output == source:
        raise TaskExecutionError("评估输出目录不能覆盖 checkpoint")
    if (output / "evaluation.json").exists():
        raise TaskExecutionError(f"已存在评估证据 {output}，请使用独立实验输出目录，避免覆盖")
    dependencies = load_runtime()
    require_frozen_runtime(r, dependencies)
    plan = {**plan, "execution_identity": execution_identity(r)}
    started = time.perf_counter()
    apply_determinism(r, dependencies, evaluation=True)
    splits, manifest = load_splits(r, dependencies)
    transformers, torch = dependencies["transformers"], dependencies["torch"]
    loader_kwargs = {"revision": revision} if zero_shot else {"local_files_only": True}
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(source), **loader_kwargs)
    if tokenizer.eos_token_id is None:
        raise TaskExecutionError("tokenizer 缺少 eos_token_id")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float32, **loader_kwargs,
    ).to("cuda")
    model.eval()
    max_tokens = r.get("train.max_completion_length")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise TaskExecutionError("train.max_completion_length 必须为正整数")
    generation_config = {
        "seed": r.get("evaluation.seed", fixed_seeds(r)[0]), "do_sample": False, "num_beams": 1,
        "max_new_tokens": max_tokens, "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "use_cache": True,
    }
    predictions = []
    with torch.inference_mode():
        for row in splits["test"]:
            encoded = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            input_length = encoded["input_ids"].shape[-1]
            generated = model.generate(**encoded, **{k: v for k, v in generation_config.items() if k != "seed"})
            completion = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
            gold = parse_gold(row["solution"], dependencies["parse"])
            scored = score_completion(completion, gold, dependencies["parse"], dependencies["verify"])
            predictions.append({
                "sample_id": row["sample_id"], "content_sha256": row["content_sha256"],
                "prompt": row["prompt"], "solution": row["solution"], "completion": completion,
                "input_tokens": input_length, "generated_tokens": len(generated[0]) - input_length,
                **scored,
            })
    torch.cuda.synchronize()
    count = len(predictions)
    correct = sum(p["accuracy"] for p in predictions)
    compliant = sum(p["format_compliance"] for p in predictions)
    model_artifact = ({"id": MODEL_ID, "revision": revision, "role": "base"} if zero_shot
                      else {**artifact_manifest(source), "role": "trained_checkpoint"})
    config = {"task_id": TASK_ID, "system_prompt": SYSTEM_PROMPT, "generation": generation_config,
              "model": model_artifact, "dataset_revision": manifest["dataset_revision"],
              "test_sha256": manifest["splits"]["test"]["sha256"]}
    evidence = completed_evidence(r, plan, "evaluation", config, manifest,
                                  runtime_evidence(dependencies, started))
    evidence.update(
        evaluation_protocol="single_checkpoint_evaluation", model_role=model_artifact["role"],
        model_artifact=model_artifact, sample_count=count,
        metrics={"accuracy": correct / count, "pass@1": correct / count,
                 "format_compliance": compliant / count},
        metric_counts={"correct": correct, "format_compliant": compliant, "total": count},
        predictions={"path": "predictions.json", "sha256": sha256_json(predictions), "count": count},
    )
    prediction_path = save_json(output / "predictions.json", predictions)
    evidence["predictions_file"] = {"path": "predictions.json", "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
    save_json(output / "manifest.json", manifest)
    return save_json(output / "evaluation.json", evidence)
