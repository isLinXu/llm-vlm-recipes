# AGENTS.md — recipe-lab 交接约定

给编码 Agent 的短入口。完整人类可读现状见 `docs/AGENT-HANDOFF.md`，机器可读清单见
`docs/AGENT-HANDOFF.json`。契约版本由 `registry/agent_contract.toml` 决定；不要凭记忆改计数。

## 先做什么

1. 确认仓库根有 `registry/taxonomy.toml`，所有命令在仓库根运行。
2. 读 `AGENT-HANDOFF.json`，再读 `AGENT-HANDOFF.md`；二者由同一份契约生成。
3. 环境：Python ≥ 3.11；核心审计零第三方依赖。未安装包时用
   `PYTHONPATH=src python3 -m recipe_lab ...`。

## 事实源与生成物

- 只修改事实源：`recipes/**/recipe.toml`、`registry/`、`src/recipe_lab/`、`tests/`。
- 不要手改 `docs/MATRIX.md`、`docs/REPRODUCIBILITY.md`、`docs/BASELINE-READINESS.md`、
  `docs/METHODOLOGY.md`、`docs/MLLM-POST-TRAINING.md`、`docs/AGENT-HANDOFF.md/json`。
- 修改后运行 `make report`，再运行 `recipe-lab handoff --check`；生成物必须能由当前源码确定性重建。

## 标准验证顺序

```bash
make ci
make report
recipe-lab handoff --check
git diff --exit-code -- docs/MATRIX.md docs/REPRODUCIBILITY.md docs/BASELINE-READINESS.md docs/METHODOLOGY.md docs/MLLM-POST-TRAINING.md docs/AGENT-HANDOFF.md docs/AGENT-HANDOFF.json
```

`make ci` 中的 `--check` 是登记完整性；`--min-*` 提升目标必须继续失败关闭，
不能把目标改成当前值来制造绿灯。当前 GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14。

## 硬边界

- 静态 JSON、hash、离线测试和 dry-run 只证明内部一致性，不认证 GPU 执行，不解锁 B2/B3。
- `blocked_zero_train` 与 `contaminated_pretraining` 是原理性阻塞，不能用证据或论文背书绕过。
- MLLM 五方向只以 preprints 202607.1494 摘要级框架为原始出处；arXiv 2503.06072 和
  `Awesome-MLLM-Tuning` 只作交叉印证 / 方法索引。
- 不要自行创建远端、push 或对外发布；只有用户明确要求时执行。
- 不要在 macOS ARM/无 torch 环境用 mock 结果冒充 CUDA 实跑或基线验收。
