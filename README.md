# RMUC 2026 决策与时序预测

比赛回放 → 可观测 WorldState → 时序预测 → 分层决策 → 行为树与执行保护。
当前正式入口为 `run_m2.py --profile mainline`，默认跳过 LLM 慢环。

## 仓库内容

- `data/`、`worldstate/`：比赛读取、雷达可见性和状态建模。
- `forecast/`、`runtime/`、`tree/`：时序预测、决策、保护和执行。
- `labels/`、`rl/`、`finetune/`：标注、强化学习、LoRA 训练。
- `eval/`、`tests/`、`scripts/`：评估、测试和验收打包。
- `docs/`：设计与实验记录；`evidence/`：清理前保存的验收报告。
- `visualizations/formal_demo/`：可直接在浏览器中打开的正式回放演示。
- `out_m2/official/`：运行时代码使用的地图、区域和通行约束，保留原路径。

完整背景见 `AGENT_PROJECT_OVERVIEW.md` 和 `CODEX_HANDOFF.md`。
历史文档中的本地已下载状态不代表仓库包含模型。

## 使用

```powershell
python -m pip install -r requirements.txt
python -m pytest tests eval -q --ignore=out_m2 --ignore=third_party
python run_m2.py --help
```

`requirements.txt` 是原项目的基础依赖列表。预测、训练与本地 LLM
还需要对应的 NumPy、PyTorch、LightGBM、XGBoost、Transformers、PEFT 等环境；
具体路线和命令见项目导读及 `docs/temporal_forecast_and_open_jev.md`。

原始比赛库不上传，需自行放入 `数据与资料/rmuc_2026_region_dataset.sqlite`，
路径由 `config.py` 配置。下载的基座模型不上传，可从 Hugging Face 恢复至
`models_hf/`，例如 `Qwen/Qwen2.5-3B-Instruct`、`Qwen/Qwen2.5-1.5B-Instruct`、
`Qwen/Qwen3.5-2B` 和 `ZefanCai/Open-Jev-2B`。
Open-Jev 本地基座使用 revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
项目训练所得权重保留在原机器的 `out_m2/`，不包含在 Git 仓库中；
换机器时需另行复制或按对应训练脚本重训。

可选 Open-Jev 上游及本地修改见 `third_party/README.md`。
API 密钥放在本地 `.env` 或环境变量中，该文件不上传。

## 验收状态

保存的报告是清理前的实验快照。LLM 候选仍有未通过的门禁，
不能据演示页面认定已达到完整部署要求；详见 `evidence/` 中的部署审计和验收记录。
