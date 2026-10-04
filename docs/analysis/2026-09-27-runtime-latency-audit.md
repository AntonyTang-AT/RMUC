# 运行延迟小样本审计

日期：2026-09-27。只做诊断，没有改生产代码或默认，也没有访问 test。

## 单次预测调用

用 validation 的同一条观测（game `1779613180708`，红方，`t=1s`，14 个实体）和冻结的 canonical-side checkpoint。每个设备 warmup 8 次后计时 30 次；CUDA 逐次同步，得到端到端墙钟延迟。

| 设备 | `predict` p50 / p95 | 特征预处理均值 | 模型 forward 均值 | 其余窗口/张量/解码均值 |
|---|---:|---:|---:|---:|
| CPU | 25.16 / 26.63 ms | 0.73 ms | 4.26 ms | 19.87 ms |
| RTX 5060 Laptop GPU | 25.54 / 27.72 ms | 0.80 ms | 1.99 ms | 24.73 ms |

模型和 runtime 首次加载分别约 42ms / 52ms。GPU forward 更快，但完整单次调用没有变快；小型模型下传输、Python 解码和其余 CPU 后处理已盖过这点收益。测后显存 allocated 约 16.7MB、reserved 48MB、可用约 6.96GB。

本地规则式自然语言组装 100 次，p50 0.070ms、p95 0.086ms。它不调用远端 LLM，因此若可视化中的文字来自该函数，文字生成不构成延迟来源；远程模型请求不在此测量内。

## 回放生成

对数据库首场比赛只构建到 `t=15s` 的预览前缀，生成 15 帧、调用预测 10 次。`build_payload` 一次耗时 3.57s；预测调用合计 2.68s，但这个总和被红方 `t=1s` 的单个 2.44s 首次调用异常值主导。全部调用 p50=24.05ms，p95=1,365ms；分开首次调用后，8 次 warm calls 的 p50/p95=23.92/24.18ms。红方第一次调用 2,441ms，蓝方第一次调用 50.85ms。当前证据只能把它称为一次冷首调用异常，不能确认根因或把它归因于 forward；其余数据准备/帧组装约 0.89s。对 2.21MB payload 做 JSON 序列化和 gzip 共约 0.037s，压缩后约 0.20MB。此构建过程仍会为所选比赛准备完整的例子集合，`max_time` 只限制输出预览前缀。没有写 HTML 文件。

所以稳态单次调用约 24ms，GPU/CPU forward 以外的部分主导 warm runtime；一次回放生成会调用多次预测，并有冷首调用异常。尚不能仅据此断定用户看到的播放卡顿来自模型：浏览器绘制、首次页面加载、内存占用及全场回放生成未测。

## 建议

优先复用已计算的每 3 秒 forecast，不要在每个 1 秒显示帧重新推理；页面播放帧间隔调整只应影响显示/定时器。若全场 HTML 生成等待时间明显，应单独优化数据准备和减小嵌入 payload；如果页面加载后播放不顺，再测浏览器每帧 JS 绘制和 DOM 更新。GPU 暂不值得作为默认提速方案，实测完整单次调用和 CPU 接近。

## 再测一次首段构建（2026-09-27）

使用当前工作树和同一首场比赛重新构建 15 秒前缀：15 帧 payload 用时 3.67 秒，JSON 序列化与 gzip 合计 0.037 秒。10 次预测调用中 8 次 warm 调用的 p50 为 23.79ms；每侧首次调用分别为 2.67 秒和 44.61ms。该冷首调用尖峰再次出现，但仍无法由这次计时确定具体根因。压缩后的 payload 为 203KB，当前数据量不是这一段构建等待的主要部分。完整记录在 `out_m2/latency_profile/replay_bounded_now.json`。

这说明稳态预测足够快，短前缀等待主要发生在 payload / 预测准备阶段并受单侧首次调用拖慢。浏览器逐帧绘制尚未测量，因此“页面播放慢”仍不能归咎于模型或本次构建。下一步应分别测冷启动前后的帧构建和浏览器 `draw()`，再决定是否做按侧预热或渐进式生成；不应仅依据单次峰值就改生产默认。

## 全场短文案接线及重复数据裁剪（2026-09-27）

把三段全局摘要接入真实回放后，首版完整文件达到 13.66MB。检查发现全局证据详情中重复保存了已在单兵表展示的 `unit_analysis`。移除这部分重复内容，保留全局现状依据、各时距概率和建议条件后，重新生成同一场 419 帧回放：耗时 15.3 秒，最终 HTML 6,639,393 bytes，内嵌 gzip 4,959,071 bytes；原首版对应为 13,659,681 / 10,224,286 bytes。HTML 体积减少约 51.4%。全场实测数据在 `out_m2/latency_profile/replay_bounded_now.json` 中是 15 秒前缀探针；419 帧全场数据由生成器 stdout 和文件大小核实，未写入额外全场 profile 文件。

本次改动没有测浏览器首次解压、绘制或交互播放 FPS；文件缩小可减轻传输和解析压力，但不等于证明播放更顺畅。完整页面位于用户可视化输出路径 `rm-battle-replay.html`。

## 回放生成复用快照（2026-09-27）

进一步检查发现，`build_game_forecast_examples` 已经从整场 timeseries 构造红蓝双方雷达快照来生成因果样本；随后 `build_payload` 又重新查询同一场比赛并完整构造一遍相同快照，用于逐秒回放帧。现改为让前者可选返回其已算好的快照供 payload builder 复用。默认调用仍只返回原有 examples；模型、标签、预测、决策节奏和帧内容契约不变。

### 数据准备微基准

首场比赛 6,704 条 timeseries，交替运行 5 次；旧路径为 examples + 再查数据库并构造 snapshots，新路径为 examples + 返回已构造 snapshots。examples 和 snapshot 对象均逐项相等。

| 路径 | 5 次中位数 | 节省 |
|---|---:|---:|
| 重复构造 | 0.558 s | — |
| 复用现有快照 | 0.398 s | 0.160 s（28.7%）|

### 回放 payload A/B

60 秒完整前缀（60 帧、20 个 3 秒决策点）重复两组。第一组受到整进程冷启动影响，故不把两组均值当作稳定的速度结论。第二组热态旧路径 3.065 s，复用路径 2.762 s，约少 0.30 s（9.9%）；只有一个热态配对，收益范围仍不确定。四次构建的预测、实际目标、实体输出和评分字段摘要 hash 完全一致；完整 payload hash 不同，因此不声称整个 payload 逐字节相同，差异来源尚未单独定位。

完整 420 秒比赛另重复两组：旧路径 13.20 / 11.81 s，复用路径 11.15 / 12.74 s。方向不一致，平均仅少 0.56 s（约 4.5%），所以全局端到端提速没有被证明。15 秒构建的 pre/post 也分别为 4.53 / 4.60 s，首个红方预测单次分别达到 3.19 / 3.45 s 的冷调用异常，压过了快照准备节省。warm predict p50 为 28.24 / 28.53 ms，没有改善。

另对 60 秒路径测了 40 次预测（20 个时刻 × 两侧）：预测总计 4.56s，warm p50 30.74ms，p95 34.84ms；首个红方调用 3.23s、蓝方 73.75ms。由于该前缀尚未达到 120 秒，首调用尖峰出现在任何预测目标可核验之前，因此这次尖峰不是由已成熟未来反馈导致；具体是模型/框架惰性初始化还是其他单次开销，仍未证实。报告见 `out_m2/latency_profile/replay_60s_prediction_profile.json`。

结论仅限**离线 payload 生成**：安全地消除了重复数据准备，但还不能宣称系统实时预测或完整页面播放变快。慢的体验需区分单次预测、回放构建等待和浏览器逐帧绘制；本次未做浏览器绘制基准，也没有证据归因用户看到的播放卡顿。整局 A/B、60 秒前缀 A/B 与逐次预测结果在 `out_m2/latency_profile/snapshot_reuse_ab_full_match.json`、`out_m2/latency_profile/snapshot_reuse_ab_60s.json`、`out_m2/latency_profile/replay_60s_prediction_profile.json`，数据准备微基准为当轮脚本 stdout 记录。目标测试 `tests/test_forecast_dataset.py tests/test_radar_visibility_replay.py`：10 passed。

## 419 帧回放字段精简（2026-09-27）

整段 HTML/JS 只读取单兵建议的 `applicable`、`action`、`action_label`、`display_summary`、触发/停止条件、建议理由、首步/目标区、评分指标和轻量 `guidance_display`。构建时仍用完整 `decision_guidance` 生成自然语言和证据；只在最后生成浏览器 payload 时，精简每秒重复的观测字段。保留用户可见建议、触发与停止条件、可行性/执行/后果/收益因果文字、路线及生命/攻击/资源状态；保留评估展开所需的指标名、目标确认状态和评分分数。删去仅供生成过程使用、页面不读取的原始 evidence 列表、重复未来文案、诊断字段和评分子树明细。阵亡/攻击/位置等其它单位字段未改。

同一 train 比赛 `1778631047459`、419 帧、140 个决策的修改前后体积：

| 项目 | 修改前 | 修改后 | 减少 |
|---|---:|---:|---:|
| 解压后 JSON payload | 78,610,181 B | 60,988,265 B | 17,621,916 B（22.4%）|
| 内嵌 gzip 数据 | 5,823,159 B | 3,959,536 B | 1,863,623 B（32.0%）|
| HTML 文件 | 7,790,809 B | 5,305,981 B | 2,484,828 B（31.9%）|

测试：`tests/test_battle_replay_visual.py tests/test_death_lifecycle.py tests/test_entity_forecast.py`，28 passed。重建后的 payload 仍为 419 帧/140 决策；解压检查确认保留了 2,357 个阵亡实体帧记录及 11,732 条单兵建议记录。此改动证明数据体积降低，不等于已测出浏览器播放帧率；播放性能仍需在实际浏览器单独计时。

## 复现

```powershell
python eval\profile_temporal_inference_latency.py --device auto --warmup 8 --samples 30 --narrative-repeats 100
python eval\profile_battle_replay_build_latency.py --max-time 15
```

第一条输出 `out_m2/latency_profile/validation_runtime_latency.json`。第二条会在 `out_m2/latency_profile/bounded_replay_build_latency.json` 保存逐调用记录，并区分两方各自第一次调用和后续 warm calls；不写 HTML，也不修改数据库。GPU 时间包含逐次同步以测墙钟，和异步纯 GPU kernel benchmark 不同。
