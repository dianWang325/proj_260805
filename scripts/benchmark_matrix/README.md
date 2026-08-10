# DeepSeek V4 CPP + SRF 长序列对比矩阵

脚本目录：

```text
/home/w00985415/proj_260805/scripts/benchmark_matrix/
```

## 数据与判据

脚本按二进制 K 生成并重新分词校验：

- `fixed_long`：输入 65,536 tokens，输出 2,560 tokens。
- `variable_long`：输入 40,960–81,920 tokens，平均 65,536 tokens，输出 2,560 tokens；采用有界近似正态样本，固定随机种子。
- `srf_mixed`：1,024 与 65,536 输入交替，输出 128；用于让请求数超过服务端 `max-num-seqs=4`，观察 SRF 对短请求 TTFT 的影响。它不是需求中的长序列吞吐数据集。

模型配置的 `max_position_embeddings` 是 1,048,576，且使用 YaRN 扩展；最坏请求需要 81,920+2,560=84,480 tokens。矩阵统一使用 `MAX_MODEL_LEN=90112`，旧值 65,536 不足以完成本测试。

自动报告使用以下判据：

1. CPP 前提：`cpp_only.variable_long / cpp_only.fixed_long` 输入吞吐保持率约 85%。
2. SRF 前提：开启 CPP 时，短请求平均 TTFT 收益约 10%；同时报告不启用 CPP 时的 SRF 收益。
3. 理论计算量：`mean(input_len²) / 65536² - 1 <= 11%`。
4. 叠加验收：`1 - cpp_srf.variable_long / cpp_srf.fixed_long < 15%`。
5. 所有请求 `failed=0`。

其中“约 85%”和“约 10%”是需求前情提要中的近似值，脚本暂按 85%/10% 给出 PASS/FAIL；正式归档前应请需求责任人确认它们是否属于硬门槛。`<=11%` 和 `<15%` 是明确判据。

## 文件说明

- `generate_long_sequence_datasets.py`：生成精确 token 长度的 JSONL，并写入 `dataset_manifest.json`。
- `generate_long_sequence_datasets_impl.py`：生成器主体，由前一个入口调用。
- `run_serving_workloads.sh`：对已启动服务运行三个工作负载。
- `run_dsv4_feature_matrix.sh`：轮换 baseline、CPP、SRF、CPP+SRF 四种服务。
- `analyze_serving_results.py`：汇总 TTFT、TPOT、E2E、输入吞吐，并生成需求判定。

四种配置如下，其他服务参数保持一致：

| variant | CPP | SRF |
| --- | --- | --- |
| baseline | false | false |
| cpp_only | true | false |
| srf_only | false | true |
| cpp_srf | true | true |

## 运行前检查

矩阵脚本需要独占 8013 端口和 NPU 0–7。先在当前服务终端按 `Ctrl+C`，等待 worker 全部退出，再检查：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8013/health
npu-smi info
```

预期：curl 连接失败，0–7 卡没有运行进程。不要用 `pkill` 清理其他用户进程。

检查脚本：

```bash
cd /home/w00985415/proj_260805/scripts/benchmark_matrix

bash -n run_serving_workloads.sh
bash -n run_dsv4_feature_matrix.sh
python -m py_compile generate_long_sequence_datasets.py \
  generate_long_sequence_datasets_impl.py analyze_serving_results.py
```

## 第一阶段：最小 smoke

先只测 `cpp_srf`，确认 90,112 上下文、64K/80K prefill 和 2.5K decode 能完成：

```bash
cd /home/w00985415/proj_260805/scripts/benchmark_matrix

MATRIX_MODE=smoke \
VARIANTS=cpp_srf \
WORKLOADS=fixed_long,variable_long \
bash run_dsv4_feature_matrix.sh 2>&1 | \
tee /home/w00985415/proj_260805/artifacts/benchmark_matrix_long_smoke.log
```

smoke 默认每种配置执行：定长 2 请求、变长 4 请求、SRF 混合 8 请求；长序列输出仍为 2,560，因而并不是秒级测试。只运行一个 variant 时，需求报告中的跨 variant 项会显示 `N/A`，属于预期。

若上述成功，再跑四种配置的完整 smoke：

```bash
MATRIX_MODE=smoke \
bash run_dsv4_feature_matrix.sh 2>&1 | \
tee /home/w00985415/proj_260805/artifacts/benchmark_matrix_long_smoke_all.log
```

## 第二阶段：正式矩阵

建议在 `tmux` 中运行：

```bash
cd /home/w00985415/proj_260805/scripts/benchmark_matrix

MATRIX_MODE=formal \
bash run_dsv4_feature_matrix.sh 2>&1 | \
tee /home/w00985415/proj_260805/artifacts/benchmark_matrix_long_formal.log
```

formal 默认值：

- 固定长序列 12 请求、并发 4。
- 变长序列 24 请求、并发 4。
- SRF 混合 32 请求、客户端并发 16。
- warmup 1，每个 workload 重复 3 轮。
- 四个 variant 依次启动，脚本只回收自己创建的服务进程组。

这是昂贵测试。应先根据 smoke 单轮耗时估算总卡时，再决定是否增加样本。若验收方要求更强的正态分布统计可信度，可显式增加变长样本，例如：

```bash
MATRIX_MODE=formal \
VARIABLE_PROMPTS=64 \
FIXED_PROMPTS=32 \
SRF_MIXED_PROMPTS=64 \
bash run_dsv4_feature_matrix.sh
```

不同 variant 必须使用完全相同的请求数、并发度、随机种子、模型与 NPU 环境。

## 结果位置与查看方式

每次运行创建独立目录：

```text
/home/w00985415/proj_260805/artifacts/benchmark_matrix/matrix_YYYYMMDD_HHMMSS/
├── matrix-config.txt
├── datasets/
│   ├── dataset_manifest.json
│   ├── fixed_64k.jsonl
│   ├── variable_40k_80k.jsonl
│   └── srf_mixed.jsonl
├── baseline/ ...
├── cpp_only/ ...
├── srf_only/ ...
├── cpp_srf/ ...
├── matrix_per_run.csv
├── matrix_summary.csv
├── matrix_summary.md
├── acceptance_report.csv
└── acceptance_report.md
```

查看最新结果：

```bash
LATEST_MATRIX="$(ls -dt /home/w00985415/proj_260805/artifacts/benchmark_matrix/matrix_* | head -n 1)"
cat "${LATEST_MATRIX}/datasets/dataset_manifest.json"
cat "${LATEST_MATRIX}/matrix_summary.md"
cat "${LATEST_MATRIX}/acceptance_report.md"
```

分析器按逐请求明细计算：

- `E2E = TTFT + sum(ITL)`
- `TPOT = (E2E - TTFT) / (output_tokens - 1)`
- `input throughput = sum(input_tokens) / benchmark duration`
- 单卡输入吞吐 = 总输入吞吐 / 8；验收比例使用总吞吐，除以 8 不改变比例。
- TTFT、TPOT、E2E 均包含 mean、P50、P90、P95、P99。

日志还应看到 CPP profiling 成功、SRF waiting queue 生效以及 startup target latency 被保留。性能 PASS 不能替代这些功能证据。

旧版 16K/128 的脚本已备份到：

```text
/home/w00985415/proj_260805/scripts/benchmark_matrix/legacy_short_workloads_20260810/
```
