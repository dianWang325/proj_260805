# DeepSeek V4 CPP + SRF 对比矩阵测试

## 目录与脚本

脚本安装目录：

```text
/home/w00985415/proj_260805/scripts/benchmark_matrix/
```

- `run_dsv4_feature_matrix.sh`：依次启动四种特性组合并执行测试。
- `run_serving_workloads.sh`：对一个已经健康的服务执行 short、long、mixed 负载。
- `analyze_serving_results.py`：生成逐轮 CSV、汇总 CSV 和 Markdown 表格。

四种组合保持 TP=4、PP=2、同步调度、模型、并发配置完全一致：

| variant | CPP | SRF |
| --- | --- | --- |
| baseline | false | false |
| cpp_only | true | false |
| srf_only | false | true |
| cpp_srf | true | true |

## 测试前检查

矩阵脚本必须独占端口 8013 和 NPU 0-7。先在原服务终端按 `Ctrl+C`，等待 Worker 退出，然后执行：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8013/health
npu-smi info
```

正确状态：curl 连接失败，NPU 0-7 没有运行进程。不要使用 `pkill` 清理其他用户任务。

检查脚本语法：

```bash
cd /home/w00985415/proj_260805/scripts/benchmark_matrix

bash -n run_serving_workloads.sh
bash -n run_dsv4_feature_matrix.sh
python analyze_serving_results.py --help
```

## 第一阶段：smoke 矩阵

先运行小规模矩阵，验证四种服务均可启动、请求均成功、进程可以自动回收：

```bash
cd /home/w00985415/proj_260805/scripts/benchmark_matrix

MATRIX_MODE=smoke \
bash run_dsv4_feature_matrix.sh 2>&1 | \
tee /home/w00985415/proj_260805/artifacts/benchmark_matrix_smoke.log
```

smoke 默认值：每种组合执行 short=20、long=8、mixed=20，每个负载一轮。脚本结束后不会保留服务进程。

只想快速验证 baseline 与 cpp_srf 的 mixed 负载时：

```bash
MATRIX_MODE=smoke \
VARIANTS=baseline,cpp_srf \
WORKLOADS=mixed \
MIXED_PROMPTS=20 \
bash run_dsv4_feature_matrix.sh
```

## 第二阶段：formal 矩阵

smoke 全部成功后再执行正式矩阵：

```bash
MATRIX_MODE=formal \
bash run_dsv4_feature_matrix.sh 2>&1 | \
tee /home/w00985415/proj_260805/artifacts/benchmark_matrix_formal.log
```

formal 默认值：

- short：输入 1024、输出 128、100 请求、并发 16。
- long：输入 16384、输出 128、40 请求、并发 8。
- mixed：输入约 1024-7168、输出 128、100 请求、并发 16。
- 每个负载 4 个 warmup，并重复 3 轮。
- 固定随机种子 0，保证四种组合使用相同的随机数据。

正式矩阵可能运行数小时，建议在稳定终端或 `tmux` 中执行。按 `Ctrl+C` 时脚本只停止自己创建的服务进程组。

可以通过环境变量缩小范围：

```bash
MATRIX_MODE=formal \
VARIANTS=baseline,cpp_srf \
WORKLOADS=mixed \
REPEATS=3 \
MIXED_PROMPTS=100 \
bash run_dsv4_feature_matrix.sh
```

## 结果目录

每次矩阵生成独立目录：

```text
/home/w00985415/proj_260805/artifacts/benchmark_matrix/matrix_YYYYMMDD_HHMMSS/
├── matrix-config.txt
├── baseline/
│   ├── service-launch.log
│   ├── service/
│   └── results/
├── cpp_only/
├── srf_only/
├── cpp_srf/
├── matrix_per_run.csv
├── matrix_summary.csv
└── matrix_summary.md
```

查看最新结果：

```bash
ls -lt /home/w00985415/proj_260805/artifacts/benchmark_matrix
```

查看汇总表：

```bash
LATEST_MATRIX="$(ls -dt /home/w00985415/proj_260805/artifacts/benchmark_matrix/matrix_* | head -n 1)"
cat "${LATEST_MATRIX}/matrix_summary.md"
```

分析器按 vLLM 官方公式计算：

- `E2E = TTFT + sum(ITL)`
- `TPOT = (E2E - TTFT) / (output_tokens - 1)`
- TTFT、TPOT、E2E 输出 mean、P50、P90、P95、P99。
- mixed 负载额外按输入 token 数 `<=4096` 和 `>4096` 分成 short/long 两组。

如需重新分析某个目录：

```bash
python /home/w00985415/proj_260805/scripts/benchmark_matrix/analyze_serving_results.py \
    "${LATEST_MATRIX}" \
    --threshold 4096
```

## 结果判定

每个 JSON 的 `failed` 应为 0。重点比较：

1. `cpp_srf` 对比 `cpp_only`：mixed/short 的 TTFT 应降低。
2. `cpp_srf` 对比 `srf_only`：long TTFT、E2E 或总吞吐用于判断 CPP 收益。
3. `cpp_srf` 对比 baseline：观察总体收益和长请求代价。
4. TPOT 通常比 TTFT 更稳定；TPOT 明显变差时应检查批处理和通信。
5. 查看服务日志中的 `ShortRequestFirst stats`、`Profiling completed successfully` 和 `Preserving startup target latency` 作为功能生效证据。

不要直接比较不同请求数、并发度、随机种子或不同 NPU 占用条件下的结果。
