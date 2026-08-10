# 需求前情提要验证说明

## 结论

前情提要可以作为性能验收的比较框架，但不能把其中全部内容当作已证明事实。

- “变长相对定长吞吐下降 <15%”与“吞吐保持率约 85%”在数学上是同一口径，逻辑一致。
- “SRF 平均 TTFT 收益约 10%”必须用包含长、短请求且产生服务端 waiting queue 的负载验证，不能用全 64K 长请求验证。
- `O(n²)` 推导只适合作为需求规定的理论口径；当前 DeepSeek V4 Flash 的实际 attention 路径是稀疏/滑窗/压缩实现，不能仅凭序列长度断言实际 FLOPs 严格按 `n²` 增长。

## O(n²) 理论口径

若强制采用需求假设，则变长集相对同均值定长集的理论计算量比例为：

```text
E[L²] / E[L]² = 1 + Var(L) / E[L]²
```

因此增幅不超过 11% 等价于变异系数 `std(L)/mean(L) <= sqrt(0.11)`。对均值 65,536 而言，标准差应不超过约 21,733 tokens。

当前生成器的最小真实比例样本已验证：

- min = 40,960
- max = 81,920
- mean = 65,536
- `E[L²]/E[L]² - 1 = 5.275%`

它满足需求的 `<=11%` 理论约束。正式数据集会将实际值写入 `datasets/dataset_manifest.json`。

## 为什么不能证明当前模型实际为 O(n²)

当前模型配置包含：

- `sliding_window = 128`
- `index_topk = 512`
- `compress_ratios` 包含 4 和 128

当前固定 vLLM-Ascend 源码还明确走 `AscendDeepseekSparseAttention`/DSA 路径，并在 prefill metadata 中使用滑窗、压缩 KV 和 `sparse_count=index_topk`：

```text
/home/w00985415/proj_260805/vllm-ascend-804317471/
  vllm_ascend/models/deepseek_v4.py
  vllm_ascend/attention/dsa_v1.py
```

所以验收报告中的 `O(n²) compute increase` 是“按需求假设计算的长度分布指标”，不是算子真实 FLOPs 测量。若验收方要求证明真实计算复杂度，应追加算子 profiling/FLOPs 或按 prefill 执行时间拟合长度曲线。

## 四项实测映射

| 前情/目标 | 实测公式 | 对照配置 |
| --- | --- | --- |
| CPP 吞吐保持约 85% | `cpp_only.variable_long / cpp_only.fixed_long` | CPP=true, SRF=false |
| SRF 短请求 TTFT 收益约 10% | `1 - cpp_srf.short_TTFT / cpp_only.short_TTFT` | CPP 固定为 true，只切换 SRF |
| 理论计算量增幅 <=11% | `mean(L²)/65536² - 1` | 数据集 manifest |
| 叠加后吞吐下降 <15% | `1 - cpp_srf.variable_long / cpp_srf.fixed_long` | CPP=true, SRF=true |

另外报告 `srf_only` 相对 `baseline` 的 TTFT 收益，用于核对“SRF 单特性约 10%”这一历史前提。

## 仍需需求方确认的歧义

“约 85%”与“约 10%”是背景经验值还是硬验收线，截图没有明确说明。脚本暂按 `>=85%` 和 `>=10%` 输出 provisional PASS/FAIL；归档正式验收结论前，应让需求责任人确认。
