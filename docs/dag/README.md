# DAG analysis tools — S1–S5

代码在仓库根目录的 `dag/`。这是第一步（随机批量）的工具交付，不包含市场数据、批量表达式或运行结果。因子求值引擎保持不变；规划器尚未接入 C++ 执行。

| 任务 | 工具 | 范围 |
|---|---|---|
| S1 / G1 | `dag_stats.py`、`cpp_probe.cpp` | 原始/规范化树节点、DAG去重、算子/层/根深度分布、最宽层、Python建图时间、C++核对 |
| 串行规模补充 | `dag_runtime.py` | 顶点V、不同依赖边E、操作数引用次数、建图和调度模拟耗时 |
| S2 / G4 | `memory_sim.py` | 保留根、逐层早释放、层内分块、贪心拓扑调度的存活面板峰值 |
| S3 框架 | `cache_sim.py` | LRU、LFU、代价感知及未来访问参照；批序列/JSONL回放 |
| S4 | `run_s4.sh`、`verify_s4.py` | 复现已有引擎的公式子批实测与评分一致性检查 |
| S5 | `plan_contract.py`、`s5/` | 单进程执行计划提案、人工示例、结构检查器 |

## 环境与测试

Python 3.10+、NumPy、pandas；C++核对另需 C++17 的 `g++`（Linux/macOS）。在仓库根目录运行：

```bash
python3 -m pip install numpy pandas
python3 -m unittest discover -s dag -p 'test_*.py' -v
```

17个测试覆盖重复/交换子表达式、全部算子与C++核对、引用计数和根消费、缓存子树剪枝、共享孩子、短固定序列Belady穷举比较、轨迹顺序和计划失效。测试只用人工合成表达式，不依赖私有fixture。

## 输入与输出

输入为交接生成器的嵌套列表JSON，例如 `[["mean","x",5],["rank",["mean","x",5]]]`。S3也支持JSONL的batch/parents/population/gene_pool事件。实际批量与成本表从数据负责人获取，不提交仓库。

下面两个环境变量自行替换为本机/集群的私有数据路径：

```bash
export GP_BATCH_DIR=/path/to/batches
export GP_COSTS=/path/to/op_costs.csv
```

所有 `--out` 放在仓库外，避免把运行输出提交到Git。

### S1：统计及跨语言核对

```bash
python3 dag/dag_stats.py "$GP_BATCH_DIR/d1_seed1.json" --out ../dag_results/d1 --check-cpp
python3 dag/dag_stats.py "$GP_BATCH_DIR/d2_seed1.json" --out ../dag_results/d2 --check-cpp
python3 dag/dag_stats.py "$GP_BATCH_DIR/d3_pool1_seed1.json" --out ../dag_results/d3 --check-cpp
python3 dag/dag_stats.py "$GP_BATCH_DIR/d3_pool1_warm30k.json" --out ../dag_results/warm --check-cpp
```

默认取1k/5k/10k/30k中存在的前缀并包含全批量，重复3次。窗口整数不算节点，叶子算节点；规范化前/后树总量分开保存。节点减少比例不是加速比。Python/C++内部ID不同，核对稳定子表达式集合、根顺序和节点层/树节点数/引用数。

```bash
python3 dag/dag_runtime.py "$GP_BATCH_DIR/d3_pool1_warm30k.json" --out ../dag_results/runtime
```

计时预热1次、测3次。E只计不同孩子-父节点对；`a*a`有一条不同依赖边、两次操作数引用。Python建图计时包含验证/规范化/邻接表，不是C++引擎计时；调度计时包含逻辑模拟检查，不含算子执行。不同机器数据不要拼成一条性能曲线。

### S2：调度与内存

```bash
python3 dag/memory_sim.py "$GP_BATCH_DIR/d3_pool1_seed1.json" --out ../dag_results/memory_d3
python3 dag/memory_sim.py "$GP_BATCH_DIR/d3_pool1_warm30k.json" --out ../dag_results/memory_warm
```

输出CSV/JSON。`peak_live_panels`为逻辑存活峰值，默认按1461天×5876股票×float32换算GiB。`--ranks P`只换算ceil(T/P)自有日期行，不含halo或真实MPI执行。`operator_groups_proxy`是执行组数，不是实测同步/消息次数。

不包含内存池保留的空闲缓冲、算子临时数组、标签、PCA和通信工作区，不能当RSS。根如果还是别人的输入不能提前删。贪心按需加载叶子且一节点一组，低内存不代表并行运行更快。交接MPI版本默认已早处理根，历史保留根策略只作对照。

### S3：跨批持久缓存

```bash
python3 dag/cache_sim.py \
  "$GP_BATCH_DIR/d3_pool1_seed1.json" "$GP_BATCH_DIR/d3_pool1_seed2.json" "$GP_BATCH_DIR/d3_pool1_seed3.json" \
  --costs "$GP_COSTS" --budgets-gib 0,1,4,16 --out ../dag_results/cache
```

- `fixed_trace`：所有算子按固定顺序请求。Belady在固定序列、等大小条目、强制插入的约定下作未加权miss参照，不保证加权成本最优。
- `dag_pruned`：批初缓存快照剪掉命中子树，仍被其他根需要的孩子继续计算。批边界按拓扑顺序更新缓存；临时工作集不计入持久预算。此时使用未来批量成员的 `future_batch_reference` 不是最优上限。
- LRU按最近访问；LFU按本次驻留的访问频次；代价感知按子树重算成本×频次。当前所有新条目强制插入，不缓存叶子。三批都是随机提案，不能称为真实连续进化三代。
- `saved_cost_seconds`是成本表的加权模型值，不是实测节省时间。默认成本列 `ns_per_cell_1thread`；rank按std、zscore按mean、delta按lag估计。忽略IO、缓存维护和通信。
- 第1批冷启动应为0；终端汇总三批成本及末批命中率，逐批值在CSV/JSON。命中与剪枝不能重复计价。
- JSONL可作为输入，保留种群/基因池事件钩子，但相关策略留待第二步实现。只按显式相同event_id去重；不按generation/phase擅自合并。
- 默认P=1、同一数据版本/日期/股票/算子语义。P>1须显式 `--ideal-shards`，仅做容量敏感性，halo校验和真实分布式缓存尚未实现。

### S4：引擎的小样本公式分块

此复现实验用的是私下交接版本的 `gpe_export`。上传时核对到目标仓库main尚无该驱动和构建目标；本次不补入或覆盖引擎代码。先使用交接版本在Vanda **计算节点**编译出的可执行文件，再指定其路径：

```bash
GPE_EXPORT=/path/to/handoff/build_s4/gpe_export \
  bash dag/run_s4.sh /path/to/sample_panels "$GP_BATCH_DIR/d3_pool1_seed1.json" ../s4_runs
```

脚本使用已有 `gpe_export`，200条公式、chunk=16/64/200各3次，记录wall time、RSS、CPU时间和退出码并核对全部评分。`--no-R`关闭矩阵写出，但原驱动仍做签名行预处理。公式子批各自建DAG，与S2同一DAG层内分块不同；此项只验证实际分块内存趋势和评分一致性，不能声称精确验证所有模拟策略。

### S5：计划格式和检查

见 [接口说明](S5_interface_zh.txt)。示例是人工表达式，数据标识仅作演示：

```bash
python3 dag/plan_contract.py check dag/s5/example_plan.json
python3 dag/plan_contract.py export "$GP_BATCH_DIR/d3_pool1_seed1.json" --context /path/to/actual_context.json --out ../dag_results/plan.json --chunk 256
```

导出的计划含输入表达式，不入库。自动导出目前仅无缓存计划；结构检查器支持cache操作测试。它不执行数值，不校验缓存字节预算，也不实现MPI。接口待引擎负责人确认，实际C++执行接入留待后续。

## 交付边界

已交付随机批量第一步的工具和接口初稿；未交付真实轨迹结论、种群/基因池缓存策略、C++计划执行器或新的MPI/CUDA实现。汇总数据和图表另行交付，不随代码提交。
