# GP Factor Miner

遗传编程因子挖掘项目。当前只运行层级 PCA-QD 的 V4；原始数据从
`/hy-tmp/autodl-tmp/EOD_Columns/eod` 读取，默认标签为
`/hy-tmp/data/factor/label_10.pkl`。

## 数据边界

- 2016-01-01 至 2021-12-31：GP 进化；
- 2022-01-01 至 2022-12-31：样本外筛选；
- 其他日期：本轮保留但不读取、不参与进化或筛选。

## 安装与测试

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
```

## EOD 输入

`eod` 是 Pickle 字典，每个键对应一张 `交易日 × 股票` 的 DataFrame。加载器会：

- 去掉标签股票代码的 `.SZ/.SH/.BJ` 后缀并和 EOD 精确取交集；
- 用 `trade_status` 和 `ST` 屏蔽不可交易、ST 样本；
- 排除 `trade_status`、`ST` 本身，不让它们成为 GP 叶子；
- 将其余数值字段转成 float32，并且只保留 2016–2022 研究窗口；
- 2016–2021 用于进化，2022 用于最终样本外筛选，其他日期不参与本轮。

## 运行

先执行真实数据 smoke test：

```bash
PYTHONPATH=src:.deps python3 -u -m gp_factor_miner.hierarchical_cli \
  --data-file /hy-tmp/autodl-tmp/EOD_Columns/eod \
  --label-file /hy-tmp/data/factor/label_10.pkl \
  --output-dir /hy-tmp/data_pyx/GP_output_eod/eod_v4_smoke \
  --smoke
```

正式搜索去掉 `--smoke`，并将输出目录改为新的运行目录：

```bash
PYTHONPATH=src:.deps python3 -u -m gp_factor_miner.hierarchical_cli
```

不传 `--output-dir` 时，正式结果默认写入
`/hy-tmp/data_pyx/GP_output_eod/run_01`。

默认 pilot 只执行一个独立运行：每个阶段种群为 10,000，阶段内 10 代，warm start 为 3 倍种群；阶段顺序严格为深度 `1 → 2 → 3`。PCA-QD 相似度阈值为 `0.90`，最终筛选使用 `|RankIC| >= 0.07` 且与进化期同方向、相关性 `< 0.75`，不设数量上限。

多进程候选评估会复用第一次表达式计算得到的因子矩阵来生成 PCA
signature，避免为同一候选重复执行完整表达式。该优化不改变
`pca_sample_step`、RankIC、PCA-QD 或筛选口径，并兼容优化前保存的 V4
checkpoint。

`--output-dir` 必须是新的、专用目录。对同一个有效 V4 目录重新运行会加载 setup 缓存和 checkpoint；配置或数据路径不匹配时必须改用新目录。

最终 RankIC 验证和相关性筛选发生在所有三个进化阶段完成之后，绝不参与阶段存活、亲子替换、晋升、fitness 或 PCA-QD 准入。V4 根目录的可比较输出包括：

- `gene_pool_depth_01.json`、`gene_pool_depth_02.json`、`gene_pool_depth_03.json`
- `pca_qd_archive.json`、`gp_history.csv`、`validation_rankic.csv`
- `correlation_decisions.csv`、`selected_gp_factors.json`、`run_config.json`

`validation_rankic.csv` 会在验证期间增量写入，`correlation_decisions.csv` 记录相关性筛选决定；`run_config.json` 记录 checkpoint 版本、完成阶段、阈值、运行参数和依赖版本。
