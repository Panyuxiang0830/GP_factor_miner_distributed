# 模块接口与数据约定

R 的行是抽样日期，列是股票；常用维度为 293×5876。现成 R 已做百分位排名、NaN 填 0.5 与逐行去均值。只有未排名抽样值才使用 `--rank-input 1`。

## 文件与 API

| 文件 | 用途 |
| --- | --- |
| core.hpp / core.cpp | CPU 排名、两种求解器、自写 Jacobi、相似度和顺序档案 |
| main.cpp | 独立批次驱动、MPI 三布局、结果和阶段计时 |
| mpi_archive.hpp | 档案签名分片、局部复核、全局 MAXLOC |
| gpu.hpp / gpu.cu | 百分位排名、批量幂迭代和手写分块相似度 |
| c_api.cpp | Python 数值研究调用的 C ABI |
| reference.py | Python 同输入幂迭代参考，仅用于验证 |
| validate.py | 与原 Python 准入逐项核对，另可指定精确 SVD 抽查 |
| synthetic.py / adversarial.py | 人工合成输入与阈值、并列等边界用例生成 |

## 与上游引擎对接

上游按原始全局交易日分块，本驱动则按已抽样的 R 行分块，两者的边界不一定相同。接入时应先在本 rank 的自有日期内按 `t % sample_step == 0` 选行，排除 halo，再按股票行排名并逐行去均值。新版主工程没有旧的 `gpe::signature_rows`，不要依赖该已移除接口。

CPU 将本地抽样行放进 `pca::Matrix`，所有 rank 使用同一初始右向量；将 `MPI_Allreduce` 包装成 `pca::Sum`，传给 `pca::solve`。按全局日期顺序汇集局部签名后统一调用 `pca::finish`，不在各 rank 独立固定符号。因子引擎可能分块打分并释放根面板，应在释放前抽样，或明确保留对应输入。本模块尚未修改该调用链。

GPU `gpu_rank_rows` 将缺失值填 0.5 并去均值，供 PCA 使用；`gpu_percentile_rows` 保留 NaN、不填充、不去均值，供 RankIC 调用方复用。采样和成对有效值掩码由调用方负责。输入是连续 `[行][股票]` 的 FP32；当前排序最多 8192 列。`gpu_power` 的 `DeviceReduce` 回调接收显存指针，使用前须确认 CUDA-aware MPI 支持。

原引擎 `rank_rows_pct` 返回 double，需逐值兼容时使用 `gpu_percentile_rows_f64`。该入口复用相同排序 kernel，在 GPU 上恢复精确平均排名后做 FP64 除法，再返回主机 double 数组，可直接交给原有 `daily_rank_ic` 的目标排名参数。详见 [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md)；提供可调用接口与消费者验证，不表示主 GP 已自动切换后端。

## 签名和准入

签名求解用 `u=normalize(Rv)`、`v=normalize(Rᵀu)` 交替更新，最后再求 `normalize(Rv)`；符号固定后转 FP32。Lanczos 通过左右基和小双对角矩阵得到近似主方向。NumPy 精确 SVD 只用于验证。

比较签名时还需重新中心化、归一化，取绝对 Pearson 相关；这是独立于签名归一化的步骤。一个非零 R 仍可能得到时间恒定的签名，这种签名不能比较，会被跳过。

准入保持候选顺序：最大相似度低于 0.90 则加入，否则只有质量严格超过最近记录加 0.002 才替换；质量为 abs RankIC。相似度并列时选字典序较大的 key。相似度粗算使用 FP32 和保守误差界，再对可能改变最大值／阈值判定的记录用 FP64 复核。前面候选的加入、替换或删除对后续候选立即可见。

默认档案由 rank 0 维护，CUDA 模式在 GPU 上粗算相似度；MPI 分片档案是另一条独立路径，保留小元数据、用 MAXLOC 统一最近记录，不假定块内档案不变。

## 布局与计时

date 每轮跨日期归约；transpose 先交换完整矩阵再本地迭代；factor 直接读取完整因子，不含上游重分布。CPU 新排名保留 double，转置也用 double；现成 FP32 R 按 float 转置。

日期布局的 power 已按批归约，Lanczos 当前逐因子归约，因此较少步数不保证较少 Allreduce。数值诊断额外做两次矩阵乘。各 rank 的阶段最大值可能来自不同进程，而且通信嵌套在求解／转置中，不可相加代替总时间。

查看结果时先检查验证标记，再看 metrics、逐候选 decisions 与最终 archive_keys；不能仅凭程序退出或文件夹存在判断正确性。所有输出留在 runs，不上传到公开仓库。

继续历史批次时传 --archive-input archive.json。集中档案和 MPI 分片档案都读取同一组 key、质量与签名，运行后导出完整快照；下一批应显式引用该快照，并保持抽样日期与算法约定一致。验证器自动读取 provenance 中的实际初始档案，不会把历史批次当成空库重放。字段和最终验收见 [DELIVERY.md](DELIVERY.md)。
