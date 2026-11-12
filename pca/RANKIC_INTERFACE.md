# RankIC 共用 GPU 百分位排名

调用方需要与原引擎 `gpe::rank_rows_pct` 的 double 排名一致时，使用 `gpu.hpp` 中的 FP64 入口：

```cpp
pca::gpu_select(local_rank);
pca::GpuTiming timing;
std::vector<double> target_ranks =
    pca::gpu_percentile_rows_f64(target.buf, target.T, target.N, timing);
auto daily = gpe::daily_rank_ic(factor, target_ranks);
```

这里 target 和 factor 是上游 `gpe::Panel`，T、N 需一致。示例仅把目标排名交给现有消费者；daily_rank_ic 内部的因子排名仍按原实现执行。接口不改变全局 GP 调用链，也不构成完整 RankIC GPU 加速测量。

输入为主机连续 FP32 数组 `[行][股票]`，输出为相同排列的主机 FP64 数组。有效值按平均排名除以该行非 NaN 数量，并列值共享排名；NaN 保持缺失，正负无穷按数值参与排序。全缺失行保持全 NaN，不做 0.5 填充或去均值。stocks 必须在 1…8192，rows 可以为零；调用方处理采样、日期对齐和成对有效掩码。

三项入口复用同一排序 kernel：

| 入口 | 返回类型 | 用途 |
| --- | --- | --- |
| gpu_rank_rows | float | PCA 预处理：排名、NaN 填 0.5、逐行去均值 |
| gpu_percentile_rows | float | 原始平均百分位，NaN 保留 |
| gpu_percentile_rows_f64 | double | 原始平均百分位，与上游 double 契约兼容 |

FP64 入口不复制排序实现。平均百分位可写为 q/(2V)，V 为非 NaN 数量、q 为整数。V≤8192 时，FP32 百分位误差乘以 2V 仍小于 1/2，因此可在 GPU 上舍入恢复精确 q，再用 FP64 对精确半整数做除法。它复用已有 FP32 排序输出，避免直接把已舍入的 float 转成 double。

本入口包含主机到 GPU 上传、两个 GPU kernel 和 FP64 下载，GpuTiming 分别记录上传、排名和下载。它目前返回主机数组；全 GPU RankIC 流程需要另行设计显存驻留与消费者接口。

`test_rankic_interface.cpp` 用原引擎的 rank_rows_pct 逐值核对，并把 GPU 输出交给真实 daily_rank_ic，核对 IC、有效日期和缺失统计。覆盖 1/17/257/5876/8192 列、并列、全 NaN、常量、±∞ 和有符号零；可附加真实原始面板。`rankic_interface.pbs` 同时运行已有 PCA GPU 检查以验证兼容性，全部通过才生成私有 SUCCESS。

编译时设置 `-DPCA_CUDA=ON -DPCA_ENGINE_BENCH=ON`；实验引擎目标用于消费者验证。接口代码与测试可以交给 RankIC 负责人接入，但采用确认需要由该模块负责人完成。
