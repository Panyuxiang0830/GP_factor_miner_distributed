# 独立 PCA 模块的交付接口

该版本从原始抽样值或已排名 R 生成签名，按给定候选顺序更新 QD，支持空档案或已有档案，输出可用于下一批的完整快照。原 Python GP 可通过显式 batch_backend 或 EOD CLI 的 --cpp-* 参数启用实际引擎衔接，见 [GP_INTEGRATION.md](GP_INTEGRATION.md)。GPU Lanczos 不在本次范围。

## 1. 历史档案与连续批次

```bash
mpirun -np 4 pca/build/pca_run \
  --manifest pca/runs/batch_01/manifest.json --out pca/runs/result_01 \
  --layout date --archive mpi --archive-input pca/runs/previous/archive.json
mpirun -np 4 pca/build/pca_run \
  --manifest pca/runs/batch_02/manifest.json --out pca/runs/result_02 \
  --layout date --archive mpi --archive-input pca/runs/result_01/archive.json
```

不指定 --archive-input 时从空档案开始。每次运行内部的所有块共享同一个档案。CPU 集中档案、GPU 相似度与 MPI 分片档案使用相同快照格式；MPI 导出时将各分片签名汇集到 rank 0。

archive.json 包含 version=1、dimension、signatures 相对路径与 records 数组。每条记录包含非空且唯一的 key、有限 quality 和唯一 slot；二进制为小端 FP32 `[slot][抽样日期]`。签名须有限且非恒定，dimension 必须等于本批 R 的行数。所有批次还须使用一致的全局抽样日期、股票定义和签名算法约定，单凭相同维度不能保证日期已对齐。

默认输出 archive.json、archive_signatures.f32、archive_keys.json、signatures.f32、decisions.json、metrics.json 和 provenance.json。完整快照在主计时结束后序列化；旧性能结果的计时范围不因此改写。输入快照只读，不覆盖旧目录。

## 2. 验证含义

```bash
python3 pca/validate.py --manifest pca/runs/batch_01/manifest.json \
  --run pca/runs/result_01
```

新运行的初始档案和实际初始向量由 provenance.json 自动定位；也可显式提供相同 --archive-input。验证包括：

- numerical_checked / numerical_passed：同预算 Python 幂迭代或 Golub–Kahan 参考，或提供签名的精确重放。未检查用 null 表示，不能当作误差为零。
- independent_python_qd_agreement：独立参考签名的候选过滤、逐候选动作/最近 key/替换 key 与最终成员是否一致。
- archive_key_agreement：用实际输出签名单独隔离并检查准入语义。
- snapshot_checked / snapshot_passed：最终快照中的 key、质量和 FP32 签名是否完整一致。
- svd_checked / svd_passed：精确 SVD 的收敛诊断；仅显式设置 --svd-tolerance 时将其作为通过条件。例如 --svd-count 4 --svd-tolerance 1e-6 用于充分预算的受控案例。

低预算近似不一定收敛，因此默认验证同预算实现是否一致，不要求每个近似都等于精确 SVD。数值输出中的 NaN 不允许通过；提供签名重放则保留并按原策略跳过不可比较的记录。

provenance.json 绑定 manifest、实际初始向量和初始档案内容。缓存 --power-reference 仅用于四步幂迭代，并核对双方输入和配置。历史缓存没有元数据时，只有显式 --allow-legacy-reference 才接受；应先人工核对对应证据。新 study 写 INPUT.json，prepare_completion.py 核对它；历史研究仅可在审计后显式使用 --allow-legacy-study。指纹为非安全用途 FNV-1a，不重新哈希数十 GB 的 R；底层 R 文件仍须保持只读不变。

## 3. 小规模最终验收

```bash
qsub -v PCA_RUN="$PWD/pca/runs/delivery_cpu_01" pca/delivery_cpu.pbs
qsub -v PCA_RUN="$PWD/pca/runs/delivery_gpu_01" pca/delivery_gpu.pbs
```

这些脚本使用人工合成小输入，不替代完整性能实验。delivery_check.py 验证历史档案、顺序替换、同 key、并列与阈值场景；将一次完整运行与两个接续批次逐项比较；验证同预算数值、充分预算 SVD 和错误输入拒绝。输出 SUCCESS 仅在全部检查通过后生成。GPU 验收额外覆盖带历史档案的 GPU 相似度和 FP32/FP64 幂迭代。

cross_node_gpu.pbs 为两节点各一卡的单独入口，记录实际 MPI 主机，拒绝落在同一个节点。默认只处理小批输入，预热一次、测量三次，同时检查一/两卡与 Python 的决定。脚本准备好不代表已经执行；交付状态必须根据实际 SUCCESS.json 记录。原有同节点双卡结果不能写成跨节点结果。资源申请与是否提交由当前任务授权和站点额度决定。

## 4. 私有论文和源码交付包

package_delivery.py 不重新运行数值实验。输入完整报告、CPU/GPU 最终验收目录和可选跨节点证据，输出新的 pca/runs 子目录：

```bash
python3 pca/package_delivery.py --report /path/to/private/report_completed \
  --cpu-check pca/runs/delivery_cpu_01 --gpu-check pca/runs/delivery_gpu_01 \
  --out pca/runs/final_delivery_01
```

它合并真实规模与整代成本章节，消除旧正文与追加说明混用；同时保留表格、图、验证摘要、新版全文 LaTeX 副本和完整仓库源码归档。--pandoc /path/to/pandoc 可生成 LaTeX 正文片段；安装 Markdown 与 WeasyPrint 后使用 --pdf 生成阅读 PDF。全文 LaTeX 是否实际编译、跨节点 GPU 是否验证、自动 GP 是否接入分别记录，不能用“打包成功”替代这些状态。

完整衔接版须另提供 --integration-cpu、--integration-gpu 和 --live-gp，分别指向 bridge_check.py 的 CPU/GPU 验收和 run_gp.py 的真实行情运行目录；三项必须一起提供。--live-gp-gpu 增加真实 GPU 进化证据，--cli-check 增加 cli_bridge_check.py 的原 CLI、样本外筛选及恢复证据，--cross-node 增加跨节点 GPU 验证。打包前须提交源码，避免源码归档与记录的提交版本不同；运行结果仍保留在忽略的私有目录。

GPU 运行验证尚在等待资源时，只能显式使用 --pending-gpu --cuda-build /path/to/verified/build 生成候选包；它要求独立的 COMPILE_CHECK.json，并将 current_gpu_interface_verified 和 independent_module_verified 保持为 false。候选包不替代通过当前 GPU 回归后的正式接口验收包。

所有行情、公式、日志、实验结果、报告和交付包均留在私有 runs；公开分支仅上传源码、作业脚本和说明。旧报告和原 PDF 保留，最终论文还需与组员统一作者、编号、交叉引用及全组结论。
