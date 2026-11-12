"""Append validated real P1 and whole-generation P2 evidence to a base report."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import save


def load(p):return json.loads(p.read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def append_completion(out,real,matched):
    out,real,matched=map(Path,(out,real,matched))
    if (out/'completion_sections.md').exists():raise ValueError('completion supplement already exists')
    rp=load(real/'SUCCESS.json');mp=load(matched/'SUCCESS.json');source=load(real/'source.json')
    allocation=load(matched.with_name(matched.name+'_resources.json'))
    assert rp['verified'] and rp['real_market_panels'] and rp['independent_subset_ranking']
    assert len(rp['configurations'])==19 and source['T']==1461 and source['N']==5876 and source['factors']==128
    assert mp['verified'] and mp['processed_factors']==9388 and mp['common_target_certified']>0
    assert mp['ranks']==8 and mp['target']==1e-6 and mp['max_steps']==10
    rows=[]
    for run in sorted(real.glob('*_r1')):
        name=run.name[:-3];axis=name[0];repeat=[]
        for r in (1,2,3):repeat.append(load(real/f'{name}_r{r}/metrics.json'))
        med={k:float(np.median([x[k] for x in repeat])) for k in repeat[0]}
        value=med[dict(m='rows',n='cols',B='factors',k='steps')[axis]]
        rows.append(dict(configuration=name,axis=axis,value=value,**med))
    assert len(rows)==19
    # n=5876 uses the same full-universe m5 case.
    base=next(x for x in rows if x['configuration']=='m5')
    curves=rows+[dict(base,axis='n',value=5876,configuration='n5876_shared_m5')]
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,axis in zip(axes.flat,('B','m','n','k')):
        series=sorted([r for r in curves if r['axis']==axis],key=lambda x:x['value'])
        for field,label in (('ranking_seconds','fresh percentile ranking'),('solve_seconds','PCA solve'),('total_seconds','sample -> archive')):
            ax.loglog([r['value'] for r in series],[r[field] for r in series],'o-',label=label)
        ax.set(xlabel=axis,ylabel='Seconds, 1 CPU thread');ax.legend(fontsize=7);ax.grid(alpha=.3)
    save(fig,out,'real_problem_size')
    with (out/'real_problem_size.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'full_accuracy_cost.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mp['methods'][0]));w.writeheader();w.writerows(mp['methods'])
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,field,label in zip(axes,('common_solve_seconds','common_allreduce_calls','common_logical_payload_bytes'),
                            ('Sum of max-rank solve seconds','Algorithm Allreduce calls','Logical payload bytes')):
        ax.bar([r['method'] for r in mp['methods']],[r[field] for r in mp['methods']]);ax.set(ylabel=label,ylim=(0,None))
        ax.grid(axis='y',alpha=.3)
    fig.suptitle(f"Same certified factors: {mp['common_target_certified']} / {mp['processed_factors']}; 8 MPI ranks")
    save(fig,out,'full_accuracy_cost')
    summary_table=table(['方法','实际处理','串行十步预算未达标','MPI 认证达标','共同达标集合求解 / s','该集合 Allreduce 次数','该集合逻辑字节'],
        [[r['method'],r['processed'],r['serial_budget_censored'],r['mpi_target_certified'],f"{r['common_solve_seconds']:.6g}",
          int(r['common_allreduce_calls']),int(r['common_logical_payload_bytes'])] for r in mp['methods']])
    text=f'''# 严格交接补充：真实规模与整代同精度成本

## P1：固定真实因子面板，重新抽样与股票子集排名

由行情字段经原引擎计算 128 个真实因子面板，固定同一批面板；每个配置重新取全局抽样日与股票前缀，再对该股票子集做平均百分位排名、NaN 填 0.5、去均值。抽样间隔为 1/2/5/10/20，股票数为 367/734/1469/2938/5876，另改变 B 和 k，共 19 个配置。每配置预热一次、测量三次；因子生成的 {source['engine_evaluation_seconds']:.3f} s 单列，不混入 PCA 时间。

因子值在完整股票池上计算并固定，改变 N 时重新排名所选股票的这些值；没有把含截面算子的因子公式在新股票池上重新求值。准入质量固定使用原 scores，不重算 RankIC。这对应 PCA 模块的股票子集输入实验。因子生成使用八线程，PCA 计时使用一个线程，含采样复制、排名、PCA 与准入，不含结果序列化。每个配置全量 QD 与三次重复一致性通过；前两个真实面板另由 pandas 重新排名并独立计算签名。

![真实输入规模](real_problem_size.png)

## P2：整代输入上的同精度 MPI 对照

两种算法都实际处理全部 {mp['processed_factors']} 张真实 R，8 个 MPI ranks 按日期分布，预热一次、三次计时。步数由先前精确 SVD 研究中首次达到 1−|cos|≤{mp['target']:g} 的预算确定；十步内未达到者用十步执行并单列。这是离线选择预算的计算成本对照，不是已实现在线自适应停止规则。

实际 PBS 分配为 {allocation['ncpus']} 个 CPU 核、{allocation['mem']} 内存；计算只启动 8 个 MPI ranks、每 rank 一线程。多申请的 CPU 核用于满足站点按核分配的内存额度，不能把这项实验说成只预留了 8 核。FP32 输入缓存约占 64.7 GB，读取与 FP64 转换成本另列。作业先核对实际分配内存，避免路由规则改写申请后内存不足。

MPI 签名与相同步数的串行签名核对，再以角度三角不等式和舍入余量给出 SVD 误差上界。仅对两算法都被认证达标的同一集合（{mp['common_target_certified']} 个因子）比较同精度成本；其余矩阵仍已运行，不混入达标集合的速度结论。

{summary_table}

![整代同精度成本](full_accuracy_cost.png)

两种方法均逐因子调用同一求解 API，power 在此没有使用生产驱动的批量归约；因此结果是控制归约粒度后的算法对照，不替代原生产强扩展结果。残差诊断不计入本次求解；每因子的对齐 Barrier、计时归约、签名汇集和记录开销位于求解计时之外。驱动总墙钟与缓存读取另存原始 metrics，不用它们冒充纯求解时间。调用数和字节量仅计求解器实际 Allreduce，字节量不是网络链路流量。

完整输入的失败／未达标数量仍需报告，不能从共同达标集合推断全部因子已经收敛。本补充替代“只有合成规模与 8 因子 MPI 代表实验”的覆盖限制；原来的合成图和 8 因子结果保留为不同实验。
'''
    (out/'completion_sections.md').write_text(text)
    for name in ('report.md','pdf_sections.md'):
        original=(out/name).read_text()
        (out/name).write_text('> 新增交接补充已验证：[真实 P1 与整代 P2](completion_sections.md)。原有合成与小样本结果保留，并与新增实验分开解释。\n\n'+original+'\n\n'+text)
    evidence=load(out/'EVIDENCE.json')
    evidence.update(completed=True,real_input_serial_run=str(real.resolve()),full_generation_accuracy_run=str(matched.resolve()),
        full_generation_reserved_cpus=allocation['ncpus'],full_generation_allocated_memory=allocation['mem'],full_generation_active_ranks=mp['ranks'],
        completion_supplement_verified=True,matched_accuracy_scope='all matrices processed; identical jointly certified cohort compared; budget-censored cases explicit')
    (out/'EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return mp


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--real-size',type=Path,required=True);p.add_argument('--matched-full',type=Path,required=True)
    a=p.parse_args();append_completion(a.report,a.real_size,a.matched_full)
