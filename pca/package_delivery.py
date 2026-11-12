#!/usr/bin/env python3
"""Package validated private evidence and current sources without rerunning benchmarks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(rows,fields):
    labels=dict(method='算法',processed='处理数量',serial_budget_censored='十步未达标',mpi_target_certified='认证达标',common_solve_seconds='共同集合求解 / s',common_allreduce_calls='归约次数',common_logical_payload_bytes='逻辑字节')
    return '\n'.join(['| '+' | '.join(labels.get(x,x) for x in fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+
        ['| '+' | '.join(str(row[x]) for x in fields)+' |' for row in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--cpu-check',type=Path,required=True)
    stage=p.add_mutually_exclusive_group(required=True);stage.add_argument('--gpu-check',type=Path);stage.add_argument('--pending-gpu',action='store_true')
    p.add_argument('--cuda-build',type=Path,help='verified compile evidence; required for a pending-GPU candidate')
    p.add_argument('--cross-node',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--integration-cpu',type=Path);p.add_argument('--integration-gpu',type=Path)
    p.add_argument('--live-gp',type=Path);p.add_argument('--live-gp-gpu',type=Path);p.add_argument('--cli-check',type=Path)
    p.add_argument('--pdf',action='store_true');p.add_argument('--pandoc');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();report=a.report.resolve()
    if out.exists() or not out.is_relative_to((repo/'pca/runs').resolve()):p.error('new private pca/runs output required')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True):p.error('commit source changes before packaging a reproducible source version')
    evidence=load(report/'EVIDENCE.json');cpu=load(a.cpu_check/'SUCCESS.json');gpu=load(a.gpu_check/'SUCCESS.json') if a.gpu_check else None
    assert evidence['completed'] and evidence['completion_supplement_verified']
    for check in (cpu,)+( (gpu,) if gpu else () ):
        assert check['verified'] and check['negative_regressions'] and check['continuation_matches_single_batch'] and check['final_snapshot_contents_exact']
    if gpu:assert gpu['cuda']
    compile_check=load(a.cuda_build/'COMPILE_CHECK.json') if a.cuda_build else None
    if a.pending_gpu:assert compile_check and compile_check['verified'] and compile_check['cuda_compiled'] and not compile_check['gpu_runtime_tested']
    cross=load(a.cross_node/'SUCCESS.json') if a.cross_node else None
    if cross:assert cross['verified'] and cross['nodes']==2 and cross['gpus']==2
    integration_paths=(a.integration_cpu,a.integration_gpu,a.live_gp)
    if any(integration_paths) and not all(integration_paths):p.error('provide integration CPU, GPU and real GP evidence together')
    integrated=all(integration_paths)
    integrated_cpu=load(a.integration_cpu/'SUCCESS.json') if integrated else None
    integrated_gpu=load(a.integration_gpu/'SUCCESS.json') if integrated else None
    live=load(a.live_gp/'SUCCESS.json') if integrated else None
    live_gpu=load(a.live_gp_gpu/'SUCCESS.json') if a.live_gp_gpu else None
    cli=load(a.cli_check/'SUCCESS.json') if a.cli_check else None
    if integrated:
        for check in (integrated_cpu,integrated_gpu):
            assert check['verified'] and check['live_cpp_engine'] and check['live_cpp_pca'] and check['live_cpp_qd']
            assert check['stages']==[1,2,3] and check['checkpoint_resume_exact'] and check['python_numerical_fallback_disabled']
        assert integrated_gpu['device']=='cuda' and live['completed'] and live['generation_events']>0
    if live_gpu:assert integrated and live_gpu['completed'] and live_gpu['device']=='cuda'
    if cli:assert integrated and cli['verified'] and cli['original_cli'] and cli['original_oos_screening'] and cli['completed_resume']
    out.mkdir();(out/'figures').mkdir();(out/'latex').mkdir();(out/'evidence').mkdir()
    for source in report.iterdir():
        if source.suffix in ('.png','.csv'):shutil.copy2(source,out/'figures'/source.name)
    for name,parent in (('cpu_acceptance',a.cpu_check),)+( (('gpu_acceptance',a.gpu_check),) if gpu else () ):
        shutil.copy2(parent/'SUCCESS.json',out/'evidence'/(name+'.json'))
    if a.cuda_build:shutil.copy2(a.cuda_build/'COMPILE_CHECK.json',out/'evidence/cuda_compile.json')
    for label,parent in (('cross_node',a.cross_node),('integration_cpu',a.integration_cpu),('integration_gpu',a.integration_gpu),('live_gp',a.live_gp),('live_gp_gpu',a.live_gp_gpu),('original_cli',a.cli_check)):
        if parent:shutil.copy2(parent/'SUCCESS.json',out/'evidence'/(label+'.json'))
    shutil.copy2(report/'EVIDENCE.json',out/'evidence/experiments.json')
    # Keep the old report as historical evidence, and produce one consolidated manuscript.
    text=(report/'pdf_sections.md').read_text().split('# 严格交接补充：')[0]
    text=re.sub(r'^>.*\n\n','',text)
    text=text.replace('完整实验及限制另见 report.md、EVIDENCE.json。','完整实验依据见 evidence/experiments.json；验收范围见 DELIVERY_STATUS.json。')
    supplement=(report/'completion_sections.md').read_text()
    real=supplement.split('## P1：',1)[1].split('## P2：',1)[0].split('\n',1)[1].strip()
    replacement='## 6.3.1 串行代价与 CPU 扩展\n\n'+real+'\n\n'
    replacement+='原有合成规模曲线作为受控输入对照保留。CPU 强扩展、弱扩展、混合并行和档案规模使用各自验证过的配置；不能将不同输入或布局的时间直接相加。\n\n'
    replacement+='![合成规模](serial_problem_size.png)\n![CPU 强扩展](strong_scaling.png)\n![弱扩展](weak_scaling.png)\n![档案规模](archive_scaling.png)\n\n'
    text=re.sub(r'## 6\.3\.1 .*?(?=## 6\.3\.2 )',lambda _:replacement,text,flags=re.S)
    with (report/'full_accuracy_cost.csv').open() as f:cost=list(csv.DictReader(f))
    replacement=f"## 6.4.2 计算代价\n\n两算法均执行全部 {int(cost[0]['processed'])} 张 R。由精确 SVD 研究离线选择预算，十步未达标者单列；同精度成本只比较共同认证的 {int(cost[0]['common_target_certified'])} 个因子。\n\n"
    replacement+=table(cost,('method','processed','serial_budget_censored','mpi_target_certified','common_solve_seconds','common_allreduce_calls','common_logical_payload_bytes'))
    replacement+=f"\n\n实际预留 {evidence['full_generation_reserved_cpus']} CPU 核、{evidence['full_generation_allocated_memory']}，启动 {evidence['full_generation_active_ranks']} 个单线程 MPI ranks。双方均逐因子归约，未使用生产 power 的批量归约；残差诊断、输入缓存和记录开销不在纯求解计时内。调用数和逻辑字节量不等于实际网络流量。该对照不表示已实现在线自适应停止。\n\n![整代同精度成本](full_accuracy_cost.png)\n\n"
    text=re.sub(r'## 6\.4\.2 .*?(?=## 6\.4\.3 )',lambda _:replacement,text,flags=re.S)
    marker='## 6.3.4 签名与准入判定的一致性'
    topology=('两节点各一卡的独立验证通过；其结果与原有同节点双卡性能实验分别记录。' if cross else
              '上述性能结果来自同一节点的两张 A40；跨节点 GPU 验证尚未执行，不能据此声称已经验证跨节点 GPU 网络通信。')
    text=text.replace(marker,topology+'\n\n'+marker)
    acceptance='''## 第 5 节附加：最终接口验收

主程序支持 --archive-input 读取历史 QD 档案，输出 archive.json 和完整 FP32 签名，以便下一批接续。快照保存 key、质量、维度和数据指纹，加载时拒绝重复 key、无效签名、维度及二进制长度不符。MPI 分片档案集体导出为同一格式。

新增小规模验收核对历史档案处理，以及一次完整批次与两个接续批次的完整最终档案。CPU、MPI 与当前 GPU 接口的验收状态分别保存在 DELIVERY_STATUS.json。验证分别报告同预算 Python 数值参考、逐候选动作/最近 key/替换 key、最终成员和快照内容。受控场景另用明确 SVD 门槛验收；低预算研究保留未收敛结果，不强行判成程序错误。错误签名、损坏档案和输入身份变化均有失败回归。

旧的大规模实验结果保留为历史证据；本次接口修正没有重跑全部 SVD 或扩展实验，也不把旧输出冒充新验证器生成的结果。新增接口检查属于独立验收，不是新的大规模性能数字。
'''
    text=text.replace('## 6.3.1 ',acceptance+'\n\n## 6.3.1 ',1)
    text+='''
## 附录 A：当前交付入口

源码位于仓库 pca/。CPU 使用 cmake -S pca -B pca/build -DPCA_MPI=ON；CUDA 另设 PCA_CUDA=ON。生产不链接 BLAS/cuBLAS，库仅用于独立性能参照。历史档案、验证字段和复现命令见 pca/DELIVERY.md、README.md 与 REPRODUCE.md。

原 Python GP 尚未自动切换至本实现；RankIC 接口已有消费者验证，正式采用由对应负责人协调。GPU Lanczos 不在本次范围。全组作者信息、原 1481 个进化因子的统计和最终全文格式需与组员整合。本包提供对应新版全文源文件的副本，不覆盖上游原稿。
'''
    if integrated:
        text=text.replace('当前交付是独立 PCA/准入子流水线，尚不构成整代 GP 与新 GPU 实现已联调的证据。',
            '本版本既提供独立 PCA/准入模块，也将其接入原 Python 分层 GP；实际 C++ 引擎、PCA、QD 与下一代的闭环证据见第 6.5 节，验证规模与完整生产预算明确区分。')
        text=text.replace('仍需与最新因子引擎在批次接口联调，核对全局日期、halo 排除、key 与返回顺序。',
            '已有实际引擎接口联调，核对全局日期抽样、halo 排除、key 和返回顺序，并验证中断恢复；完整生产预算的重复性能测量仍可进一步开展。')
        text=text.replace('原 Python GP 尚未自动切换至本实现；',
            '原 Python GP 的可选 batch_backend 与原 EOD CLI 的 --cpp-* 参数已接入本实现；默认原后端保持不变。')
        text=text.replace('源码位于仓库 pca/。',
            '数值与衔接实现位于仓库 pca/，原 Python 的 hierarchical_engine.py 和 hierarchical_cli.py 增加可选后端接口。')
        rows=[]
        for label,run in (('CPU/MPI',live),('CUDA',live_gpu)):
            if run:rows.append(dict(backend=label,T=run['T'],N=run['N'],fields=len(run['fields']),ranks=run['ranks'],generations=run['generation_events'],archive=run['archive_size'],wall_seconds=round(run['evolution_wall_seconds'],3)))
        body='''## 6.5 实际 GP 衔接与完整闭环

原 Python GP 仍负责分层提案、父代选择、交叉、变异、缓存和检查点。启用可选后端后，一个进度块交给原 C++ MPI 引擎计算 DAG、RankIC 与适应度；按全局日期抽取自有行、排除 halo 后执行 PCA，再将同序结果交回 Python。准入另调用 C++ 顺序档案，并将完整新档案逐项同步给原对象，继续生成下一代。无需预先导出 R，也不把整代完整因子值写入磁盘。

合成闭环验收实际完成深度 1、2、3 及下一代；将旧 Python 评估、PCA、准入入口设为调用即失败，证明没有隐蔽回退。固定表达式与 Python 参考核对；中断后恢复与未中断结果一致，后端或数据身份改变时拒绝恢复。CPU/MPI 和 GPU 分别通过。

真实行情字段上的小规模实际进化如下。单次墙钟包含 MPI 子进程启动、原 Python 控制和检查点，不能作为完整生产规模的重复加速基准：

'''+table(rows,('backend','T','N','fields','ranks','generations','archive','wall_seconds'))
        if cli:body+='\n\n原 EOD 命令行还通过了带 C++ 后端的完整运行和已完成检查点恢复，输出三个深度基因池、QD 档案、原样本外 RankIC 筛选、相关性筛选和最终因子文件。筛选阶段沿用原 Python 实现，不将它描述为已全部迁移到 CUDA。'
        body+='\n\n目前为有界子批与进度块级的进程调用，启动、数据准备及 I/O 会限制小批次效率。不同 MPI 规模或精度可能放大细小评分排序差异，因此固定候选顺序的 QD 一致性与跨配置整次进化集合相同分别检查，不作普遍等价保证。正式人口和代数预算需在对应资源配置下另做性能测量。\n\n'
        text=text.replace('## 7.1 ',body+'## 7.1 ',1)
    for image in re.findall(r'!\[[^]]*\]\(([^)]+)\)',text):
        assert (out/'figures'/image).exists(),image
        text=text.replace(']('+image+')','](figures/'+image+')')
    if a.pending_gpu:text='> 交付候选：当前 CPU/MPI 接口验收与 CUDA 编译通过，新增 GPU 接口运行验证仍在排队；历史 GPU 性能结果保留，不等于本次接口已实测。跨节点状态另列。\n\n'+text
    (out/'pca_paper.md').write_text(text)
    shutil.copy2(repo/'report/main_zh.tex',out/'upstream_main_zh.tex')
    # A source archive is private too; only repository source, never ignored runs/builds.
    names=subprocess.check_output(['git','-C',str(repo),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    sources=[name for name in names if name and (repo/name).is_file() and '.git' not in Path(name).parts]
    assert all('runs' not in Path(name).parts and not any(x.startswith('build') for x in Path(name).parts[:-1]) for name in sources)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for name in sorted(sources):tar.add(repo/name,arcname='GP_factor_miner_distributed/'+name)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if a.pandoc:
        subprocess.run([a.pandoc,'-f','markdown','-t','latex','--wrap=none','-o',str(out/'latex/pca_sections.tex')],input=text.encode(),check=True)
    if a.pdf:
        import markdown
        from weasyprint import HTML
        font=Path('/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf')
        css='@font-face{font-family:Chinese;src:url("'+font.as_uri()+'")} @page{size:A4;margin:18mm} body{font-family:"DejaVu Sans",Chinese;font-size:10pt;line-height:1.5} img{max-width:100%} table{width:100%;border-collapse:collapse;font-size:7pt} td,th{border:1px solid #bcc;padding:3px;overflow-wrap:anywhere} tr{break-inside:avoid} h1,h2{break-after:avoid} p{overflow-wrap:anywhere}'
        html='<meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown(text,extensions=['tables','fenced_code'])
        (out/'pca_paper.html').write_text(html);HTML(string=html,base_url=str(out)).write_pdf(out/'pca_paper.pdf')
    status=dict(independent_module_verified=bool(gpu),cpu_interface_verified=True,current_gpu_interface_verified=bool(gpu),cuda_compiled=bool(compile_check) or bool(gpu),
        source_commit=commit,cpu_acceptance=str(a.cpu_check.resolve()),gpu_acceptance=str(a.gpu_check.resolve()) if a.gpu_check else None,
        historical_experiments=str(report),cross_node_gpu_verified=bool(cross),automatic_gp_integration=integrated,
        original_cli_verified=bool(cli),original_oos_pipeline_verified=bool(cli),real_gpu_gp_verified=bool(live_gpu),
        full_production_budget_measured=False,gpu_lanczos_in_scope=False,
        latest_latex_available=True,full_latex_compiled=False)
    (out/'DELIVERY_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    (out/'README.md').write_text('# PCA 模块交付包\n\n先读 pca_paper.md 或 pca_paper.pdf；源码为 source.tar.gz，需在完整仓库布局中编译。\n\n'
        '实验数字来自已完成私有报告；独立接口、跨节点、GP 闭环、原 EOD CLI 和真实数据验证分别见 evidence/ 与 DELIVERY_STATUS.json，未提供运行证据时保持未完成。完整生产预算的性能测量与全组最终合稿另行协作。\n\n'
        'upstream_main_zh.tex 是现有新版源文件副本；latex/pca_sections.tex 可供合并，未宣称已编译最终全文。数据、表达式和本包均不上传公开仓库。\n')
    hashes={str(x.relative_to(out)):sha(x) for x in out.rglob('*') if x.is_file()}
    (out/'MANIFEST_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n');print('Delivery package written:',out)


if __name__=='__main__':main()
