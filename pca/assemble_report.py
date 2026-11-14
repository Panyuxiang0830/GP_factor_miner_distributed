#!/usr/bin/env python3
"""Assemble the complete P0--P6 report only after all experiment gates pass."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report import save
from paper_sections import write_pdf_sections


def load(path): return json.loads(path.read_text())


def table(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def measured(parent, pattern):
    if not (parent/"SUCCESS").exists(): raise ValueError(f"incomplete job: {parent.name}")
    groups={}
    for path in parent.glob("*/metrics.json"):
        match=re.fullmatch(pattern,path.parent.name)
        if match:
            key=match.groups()[:-1];repeat=int(match.groups()[-1])
            if repeat in (1,2,3):groups.setdefault(key,{})[repeat]=load(path)
    answer={}
    for key,values in groups.items():
        if set(values)!={1,2,3}:raise ValueError(f"missing repetitions: {parent}:{key}")
        answer[key]={k:float(np.median([v[k] for v in values.values()]))
                     for k in values[1] if isinstance(values[1][k],(int,float))}
    return answer


def mdtable(rows, keys, labels=None):
    def cell(v):return f"{v:.6g}" if isinstance(v,float) else str(v)
    return "\n".join(["| "+" | ".join(labels or keys)+" |","| "+" | ".join(["---"]*len(keys))+" |"]+
                     ["| "+" | ".join(cell(row[k]) for k in keys)+" |" for row in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path(__file__).resolve().parent/"runs")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--precision",default="precision_full_03")
    parser.add_argument("--adversarial",default="adversarial_02")
    parser.add_argument("--scaling-validation",default="scaling_validation_01")
    parser.add_argument("--synthetic-validation",default="synthetic_validation_01")
    parser.add_argument("--cublas-reference",default="cublas_reference_01")
    parser.add_argument("--real-size",type=Path,help="verified real-panel size experiment")
    parser.add_argument("--matched-full",type=Path,help="verified whole-generation MPI accuracy-cost experiment")
    parser.add_argument("--require-completion",action="store_true",help="require both real-input and whole-generation supplements")
    args=parser.parse_args();root=args.runs.resolve();out=args.out.resolve()
    if bool(args.real_size)!=bool(args.matched_full) or (args.require_completion and not args.real_size):
        parser.error("provide both --real-size and --matched-full for the completion supplement")
    if out.exists() or not out.is_relative_to(root):parser.error("output must be new and inside runs/")
    required=["serial_sizes_01","archive_sizes_01","gpu_two_01","gpu_roofline_01","iteration_cost_01","gpu_api_01",args.adversarial,args.cublas_reference]
    required += [f"{kind}_{n}node_01" for kind in ("scaling","weak","raw_scaling") for n in (1,2,4)]
    missing=[name for name in required if not (root/name/"SUCCESS").exists()]
    for name in ("full_validation_01",args.scaling_validation,args.precision,args.synthetic_validation):
        path=root/name/"SUCCESS.json"
        if not path.exists() or not load(path).get("verified",load(path).get("validated",False)):missing.append(name)
    if missing:raise ValueError("incomplete requirements: "+", ".join(missing))
    for kind,expected_count in (("weak",27),("raw",9)):
        verified=load(root/args.synthetic_validation/kind/"SUCCESS.json")
        assert verified["verified"] and verified["kind"]==kind
        assert len(verified["configurations"])==expected_count
        assert max(x["ranks"] for x in verified["configurations"])==288
        assert all(x["python_qd_passed"] and x["numerical_sample_count"]>=8
                   for x in verified["configurations"])
    for case in ("threshold","near_tie","duplicate_margin","invalid"):
        for device in ("cpu","cuda","mpi"):assert load(root/args.adversarial/f"{case}_{device}/validation.json")["passed"]
    api=load(root/"gpu_api_01/validation.json")
    assert api["passed"] and api["raw_percentile_mask_preserved"] and api["percentile_error"]<1e-7
    provided=load(root/"gpu_api_01/PROVIDED_REFERENCE.json")
    assert provided["verified"] and provided["cpu_cuda_mpi_exact_decisions"]
    link=load(root/args.cublas_reference/"PRODUCTION_LINK_CHECK.json")
    assert link["verified"] and not link["cublas_dependency"]
    assert load(root/args.cublas_reference/"gpu_validation.json")["passed"]
    sensitivity=load(root/"sensitivity_full_01/summary.json")
    assert sensitivity["completed"] and sensitivity["factors"]==9388 and len(sensitivity["decision_sensitivity"])==60
    subprocess.run([sys.executable,str(Path(__file__).with_name("report.py")),"--study",str(root/"study_full_01"),
        "--sensitivity",str(root/"sensitivity_full_01"),"--scaling",*[str(root/f"scaling_{n}node_01") for n in (1,2,4)],
        "--roofline",str(root/"gpu_roofline_01"),"--out",str(out),"--runs-root",str(root)],check=True)

    serial=measured(root/"serial_sizes_01",r"([mBnk]\d+)_r([123])");assert len(serial)==19
    series={k:[] for k in "Bmnk"};slopes={};size_rows=[]
    for (tag,),row in serial.items():
        axis=tag[0];series[axis].append((row[dict(B="factors",m="rows",n="cols",k="steps")[axis]],row))
    series["n"].append((5876,serial[("m5",)]))
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,(axis,points) in zip(axes.flat,series.items()):
        points=sorted(points);x=np.array([p[0] for p in points]);y=np.array([p[1]["solve_seconds"] for p in points])
        work=2*x+1 if axis=="k" else x;slopes[axis]=float(np.polyfit(np.log(work),np.log(y),1)[0])
        ax.loglog(x,y,"o-",label=f"measured slope {slopes[axis]:.2f}")
        ax.loglog(x,y[len(y)//2]*work/work[len(work)//2],"--",label="linear theoretical work")
        ax.set(xlabel=axis,ylabel="Signature compute seconds");ax.legend(fontsize=8);ax.grid(alpha=.3)
        for xx,row in points:size_rows.append(dict(axis=axis,value=xx,solve_seconds=row["solve_seconds"],total_seconds=row["total_seconds"]))
    save(fig,out,"serial_problem_size");table(out/"serial_problem_size.csv",size_rows)

    strong={};hybrid=[];phases=[]
    for nodes in (1,2,4):
        groups=measured(root/f"scaling_{nodes}node_01",r"(strong|hybrid)_(date|transpose|factor)_p(\d+)_t(\d+)_r([123])")
        for (kind,layout,p,t),row in groups.items():
            entry=dict(nodes=nodes,kind=kind,layout=layout,ranks=int(p),threads=int(t),**{k:row[k] for k in
                ("total_seconds","load_seconds","solve_seconds","communication_seconds","transpose_seconds","gather_seconds","similarity_seconds","admission_seconds")})
            phases.append(entry)
            if kind=="strong":strong[(layout,int(p))]=entry
            else:hybrid.append(entry)
    assert len(strong)==27 and len(hybrid)==9
    assert all((l,p) in strong for l in ("date","transpose","factor") for p in (1,2,4,8,16,36,72,144,288))
    table(out/"strong_and_hybrid.csv",phases)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,layout in zip(axes,("date","transpose","factor")):
        pure=[strong[(layout,p)] for p in (72,144,288)];mixed=sorted([x for x in hybrid if x["layout"]==layout],key=lambda x:x["nodes"])
        for rows,label in ((pure,"MPI"),(mixed,"4 ranks/node x 18 threads")):
            ax.loglog([x["nodes"]*72 for x in rows],[x["total_seconds"] for x in rows],"o-",label=label)
        ax.set(title=layout,xlabel="CPU cores",ylabel="Total seconds");ax.legend(fontsize=8);ax.grid(alpha=.3)
    save(fig,out,"hybrid_comparison")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for layout in ("date","transpose","factor"):
        rows=[strong[(layout,p)] for p in (1,2,4,8,16,36,72,144,288)];x=[r["ranks"] for r in rows]
        if any(r["communication_seconds"]>0 for r in rows):axes[0].loglog(x,[r["communication_seconds"] for r in rows],"o-",label=layout)
        axes[1].semilogx(x,[r["admission_seconds"]/r["total_seconds"] for r in rows],"o-",label=layout)
    axes[0].set(xlabel="MPI ranks",ylabel="Allreduce / Alltoallv seconds (max rank)")
    axes[1].set(xlabel="MPI ranks",ylabel="Sequential decision loop / total")
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    save(fig,out,"communication_and_sequential_fraction")

    raw_rows=[];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for nodes in (1,2,4):
        groups=measured(root/f"raw_scaling_{nodes}node_01",r"raw_(date|transpose|factor)_p(\d+)_t(\d+)_r([123])");assert len(groups)==3
        for (layout,p,t),r in groups.items():
            assert r["rank_input"]==1 and r["ranking_seconds"]>0 and r["factors"]==9216
            raw_rows.append(dict(nodes=nodes,layout=layout,ranks=int(p),**{k:r[k] for k in
                ("ranking_seconds","solve_seconds","total_seconds","communication_seconds","transpose_seconds")}))
    for layout in ("date","transpose","factor"):
        rows=sorted([r for r in raw_rows if r["layout"]==layout],key=lambda r:r["ranks"])
        axes[0].loglog([r["ranks"] for r in rows],[r["ranking_seconds"] for r in rows],"o-",label=layout)
        axes[1].loglog([r["ranks"] for r in rows],[r["total_seconds"] for r in rows],"o-",label=layout)
    axes[0].set(xlabel="CPU cores",ylabel="Ranking seconds");axes[1].set(xlabel="CPU cores",ylabel="Raw panels -> archive seconds")
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    save(fig,out,"raw_ranking_pipeline");table(out/"raw_ranking_pipeline.csv",raw_rows)

    weak={};weak_rows=[];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for nodes in (1,2,4):
        groups=measured(root/f"weak_{nodes}node_01",r"weak_(date|transpose|factor)_p(\d+)_r([123])")
        for (layout,p),row in groups.items():assert row["factors"]==32*int(p);weak[(layout,int(p))]=row
    assert len(weak)==27
    for layout in ("date","transpose","factor"):
        p=sorted(p for l,p in weak if l==layout);times=np.array([weak[(layout,r)]["total_seconds"] for r in p]);eff=times[0]/times
        axes[0].loglog(p,times,"o-",label=layout);axes[1].semilogx(p,eff,"o-",label=layout)
        for r,t,e in zip(p,times,eff):weak_rows.append(dict(layout=layout,ranks=r,factors=32*r,total_seconds=t,weak_efficiency=e))
    axes[0].set(xlabel="MPI ranks",ylabel="Total seconds");axes[1].set(xlabel="MPI ranks",ylabel="Weak efficiency T1 / TP")
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    save(fig,out,"weak_scaling");table(out/"weak_scaling.csv",weak_rows)

    archive={};archive_rows=[]
    for path in (root/"archive_sizes_01").glob("*.json"):
        match=re.fullmatch(r"(root|mpi)_A(\d+)_p(\d+)_r([123])\.json",path.name)
        if match:
            mode,A,p,_=match.groups();archive.setdefault((mode,int(A),int(p)),[]).append(load(path))
    assert len(archive)==16 and all(len(v)==3 for v in archive.values())
    fig,grid=plt.subplots(2,2,figsize=(11,8));axes=grid.ravel()
    for p in (1,4,16,72):
        mode="root" if p==1 else "mpi";sizes=[1000,5000,10000,50000];times=[];comm=[];similarity=[];admission=[]
        for A in sizes:
            rows=archive[(mode,A,p)];t=float(np.median([x["total_seconds"] for x in rows]));c=float(np.median([x["communication_seconds"] for x in rows]));times.append(t);comm.append(c)
            expected=archive[("root",A,1)][0]
            assert all(tuple(x[k] for k in ("admit","replace","reject","archive_final_size"))==tuple(expected[k] for k in ("admit","replace","reject","archive_final_size")) for x in rows)
            s=float(np.median([x["similarity_seconds"] for x in rows]));a=float(np.median([x["admission_seconds"] for x in rows]))
            similarity.append(s);admission.append(a)
            archive_rows.append(dict(mode=mode,archive_input_size=A,ranks=p,total_seconds=t,
                                     similarity_seconds=s,admission_seconds=a,communication_seconds=c))
        axes[0].loglog(sizes,times,"o-",label=f"{mode} P={p}")
        if p>1:axes[1].loglog(sizes,comm,"o-",label=f"P={p}")
        axes[2].loglog(sizes,similarity,"o-",label=f"{mode} P={p}")
        axes[3].loglog(sizes,admission,"o-",label=f"{mode} P={p}")
    axes[0].set(xlabel="Archive records A",ylabel="128-candidate seconds");axes[1].set(xlabel="Archive records A",ylabel="MAXLOC communication seconds")
    axes[2].set(xlabel="Archive records A",ylabel="Similarity seconds")
    axes[3].set(xlabel="Archive records A",ylabel="Sequential decision seconds")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3)
    save(fig,out,"archive_scaling");table(out/"archive_scaling.csv",archive_rows)

    gpu=measured(root/"gpu_two_01",r"strong_gpu_p(\d+)_r([123])");assert set(gpu)=={("1",),("2",)}
    values=[strong[("date",72)]["total_seconds"],gpu[("1",)]["total_seconds"],gpu[("2",)]["total_seconds"]]
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(["72 CPU cores","1 A40","2 A40"],values);ax.set(yscale="log",ylabel="End-to-end seconds, 9388 factors")
    save(fig,out,"cpu_gpu_end_to_end")
    kernel_rows=[]
    for dtype in ("float","double"):
        for B in (8,32,64):
            repeated=[load(root/f"gpu_roofline_01/{dtype}_B{B}_r{repeat}.json") for repeat in (1,2,3)]
            assert all((x["B"],x["m"],x["n"],x["k"],x["A"])==(B,293,5876,4,16384) for x in repeated)
            kernel_rows.append(dict(iteration_precision=dtype,B=B,
                **{field:float(np.median([x[field] for x in repeated])) for field in
                   ("ranking_seconds","gemv_seconds","iteration_seconds","similarity_seconds",
                    "gemv_effective_bytes_per_second","gemv_flops_per_second","similarity_flops_per_second")}))
    table(out/"gpu_kernel_times.csv",kernel_rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,field,title in zip(axes,("ranking_seconds","gemv_seconds","similarity_seconds"),
                              ("Percentile ranking (FP32)","GEMV (iteration precision)","Similarity (FP32)")):
        for dtype in ("float","double"):
            rows=[x for x in kernel_rows if x["iteration_precision"]==dtype]
            ax.loglog([x["B"] for x in rows],[x[field] for x in rows],"o-",label=f"{dtype} iteration run")
        ax.set(title=title,xlabel="Batch factors B",ylabel="Kernel seconds");ax.legend(fontsize=8);ax.grid(alpha=.3)
    save(fig,out,"gpu_kernel_times")
    cublas_rows=[]
    for path in sorted((root/args.cublas_reference).glob("B*_A*.json")):
        data=load(path)
        assert data["passed"] and not data["production_uses_cublas"] and data["dimension"]==293
        assert data["compute_mode"]=="CUBLAS_COMPUTE_32F_PEDANTIC" and data["math_mode"]=="CUBLAS_PEDANTIC_MATH"
        repeated=[x for x in data["repetitions"] if x["repeat"] in (1,2,3)]
        assert len(repeated)==3 and {x["repeat"] for x in repeated}=={1,2,3}
        assert max(data[k] for k in ("max_abs_hand_vs_cublas","sampled_fp64_hand_error","sampled_fp64_cublas_error"))<=1e-5
        row=dict(B=data["B"],A=data["A"],dimension=data["dimension"],
                 **{field:float(np.median([x[field] for x in repeated])) for field in
                    ("hand_similarity_seconds","cublas_gemm_seconds","cublas_similarity_seconds")})
        row["cublas_full_semantics_speedup"]=row["hand_similarity_seconds"]/row["cublas_similarity_seconds"]
        row["max_abs_hand_vs_cublas"]=data["max_abs_hand_vs_cublas"]
        row["sampled_fp64_hand_error"]=data["sampled_fp64_hand_error"]
        row["sampled_fp64_cublas_error"]=data["sampled_fp64_cublas_error"]
        cublas_rows.append(row)
    assert {(x["B"],x["A"]) for x in cublas_rows}=={(8,16384),(32,16384),(64,16384),(128,50000)}
    table(out/"cublas_comparison.csv",cublas_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4));positions=np.arange(len(cublas_rows))
    for offset,field,label in ((-.25,"hand_similarity_seconds","hand GEMM+abs"),(0,"cublas_gemm_seconds","cuBLAS GEMM only"),(.25,"cublas_similarity_seconds","cuBLAS GEMM+abs")):
        axes[0].bar(positions+offset,[x[field] for x in cublas_rows],width=.25,label=label)
    axes[0].set(yscale="log",ylabel="Resident device operation seconds");axes[0].legend(fontsize=8)
    axes[1].bar(positions,[x["cublas_full_semantics_speedup"] for x in cublas_rows]);axes[1].axhline(1,color="black",linestyle="--")
    axes[1].set(ylabel="Hand / cuBLAS+abs time ratio")
    for ax in axes:ax.set_xticks(positions,[f"B={x['B']}\nA={x['A']}" for x in cublas_rows]);ax.grid(axis="y",alpha=.3)
    save(fig,out,"cublas_comparison")
    precision=load(root/args.precision/"SUCCESS.json");precision_rows=[]
    for dtype in ("float","double"):
        v=load(root/args.precision/f"full_{dtype}/validation.json");m=load(root/args.precision/f"full_{dtype}/metrics.json");assert v["passed"]
        precision_rows.append(dict(precision=dtype,max_cosine_error=v["max_one_minus_abs_cosine_vs_power"],total_seconds=m["total_seconds"],gpu_iteration_seconds=m["gpu_iteration_seconds"]))
    table(out/"gpu_precision.csv",precision_rows)

    metrics=np.load(root/"study_full_01/metrics.npy",mmap_mode="r");saved=np.load(root/"study_full_01/signatures.npy",mmap_mode="r")
    cost=[];matched=[];iterator=measured(root/"iteration_cost_01",r"(power|lanczos)_k(\d+)_r([123])");assert len(iterator)==20
    for method,name in enumerate(("power","lanczos")):
        error=np.asarray(metrics[:,0,method,:,0]);hit=error<=1e-6;success=hit.any(axis=1);first=hit.argmax(axis=1);ids=np.where(success)[0]
        cost.append(dict(method=name,target=1e-6,converged=int(success.sum()),not_converged_by_k10=int((~success).sum()),median_steps=float(np.median(first[success]+1)),
            median_matvecs=float(np.median(metrics[ids,0,method,first[ids],3])),median_seconds_including_residual=float(np.median(metrics[ids,0,method,first[ids],4]))))
        eligible=np.where((error[:8]<=1e-6).all(axis=0))[0];assert len(eligible);k=int(eligible[0]+1);r=iterator[(name,str(k))]
        reference=np.asarray(saved[:8,0,method,k-1,:],dtype=float);bound=0.
        for repeat in (1,2,3):
            mpi=np.fromfile(root/f"iteration_cost_01/{name}_k{k}_r{repeat}/signatures.f32",dtype="<f4").reshape(reference.shape).astype(float)
            cosine=np.abs(np.sum(mpi*reference,axis=1)/(np.linalg.norm(mpi,axis=1)*np.linalg.norm(reference,axis=1)))
            angle=np.arccos(np.clip(1-error[:8,k-1]-1e-12,-1,1))+np.arccos(np.clip(cosine-1e-12,-1,1))
            bound=max(bound,float(np.max(1-np.cos(np.minimum(np.pi/2,angle)))))
        assert bound<=1e-6
        matched.append(dict(method=name,k=k,factors=8,ranks=8,solve_seconds=r["solve_seconds"],allreduce_calls=int(r["collective_calls"]),
                            communication_seconds=r["communication_seconds"],logical_payload_bytes=int(r["collective_elements"])*8,max_mpi_svd_error_upper_bound=bound))
    table(out/"accuracy_cost.csv",cost);table(out/"mpi_matched_accuracy.csv",matched)
    table(out/"mpi_iteration_cost.csv",[dict(method=method,k=int(k),solve_seconds=r["solve_seconds"],allreduce_calls=int(r["collective_calls"]),communication_seconds=r["communication_seconds"]) for (method,k),r in iterator.items()])
    sm=np.load(root/"study_synthetic_01/metrics.npy");ratios=np.load(root/"study_synthetic_01/sigma2_over_sigma1.npy");fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,ratio in zip(axes,(.5,.9,.99)):
        select=np.abs(ratios-ratio)<.02;assert select.sum()==8
        for method,name in enumerate(("power","lanczos")):ax.loglog(np.arange(1,11),np.maximum(np.median(sm[select,0,method,:,0],axis=0),1e-16),"o-",label=name)
        ax.set(title=f"sigma2/sigma1 ~ {ratio}",xlabel="Iterations",ylabel="Median angle error");ax.legend();ax.grid(alpha=.3)
    save(fig,out,"controlled_spectrum_convergence")

    selected=[d for d in sensitivity["decision_sensitivity"] if d["seed"]==0 and d["k"] in (2,4,8)];table(out/"selected_qd_sensitivity.csv",selected)
    manifest=load(root/"real_input/manifest.json")
    candidates=[row for row in manifest["records"] if row["candidate"]]
    final_keys=load(root/"scaling_1node_01/strong_date_p1_t1_r0/archive_keys.json")
    qualities={row["key"]:row["quality"] for row in candidates}
    assert len(candidates)==provided["candidates"] and all(key in qualities for key in final_keys)
    populations={"valid proposals":np.asarray([row["quality"] for row in candidates]),
                 "power4 QD archive":np.asarray([qualities[key] for key in final_keys])}
    rankic_rows=[];fig,ax=plt.subplots(figsize=(7,4))
    bins=np.linspace(0,max(float(x.max()) for x in populations.values())*1.01,41)
    for population,x in populations.items():
        assert np.isfinite(x).all() and np.all(x>=0)
        q=np.quantile(x,[0,.25,.5,.75,1])
        rankic_rows.append(dict(population=population,count=len(x),minimum=float(q[0]),p25=float(q[1]),
                               median=float(q[2]),p75=float(q[3]),maximum=float(q[4])))
        ax.hist(x,bins=bins,density=True,histtype="step",label=f"{population} (n={len(x)})")
    ax.set(xlabel="In-sample absolute RankIC",ylabel="Density");ax.legend();ax.grid(alpha=.3)
    save(fig,out,"rankic_distributions");table(out/"rankic_distributions.csv",rankic_rows)
    write_pdf_sections(out,dict(sensitivity=sensitivity,gpu_times=values,precision=precision,provided=provided,
        kernels=kernel_rows,cublas_rows=cublas_rows,cost=cost,matched=matched,selected=selected,rankic_rows=rankic_rows,
        power4_not_converged=int(np.sum(metrics[:,0,0,3,0]>1e-6))))
    strong_rows=[]
    for layout in ("date","transpose","factor"):
        for p in (1,72,144,288):
            r=strong[(layout,p)];speed=strong[(layout,1)]["total_seconds"]/r["total_seconds"]
            strong_rows.append(dict(layout=layout,ranks=p,seconds=r["total_seconds"],speedup=speed,efficiency=speed/p))
    sections=["\n## P0：正确性证据\n",
        f"真实交接有 9388 个非恒定 R、8866 个有效且签名可比较的候选。full_validation_01 验证 144 核、混合、单/双 GPU；{args.scaling_validation} 验证单节点各配置及 288 核。核对角度误差、逐候选动作/最近 key/替换 key、最终成员和三次重复。factor/transpose 还要求同输入串行签名逐位一致。",
        f"{args.adversarial} 覆盖 0.90 附近、最大值近似并列、同 key、质量差恰好 0.002、无效签名，CPU/CUDA/MPI 分别与 Python 比较。{args.precision} 用 257 列验证新排名的 double 转置，避免只测试二进制分母。GPU API 另验证不填充、不去均值的百分位排名保留 NaN，供 RankIC 复用。",
        f"另外对交接提供的整代 signatures.f32 独立重放，CPU/CUDA/分片 MPI 与 Python 决定一致。与从 FP32 R 重算签名相比，动作变化 {provided['provided_vs_exported_R_changed_actions']}，三元组变化 {provided['provided_vs_exported_R_changed_tuples']}，最终成员对称差 {provided['provided_vs_exported_R_archive_key_difference']}；签名最大角度误差 {provided['max_signature_cosine_error_from_R_export']:.3g}。这是导出精度差异，不能直接当作并行实现错误。",
        "\n## P1：串行规模\n![串行规模](serial_problem_size.png)\n",
        f"19 组 B/m/n/k 合成规模实验，预热一次、三次中位数。m=ceil(1461/sample_step)，并非对同一真实因子重新排名。k 的工作量为 2k+1 次矩阵遍历。拟合斜率：{json.dumps(slopes)}。",
        "\n## P2：收敛、精度与准入\n![谱比](spectral_ratios.png)\n![收敛](convergence.png)\n![受控谱](controlled_spectrum_convergence.png)\n![准入变化](decision_sensitivity.png)\n",
        "目标角度误差 1e-6，seed=0，未在 k<=10 达标者单列，不能假装全部收敛：\n"+mdtable(cost,["method","converged","not_converged_by_k10","median_steps","median_matvecs","median_seconds_including_residual"]),
        "时间来自并发数值研究任务，含 C API 复制与两次额外残差矩阵乘，不等同于独立单核基准。前 8 个真实因子、8 ranks 的同精度实测如下：\n"+mdtable(matched,["method","k","solve_seconds","allreduce_calls","communication_seconds","logical_payload_bytes"]),
        "MPI 签名与已验证串行签名比较，再用角度三角不等式和 1e-12 舍入余量确认 SVD 误差上界。power 是批量归约，Lanczos 逐因子且两遍重正交化；调用次数、字节数和时间必须一起看，不能把差异全归因于算法。",
        "相对 power k=4 seed=0：动作变化与最近 key 变化不同，最终成员使用对称差：\n"+mdtable(selected,["method","k","changed_actions","changed_closest_keys","archive_key_symmetric_difference"]),
        "60 条 seed/方法/k 轨迹均保留。四次迭代是预算选择，不是所有因子已收敛的保证。受控合成使用四个解析正交 DCT 方向，奇异值 (1,ratio,0.2,0.1)，ratio=0.5/0.9/0.99，加 seed=2026、标准差 1e-8 的高斯噪声；按噪声后精确谱比复核分组。其近似低秩性质不代表真实因子也四步收敛。",
        "\n## P3/P5：强扩展与 MPI+OpenMP\n![强扩展](strong_scaling.png)\n![混合](hybrid_comparison.png)\n![通信与顺序部分](communication_and_sequential_fraction.png)\n"+mdtable(strong_rows,["layout","ranks","seconds","speedup","efficiency"]),
        "288 核使用四个不同节点，每节点 72 核。date 的 batch*n 归约量不随 P 缩小；transpose 包括重分布，factor 是直接按完整因子读文件的独立场景。默认 rank 0 顺序准入构成下界。",
        "communication_seconds 只计 Allreduce/Alltoallv，factor 仍有单列的 Gatherv。通信是 solve/transpose 的子集，各阶段取不同 rank 最大值，不可相加代替总时间。admission_seconds 不含粗算前的单线程准备，不能当作全部串行比例。",
        "真实输入已经是 R，排名耗时为零，不把外部预计算隐含计入。另以 9216 张合成数值面板为原始抽样值，实际排名后补测 72/144/288 核，并捕获节点 lscpu：\n![含排名流水线](raw_ranking_pipeline.png)",
        "排序会改变合成矩阵的谱，不套用受控谱的 sigma 比。CPU 新排名是 double，transpose 用 double 保持精度，与现成 FP32 R 的字节量不同。",
        "\n## P4：弱扩展\n![弱扩展](weak_scaling.png)\n每 rank 固定 32 张完整矩阵等价工作量，B=32P，m=293，n=5876，至 P=288，输入为同一合成序列前缀。读取、批次数增长、Allreduce、日期块不均以及全局档案增长均影响端到端效率。",
        "\n## P5：GPU\n![CPU/GPU](cpu_gpu_end_to_end.png)\n![Kernel 时间](gpu_kernel_times.png)\n![Roofline](gpu_roofline.png)\n",
        "三个 kernel 各自计时，B=8/32/64，预热一次、三次中位数；相似度档案 A=16384。float/double 只指幂迭代的矩阵和向量精度，两个配置的排名和粗算相似度本身均为 FP32。相似度仍通过 CPU FP64 复核保证最终决定。完整计时与吞吐在 gpu_kernel_times.csv。\n"+
            mdtable(kernel_rows,["iteration_precision","B","ranking_seconds","gemv_seconds","similarity_seconds"]),
        f"同一批 9388 个 R：72 核 CPU date={values[0]:.3f}s，单 A40={values[1]:.3f}s，双 A40={values[2]:.3f}s；双卡加速比 {values[1]/values[2]:.2f}。显存范数和 Rᵀu 直接传给 MPI_Allreduce，运行时核对 CUDA-aware 支持。",
        f"整代 FP32/FP64 动作变化={precision['fp32_vs_fp64_changed_actions']}，三元组变化={precision['fp32_vs_fp64_changed_decision_tuples']}，成员对称差={precision['archive_key_symmetric_difference']}。两种精度各自与 CPU 签名和 Python 准入比较，误差见 gpu_precision.csv。",
        "gpu_precision.csv 的时间来自单次正确性运行，受文件缓存和数据准备影响，不用于宣称 FP32/FP64 的加速比。精度间的 kernel 成本采用上面的重复测量。",
        "cuBLAS 仅作独立参照，生产程序的动态依赖检查确认不链接 cuBLAS。输入由固定整数混合初值 20261111/987654321 生成均匀行，CPU 中心化、单位化后转 FP32，并放入相同/相反两对向量检查 abs。对照复用 gpu.cu 的同一个 kernel，输入预先在显存上就绪，计时排除传输、分配和归一化；每个计时块 50 次，预热块不计、三块取中位数，交替执行顺序。手写 kernel 融合 abs，因此主要比较 cuBLAS GEMM+单独 abs 的完整等价操作，纯 GEMM 另列。两者全矩阵最大差另加 512 对 FP64 抽查，不声称 FP64 全矩阵校验。采用 CUBLAS_COMPUTE_32F_PEDANTIC/CUBLAS_PEDANTIC_MATH，关闭 TF32，不代表库的所有最高吞吐模式；精度模式见 [NVIDIA cuBLAS 12.9 文档](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cublascomputetype-t)。\n![cuBLAS 独立参照](cublas_comparison.png)\n"+
            mdtable(cublas_rows,["B","A","hand_similarity_seconds","cublas_gemm_seconds","cublas_similarity_seconds","cublas_full_semantics_speedup"]),
        "单卡的读取、类型转换、传输、归一化和顺序准入计入端到端时间。快速 kernel 不自动等于最快流水线。Roofline 为名义流量估计而非 profiler DRAM 流量；FP32 ceiling 不套用 FP64。排序独立报告耗时，不套 GEMV FLOP 模型。",
        "\n## P6：档案规模\n![档案规模](archive_scaling.png)\nA=1000/5000/10000/50000，每块 128 候选，集中 CPU 对比分片 4/16/72 ranks，固定整数混合输入，初始化不计时。分别给出总时间、相似度、顺序决定和通信，各阶段最大值不直接相加。key/quality 复制、签名按 FNV hash 分片。每候选 MAXLOC 和构建 key 顺序的重复元数据工作都会限制扩展，并非所有损失来自网络。",
        "\n## 与 report/main_zh.tex 合并\n按原 PDF 4.3、4.4、6.3、6.4、7 及附录组织的补稿见 [pdf_sections.md](pdf_sections.md)。"
        "提供本批随机提案与最终 QD 档案的样本内 RankIC 分布，不能替代原稿 1481 个进化因子的分布或样本外结果：\n"
        "![本批 RankIC](rankic_distributions.png)\n"+mdtable(rankic_rows,["population","count","median","p25","p75"]),
        "\n## 复现与边界\n源码、构建命令和实验脚本见 README/WALKTHROUGH。数据、公式、日志、CSV 和图只放在忽略的 runs，不修改共享输入。流水线从抽样因子值或 R 到签名和档案结束，不改写 GP 选择/变异或交易回测。真实提案的基因池随机生成，不能据此声称投资有效。AI 使用声明：使用 OpenAI Codex 协助实现、编排和初稿；数值均来自实际运行。"]
    with (out/"report.md").open("a") as handle:handle.write("\n\n".join(sections)+"\n")
    (out/"EVIDENCE.json").write_text(json.dumps(dict(completed=not bool(args.real_size),source_runs=str(root),numerical_study="study_full_01",qd_replay="sensitivity_full_01",
        strong_configurations=len(strong),hybrid_configurations=len(hybrid),weak_configurations=len(weak),raw_configurations=len(raw_rows),
        serial_configurations=len(serial),archive_configurations=len(archive),precision_run=args.precision,adversarial_run=args.adversarial,
        scaling_validation=args.scaling_validation,synthetic_validation=args.synthetic_validation,
        paper_sections="pdf_sections.md",paper_structure_reference="report/main_zh.tex",cublas_reference_run=args.cublas_reference),indent=2)+"\n")
    if args.real_size:
        from completion_report import append_completion
        append_completion(out,args.real_size,args.matched_full)
    print(f"Full report assembled: {out}")


if __name__=="__main__":main()
