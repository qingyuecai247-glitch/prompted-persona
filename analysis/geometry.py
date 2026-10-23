"""Person-cluster inference for Figure 2 centroids, conditional on the 62 situations and fitted human PCA."""
from pathlib import Path
import argparse,ast,hashlib,json,os,platform,time
import numpy as np
import pandas as pd

REPO=next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
TRAITS=['E','A','C','N','O']


def load_inputs(root):
    maps=json.loads((REPO/'config/figure2_datasets.json').read_text())
    frames={};audit=[]
    for source,name in {'Human':'data_long.txt',**maps['DATASETS_RAW']}.items():
        path=root/name if source=='Human' else root/'llm_output'/name
        try:df=pd.read_csv(path,sep=r'\s+' if source=='Human' else ',',encoding='utf-8')
        except UnicodeDecodeError:df=pd.read_csv(path,encoding='gbk')
        if source=='Human':df=df.rename(columns={f'State_{t}_sum':f'State_{t}' for t in TRAITS})
        cols=[f'State_{t}' for t in TRAITS]
        df[cols]=df[cols].apply(pd.to_numeric,errors='coerce')
        keep=np.isfinite(df[cols].to_numpy()).all(axis=1)&df.id_p.notna()&df.id_s.notna()
        clean=df.loc[keep,['id_p','id_s',*cols]].copy()
        if clean.duplicated(['id_p','id_s']).any():raise ValueError(f'{source}: duplicate person-situation keys')
        frames[source]=clean
        audit.append(dict(source=source,file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            input_rows=len(df),complete_rows=len(clean),dropped_rows=int((~keep).sum()),
            people=int(clean.id_p.nunique()),situations=int(clean.id_s.nunique()),
            min_rows_per_person=int(clean.groupby('id_p').size().min()),max_rows_per_person=int(clean.groupby('id_p').size().max())))
    return frames,maps,audit


def cluster_summary(x,ids):
    codes,labels=pd.factorize(np.asarray(ids),sort=True);g=len(labels)
    counts=np.bincount(codes).astype(float)
    sums=np.column_stack([np.bincount(codes,weights=x[:,j]) for j in range(2)])
    mean=sums.sum(axis=0)/counts.sum()
    score=sums-counts[:,None]*mean
    covariance=(g/(g-1))*(score.T@score)/(counts.sum()**2)
    return mean,covariance,sums,counts


def boot_means(sums,counts,reps,rng,batch=1000):
    g=len(counts);result=np.empty((reps,2))
    # Sampling cluster indices with replacement is equivalent to keeping every
    # original row of a sampled person; sum/count retains the original record-weighted estimand.
    for start in range(0,reps,batch):
        n=min(batch,reps-start);indices=rng.integers(g,size=(n,g))
        denominator=counts[indices].sum(axis=1)
        for j in range(2):result[start:start+n,j]=sums[:,j][indices].sum(axis=1)/denominator
    return result


def holm_log(logp):
    logp=np.asarray(logp);order=np.argsort(logp);out=np.empty(len(logp));running=-np.inf
    for rank,index in enumerate(order):
        running=max(running,logp[index]+np.log(len(logp)-rank))
        out[index]=min(0.,running)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps',type=int,default=199999)
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('PERSONA_DATA_ROOT',REPO/'data/raw')))
    parser.add_argument('--output-dir',type=Path,default=Path(os.environ.get('PERSONA_STAT_OUTPUT',REPO/'outputs/statistics'))/'figure2_cluster')
    args=parser.parse_args()
    if args.reps<999:parser.error('At least 999 resamples are required.')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);started=time.time()
    frames,maps,audit=load_inputs(args.data_root)
    columns=[f'State_{t}' for t in TRAITS];human=frames['Human'][columns].to_numpy()
    mean=human.mean(axis=0);sd=human.std(axis=0,ddof=0);z=(human-mean)/sd
    center=z.mean(axis=0);_,singular,vt=np.linalg.svd(z-center,full_matrices=False);basis=vt[:2].T
    # Fix SVD signs for portable reporting of signed coordinate differences.
    for j in range(2):
        if basis[np.abs(basis[:,j]).argmax(),j]<0:basis[:,j]*=-1
    projected={key:((df[columns].to_numpy()-mean)/sd-center)@basis for key,df in frames.items()}
    summaries={key:cluster_summary(projected[key],df.id_p) for key,df in frames.items()}
    seeds=np.random.SeedSequence(args.seed).spawn(len(frames))
    hm,hcov,hsum,hcount=summaries['Human']
    human_boot=boot_means(hsum,hcount,args.reps,np.random.default_rng(seeds[0]))
    print(f'Human: {len(hcount)} clusters, {len(human)} rows; {args.reps} cluster resamples complete.',flush=True)
    results=[]
    for index,(key,df) in enumerate(list(frames.items())[1:],1):
        mu,cov,sums,counts=summaries[key];delta=mu-hm;variance=cov+hcov
        eigen=np.linalg.eigvalsh(variance)
        if eigen.min()<=0:raise ValueError(f'{key}: nonpositive centroid covariance')
        inverse=np.linalg.inv(variance);wald=float(delta@inverse@delta)
        boot=boot_means(sums,counts,args.reps,np.random.default_rng(seeds[index]))-human_boot
        errors=boot-delta
        null_wald=np.einsum('bi,ij,bj->b',errors,inverse,errors)
        exceeds=int(np.count_nonzero(null_wald>=wald));bootstrap_p=(1+exceeds)/(args.reps+1)
        q95=float(np.quantile(null_wald,.95));lo,hi=np.quantile(boot,[.025,.975],axis=0)
        distances=np.linalg.norm(boot,axis=1);dist_ci=np.quantile(distances,[.025,.975])
        empirical_cov=np.cov(boot,rowvar=False)
        row=dict(model=key,display=maps['MODEL_DISPLAY_NAMES'][key],n_human_people=len(hcount),n_model_people=len(counts),
            n_human_records=len(human),n_model_records=len(df),centroid_distance=float(np.linalg.norm(delta)),
            delta_pc1=float(delta[0]),delta_pc2=float(delta[1]),pc1_ci_low=lo[0],pc1_ci_high=hi[0],
            pc2_ci_low=lo[1],pc2_ci_high=hi[1],distance_ci_low=dist_ci[0],distance_ci_high=dist_ci[1],
            cluster_wald=wald,wald_df=2,wald_log_p=-wald/2,wald_log10_p=-wald/2/np.log(10),
            bootstrap_reps=args.reps,bootstrap_exceedances=exceeds,bootstrap_p=bootstrap_p,
            joint95_critical=q95,joint95_excludes_zero=bool(wald>q95),
            covariance_11=variance[0,0],covariance_12=variance[0,1],covariance_22=variance[1,1],
            bootstrap_covariance_11=empirical_cov[0,0],bootstrap_covariance_12=empirical_cov[0,1],bootstrap_covariance_22=empirical_cov[1,1])
        results.append(row)
        print(f'{key}: distance={row["centroid_distance"]:.4f}, W={wald:.2f}, bootstrap exceedances={exceeds}/{args.reps}, joint95 excludes zero={wald>q95}',flush=True)
    result=pd.DataFrame(results)
    result['wald_log_p_holm']=holm_log(result.wald_log_p)
    result['wald_log10_p_holm']=result.wald_log_p_holm/np.log(10)
    result['wald_p']=np.exp(result.wald_log_p);result['wald_p_holm']=np.exp(result.wald_log_p_holm)
    result['bootstrap_p_holm']=np.exp(holm_log(np.log(result.bootstrap_p)))
    result['bootstrap_p_resolution']=1/(args.reps+1)
    # Pointwise 95% upper bounds on Monte Carlo tail probabilities for zero exceedances.
    result['bootstrap_tail_upper95_if_zero']=np.where(result.bootstrap_exceedances.eq(0),-np.expm1(np.log(.05)/args.reps),np.nan)
    result.to_csv(out/'figure2_cluster_centroid_tests.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False)
    pd.DataFrame(basis,index=TRAITS,columns=['PC1','PC2']).to_csv(out/'human_pca_loadings.csv')
    manifest=dict(seed=args.seed,bootstrap_reps=args.reps,bootstrap_resolution=1/(args.reps+1),
        test='two-dimensional unequal-covariance person-cluster sandwich Wald test; asymptotic chi-square(2)',
        bootstrap_test='independent whole-person resampling within each source; centered centroid errors; fixed sandwich covariance quadratic form; plus-one empirical upper-tail P',
        bootstrap_ci='pointwise percentile coordinate and distance intervals; joint confidence ellipse calibrated by 95th percentile of centered bootstrap quadratic form',
        estimand='record-weighted source mean difference in fixed human-fitted PC1-PC2 space',
        situations='conditional on the observed 62 situations; situations are not resampled',
        weighting='unequal human cluster sizes retained through ratio of resampled cluster sums to counts',
        fixed_reference='human standardization and PCA held fixed across resamples; not a full-pipeline bootstrap',
        human_bootstrap='same independently drawn human bootstrap used across model contrasts; Holm valid under arbitrary dependence',
        covariance='G/(G-1) sum_g (sum_{i in g}(z_i-zbar))(same)^T / N^2 within each independent source; sum source covariances for contrast',
        multiplicity='Holm across 11 contrasts separately for Wald and bootstrap P values; confidence intervals are pointwise, not simultaneous',
        primary_p_value='asymptotic cluster-Wald P; centered cluster-bootstrap P is a sensitivity analysis',
        original_hotelling='original record-level pooled-covariance Hotelling test not overwritten or relabelled',
        sign_rule='largest-magnitude loading positive for each PC; signs do not affect distances or joint tests',
        explained_variance=(singular**2/(singular**2).sum()).tolist(),runtime_seconds=time.time()-started,
        versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(result[['display','centroid_distance','wald_log10_p_holm','bootstrap_p_holm','joint95_excludes_zero']].to_string(index=False),flush=True)
    print('Saved',out,flush=True)

if __name__=='__main__':main()
