"""Synthetic demonstration of person-cluster mean covariance; no study data."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
spec=importlib.util.spec_from_file_location('cluster',ROOT/'analysis/geometry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
df=pd.read_csv(ROOT/'examples/synthetic_cluster_profiles.csv')
x=df[['coordinate1','coordinate2']].to_numpy()
mu,cov,sums,counts=m.cluster_summary(x,df.synthetic_person)
boot=m.boot_means(sums,counts,1999,np.random.default_rng(20261022))
out=ROOT/'outputs/demo_cluster';out.mkdir(parents=True,exist_ok=True)
result=dict(synthetic=True,people=len(counts),records=len(df),mean=mu.tolist(),cluster_covariance=cov.tolist(),coordinate_percentile95=np.quantile(boot,[.025,.975],axis=0).tolist(),bootstrap_reps=1999)
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
