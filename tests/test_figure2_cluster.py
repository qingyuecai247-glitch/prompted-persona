"""Cluster covariance and resampling checks independent of the paper's effects."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
spec=importlib.util.spec_from_file_location('cluster',ROOT/'analysis/geometry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ClusterInference(unittest.TestCase):
    def test_equal_cluster_sizes_reduce_to_covariance_of_person_means(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(70,2));ids=np.repeat(np.arange(10),7)
        mean,cov,sums,counts=m.cluster_summary(x,ids)
        np.testing.assert_allclose(mean,x.mean(axis=0))
        np.testing.assert_allclose(cov,np.cov(x.reshape(10,7,2).mean(axis=1),rowvar=False)/10)

    def test_unequal_cluster_size_keeps_record_weighting(self):
        x=np.array([[0.,0.],[10.,4.],[10.,4.]])
        mean,cov,sums,counts=m.cluster_summary(x,['a','b','b'])
        np.testing.assert_allclose(mean,[20/3,8/3])
        boot=m.boot_means(sums,counts,20000,np.random.default_rng(5))
        outcomes=np.array([[0.,0.],[20/3,8/3],[10.,4.]])
        self.assertTrue(all(np.min(np.linalg.norm(outcomes-row,axis=1))<1e-12 for row in boot))

    def test_cluster_se_is_invariant_to_duplicating_rows_within_people(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(60,2));ids=np.repeat(np.arange(12),5)
        _,v,_,_=m.cluster_summary(x,ids)
        _,duplicated,_,_=m.cluster_summary(np.repeat(x,3,axis=0),np.repeat(ids,3))
        np.testing.assert_allclose(v,duplicated,rtol=1e-12,atol=1e-12)

    def test_log_holm_matches_known_correction(self):
        np.testing.assert_allclose(np.exp(m.holm_log(np.log([.03,.01,.04]))),[.06,.03,.06])

if __name__=='__main__':unittest.main()
