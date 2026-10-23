"""Recompute formal v7 Figure 6d statistics from the three retained server CSV shards."""
from pathlib import Path
import os, subprocess, sys
ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
reference=Path(os.environ.get('PERSONA_REFERENCE_ROOT',ROOT/'data/reference'))
output=Path(os.environ.get('PERSONA_STAT_OUTPUT',ROOT/'outputs/statistics'))/'figure6_v7'
analyzer=ROOT/'sae/interventions.py'
shards=[reference/'sae_server_outputs/intervention'/f'intervention_results_long_v7_confirmatory_greedy_last4_doses-8_shard{i:02d}.csv' for i in range(3)]
for path in [analyzer,*shards]:
    if not path.is_file():raise FileNotFoundError(path)
subprocess.run([sys.executable,str(analyzer),*[str(p) for p in shards],'--bootstrap','5000','--seed','20260720','--output-dir',str(output)],check=True)
