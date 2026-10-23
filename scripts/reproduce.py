#!/usr/bin/env python3
"""Explicit, offline reproduction entry points; never launches model generation."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'figure2': '03_behavior_analysis/figures/main/joint_distribution_figure2.py',
    'figure2-cluster': '03_behavior_analysis/statistics/geometry/cluster_figure2_centroids.py',
    'figure3': '03_behavior_analysis/figures/main/figure3.py',
    'figure5': '03_behavior_analysis/figures/main/random_slope_figure5.py',
    'figure6': '03_behavior_analysis/figures/main/figure6_polished.py',
    'figure6-v7-statistics': '03_behavior_analysis/statistics/recompute_figure6_v7.py',
}


def check():
    problems = []
    files = list(ROOT.rglob('*.py'))
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            problems.append(f'{path.relative_to(ROOT)}: {exc}')
    cfg = json.loads((ROOT / 'config/paths.json').read_text())
    for key in ('data_root', 'reference_root'):
        if not (ROOT / cfg[key]).is_dir():
            problems.append(f'Missing {key}: {cfg[key]}')
    inventory = {name: importlib.util.find_spec(name) is not None for name in
                 ['numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'seaborn', 'statsmodels']}
    result = {'python_files_parsed': len(files), 'syntax_and_paths_ok': not problems,
              'problems': problems, 'available_python_modules': inventory,
              'Rscript_available': shutil.which('Rscript') is not None,
              'scope': 'Syntax and path checks only; no statistical validation.'}
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/check_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return bool(problems)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['check', 'figure3-export', *TASKS])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.task == 'check':
        return check()
    cfg = json.loads((ROOT / 'config/paths.json').read_text())
    env = os.environ.copy()
    data = (args.data_root or ROOT / cfg['data_root']).resolve()
    reference = (args.reference_root or ROOT / cfg['reference_root']).resolve()
    out = (args.output_root or ROOT / cfg['output_root']).resolve()
    env.update(PERSONA_DATA_ROOT=str(data), PERSONA_REFERENCE_ROOT=str(reference),
               PERSONA_FIGURE_OUTPUT=str(out / 'figures'), PERSONA_STAT_OUTPUT=str(out / 'statistics'))
    env.setdefault('SAE_PROJECT_ROOT', str(reference / 'figure6'))
    env.setdefault('SAE_FIGURE_OUTPUT_DIR', str(out / 'figures'))
    if args.task == 'figure3-export':
        node = shutil.which('node')
        if not node:
            parser.error('Node.js is required. Install Node and run npm install in the repository.')
        return subprocess.run([node, str(ROOT / 'scripts/export_figure3.cjs'), str(out / 'figures')], env=env, cwd=ROOT).returncode
    return subprocess.run([sys.executable, str(ROOT / TASKS[args.task])], env=env, cwd=ROOT).returncode

if __name__ == '__main__':
    raise SystemExit(main())
