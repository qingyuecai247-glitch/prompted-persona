"""Offline regression checks against archived numerical outputs, with isolated output roots."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/reproduce.py'

class ReproductionRegression(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'data/reference/analysis/results/factor_modularity_centered_diamonds_phi_medium/representative_factor_loadings.csv').exists(), 'Restore the separate reference-data archive to run the Figure3 regression.')
    def test_figure3_preserves_human_uncertainty_and_full_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ENTRY), 'figure3', '--output-root', tmp],
                           cwd=tmp, check=True, capture_output=True)
            with (Path(tmp) / 'figures/figure3_compact_source_data.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]['model'], 'human')
            self.assertAlmostEqual(float(rows[0]['FMI']), 0.6166666666666667)
            self.assertEqual((float(rows[0]['ci_low']), float(rows[0]['ci_high'])), (.537, .668))
            svg = (Path(tmp) / 'figures/figure3_compact.svg').read_text()
            self.assertIn('width="183mm" height="121mm"', svg)
            self.assertEqual(svg.count('stroke-dasharray="4 3"'), 1)

if __name__ == '__main__':
    unittest.main()
