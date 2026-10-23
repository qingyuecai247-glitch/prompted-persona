"""Check that reference rendering retains the source estimand and the off-scale value."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
SCRIPT = ROOT / 'figures/figure5.py'

class Figure5Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            spec = importlib.util.spec_from_file_location('figure5', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f'Plotting dependency missing: {exc}')

    def test_reference_data_preserved(self):
        m = self.module
        data = m.compute_relative_variability()
        self.assertEqual(len(data), 280)
        self.assertAlmostEqual(float(data.loc[data.source.eq('gpt52'), 'relative_sd'].max()), 1.96149, places=5)
        contexts = data.groupby(['source','context']).relative_sd.median().unstack().median().sort_values(ascending=False)
        self.assertEqual(contexts.index.tolist(), ['pos','int','neg','soc','mat','dut','adv','dec'])
        self.assertEqual([round(x, 2) for x in contexts], [.25,.20,.20,.17,.12,.12,.09,.07])

    def test_outlier_annotation_and_unclipped_source(self):
        m = self.module
        data = m.compute_relative_variability()
        fig, ax = m.plt.subplots()
        order = ['deepseek','grok3','gpt52','qwen32','phimedium','gpt4omini','llama']
        try:
            m.draw_panel_a(ax, data, order)
            self.assertEqual(ax.get_ylim(), (0., 1.2))
            self.assertIn('1.96', [item.get_text() for item in ax.texts])
            self.assertGreater(data.relative_sd.max(), 1.9)
        finally:
            m.plt.close(fig)

if __name__ == '__main__':
    unittest.main()
