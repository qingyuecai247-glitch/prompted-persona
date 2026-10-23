"""Checks for full-instrument BFI-2 coverage and reverse scoring."""
from pathlib import Path
import csv
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from generation.bfi2 import DOMAIN_ITEMS, REVERSE_ITEMS, score_bfi2

class BFI2Scoring(unittest.TestCase):
    def test_all_sixty_items_used_once_with_twelve_per_domain(self):
        flattened = [item for items in DOMAIN_ITEMS.values() for item in items]
        self.assertEqual(sorted(flattened), list(range(1, 61)))
        self.assertTrue(all(len(items) == 12 for items in DOMAIN_ITEMS.values()))

    def test_fully_high_keyed_profile_scores_five_in_every_domain(self):
        answers = {f"Q{i}": 1 if i in REVERSE_ITEMS else 5 for i in range(1, 61)}
        self.assertEqual(score_bfi2(answers), dict.fromkeys(["E", "A", "C", "N", "O"], 5.0))

    def test_late_forward_and_reverse_items_affect_their_domains(self):
        for item, domain, expected in [(60, "O", 3 + 2/12), (59, "N", 3 + 2/12), (58, "C", 3 - 2/12), (36, "E", 3 - 2/12), (57, "A", 3 + 2/12)]:
            answers = {f"Q{i}": 3 for i in range(1, 61)}
            answers[f"Q{item}"] = 5
            self.assertAlmostEqual(score_bfi2(answers)[domain], expected)

    def test_missing_out_of_range_and_unknown_labels_are_rejected(self):
        for bad in [None, 6, float("nan"), "unknown"]:
            answers = {f"Q{i}": 3 for i in range(1, 61)}
            answers["Q60"] = bad
            with self.assertRaises(ValueError):
                score_bfi2(answers)

    def test_csv_rescoring_retains_original_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"responses.csv"
            output = Path(tmp)/"scored.csv"
            fields = ["synthetic_id"] + [f"Q{i}" for i in range(1, 61)]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"synthetic_id":"example", **{f"Q{i}":3 for i in range(1, 61)}})
            before = source.read_bytes()
            subprocess.run([sys.executable, str(ROOT/"generation/score_bfi2.py"), "--input", str(source), "--output", str(output)], check=True, capture_output=True)
            self.assertEqual(source.read_bytes(), before)
            with output.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["BFI2_O"]), 3.0)
            self.assertEqual(row["Q60"], "3")

if __name__ == "__main__":
    unittest.main()
