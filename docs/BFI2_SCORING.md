# BFI-2 60-item scoring

Added2026-10-23. The domain membership and reverse-scoring key were checked against the official BFI-2 form/key: https://www.colby.edu/wp-content/uploads/2013/08/bfi2-form.pdf . The supplied instrument file follows this numbered60-item order. Code labels N and O correspond to the official Negative Emotionality and Open-Mindedness domains. Questionnaire wording is not redistributed here.

Implementation: generation/bfi2.py. Machine-readable item mapping: config/bfi2_scoring.json. Both generation/traits.py and generation/responses.py use the shared implementation: reverse-keyed scores are6-x; each domain averages all12 items. Numeric means within1–5 and the configured five English response labels are accepted. Missing, unknown, non-finite and out-of-range values raise errors; no neutral-score imputation is performed. The existing traits CSV wrapper retains its two-decimal output rounding.

Rescore a retained CSV with Q1–Q60 columns, preserving the original records:

```sh
python generation/score_bfi2.py --input /path/to/item_responses.csv --output /path/to/rescored.csv
```

Columns BFI2_E/A/C/N/O are appended. Preserve raw responses and scoring-version provenance. If the file contains only five aggregated trait columns, the missing item-level information cannot be reconstructed by this utility. Repeated-generation averaging and item-order verification must be performed according to the actual study design before domain scoring.

This commit adds a scoring implementation; it does not establish which scorer produced previously archived data, and it does not retrospectively regenerate or validate those empirical results.
