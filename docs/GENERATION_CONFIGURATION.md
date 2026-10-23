# Generation configuration

The maintained situation-conditioned generator is generation/responses.py, exposed by python run.py generate-responses. It uses complete60-item BFI-2 scoring from generation/bfi2.py and config/bfi2_scoring.json; all five domains contain12 items. There is no abbreviated-scoring switch. Questionnaires with a different item count or incomplete baseline response dictionaries are rejected.

Endpoints, credentials and input/output locations are configured through environment variables. Prompts and the15-item state aggregation remain as supplied. Chinese comments are excluded. Baseline assessment in generation/traits.py uses the same scoring module.

Offline tests exercise response parsing and the actual generator-to-scorer connection. They do not make model requests or regenerate archived data. See BFI2_SCORING.md for scoring-key provenance and item-level rescoring.
